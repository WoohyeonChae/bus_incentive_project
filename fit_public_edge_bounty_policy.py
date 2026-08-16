from __future__ import annotations

"""Fit the public hour-by-edge bus bounty decision function.

This is stage 2 of the split pipeline.

Required upstream outputs
-------------------------
1. ``build_model_input_purpose_candidates15.py`` output directory
2. ``fit_route_choice_top5.py`` output directory

The passenger-choice parameters and the final top-5 path set are treated as
fixed inputs.  This script estimates only the global bounty-function
parameters

    score_e = theta0
            + theta_crowding * low_crowding_e
            + theta_altprob * AltProb_e

    reward_e = Rmax * sigmoid(score_e)

where the deployable reward is fixed by hour, route, and directed edge.  The
same hour-edge reward table is used on every calendar date.  Calendar dates are
training/evaluation replications, not policy dimensions.

The objective is total congestion relief over the training dates subject to a
per-day budget constraint. Capacity is not enforced as a hard constraint; the
55-passenger value is retained only as a crowding reference and post-policy QA
threshold.
Cost efficiency is reported but is not the optimization objective.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import expit

EPS = 1.0e-12
SCRIPT_VERSION = "public-edge-bounty-split-v1.3.1"
CACHE_COMPAT_VERSION = "public-edge-bounty-split-v1.2.0"
POLICY_SIMULATION_VERSION = "no-capacity-constraint-v1"
SegmentKey = Tuple[int, str, str, str]


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(round(seconds))
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def normalize_id(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    if text.endswith(".0") and text[:-2].replace("-", "").isdigit():
        text = text[:-2]
    return text


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_signature(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def hash_payload(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def optional_float(value: object, default: float = math.inf) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def load_reference_value(config: Mapping[str, object]) -> float:
    """Return the average-onboard reference used only for features and QA.

    This value is not a feasibility constraint.  The legacy ``capacity`` block
    is accepted so an older configuration can still reuse prepared caches.
    """
    reference_cfg = config.get("load_reference", {})
    legacy_cfg = config.get("capacity", {})
    value = reference_cfg.get(
        "average_onboard", legacy_cfg.get("max_average_load", 55.0)
    )
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(
            "load_reference.average_onboard must be a finite positive number"
        )
    return result


class Progress:
    def __init__(self, label: str, total: Optional[int], interval_seconds: float = 15.0):
        self.label = label
        self.total = int(total) if total is not None else None
        self.interval = max(0.1, float(interval_seconds))
        self.started = time.perf_counter()
        self.last_print = self.started

    def update(self, done: int, extra: str = "", force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last_print < self.interval:
            return
        elapsed = max(now - self.started, EPS)
        rate = done / elapsed
        if self.total is not None and self.total > 0:
            pct = 100.0 * done / self.total
            eta = max(0, self.total - done) / max(rate, EPS)
            text = (
                f"[{self.label}] {done:,}/{self.total:,} ({pct:5.1f}%), "
                f"{rate:,.1f}/s, elapsed {format_seconds(elapsed)}, ETA {format_seconds(eta)}"
            )
        else:
            text = (
                f"[{self.label}] {done:,}, {rate:,.1f}/s, "
                f"elapsed {format_seconds(elapsed)}"
            )
        if extra:
            text += f", {extra}"
        print(text, flush=True)
        self.last_print = now


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ChoiceParameters:
    beta_total_time: float
    beta_walk_time_extra: float
    beta_transfer: float
    beta_discount_per_won: float
    value_of_time_won_per_hour: float


@dataclass(frozen=True)
class PolicyTheta:
    intercept: float
    low_crowding: float
    altprob: float


@dataclass
class HourTopology:
    hour: int
    edge_frame: pd.DataFrame
    path_edge: csr_matrix
    path_board: Optional[csr_matrix]
    group_starts: np.ndarray
    group_lengths: np.ndarray
    group_index: np.ndarray
    walk_time: np.ndarray
    transfers: np.ndarray
    source_rows: np.ndarray
    groups: int
    paths: int
    edge_nnz: int
    board_nnz: int


@dataclass
class DayHourState:
    date: str
    hour: int
    base_utility: np.ndarray
    p0: np.ndarray
    total_time: np.ndarray
    group_demand: np.ndarray
    travel_time: np.ndarray
    trips: np.ndarray
    load0: np.ndarray
    altprob: np.ndarray
    h0: float


@dataclass
class DayPolicyResult:
    date: str
    h0: float
    h1: float
    improvement: float
    cost: float
    efficiency_per_1000: float
    extra_passenger_minutes: float
    rewarded_edges: int
    expected_discounted_passengers: float
    response_scale: float
    max_policy_avg_onboard: float
    overloaded_edges_above_reference: int
    new_overloaded_edges_above_reference: int
    increased_existing_overload_edges: int
    max_load_increase: float
    feasible: bool


# -----------------------------------------------------------------------------
# CLI and configuration
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit fixed hour-by-edge public bus bounty decision function"
    )
    parser.add_argument(
        "--model-input",
        required=True,
        help="build_model_input_purpose_candidates15.py output directory",
    )
    parser.add_argument(
        "--route-choice-output",
        required=True,
        help="fit_route_choice_top5.py output directory",
    )
    parser.add_argument("--output", required=True, help="bounty-policy output directory")
    parser.add_argument("--config", required=True, help="shared optimization config JSON")
    parser.add_argument(
        "--top-paths",
        default="",
        help="override candidate_paths_top5.csv.gz",
    )
    parser.add_argument(
        "--route-choice-parameters",
        default="",
        help="override route_choice_parameters.json",
    )
    parser.add_argument("--policy-trials", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--altprob-group-block",
        type=int,
        default=20_000,
        help="hour-OD groups per sparse AltProb block",
    )
    parser.add_argument("--progress-seconds", type=float, default=15.0)
    parser.add_argument(
        "--rebuild-topology-cache",
        action="store_true",
        help="reparse the top-5 path topology",
    )
    parser.add_argument(
        "--rebuild-daily-cache",
        action="store_true",
        help="rebuild date-hour demand, network, probability, and AltProb arrays",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "build/reuse topology and daily AltProb caches, write diagnostics, "
            "then stop before Bayesian policy fitting"
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete output directory and all reusable caches/studies before running",
    )
    return parser.parse_args()


def parse_dates(model_input: Path) -> List[str]:
    manifest_path = model_input / "model_input_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        values = (
            manifest.get("target_dates", [])
            or manifest.get("used_dates", [])
            or manifest.get("analysis_dates", [])
        )
        dates = sorted(
            {
                normalize_id(value).replace("-", "")
                for value in values
                if normalize_id(value)
            }
        )
        if dates:
            return dates
    values: set[str] = set()
    for chunk in pd.read_csv(
        model_input / "od_hourly.csv",
        usecols=["date"],
        dtype=str,
        chunksize=250_000,
    ):
        values.update(
            normalize_id(value).replace("-", "")
            for value in chunk["date"].dropna()
        )
    return sorted(value for value in values if value)


def split_dates(
    dates: Sequence[str], config: Mapping[str, object]
) -> Tuple[List[str], List[str]]:
    cfg = dict(config.get("split", {}))
    available = sorted(set(dates))
    train_requested = [
        normalize_id(value).replace("-", "")
        for value in cfg.get("train_dates", [])
        if normalize_id(value)
    ]
    test_requested = [
        normalize_id(value).replace("-", "")
        for value in cfg.get("test_dates", [])
        if normalize_id(value)
    ]
    if train_requested or test_requested:
        train = [value for value in train_requested if value in available]
        test = [value for value in test_requested if value in available]
        if not train:
            raise ValueError("split.train_dates has no available date")
        if not test:
            test = [value for value in available if value not in set(train)]
        if not test:
            raise ValueError("split.test_dates has no available date")
        overlap = sorted(set(train).intersection(test))
        if overlap:
            raise ValueError(f"train/test dates overlap: {overlap}")
        return train, test
    fraction = float(cfg.get("train_fraction", 0.67))
    min_test = int(cfg.get("min_test_dates", 2))
    if len(available) <= min_test + 1:
        return available, available
    count = max(1, min(int(math.floor(len(available) * fraction)), len(available) - min_test))
    return available[:count], available[count:]


def load_choice_parameters(path: Path) -> ChoiceParameters:
    raw = load_json(path)
    return ChoiceParameters(
        beta_total_time=float(raw["beta_total_time"]),
        beta_walk_time_extra=float(
            raw.get("beta_walk_time_extra", raw.get("beta_walk_time", 0.0))
        ),
        beta_transfer=float(raw.get("beta_transfer", 0.0)),
        beta_discount_per_won=float(raw["beta_discount_per_won"]),
        value_of_time_won_per_hour=float(
            raw.get("value_of_time_won_per_hour", 1101.0)
        ),
    )


# -----------------------------------------------------------------------------
# Baseline edge universe
# -----------------------------------------------------------------------------


def load_baseline_segments(model_input: Path, hours: set[int]) -> pd.DataFrame:
    path = model_input / "segments_baseline.csv.gz"
    if path.exists():
        frame = pd.read_csv(path, compression="gzip", dtype=str, low_memory=False)
    else:
        frame = pd.read_csv(model_input / "segments.csv", dtype=str, low_memory=False)
    required = [
        "hour",
        "route_id",
        "from_stop_id",
        "to_stop_id",
        "travel_time",
        "distance",
        "avg_onboard",
        "trips",
    ]
    require_columns(frame, required, "segments_baseline")
    frame = frame.copy()
    frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce")
    for column in ["travel_time", "distance", "avg_onboard", "trips"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["route_id", "from_stop_id", "to_stop_id"]:
        frame[column] = frame[column].map(normalize_id)
    frame = frame.dropna(subset=required)
    frame = frame[
        frame["hour"].isin(hours)
        & frame["travel_time"].gt(0)
        & frame["distance"].gt(0)
        & frame["avg_onboard"].ge(0)
        & frame["trips"].gt(0)
        & frame["route_id"].ne("")
        & frame["from_stop_id"].ne("")
        & frame["to_stop_id"].ne("")
    ].copy()
    frame["hour"] = frame["hour"].astype(int)
    key = ["hour", "route_id", "from_stop_id", "to_stop_id"]
    aggregations: Dict[str, Tuple[str, str]] = {
        "travel_time": ("travel_time", "mean"),
        "distance": ("distance", "mean"),
        "avg_onboard": ("avg_onboard", "mean"),
        "trips": ("trips", "mean"),
    }
    if frame.duplicated(key).any():
        frame = frame.groupby(key, as_index=False).agg(**aggregations)
    return frame.sort_values(key, kind="stable").reset_index(drop=True)


def baseline_by_hour(frame: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    output: Dict[int, pd.DataFrame] = {}
    for hour, group in frame.groupby("hour", sort=True):
        value = group.reset_index(drop=True).copy()
        value["edge_index"] = np.arange(len(value), dtype=np.int32)
        output[int(hour)] = value
    return output


# -----------------------------------------------------------------------------
# Top-5 path topology cache
# -----------------------------------------------------------------------------


TOPOLOGY_REQUIRED = [
    "od_index",
    "hour",
    "origin_stop_id",
    "destination_stop_id",
    "signature",
    "segment_keys",
    "boarding_edges",
    "walk_time",
    "transfers",
]


def parse_edge_tokens(value: object) -> List[Tuple[str, str, str]]:
    text = normalize_id(value)
    if not text:
        return []
    output: List[Tuple[str, str, str]] = []
    for token in text.split(";"):
        token = token.strip()
        if not token:
            continue
        parts = token.split("|")
        if len(parts) != 3:
            raise ValueError(f"invalid edge token: {token!r}")
        output.append((normalize_id(parts[0]), normalize_id(parts[1]), normalize_id(parts[2])))
    return output


class HourTopologyWriter:
    def __init__(
        self,
        root: Path,
        hour: int,
        edge_frame: pd.DataFrame,
    ) -> None:
        self.hour = int(hour)
        self.root = root / f"hour_{self.hour:02d}"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.edge_frame = edge_frame
        self.edge_lookup = {
            (str(row.route_id), str(row.from_stop_id), str(row.to_stop_id)): int(row.edge_index)
            for row in edge_frame.itertuples(index=False)
        }
        self.edge_indices_handle = open(self.root / "path_edge_indices.i32", "wb")
        self.edge_indptr_handle = open(self.root / "path_edge_indptr.i64", "wb")
        self.board_indices_handle = open(self.root / "path_board_indices.i32", "wb")
        self.board_indptr_handle = open(self.root / "path_board_indptr.i64", "wb")
        self.walk_handle = open(self.root / "walk_time.f32", "wb")
        self.transfer_handle = open(self.root / "transfers.i16", "wb")
        self.source_handle = open(self.root / "source_rows.i64", "wb")
        self.group_starts_handle = open(self.root / "group_starts.i64", "wb")
        self.group_handle = gzip.open(
            self.root / "groups.csv.gz", "wt", encoding="utf-8-sig", newline=""
        )
        self.group_writer = csv.DictWriter(
            self.group_handle,
            fieldnames=[
                "group_index",
                "od_index",
                "hour",
                "origin_stop_id",
                "destination_stop_id",
                "path_start",
                "path_count",
            ],
        )
        self.group_writer.writeheader()
        np.asarray([0], dtype=np.int64).tofile(self.edge_indptr_handle)
        np.asarray([0], dtype=np.int64).tofile(self.board_indptr_handle)
        np.asarray([0], dtype=np.int64).tofile(self.group_starts_handle)
        self.paths = 0
        self.groups = 0
        self.edge_nnz = 0
        self.board_nnz = 0
        self.invalid_missing_edge = 0
        self.invalid_empty = 0
        self.skipped_single_path_groups = 0

    def parse_path(self, row: Tuple[object, ...], source_row: int) -> Optional[Tuple[np.ndarray, np.ndarray, float, int, int]]:
        segment_tokens = parse_edge_tokens(row[5])
        board_tokens = parse_edge_tokens(row[6])
        if not segment_tokens or not board_tokens:
            self.invalid_empty += 1
            return None
        try:
            segment_indices = [self.edge_lookup[token] for token in segment_tokens]
            board_indices = [self.edge_lookup[token] for token in board_tokens]
        except KeyError:
            self.invalid_missing_edge += 1
            return None
        # A path is loopless in the upstream generator.  Deduplicate defensively so a
        # repeated edge cannot receive the same bounty twice on one path.
        segment_unique = np.asarray(list(dict.fromkeys(segment_indices)), dtype=np.int32)
        board_unique = np.asarray(list(dict.fromkeys(board_indices)), dtype=np.int32)
        walk_time = float(row[7])
        transfers = int(float(row[8]))
        if not math.isfinite(walk_time) or walk_time < 0 or transfers < 0:
            self.invalid_empty += 1
            return None
        return segment_unique, board_unique, walk_time, transfers, int(source_row)

    def write_group(
        self,
        key: Tuple[int, int, str, str],
        rows: Sequence[Tuple[Tuple[object, ...], int]],
    ) -> None:
        parsed: List[Tuple[np.ndarray, np.ndarray, float, int, int]] = []
        for row, source_row in rows:
            value = self.parse_path(row, source_row)
            if value is not None:
                parsed.append(value)
        if len(parsed) < 2:
            self.skipped_single_path_groups += 1
            return
        _, od_index, origin, destination = key
        start = self.paths
        for edge_indices, board_indices, walk_time, transfers, source_row in parsed:
            edge_indices.tofile(self.edge_indices_handle)
            self.edge_nnz += len(edge_indices)
            np.asarray([self.edge_nnz], dtype=np.int64).tofile(self.edge_indptr_handle)
            board_indices.tofile(self.board_indices_handle)
            self.board_nnz += len(board_indices)
            np.asarray([self.board_nnz], dtype=np.int64).tofile(self.board_indptr_handle)
            np.asarray([walk_time], dtype=np.float32).tofile(self.walk_handle)
            np.asarray([transfers], dtype=np.int16).tofile(self.transfer_handle)
            np.asarray([source_row], dtype=np.int64).tofile(self.source_handle)
            self.paths += 1
        np.asarray([self.paths], dtype=np.int64).tofile(self.group_starts_handle)
        self.group_writer.writerow(
            {
                "group_index": self.groups,
                "od_index": od_index,
                "hour": self.hour,
                "origin_stop_id": origin,
                "destination_stop_id": destination,
                "path_start": start,
                "path_count": len(parsed),
            }
        )
        self.groups += 1

    def close(self) -> Dict[str, object]:
        for handle in [
            self.edge_indices_handle,
            self.edge_indptr_handle,
            self.board_indices_handle,
            self.board_indptr_handle,
            self.walk_handle,
            self.transfer_handle,
            self.source_handle,
            self.group_starts_handle,
            self.group_handle,
        ]:
            handle.close()
        metadata = {
            "hour": self.hour,
            "edges": len(self.edge_frame),
            "groups": self.groups,
            "paths": self.paths,
            "edge_nnz": self.edge_nnz,
            "board_nnz": self.board_nnz,
            "invalid_missing_edge_paths": self.invalid_missing_edge,
            "invalid_empty_paths": self.invalid_empty,
            "skipped_groups_with_fewer_than_two_valid_paths": self.skipped_single_path_groups,
        }
        (self.root / "meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return metadata


def topology_cache_signature(
    top_paths: Path,
    baseline: pd.DataFrame,
    hours: Sequence[int],
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "script_version": CACHE_COMPAT_VERSION,
                "top_paths": file_signature(top_paths),
                "hours": list(hours),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    columns = [
        "hour",
        "route_id",
        "from_stop_id",
        "to_stop_id",
        "travel_time",
        "avg_onboard",
        "trips",
    ]
    ordered = baseline[columns].sort_values(columns[:4], kind="stable")
    digest.update(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def build_topology_cache(
    top_paths: Path,
    baseline_hours: Mapping[int, pd.DataFrame],
    cache_root: Path,
    signature: str,
    chunksize: int,
    progress_seconds: float,
) -> Dict[str, object]:
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    header = pd.read_csv(top_paths, compression="infer", nrows=0)
    require_columns(header, TOPOLOGY_REQUIRED, "candidate_paths_top5")

    writers: Dict[int, HourTopologyWriter] = {}
    hour_meta: Dict[str, object] = {}
    current_key: Optional[Tuple[int, int, str, str]] = None
    current_rows: List[Tuple[Tuple[object, ...], int]] = []
    previous_sort_key: Optional[Tuple[int, int]] = None
    source_row = 0
    meter = Progress("topology-cache", None, progress_seconds)

    def flush_group() -> None:
        nonlocal current_key, current_rows
        if current_key is None:
            return
        hour = current_key[0]
        writer = writers.get(hour)
        if writer is None:
            if hour not in baseline_hours:
                current_key = None
                current_rows = []
                return
            writer = HourTopologyWriter(cache_root, hour, baseline_hours[hour])
            writers[hour] = writer
        writer.write_group(current_key, current_rows)
        current_key = None
        current_rows = []

    dtype = {
        "od_index": "int64",
        "hour": "int16",
        "origin_stop_id": "string",
        "destination_stop_id": "string",
        "signature": "string",
        "segment_keys": "string",
        "boarding_edges": "string",
        "walk_time": "float64",
        "transfers": "float64",
    }
    for chunk in pd.read_csv(
        top_paths,
        compression="infer",
        usecols=TOPOLOGY_REQUIRED,
        dtype=dtype,
        chunksize=max(1, int(chunksize)),
        low_memory=False,
    ):
        for row in chunk.itertuples(index=False, name=None):
            od_index = int(row[0])
            hour = int(row[1])
            origin = normalize_id(row[2])
            destination = normalize_id(row[3])
            key = (hour, od_index, origin, destination)
            sort_key = (hour, od_index)
            if previous_sort_key is not None and sort_key < previous_sort_key:
                raise ValueError(
                    "candidate_paths_top5 must be sorted by hour and od_index. "
                    "Use the direct output from fit_route_choice_top5.py."
                )
            previous_sort_key = sort_key
            if current_key is None:
                current_key = key
            elif key != current_key:
                flush_group()
                current_key = key
            current_rows.append((row, source_row))
            source_row += 1
        meter.update(
            source_row,
            extra=(
                f"valid paths {sum(writer.paths for writer in writers.values()):,}, "
                f"groups {sum(writer.groups for writer in writers.values()):,}"
            ),
        )
    flush_group()
    for hour, writer in sorted(writers.items()):
        hour_meta[str(hour)] = writer.close()
    missing_hours = sorted(set(baseline_hours) - set(writers))
    if missing_hours:
        raise RuntimeError(f"No top-5 path rows for configured hours: {missing_hours}")
    metadata = {
        "script_version": SCRIPT_VERSION,
        "signature": signature,
        "top_paths": file_signature(top_paths),
        "source_rows": source_row,
        "hours": hour_meta,
        "total_groups": int(sum(int(value["groups"]) for value in hour_meta.values())),
        "total_paths": int(sum(int(value["paths"]) for value in hour_meta.values())),
        "total_edge_nnz": int(sum(int(value["edge_nnz"]) for value in hour_meta.values())),
        "total_board_nnz": int(sum(int(value["board_nnz"]) for value in hour_meta.values())),
    }
    (cache_root / "topology_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meter.update(
        source_row,
        extra=f"valid paths {metadata['total_paths']:,}, groups {metadata['total_groups']:,}",
        force=True,
    )
    return metadata


def prepare_topology_cache(
    top_paths: Path,
    baseline: pd.DataFrame,
    baseline_hours: Mapping[int, pd.DataFrame],
    cache_root: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], str]:
    signature = topology_cache_signature(top_paths, baseline, sorted(baseline_hours))
    meta_path = cache_root / "topology_meta.json"
    reusable = False
    if not args.rebuild_topology_cache and meta_path.exists():
        try:
            reusable = load_json(meta_path).get("signature") == signature
        except Exception:
            reusable = False
    if reusable:
        metadata = load_json(meta_path)
        print(
            f"[topology-cache] reused: {int(metadata['total_paths']):,} paths, "
            f"{int(metadata['total_groups']):,} groups",
            flush=True,
        )
        return metadata, signature
    return (
        build_topology_cache(
            top_paths,
            baseline_hours,
            cache_root,
            signature,
            args.chunksize,
            args.progress_seconds,
        ),
        signature,
    )


def load_hour_topology(
    cache_root: Path,
    hour: int,
    edge_frame: pd.DataFrame,
    include_board: bool,
) -> HourTopology:
    root = cache_root / f"hour_{int(hour):02d}"
    meta = load_json(root / "meta.json")
    paths = int(meta["paths"])
    groups = int(meta["groups"])
    edges = int(meta["edges"])
    edge_nnz = int(meta["edge_nnz"])
    board_nnz = int(meta["board_nnz"])
    if edges != len(edge_frame):
        raise RuntimeError(f"hour {hour}: baseline edge count changed after topology cache")

    edge_indices = (
        np.memmap(
            root / "path_edge_indices.i32",
            dtype=np.int32,
            mode="r",
            shape=(edge_nnz,),
        )
        if edge_nnz
        else np.empty(0, dtype=np.int32)
    )
    edge_indptr = np.memmap(
        root / "path_edge_indptr.i64", dtype=np.int64, mode="r", shape=(paths + 1,)
    )
    edge_data = np.ones(edge_nnz, dtype=np.float32)
    path_edge = csr_matrix(
        (edge_data, edge_indices, edge_indptr), shape=(paths, edges), copy=False
    )
    path_board: Optional[csr_matrix] = None
    if include_board:
        board_indices = (
            np.memmap(
                root / "path_board_indices.i32",
                dtype=np.int32,
                mode="r",
                shape=(board_nnz,),
            )
            if board_nnz
            else np.empty(0, dtype=np.int32)
        )
        board_indptr = np.memmap(
            root / "path_board_indptr.i64", dtype=np.int64, mode="r", shape=(paths + 1,)
        )
        board_data = np.ones(board_nnz, dtype=np.float32)
        path_board = csr_matrix(
            (board_data, board_indices, board_indptr), shape=(paths, edges), copy=False
        )
    starts = np.memmap(
        root / "group_starts.i64", dtype=np.int64, mode="r", shape=(groups + 1,)
    )
    lengths = np.diff(starts).astype(np.int16, copy=False)
    group_index = np.repeat(np.arange(groups, dtype=np.int32), lengths)
    walk_time = (
        np.memmap(root / "walk_time.f32", dtype=np.float32, mode="r", shape=(paths,))
        if paths
        else np.empty(0, dtype=np.float32)
    )
    transfers = (
        np.memmap(root / "transfers.i16", dtype=np.int16, mode="r", shape=(paths,))
        if paths
        else np.empty(0, dtype=np.int16)
    )
    source_rows = (
        np.memmap(root / "source_rows.i64", dtype=np.int64, mode="r", shape=(paths,))
        if paths
        else np.empty(0, dtype=np.int64)
    )
    return HourTopology(
        hour=int(hour),
        edge_frame=edge_frame.reset_index(drop=True),
        path_edge=path_edge,
        path_board=path_board,
        group_starts=starts,
        group_lengths=lengths,
        group_index=group_index,
        walk_time=walk_time,
        transfers=transfers,
        source_rows=source_rows,
        groups=groups,
        paths=paths,
        edge_nnz=edge_nnz,
        board_nnz=board_nnz,
    )


# -----------------------------------------------------------------------------
# Daily data cache
# -----------------------------------------------------------------------------


def state_dir(cache_root: Path, date: str, hour: int) -> Path:
    return cache_root / str(date) / f"hour_{int(hour):02d}"


def grouped_softmax(
    utility: np.ndarray,
    starts: np.ndarray,
    group_index: np.ndarray,
) -> np.ndarray:
    if len(utility) == 0:
        return np.empty(0, dtype=np.float64)
    maxima = np.maximum.reduceat(utility, starts[:-1])
    shifted = utility - maxima[group_index]
    exponent = np.exp(np.clip(shifted, -700.0, 50.0))
    denominators = np.add.reduceat(exponent, starts[:-1])
    return exponent / np.maximum(denominators[group_index], EPS)


def compute_altprob_sparse(
    topology: HourTopology,
    p0: np.ndarray,
    demand: np.ndarray,
    travel_time: np.ndarray,
    load0: np.ndarray,
    capacity: float,
    group_block: int,
    progress_seconds: float,
    label: str,
) -> np.ndarray:
    if topology.groups == 0 or topology.paths == 0:
        return np.zeros(topology.path_edge.shape[1], dtype=np.float64)
    load_ratio = np.clip(load0 / max(capacity, EPS), 0.0, 3.0)
    crowd_numerator = np.asarray(
        topology.path_edge @ (travel_time * load_ratio), dtype=np.float64
    ).reshape(-1)
    ride_time = np.asarray(topology.path_edge @ travel_time, dtype=np.float64).reshape(-1)
    path_crowding = np.divide(
        crowd_numerator,
        np.maximum(ride_time, EPS),
        out=np.zeros_like(crowd_numerator),
        where=ride_time > EPS,
    )
    numerator = np.zeros(topology.path_edge.shape[1], dtype=np.float64)
    denominator = np.zeros(topology.path_edge.shape[1], dtype=np.float64)
    block_size = max(100, int(group_block))
    meter = Progress(label, topology.groups, progress_seconds)

    for g0 in range(0, topology.groups, block_size):
        g1 = min(topology.groups, g0 + block_size)
        r0 = int(topology.group_starts[g0])
        r1 = int(topology.group_starts[g1])
        if r1 <= r0:
            continue
        lengths = topology.group_lengths[g0:g1].astype(np.int64, copy=False)
        local_indptr = np.r_[0, np.cumsum(lengths)].astype(np.int64, copy=False)
        n_paths = r1 - r0
        local_columns = np.arange(n_paths, dtype=np.int32)
        local_p = np.asarray(p0[r0:r1], dtype=np.float64)
        local_pc = local_p * path_crowding[r0:r1]
        p_matrix = csr_matrix(
            (local_p, local_columns, local_indptr),
            shape=(g1 - g0, n_paths),
        )
        pc_matrix = csr_matrix(
            (local_pc, local_columns, local_indptr),
            shape=(g1 - g0, n_paths),
        )
        path_edge_block = topology.path_edge[r0:r1]
        q = (p_matrix @ path_edge_block).tocsr()
        s = (pc_matrix @ path_edge_block).tocsr()
        group_crowding = np.add.reduceat(local_pc, local_indptr[:-1])
        repeated_crowding = np.repeat(group_crowding, np.diff(q.indptr))
        r = csr_matrix(
            (repeated_crowding, q.indices.copy(), q.indptr.copy()),
            shape=q.shape,
        )
        r = r - s
        r.eliminate_zeros()
        local_demand = np.asarray(demand[g0:g1], dtype=np.float64)
        weighted_r = r.multiply(local_demand[:, None])
        denominator += np.asarray(weighted_r.sum(axis=0)).reshape(-1)
        numerator += np.asarray(q.multiply(r).multiply(local_demand[:, None]).sum(axis=0)).reshape(-1)
        meter.update(g1)
    meter.update(topology.groups, force=True)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > EPS,
    )


def build_daily_edge_arrays(
    model_input: Path,
    baseline_hours: Mapping[int, pd.DataFrame],
    dates: Sequence[str],
    cache_root: Path,
    chunksize: int,
    progress_seconds: float,
) -> None:
    date_set = set(dates)
    hours = set(baseline_hours)
    fields = ["travel_time", "trips", "avg_onboard"]
    sums: Dict[Tuple[str, int, str], np.ndarray] = {}
    counts: Dict[Tuple[str, int], np.ndarray] = {}
    edge_indexes: Dict[int, pd.MultiIndex] = {}
    for hour, frame in baseline_hours.items():
        edge_indexes[hour] = pd.MultiIndex.from_frame(
            frame[["route_id", "from_stop_id", "to_stop_id"]]
        )
        for date in dates:
            counts[(date, hour)] = np.zeros(len(frame), dtype=np.int32)
            for field in fields:
                sums[(date, hour, field)] = np.zeros(len(frame), dtype=np.float64)

    required = [
        "date",
        "hour",
        "route_id",
        "from_stop_id",
        "to_stop_id",
        "travel_time",
        "avg_onboard",
        "trips",
    ]
    processed = 0
    meter = Progress("daily-segments", None, progress_seconds)
    for chunk in pd.read_csv(
        model_input / "segments.csv",
        usecols=lambda column: column in set(required),
        dtype=str,
        chunksize=max(1, int(chunksize)),
        low_memory=False,
    ):
        require_columns(chunk, required, "segments.csv")
        processed += len(chunk)
        chunk["date"] = chunk["date"].map(normalize_id).str.replace("-", "", regex=False)
        chunk["hour"] = pd.to_numeric(chunk["hour"], errors="coerce")
        for field in fields:
            chunk[field] = pd.to_numeric(chunk[field], errors="coerce")
        for column in ["route_id", "from_stop_id", "to_stop_id"]:
            chunk[column] = chunk[column].map(normalize_id)
        chunk = chunk[
            chunk["date"].isin(date_set)
            & chunk["hour"].isin(hours)
            & chunk["travel_time"].gt(0)
            & chunk["trips"].gt(0)
            & chunk["avg_onboard"].ge(0)
        ].copy()
        if chunk.empty:
            meter.update(processed)
            continue
        chunk["hour"] = chunk["hour"].astype(int)
        for (date, hour), group in chunk.groupby(["date", "hour"], sort=False):
            indexer = edge_indexes[int(hour)].get_indexer(
                pd.MultiIndex.from_frame(
                    group[["route_id", "from_stop_id", "to_stop_id"]]
                )
            )
            valid = indexer >= 0
            if not np.any(valid):
                continue
            indices = indexer[valid]
            np.add.at(counts[(str(date), int(hour))], indices, 1)
            for field in fields:
                values = group[field].to_numpy(float)[valid]
                np.add.at(sums[(str(date), int(hour), field)], indices, values)
        meter.update(processed)
    meter.update(processed, force=True)

    for date in dates:
        for hour, baseline in baseline_hours.items():
            root = state_dir(cache_root, date, hour)
            root.mkdir(parents=True, exist_ok=True)
            count = counts[(date, hour)]
            for field, filename in [
                ("travel_time", "travel_time.npy"),
                ("trips", "trips.npy"),
                ("avg_onboard", "load0.npy"),
            ]:
                base = baseline[field].to_numpy(float)
                values = np.divide(
                    sums[(date, hour, field)],
                    count,
                    out=base.copy(),
                    where=count > 0,
                )
                np.save(root / filename, values.astype(np.float32), allow_pickle=False)


def build_daily_demand_arrays(
    model_input: Path,
    topology_root: Path,
    topology_meta: Mapping[str, object],
    dates: Sequence[str],
    cache_root: Path,
    chunksize: int,
    progress_seconds: float,
) -> None:
    date_set = set(dates)
    hours = sorted(int(value) for value in topology_meta["hours"])
    group_indexes: Dict[int, pd.MultiIndex] = {}
    group_counts: Dict[int, int] = {}
    demand: Dict[Tuple[str, int], np.ndarray] = {}
    for hour in hours:
        groups = pd.read_csv(
            topology_root / f"hour_{hour:02d}" / "groups.csv.gz",
            compression="gzip",
            dtype={"origin_stop_id": str, "destination_stop_id": str},
        )
        group_indexes[hour] = pd.MultiIndex.from_frame(
            groups[["origin_stop_id", "destination_stop_id"]].astype(str)
        )
        group_counts[hour] = len(groups)
        for date in dates:
            demand[(date, hour)] = np.zeros(len(groups), dtype=np.float64)

    required = ["date", "hour", "origin_stop_id", "destination_stop_id", "passengers"]
    processed = 0
    meter = Progress("daily-od", None, progress_seconds)
    for chunk in pd.read_csv(
        model_input / "od_hourly.csv",
        usecols=required,
        dtype=str,
        chunksize=max(1, int(chunksize)),
        low_memory=False,
    ):
        processed += len(chunk)
        chunk["date"] = chunk["date"].map(normalize_id).str.replace("-", "", regex=False)
        chunk["hour"] = pd.to_numeric(chunk["hour"], errors="coerce")
        chunk["passengers"] = pd.to_numeric(chunk["passengers"], errors="coerce")
        for column in ["origin_stop_id", "destination_stop_id"]:
            chunk[column] = chunk[column].map(normalize_id)
        chunk = chunk[
            chunk["date"].isin(date_set)
            & chunk["hour"].isin(hours)
            & chunk["passengers"].gt(0)
        ].copy()
        if chunk.empty:
            meter.update(processed)
            continue
        chunk["hour"] = chunk["hour"].astype(int)
        for (date, hour), group in chunk.groupby(["date", "hour"], sort=False):
            indexer = group_indexes[int(hour)].get_indexer(
                pd.MultiIndex.from_frame(
                    group[["origin_stop_id", "destination_stop_id"]].astype(str)
                )
            )
            valid = indexer >= 0
            if not np.any(valid):
                continue
            np.add.at(
                demand[(str(date), int(hour))],
                indexer[valid],
                group["passengers"].to_numpy(float)[valid],
            )
        meter.update(processed)
    meter.update(processed, force=True)
    for date in dates:
        for hour in hours:
            root = state_dir(cache_root, date, hour)
            root.mkdir(parents=True, exist_ok=True)
            np.save(
                root / "group_demand.npy",
                demand[(date, hour)].astype(np.float32),
                allow_pickle=False,
            )


def daily_cache_signature(
    model_input: Path,
    topology_signature: str,
    parameters: ChoiceParameters,
    dates: Sequence[str],
    config: Mapping[str, object],
) -> str:
    # Keep cache compatibility with the preceding v1.2 preparation because the
    # cached arrays themselves do not depend on hard-cap enforcement. AltProb and
    # low-crowding still use the same 55-passenger reference load.
    reference_load = load_reference_value(config)
    payload = {
        "script_version": CACHE_COMPAT_VERSION,
        "topology_signature": topology_signature,
        "segments": file_signature(model_input / "segments.csv"),
        "od_hourly": file_signature(model_input / "od_hourly.csv"),
        "choice_parameters": asdict(parameters),
        "dates": list(dates),
        "capacity": {
            "max_average_load": reference_load,
            "constraint_mode": "no_new_overload",
        },
    }
    return hash_payload(payload)


def build_daily_cache(
    model_input: Path,
    topology_root: Path,
    topology_meta: Mapping[str, object],
    baseline_hours: Mapping[int, pd.DataFrame],
    dates: Sequence[str],
    parameters: ChoiceParameters,
    config: Mapping[str, object],
    cache_root: Path,
    signature: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    print("[stage] daily edge arrays", flush=True)
    build_daily_edge_arrays(
        model_input,
        baseline_hours,
        dates,
        cache_root,
        args.chunksize,
        args.progress_seconds,
    )
    print("[stage] daily OD demand arrays", flush=True)
    build_daily_demand_arrays(
        model_input,
        topology_root,
        topology_meta,
        dates,
        cache_root,
        args.chunksize,
        args.progress_seconds,
    )

    capacity = load_reference_value(config)
    hours = sorted(baseline_hours)
    state_meta: Dict[str, object] = {}
    total_states = len(hours) * len(dates)
    state_counter = 0
    state_meter = Progress("daily-state", total_states, args.progress_seconds)
    for hour in hours:
        topology = load_hour_topology(
            topology_root, hour, baseline_hours[hour], include_board=True
        )
        if topology.path_board is None:
            raise RuntimeError("boarding matrix missing while building daily cache")
        for date in dates:
            root = state_dir(cache_root, date, hour)
            travel = np.load(root / "travel_time.npy", mmap_mode="r").astype(np.float64)
            trips = np.load(root / "trips.npy", mmap_mode="r").astype(np.float64)
            load0 = np.load(root / "load0.npy", mmap_mode="r").astype(np.float64)
            demand = np.load(root / "group_demand.npy", mmap_mode="r").astype(np.float64)
            ride = np.asarray(topology.path_edge @ travel, dtype=np.float64).reshape(-1)
            wait = np.asarray(
                topology.path_board @ (30.0 / np.maximum(trips, EPS)),
                dtype=np.float64,
            ).reshape(-1)
            total_time = ride + wait + np.asarray(topology.walk_time, dtype=np.float64)
            base_utility = (
                parameters.beta_total_time * total_time
                + parameters.beta_walk_time_extra
                * np.asarray(topology.walk_time, dtype=np.float64)
                + parameters.beta_transfer
                * np.asarray(topology.transfers, dtype=np.float64)
            )
            p0 = grouped_softmax(
                base_utility, topology.group_starts, topology.group_index
            )
            altprob = compute_altprob_sparse(
                topology,
                p0,
                demand,
                travel,
                load0,
                capacity,
                args.altprob_group_block,
                args.progress_seconds,
                f"altprob-{date}-{hour:02d}",
            )
            h0 = float(np.sum(travel * trips * load0 * load0))
            np.save(root / "base_utility.npy", base_utility.astype(np.float32), allow_pickle=False)
            np.save(root / "p0.npy", p0.astype(np.float32), allow_pickle=False)
            np.save(root / "total_time.npy", total_time.astype(np.float32), allow_pickle=False)
            np.save(root / "altprob.npy", altprob.astype(np.float32), allow_pickle=False)
            active_groups = int(np.count_nonzero(demand > 0))
            key = f"{date}-{hour:02d}"
            state_meta[key] = {
                "date": date,
                "hour": hour,
                "paths": topology.paths,
                "groups": topology.groups,
                "active_groups": active_groups,
                "demand": float(demand.sum()),
                "h0": h0,
            }
            state_counter += 1
            state_meter.update(
                state_counter,
                extra=f"paths {topology.paths:,}, active groups {active_groups:,}",
            )
        del topology
    state_meter.update(total_states, force=True)
    metadata = {
        "script_version": SCRIPT_VERSION,
        "signature": signature,
        "dates": list(dates),
        "hours": hours,
        "choice_parameters": asdict(parameters),
        "states": state_meta,
    }
    (cache_root / "daily_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def prepare_daily_cache(
    model_input: Path,
    topology_root: Path,
    topology_meta: Mapping[str, object],
    topology_signature: str,
    baseline_hours: Mapping[int, pd.DataFrame],
    dates: Sequence[str],
    parameters: ChoiceParameters,
    config: Mapping[str, object],
    cache_root: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], str]:
    signature = daily_cache_signature(
        model_input, topology_signature, parameters, dates, config
    )
    meta_path = cache_root / "daily_meta.json"
    reusable = False
    if not args.rebuild_daily_cache and meta_path.exists():
        try:
            reusable = load_json(meta_path).get("signature") == signature
        except Exception:
            reusable = False
    if reusable:
        metadata = load_json(meta_path)
        print(f"[daily-cache] reused: {len(metadata['states']):,} date-hour states", flush=True)
        return metadata, signature
    return (
        build_daily_cache(
            model_input,
            topology_root,
            topology_meta,
            baseline_hours,
            dates,
            parameters,
            config,
            cache_root,
            signature,
            args,
        ),
        signature,
    )


def summarize_baseline_capacity(
    daily_root: Path,
    dates: Sequence[str],
    hours: Sequence[int],
    capacity: float,
    mode: str,
) -> Dict[str, object]:
    """Summarize observed overload before fitting a policy.

    This report is intentionally separate from the policy result.  Under the
    default ``no_new_overload`` rule, pre-existing overload is allowed but may
    not be increased.  Under ``absolute`` it makes the no-policy baseline
    infeasible and is therefore a critical model diagnostic.
    """
    state_rows: List[Dict[str, object]] = []
    total_edges = 0
    overloaded_edges = 0
    maximum_load = 0.0
    maximum_absolute_violation = 0.0
    for date in dates:
        for hour in hours:
            load0 = np.asarray(
                np.load(state_dir(daily_root, date, int(hour)) / "load0.npy", mmap_mode="r"),
                dtype=np.float64,
            )
            local_over = load0 > capacity + 1.0e-9
            local_count = int(np.count_nonzero(local_over))
            local_max = float(np.max(load0)) if len(load0) else 0.0
            local_violation = max(0.0, local_max - capacity)
            state_rows.append(
                {
                    "date": str(date),
                    "hour": int(hour),
                    "edge_count": int(len(load0)),
                    "nominal_overloaded_edges": local_count,
                    "max_observed_avg_onboard": local_max,
                    "max_absolute_capacity_violation": local_violation,
                }
            )
            total_edges += int(len(load0))
            overloaded_edges += local_count
            maximum_load = max(maximum_load, local_max)
            maximum_absolute_violation = max(
                maximum_absolute_violation, local_violation
            )
    return {
        "capacity": float(capacity),
        "constraint_mode": mode,
        "date_hour_edge_observations": total_edges,
        "nominal_overloaded_date_hour_edges": overloaded_edges,
        "nominal_overloaded_share": overloaded_edges / max(total_edges, 1),
        "max_observed_avg_onboard": maximum_load,
        "max_absolute_capacity_violation": maximum_absolute_violation,
        "baseline_feasible_under_configured_mode": True if mode == "diagnostic_only" else bool(
            mode == "no_new_overload" or maximum_absolute_violation <= 1.0e-6
        ),
        "states": state_rows,
    }


def load_day_hour_state(cache_root: Path, date: str, hour: int) -> DayHourState:
    root = state_dir(cache_root, date, hour)
    meta = load_json(cache_root / "daily_meta.json")["states"][f"{date}-{hour:02d}"]
    return DayHourState(
        date=date,
        hour=hour,
        base_utility=np.load(root / "base_utility.npy", mmap_mode="r"),
        p0=np.load(root / "p0.npy", mmap_mode="r"),
        total_time=np.load(root / "total_time.npy", mmap_mode="r"),
        group_demand=np.load(root / "group_demand.npy", mmap_mode="r"),
        travel_time=np.load(root / "travel_time.npy", mmap_mode="r"),
        trips=np.load(root / "trips.npy", mmap_mode="r"),
        load0=np.load(root / "load0.npy", mmap_mode="r"),
        altprob=np.load(root / "altprob.npy", mmap_mode="r"),
        h0=float(meta["h0"]),
    )


# -----------------------------------------------------------------------------
# Policy features and simulation
# -----------------------------------------------------------------------------


def build_operational_features(
    baseline_hours: Mapping[int, pd.DataFrame],
    daily_root: Path,
    train_dates: Sequence[str],
    config: Mapping[str, object],
) -> pd.DataFrame:
    capacity = load_reference_value(config)
    rows: List[pd.DataFrame] = []
    for hour, edges in sorted(baseline_hours.items()):
        low_values: List[np.ndarray] = []
        alt_values: List[np.ndarray] = []
        for date in train_dates:
            state = load_day_hour_state(daily_root, date, hour)
            low_values.append(
                np.clip(
                    1.0 - np.asarray(state.load0, dtype=np.float64) / max(capacity, EPS),
                    0.0,
                    1.0,
                )
            )
            alt_values.append(np.asarray(state.altprob, dtype=np.float64))
        frame = edges[
            ["hour", "route_id", "from_stop_id", "to_stop_id"]
        ].copy()
        frame["low_crowding_train_mean"] = np.mean(np.vstack(low_values), axis=0)
        frame["altprob_train_mean"] = np.mean(np.vstack(alt_values), axis=0)
        frame["training_date_count"] = len(train_dates)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["hour", "route_id", "from_stop_id", "to_stop_id"], kind="stable"
    ).reset_index(drop=True)


def score_features(
    features: pd.DataFrame,
    theta: PolicyTheta,
    config: Mapping[str, object],
) -> pd.DataFrame:
    scored = features.copy()
    score = (
        theta.intercept
        + theta.low_crowding
        * scored["low_crowding_train_mean"].to_numpy(float)
        + theta.altprob * scored["altprob_train_mean"].to_numpy(float)
    )
    maximum = float(config.get("policy", {}).get("max_edge_reward_won", 500.0))
    reward = np.clip(maximum * expit(score), 0.0, maximum)
    minimum_altprob = float(config.get("policy", {}).get("minimum_altprob", 0.0))
    eligible = scored["altprob_train_mean"].to_numpy(float) > minimum_altprob
    reward = np.where(eligible, reward, 0.0)
    scored["policy_score"] = score
    scored["reward_won"] = reward
    scored["eligible"] = eligible.astype(np.int8)
    return scored


def rewards_by_hour(
    scored: pd.DataFrame,
    baseline_hours: Mapping[int, pd.DataFrame],
) -> Dict[int, np.ndarray]:
    output: Dict[int, np.ndarray] = {}
    for hour, edges in baseline_hours.items():
        local = scored[scored["hour"] == hour]
        if len(local) != len(edges):
            raise RuntimeError(f"hour {hour}: operational feature edge count mismatch")
        output[hour] = local["reward_won"].to_numpy(float)
    return output



class PolicySimulator:
    def __init__(
        self,
        topologies: Mapping[int, HourTopology],
        daily_root: Path,
        dates: Sequence[str],
        parameters: ChoiceParameters,
        config: Mapping[str, object],
    ) -> None:
        self.topologies = dict(topologies)
        self.daily_root = daily_root
        self.dates = list(dates)
        self.parameters = parameters
        self.config = config
        self._states: Dict[Tuple[str, int], DayHourState] = {}

    def state(self, date: str, hour: int) -> DayHourState:
        key = (date, hour)
        if key not in self._states:
            self._states[key] = load_day_hour_state(self.daily_root, date, hour)
        return self._states[key]

    def path_discounts(
        self, rewards: Mapping[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Compute path-level cumulative bounty once for a fixed reward table.

        Rewards are operationally fixed by hour and directed route edge, so this
        sparse matrix multiplication is identical on every calendar date.  Doing
        it once per theta rather than once per theta-date materially reduces the
        policy-search cost on the 12-date model.
        """
        output: Dict[int, np.ndarray] = {}
        for hour, topology in self.topologies.items():
            reward = np.asarray(rewards[hour], dtype=np.float64)
            if topology.paths == 0 or not np.any(reward > 0.0):
                output[hour] = np.zeros(topology.paths, dtype=np.float64)
            else:
                output[hour] = np.asarray(
                    topology.path_edge @ reward, dtype=np.float64
                ).reshape(-1)
        return output

    def simulate_day(
        self,
        date: str,
        rewards: Mapping[int, np.ndarray],
        detailed: bool = False,
        path_discount_by_hour: Optional[Mapping[int, np.ndarray]] = None,
    ) -> Tuple[DayPolicyResult, Dict[int, Dict[str, np.ndarray]]]:
        # ``reference_load`` is NOT a feasibility constraint.  It is used only
        # for load diagnostics (and elsewhere as the crowding-feature scale).
        reference_load = load_reference_value(self.config)
        budget = float(self.config.get("budget_per_day", 3_000_000.0))
        max_extra = optional_float(
            self.config.get("policy", {}).get(
                "max_extra_passenger_minutes_per_day", None
            ),
            math.inf,
        )
        target_by_hour: Dict[int, Dict[str, np.ndarray]] = {}
        h0 = 0.0
        extra_target = 0.0

        # First compute the unconstrained MNL response for every hour.
        # No common capacity-response scaling is applied.
        for hour, topology in self.topologies.items():
            state = self.state(date, hour)
            reward = np.asarray(rewards[hour], dtype=np.float64)
            p0 = np.asarray(state.p0, dtype=np.float64)
            if not np.any(reward > 0.0):
                path_discount = np.zeros(topology.paths, dtype=np.float64)
                p_target = p0.copy()
                diff = np.zeros_like(p0)
            else:
                if path_discount_by_hour is None:
                    path_discount = np.asarray(
                        topology.path_edge @ reward, dtype=np.float64
                    ).reshape(-1)
                else:
                    path_discount = np.asarray(
                        path_discount_by_hour[hour], dtype=np.float64
                    )
                utility1 = (
                    np.asarray(state.base_utility, dtype=np.float64)
                    + self.parameters.beta_discount_per_won * path_discount
                )
                p_target = grouped_softmax(
                    utility1, topology.group_starts, topology.group_index
                )
                diff = p_target - p0

            demand_path = np.asarray(state.group_demand, dtype=np.float64)[
                topology.group_index
            ]
            target_delta_edge = np.asarray(
                topology.path_edge.T @ (demand_path * diff), dtype=np.float64
            ).reshape(-1)
            extra_target += float(
                np.dot(
                    demand_path * diff,
                    np.asarray(state.total_time, dtype=np.float64),
                )
            )
            h0 += state.h0
            target_by_hour[hour] = {
                "target_delta_edge": target_delta_edge,
                "path_discount": path_discount,
                "p_target": p_target if detailed else np.empty(0),
                "diff": diff if detailed else np.empty(0),
            }

        h1 = 0.0
        cost = 0.0
        expected_discounted = 0.0
        rewarded_edges = 0
        max_policy_load = 0.0
        max_load_increase = 0.0
        overloaded_edges = 0
        new_overloaded_edges = 0
        increased_existing_overload_edges = 0
        details: Dict[int, Dict[str, np.ndarray]] = {}

        for hour, topology in self.topologies.items():
            state = self.state(date, hour)
            reward = np.asarray(rewards[hour], dtype=np.float64)
            delta_edge = target_by_hour[hour]["target_delta_edge"]
            trips = np.asarray(state.trips, dtype=np.float64)
            load0 = np.asarray(state.load0, dtype=np.float64)
            load1 = np.maximum(
                load0 + delta_edge / np.maximum(trips, EPS), 0.0
            )
            travel = np.asarray(state.travel_time, dtype=np.float64)
            h1 += float(np.sum(travel * trips * load1 * load1))

            # Payment rule A: every passenger using a rewarded edge after the
            # policy receives that edge's bounty, not only newly switched users.
            edge_users_after = load1 * trips
            cost += float(np.dot(reward, edge_users_after))
            positive = reward > 0
            rewarded_edges += int(np.count_nonzero(positive))
            expected_discounted += float(edge_users_after[positive].sum())

            # 55-passenger reference is diagnostic only; it never changes p_target
            # or feasibility.
            if len(load1):
                max_policy_load = max(max_policy_load, float(np.max(load1)))
                max_load_increase = max(
                    max_load_increase, float(np.max(load1 - load0))
                )
            overloaded = load1 > reference_load + 1.0e-9
            new_overloaded = (load0 <= reference_load + 1.0e-9) & overloaded
            increased_existing = (load0 > reference_load + 1.0e-9) & (load1 > load0 + 1.0e-9)
            overloaded_edges += int(np.count_nonzero(overloaded))
            new_overloaded_edges += int(np.count_nonzero(new_overloaded))
            increased_existing_overload_edges += int(np.count_nonzero(increased_existing))

            if detailed:
                p1 = np.asarray(target_by_hour[hour]["p_target"], dtype=np.float64)
                details[hour] = {
                    "load1": load1,
                    "p1": p1,
                    "path_discount": target_by_hour[hour]["path_discount"],
                    "reward": reward,
                }

        improvement = float(h0 - h1)
        extra = float(max(0.0, extra_target))
        feasible = bool(
            cost <= budget + 1.0e-6
            and extra <= max_extra + 1.0e-6
        )
        result = DayPolicyResult(
            date=date,
            h0=float(h0),
            h1=float(h1),
            improvement=improvement,
            cost=float(cost),
            efficiency_per_1000=(
                improvement / cost * 1000.0 if cost > 0 else 0.0
            ),
            extra_passenger_minutes=extra,
            rewarded_edges=rewarded_edges,
            expected_discounted_passengers=expected_discounted,
            response_scale=1.0,
            max_policy_avg_onboard=float(max_policy_load),
            overloaded_edges_above_reference=int(overloaded_edges),
            new_overloaded_edges_above_reference=int(new_overloaded_edges),
            increased_existing_overload_edges=int(increased_existing_overload_edges),
            max_load_increase=float(max(0.0, max_load_increase)),
            feasible=feasible,
        )
        return result, details

    def evaluate(
        self,
        rewards: Mapping[int, np.ndarray],
        dates: Sequence[str],
        stop_on_infeasible: bool = False,
    ) -> Tuple[Dict[str, float], List[DayPolicyResult]]:
        path_discount_by_hour = self.path_discounts(rewards)
        results: List[DayPolicyResult] = []
        for date in dates:
            result, _ = self.simulate_day(
                date,
                rewards,
                detailed=False,
                path_discount_by_hour=path_discount_by_hour,
            )
            results.append(result)
            if stop_on_infeasible and not result.feasible:
                break
        improvement = float(sum(value.improvement for value in results))
        cost = float(sum(value.cost for value in results))
        extra = float(sum(value.extra_passenger_minutes for value in results))
        daily_costs = [value.cost for value in results] or [0.0]
        daily_improvements = [value.improvement for value in results] or [0.0]
        feasible = bool(len(results) == len(dates) and all(value.feasible for value in results))
        record = {
            "objective_improvement": improvement,
            "improvement": improvement,
            "cost": cost,
            "efficiency_per_1000": improvement / cost * 1000.0 if cost > 0 else 0.0,
            "mean_daily_cost": float(np.mean(daily_costs)),
            "max_daily_cost": float(np.max(daily_costs)),
            "min_daily_improvement": float(np.min(daily_improvements)),
            "extra_passenger_minutes": extra,
            "max_policy_avg_onboard": float(
                max((value.max_policy_avg_onboard for value in results), default=0.0)
            ),
            "max_daily_new_overloaded_edges": float(
                max((value.new_overloaded_edges_above_reference for value in results), default=0)
            ),
            "max_daily_increased_existing_overload_edges": float(
                max((value.increased_existing_overload_edges for value in results), default=0)
            ),
            "max_load_increase": float(
                max((value.max_load_increase for value in results), default=0.0)
            ),
            "feasible": float(feasible),
            "dates_evaluated": float(len(results)),
        }
        return record, results


# -----------------------------------------------------------------------------
# Bayesian bounty-function fitting
# -----------------------------------------------------------------------------


def policy_search_signature(
    daily_signature: str,
    features: pd.DataFrame,
    parameters: ChoiceParameters,
    train_dates: Sequence[str],
    config: Mapping[str, object],
) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "policy_simulation_version": POLICY_SIMULATION_VERSION,
        "daily_signature": daily_signature,
        "parameters": asdict(parameters),
        "train_dates": list(train_dates),
        "budget": config.get("budget_per_day"),
        "load_reference": load_reference_value(config),
        "capacity_constraint_enforced": False,
        "policy": config.get("policy", {}),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    columns = [
        "hour",
        "route_id",
        "from_stop_id",
        "to_stop_id",
        "low_crowding_train_mean",
        "altprob_train_mean",
    ]
    digest.update(
        pd.util.hash_pandas_object(features[columns], index=False).to_numpy().tobytes()
    )
    return digest.hexdigest()[:20]


def fit_policy_theta(
    features: pd.DataFrame,
    baseline_hours: Mapping[int, pd.DataFrame],
    simulator: PolicySimulator,
    train_dates: Sequence[str],
    parameters: ChoiceParameters,
    config: MutableMapping[str, object],
    output: Path,
    daily_signature: str,
    override_trials: int,
) -> Tuple[Optional[PolicyTheta], pd.DataFrame, pd.DataFrame, Dict[str, object], bool]:
    cfg = dict(config.get("policy", {}).get("bayesian", {}))
    ranges = dict(cfg.get("ranges", {}))
    defaults = {
        "intercept": [-20.0, 0.0],
        "low_crowding": [0.0, 12.0],
        "altprob": [0.0, 20.0],
    }
    bounds: Dict[str, Tuple[float, float]] = {}
    for name, default in defaults.items():
        values = ranges.get(name, default)
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"policy.bayesian.ranges.{name} must be [min,max]")
        low, high = float(values[0]), float(values[1])
        if high <= low:
            raise ValueError(f"invalid theta range for {name}: {values}")
        bounds[name] = (low, high)

    signature = policy_search_signature(
        daily_signature, features, parameters, train_dates, config
    )
    study_name = f"public_edge_bounty_split_{signature}"
    storage_path = output / "edge_bounty_study.sqlite3"
    sampler = optuna.samplers.TPESampler(
        seed=int(cfg.get("seed", 42)),
        n_startup_trials=int(cfg.get("n_startup_trials", 16)),
        multivariate=bool(cfg.get("multivariate", True)),
        group=bool(cfg.get("group", True)),
    )
    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///" + storage_path.as_posix(),
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    if len(study.trials) == 0:
        enqueue = cfg.get(
            "enqueue",
            [
                {"intercept": -14.0, "low_crowding": 4.0, "altprob": 8.0},
                {"intercept": -18.0, "low_crowding": 2.0, "altprob": 4.0},
                {"intercept": -10.0, "low_crowding": 6.0, "altprob": 12.0},
            ],
        )
        for point in enqueue:
            study.enqueue_trial({name: float(point[name]) for name in bounds})

    target = int(override_trials or cfg.get("n_trials", 100))
    completed_before = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining = max(0, target - completed_before)
    print(
        f"[Bayes bounty function] completed {completed_before}/{target}; running {remaining}",
        flush=True,
    )
    started = time.perf_counter()
    completed_this_run = 0
    stop_on_infeasible = bool(
        config.get("policy", {}).get("early_stop_infeasible_trials", True)
    )

    def objective(trial: optuna.Trial) -> float:
        nonlocal completed_this_run
        theta = PolicyTheta(
            intercept=trial.suggest_float("intercept", *bounds["intercept"]),
            low_crowding=trial.suggest_float(
                "low_crowding", *bounds["low_crowding"]
            ),
            altprob=trial.suggest_float("altprob", *bounds["altprob"]),
        )
        scored = score_features(features, theta, config)
        rewards = rewards_by_hour(scored, baseline_hours)
        record, _ = simulator.evaluate(
            rewards, train_dates, stop_on_infeasible=stop_on_infeasible
        )
        positive = scored.loc[scored["reward_won"] > 0, "reward_won"]
        record["rewarded_edges"] = float(len(positive))
        record["mean_positive_reward_won"] = (
            float(positive.mean()) if len(positive) else 0.0
        )
        record["max_reward_won"] = float(positive.max()) if len(positive) else 0.0
        for key, value in record.items():
            trial.set_user_attr(key, float(value))
        value = (
            float(record["objective_improvement"])
            if record["feasible"] > 0.5
            else -1.0e30
        )
        completed_this_run += 1
        completed_total = completed_before + completed_this_run
        elapsed = time.perf_counter() - started
        rate = completed_this_run / max(elapsed, EPS)
        eta = max(0, target - completed_total) / max(rate, EPS)
        print(
            "[Bayes] {0}/{1}, elapsed {2}, ETA {3}: relief={4:,.3f}, "
            "max daily cost={5:,.0f}, efficiency={6:.6f}, feasible={7}, "
            "theta=({8:.4f},{9:.4f},{10:.4f})".format(
                completed_total,
                target,
                format_seconds(elapsed),
                format_seconds(eta),
                record["objective_improvement"],
                record["max_daily_cost"],
                record["efficiency_per_1000"],
                bool(record["feasible"] > 0.5),
                theta.intercept,
                theta.low_crowding,
                theta.altprob,
            ),
            flush=True,
        )
        return value

    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True, n_jobs=1)

    completed = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    rows: List[Dict[str, object]] = []
    for trial in completed:
        row: Dict[str, object] = {
            "trial": int(trial.number),
            "policy_type": "SIGMOID_BOUNTY",
            "intercept": float(trial.params["intercept"]),
            "low_crowding": float(trial.params["low_crowding"]),
            "altprob": float(trial.params["altprob"]),
            "study_objective": float(trial.value) if trial.value is not None else np.nan,
        }
        row.update(trial.user_attrs)
        rows.append(row)

    # A public authority can always choose not to deploy a bounty.  The finite
    # logistic parameter box cannot represent exactly zero reward, so include an
    # explicit null-policy alternative.  This prevents a tiny but harmful reward
    # from being selected merely because every sampled theta has negative relief.
    zero_rewards = {
        hour: np.zeros(len(edges), dtype=np.float64)
        for hour, edges in baseline_hours.items()
    }
    null_record, _ = simulator.evaluate(
        zero_rewards, train_dates, stop_on_infeasible=False
    )
    null_row: Dict[str, object] = {
        "trial": -1,
        "policy_type": "NO_BOUNTY",
        "intercept": np.nan,
        "low_crowding": np.nan,
        "altprob": np.nan,
        "study_objective": (
            float(null_record["objective_improvement"])
            if null_record["feasible"] > 0.5
            else -1.0e30
        ),
        **null_record,
        "rewarded_edges": 0.0,
        "mean_positive_reward_won": 0.0,
        "max_reward_won": 0.0,
    }
    allow_no_bounty = bool(config.get("policy", {}).get("allow_no_bounty", True))
    if allow_no_bounty:
        rows.append(null_row)
    if not rows:
        raise RuntimeError("No completed bounty-function trial")

    trials = pd.DataFrame(rows).sort_values(
        ["feasible", "objective_improvement", "max_daily_cost"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    feasible = trials[trials["feasible"] > 0.5]
    if feasible.empty:
        raise RuntimeError(
            "No budget-feasible bounty function. Enable policy.allow_no_bounty or "
            "expand the negative intercept range."
        )

    minimum_improvement = float(
        config.get("policy", {}).get("minimum_training_improvement", 0.0)
    )
    best = feasible.iloc[0]
    policy_active = bool(
        str(best["policy_type"]) == "SIGMOID_BOUNTY"
        and float(best["objective_improvement"]) > minimum_improvement
    )
    if policy_active:
        theta: Optional[PolicyTheta] = PolicyTheta(
            intercept=float(best["intercept"]),
            low_crowding=float(best["low_crowding"]),
            altprob=float(best["altprob"]),
        )
        scored = score_features(features, theta, config)
    else:
        theta = None
        scored = features.copy()
        scored["policy_score"] = -np.inf
        scored["reward_won"] = 0.0
        minimum_altprob = float(
            config.get("policy", {}).get("minimum_altprob", 0.0)
        )
        scored["eligible"] = (
            scored["altprob_train_mean"].to_numpy(float) > minimum_altprob
        ).astype(np.int8)

    selected_trial = int(best["trial"])
    metadata = {
        "method": "Optuna TPE Bayesian optimization with explicit no-bounty alternative",
        "study_name": study_name,
        "storage": str(storage_path),
        "completed_trials": len(completed),
        "target_trials": target,
        "selected_policy": "SIGMOID_BOUNTY" if policy_active else "NO_BOUNTY",
        "policy_active": policy_active,
        "selected_trial": selected_trial,
        "best_trial": selected_trial if policy_active else None,
        "best_objective_improvement": float(best["objective_improvement"]),
        "best_max_daily_cost": float(best["max_daily_cost"]),
        "best_efficiency_per_1000": float(best["efficiency_per_1000"]),
        "minimum_training_improvement": minimum_improvement,
        "allow_no_bounty": allow_no_bounty,
        "ranges": {name: list(value) for name, value in bounds.items()},
        "objective": "maximize total congestion relief on training dates",
        "constraints": {
            "budget_per_day": config.get("budget_per_day", 3_000_000.0),
            "capacity_constraint": None,
            "load_reference_for_diagnostics": load_reference_value(config),
            "max_extra_passenger_minutes_per_day": config.get("policy", {}).get(
                "max_extra_passenger_minutes_per_day", None
            ),
        },
        "deployment_dimension": "hour x route x directed edge",
        "date_specific_rewards": False,
        "feature_aggregation": "arithmetic mean across training dates only",
        "early_stop_infeasible_trials": stop_on_infeasible,
    }
    return theta, scored, trials, metadata, policy_active



# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------


def aggregate_scenarios(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["split", "scenario"], as_index=False).agg(
        H0=("h0", "sum"),
        H1=("h1", "sum"),
        improvement=("improvement", "sum"),
        cost=("cost", "sum"),
        rewarded_edges=("rewarded_edges", "sum"),
        expected_discounted_passengers=("expected_discounted_passengers", "sum"),
        max_policy_avg_onboard=("max_policy_avg_onboard", "max"),
        overloaded_edges_above_reference=("overloaded_edges_above_reference", "sum"),
        new_overloaded_edges_above_reference=("new_overloaded_edges_above_reference", "sum"),
        increased_existing_overload_edges=("increased_existing_overload_edges", "sum"),
        max_load_increase=("max_load_increase", "max"),
        all_dates_feasible=("feasible", "min"),
        dates=("date", "nunique"),
    )
    grouped["improvement_pct"] = (
        100.0 * grouped["improvement"] / grouped["H0"].clip(lower=EPS)
    )
    grouped["efficiency_per_1000"] = (
        grouped["improvement"]
        / grouped["cost"].replace(0, np.nan)
        * 1000.0
    ).fillna(0.0)
    return grouped


def write_edge_altprob(
    destination: Path,
    baseline_hours: Mapping[int, pd.DataFrame],
    daily_root: Path,
    dates: Sequence[str],
    capacity: float,
) -> int:
    if destination.exists():
        destination.unlink()
    first = True
    rows = 0
    for date in dates:
        for hour, edges in sorted(baseline_hours.items()):
            state = load_day_hour_state(daily_root, date, hour)
            frame = edges[
                ["hour", "route_id", "from_stop_id", "to_stop_id"]
            ].copy()
            frame.insert(0, "date", date)
            frame["avg_onboard"] = np.asarray(state.load0)
            frame["trips"] = np.asarray(state.trips)
            frame["low_crowding"] = np.clip(
                1.0 - np.asarray(state.load0, dtype=float) / max(capacity, EPS),
                0.0,
                1.0,
            )
            frame["altprob"] = np.asarray(state.altprob)
            frame.to_csv(
                destination,
                mode="wt" if first else "at",
                header=first,
                index=False,
                encoding="utf-8-sig" if first else "utf-8",
                compression="gzip",
            )
            first = False
            rows += len(frame)
    return rows


def write_final_results(
    output: Path,
    theta: Optional[PolicyTheta],
    scored_features: pd.DataFrame,
    baseline_hours: Mapping[int, pd.DataFrame],
    simulator: PolicySimulator,
    train_dates: Sequence[str],
    test_dates: Sequence[str],
) -> Tuple[pd.DataFrame, int]:
    rewards = rewards_by_hour(scored_features, baseline_hours)
    path_discount_by_hour = simulator.path_discounts(rewards)
    zero_rewards = {
        hour: np.zeros(len(edges), dtype=np.float64)
        for hour, edges in baseline_hours.items()
    }
    zero_path_discounts = simulator.path_discounts(zero_rewards)
    daily_rows: List[Dict[str, object]] = []
    edge_destination = output / "edge_rewards.csv.gz"
    if edge_destination.exists():
        edge_destination.unlink()
    first_edge = True
    edge_rows_written = 0
    train_set = set(train_dates)

    for date in simulator.dates:
        split = "train" if date in train_set else "test"
        policy_result, details = simulator.simulate_day(
            date,
            rewards,
            detailed=True,
            path_discount_by_hour=path_discount_by_hour,
        )
        baseline_result, _ = simulator.simulate_day(
            date,
            zero_rewards,
            detailed=False,
            path_discount_by_hour=zero_path_discounts,
        )
        baseline_result.response_scale = 0.0
        daily_rows.append(
            {"date": date, "split": split, "scenario": "S0", **asdict(baseline_result)}
        )
        daily_rows.append(
            {
                "date": date,
                "split": split,
                "scenario": "PUBLIC_EDGE_BOUNTY",
                **asdict(policy_result),
            }
        )
        for hour, edges in sorted(baseline_hours.items()):
            detail = details[hour]
            reward = detail["reward"]
            positive = reward > 0
            if not np.any(positive):
                continue
            state = simulator.state(date, hour)
            local_features = scored_features[scored_features["hour"] == hour].reset_index(drop=True)
            frame = edges.loc[
                positive,
                ["hour", "route_id", "from_stop_id", "to_stop_id"],
            ].copy()
            frame.insert(0, "split", split)
            frame.insert(0, "date", date)
            frame["reward_won"] = reward[positive]
            frame["policy_score"] = local_features.loc[positive, "policy_score"].to_numpy(float)
            frame["train_mean_low_crowding"] = local_features.loc[
                positive, "low_crowding_train_mean"
            ].to_numpy(float)
            frame["train_mean_altprob"] = local_features.loc[
                positive, "altprob_train_mean"
            ].to_numpy(float)
            frame["date_altprob_diagnostic"] = np.asarray(state.altprob)[positive]
            frame["baseline_avg_onboard"] = np.asarray(state.load0)[positive]
            frame["policy_avg_onboard"] = detail["load1"][positive]
            frame["trips"] = np.asarray(state.trips)[positive]
            frame["edge_users_after"] = (
                detail["load1"][positive] * np.asarray(state.trips)[positive]
            )
            frame.to_csv(
                edge_destination,
                mode="wt" if first_edge else "at",
                header=first_edge,
                index=False,
                encoding="utf-8-sig" if first_edge else "utf-8",
                compression="gzip",
            )
            first_edge = False
            edge_rows_written += len(frame)
    if first_edge:
        # Keep the output contract even when the selected optimum is no bounty.
        pd.DataFrame(
            columns=[
                "date",
                "split",
                "hour",
                "route_id",
                "from_stop_id",
                "to_stop_id",
                "reward_won",
                "policy_score",
                "train_mean_low_crowding",
                "train_mean_altprob",
                "date_altprob_diagnostic",
                "baseline_avg_onboard",
                "policy_avg_onboard",
                "trips",
                "edge_users_after",
            ]
        ).to_csv(
            edge_destination,
            index=False,
            encoding="utf-8-sig",
            compression="gzip",
        )

    daily = pd.DataFrame(daily_rows)
    daily.to_csv(output / "daily_policy_results.csv", index=False, encoding="utf-8-sig")
    aggregate_scenarios(daily).to_csv(
        output / "scenario_summary.csv", index=False, encoding="utf-8-sig"
    )
    return daily, edge_rows_written


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    model_input = Path(args.model_input).resolve()
    route_choice_output = Path(args.route_choice_output).resolve()
    output = Path(args.output).resolve()
    config_path = Path(args.config).resolve()
    config: MutableMapping[str, object] = load_json(config_path)
    if args.fresh and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    # Remove the obsolete hard-cap QA filename so it cannot be mistaken for a
    # constraint produced by this no-capacity version. Reusable binary caches
    # and the Optuna SQLite database are intentionally preserved.
    obsolete_capacity_qa = output / "baseline_capacity_qa.json"
    if obsolete_capacity_qa.exists():
        obsolete_capacity_qa.unlink()

    top_paths = (
        Path(args.top_paths).resolve()
        if args.top_paths
        else route_choice_output / "candidate_paths_top5.csv.gz"
    )
    parameter_path = (
        Path(args.route_choice_parameters).resolve()
        if args.route_choice_parameters
        else route_choice_output / "route_choice_parameters.json"
    )
    if not top_paths.exists():
        raise FileNotFoundError(top_paths)
    if not parameter_path.exists():
        raise FileNotFoundError(parameter_path)
    parameters = load_choice_parameters(parameter_path)

    dates = parse_dates(model_input)
    train_dates, test_dates = split_dates(dates, config)
    hours = {int(value) for value in config.get("hours", [7, 8, 9, 17, 18, 19])}
    print("analysis dates:", dates, flush=True)
    print("training dates:", train_dates, flush=True)
    print("evaluation dates:", test_dates, flush=True)
    print("hours:", sorted(hours), flush=True)
    print("route-choice parameters:", asdict(parameters), flush=True)

    baseline = load_baseline_segments(model_input, hours)
    baseline_hours = baseline_by_hour(baseline)
    missing_hours = sorted(hours - set(baseline_hours))
    if missing_hours:
        raise RuntimeError(f"No baseline segments for configured hours: {missing_hours}")

    topology_root = output / "_topology_cache"
    print("[stage] top-5 topology cache", flush=True)
    topology_meta, topology_signature = prepare_topology_cache(
        top_paths,
        baseline,
        baseline_hours,
        topology_root,
        args,
    )
    daily_root = output / "_daily_cache"
    print("[stage] daily choice/network cache", flush=True)
    daily_meta, daily_signature = prepare_daily_cache(
        model_input,
        topology_root,
        topology_meta,
        topology_signature,
        baseline_hours,
        dates,
        parameters,
        config,
        daily_root,
        args,
    )

    print("[stage] fixed operational hour-edge features", flush=True)
    operational_features = build_operational_features(
        baseline_hours, daily_root, train_dates, config
    )
    operational_features.to_csv(
        output / "operational_hour_edge_features.csv.gz",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )
    capacity = load_reference_value(config)
    # The 55-passenger value is now a descriptive reference only, never a hard
    # feasibility constraint.  Keep baseline QA so policy side-effects can be
    # compared with the observed network.
    capacity_qa = summarize_baseline_capacity(
        daily_root,
        dates,
        sorted(baseline_hours),
        capacity,
        "diagnostic_only",
    )
    capacity_qa["constraint_enforced"] = False
    capacity_qa["interpretation"] = (
        "Reference load used only for crowding features and post-policy QA; "
        "no capacity constraint is imposed during policy simulation."
    )
    (output / "baseline_load_qa.json").write_text(
        json.dumps(capacity_qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "baseline load QA (diagnostic only): "
        f"state-edges above {capacity:g} = "
        f"{capacity_qa['nominal_overloaded_date_hour_edges']:,}, "
        f"max observed load {capacity_qa['max_observed_avg_onboard']:,.3f}; "
        "NO capacity constraint is enforced",
        flush=True,
    )
    altprob_rows = write_edge_altprob(
        output / "edge_altprob.csv.gz",
        baseline_hours,
        daily_root,
        dates,
        capacity,
    )
    print(f"edge_altprob rows: {altprob_rows:,}", flush=True)

    if args.prepare_only:
        prepare_manifest = {
            "created_at": pd.Timestamp.now().isoformat(),
            "script_version": SCRIPT_VERSION,
            "stage": "policy_cache_preparation_only",
            "model_input": str(model_input),
            "route_choice_output": str(route_choice_output),
            "top_paths": str(top_paths),
            "route_choice_parameters": str(parameter_path),
            "config": str(config_path),
            "dates": dates,
            "train_dates": train_dates,
            "test_dates": test_dates,
            "hours": sorted(hours),
            "topology_signature": topology_signature,
            "daily_cache_signature": daily_signature,
            "topology": topology_meta,
            "daily_states": len(daily_meta.get("states", {})),
            "baseline_load_qa": "baseline_load_qa.json",
            "capacity_constraint_enforced": False,
            "edge_altprob_rows": altprob_rows,
            "next_step": "rerun the same command without --prepare-only and without --fresh",
        }
        (output / "prepare_only_manifest.json").write_text(
            json.dumps(prepare_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("\nprepare-only completed:", output, flush=True)
        print(" -", output / "prepare_only_manifest.json", flush=True)
        print(" -", output / "baseline_load_qa.json", flush=True)
        print(" -", output / "operational_hour_edge_features.csv.gz", flush=True)
        print(" -", output / "edge_altprob.csv.gz", flush=True)
        print(
            "total elapsed:", format_seconds(time.perf_counter() - run_started), flush=True
        )
        return

    print("[stage] loading policy matrices", flush=True)
    topologies = {
        hour: load_hour_topology(
            topology_root, hour, baseline_hours[hour], include_board=False
        )
        for hour in sorted(baseline_hours)
    }
    simulator = PolicySimulator(topologies, daily_root, dates, parameters, config)

    print("[stage] bounty decision-function fit", flush=True)
    theta, scored_features, trials, search_meta, policy_active = fit_policy_theta(
        operational_features,
        baseline_hours,
        simulator,
        train_dates,
        parameters,
        config,
        output,
        daily_signature,
        args.policy_trials,
    )
    trials.to_csv(output / "policy_bayes_results.csv", index=False, encoding="utf-8-sig")
    scored_features.to_csv(
        output / "operational_hour_edge_rewards.csv.gz",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )
    learned = {
        "created_at": pd.Timestamp.now().isoformat(),
        "script_version": SCRIPT_VERSION,
        "policy_active": policy_active,
        "selected_policy": "SIGMOID_BOUNTY" if policy_active else "NO_BOUNTY",
        "theta": asdict(theta) if theta is not None else None,
        "score_formula": (
            "score = intercept + low_crowding*low_crowding_train_mean "
            "+ altprob*altprob_train_mean"
        ),
        "reward_formula": (
            "reward_won = policy_active * max_edge_reward_won * sigmoid(score)"
        ),
        "max_edge_reward_won": config.get("policy", {}).get(
            "max_edge_reward_won", 500.0
        ),
        "deployment_dimension": "hour x route x directed edge",
        "date_specific_rewards": False,
        "feature_source": "training-date arithmetic mean only",
        "positive_reward_edges": int(
            np.count_nonzero(scored_features["reward_won"].to_numpy(float) > 0)
        ),
        "reward_file": "operational_hour_edge_rewards.csv.gz",
        "payment_rule": "all passengers traversing each rewarded edge after policy receive that edge bounty",
        "capacity_constraint_enforced": False,
        "load_reference_for_diagnostics": capacity,
    }
    (output / "learned_bounty_function.json").write_text(
        json.dumps(learned, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "policy_search_metadata.json").write_text(
        json.dumps(search_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if theta is None:
        print("selected policy: NO_BOUNTY (all admissible sampled rewards failed to improve the training objective)", flush=True)
    else:
        print("learned theta:", asdict(theta), flush=True)

    print("[stage] fixed-policy evaluation on train and test dates", flush=True)
    daily_results, edge_rows = write_final_results(
        output,
        theta,
        scored_features,
        baseline_hours,
        simulator,
        train_dates,
        test_dates,
    )
    policy_rows = daily_results[daily_results["scenario"] == "PUBLIC_EDGE_BOUNTY"].copy()
    load_qa = {
        "reference_load": capacity,
        "capacity_constraint_enforced": False,
        "max_policy_avg_onboard": float(policy_rows["max_policy_avg_onboard"].max()) if len(policy_rows) else 0.0,
        "total_overloaded_date_hour_edges_above_reference": int(policy_rows["overloaded_edges_above_reference"].sum()) if len(policy_rows) else 0,
        "total_new_overloaded_date_hour_edges_above_reference": int(policy_rows["new_overloaded_edges_above_reference"].sum()) if len(policy_rows) else 0,
        "total_increased_existing_overload_edges": int(policy_rows["increased_existing_overload_edges"].sum()) if len(policy_rows) else 0,
        "max_load_increase": float(policy_rows["max_load_increase"].max()) if len(policy_rows) else 0.0,
        "note": "These are diagnostics only and do not affect policy feasibility or route-choice response.",
    }
    (output / "policy_load_qa.json").write_text(
        json.dumps(load_qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "created_at": pd.Timestamp.now().isoformat(),
        "script_version": SCRIPT_VERSION,
        "stage": "public_edge_bounty_function_fit_and_evaluation",
        "model_input": str(model_input),
        "route_choice_output": str(route_choice_output),
        "top_paths": str(top_paths),
        "route_choice_parameters": str(parameter_path),
        "config": str(config_path),
        "dates": dates,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "hours": sorted(hours),
        "topology_signature": topology_signature,
        "daily_cache_signature": daily_signature,
        "topology": topology_meta,
        "daily_cache": {
            "states": len(daily_meta.get("states", {})),
        },
        "baseline_load_qa": capacity_qa,
        "route_choice_parameters_value": asdict(parameters),
        "learned_bounty_function": learned,
        "policy_search": search_meta,
        "output_rows": {
            "edge_altprob": altprob_rows,
            "edge_rewards": edge_rows,
            "daily_policy_results": len(daily_results),
        },
        "model_notes": [
            "Passenger route-choice coefficients and the final top-5 set are fixed upstream inputs.",
            "The bounty function is global, while rewards are deployed by hour x route x directed edge.",
            "Low crowding and AltProb are averaged over training dates only; test dates do not affect rewards.",
            "Payment rule A is used: every passenger traversing a rewarded edge receives that edge bounty, including passengers who would have used it without the policy.",
            "The objective is total congestion relief under a per-day budget constraint; no capacity constraint is enforced.",
            "An explicit no-bounty alternative is selected when every admissible fitted reward rule has non-positive training relief.",
            "The 55-passenger load threshold is diagnostic only; MNL path-share responses are applied without capacity-based scaling.",
            "Hour-OD groups with fewer than two valid paths are omitted from route-switch calculations because their MNL share is identically one; they remain represented in observed edge loads and public expenditure.",
        ],
    }
    (output / "bounty_model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\ncompleted:", output, flush=True)
    for name in [
        "learned_bounty_function.json",
        "operational_hour_edge_features.csv.gz",
        "operational_hour_edge_rewards.csv.gz",
        "edge_altprob.csv.gz",
        "policy_bayes_results.csv",
        "daily_policy_results.csv",
        "scenario_summary.csv",
        "edge_rewards.csv.gz",
        "policy_search_metadata.json",
        "baseline_load_qa.json",
        "policy_load_qa.json",
        "bounty_model_manifest.json",
    ]:
        print(" -", output / name, flush=True)
    print("total elapsed:", format_seconds(time.perf_counter() - run_started), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted by user. Topology/daily caches and the Optuna SQLite study are reusable.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
