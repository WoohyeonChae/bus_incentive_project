from __future__ import annotations

"""Fit the passenger route-choice model and create a self-consistent top-5 choice set.

Input
-----
``build_model_input_purpose_candidates15.py`` output directory.  The important
file is ``candidate_pool.csv.gz`` with at most 15 paths per hour-OD.

Output
------
* ``route_choice_parameters.json``
* ``candidate_paths_top5.csv.gz``
* ``route_choice_fit_history.csv``
* ``choice_set_qa.json``
* ``route_choice_manifest.json``

The utility used for baseline route choice is

    V = beta_total_time * total_time
      + beta_walk_time_extra * walk_time
      + beta_transfer * transfers

The monetary coefficient is not estimated from the observational data.  It is
calculated from the fitted total-time coefficient and the configured value of
time.  The script first fits the model on the full candidate pool, selects five
paths per hour-OD, refits on the retained set, and repeats until the retained
set is stable (or the configured iteration limit is reached).

This program deliberately does *not* fit the public-edge bounty function.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

EPS = 1.0e-12
SCRIPT_VERSION = "route-choice-top5-v1.1.0"
FEATURE_COLUMNS = ["total_time", "walk_time", "transfers"]
REQUIRED_COLUMNS = [
    "od_index",
    "hour",
    "origin_stop_id",
    "destination_stop_id",
    "signature",
    "total_time",
    "walk_time",
    "transfers",
    "observed_passengers_train",
]


# -----------------------------------------------------------------------------
# General utilities
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


def require_columns(columns: Sequence[str], required: Sequence[str], label: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


class Progress:
    def __init__(
        self,
        label: str,
        total: Optional[int],
        interval_seconds: float = 15.0,
    ) -> None:
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
            prefix = (
                f"[{self.label}] {done:,}/{self.total:,} ({pct:5.1f}%), "
                f"{rate:,.1f}/s, elapsed {format_seconds(elapsed)}, "
                f"ETA {format_seconds(eta)}"
            )
        else:
            prefix = (
                f"[{self.label}] {done:,}, {rate:,.1f}/s, "
                f"elapsed {format_seconds(elapsed)}"
            )
        if extra:
            prefix += f", {extra}"
        print(prefix, flush=True)
        self.last_print = now


# -----------------------------------------------------------------------------
# CLI/configuration
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit weighted MNL route-choice parameters on the 15-path pool and "
            "create a self-consistent top-5 choice set"
        )
    )
    parser.add_argument(
        "--model-input",
        required=True,
        help="build_model_input_purpose_candidates15.py output directory",
    )
    parser.add_argument("--output", required=True, help="route-choice output directory")
    parser.add_argument(
        "--config",
        required=True,
        help="JSON configuration shared with the bounty model",
    )
    parser.add_argument(
        "--candidate-pool",
        default="",
        help="candidate_pool.csv.gz; default: <model-input>/candidate_pool.csv.gz",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="override final choice-set size; 0 uses config/choice_set_spec.json",
    )
    parser.add_argument(
        "--max-fixed-point-iterations",
        type=int,
        default=0,
        help="override fixed-point selection/refit iteration limit",
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument(
        "--fit-block-rows",
        type=int,
        default=2_000_000,
        help="maximum candidate rows processed per likelihood block",
    )
    parser.add_argument("--progress-seconds", type=float, default=15.0)
    parser.add_argument(
        "--strict-probability-top-k",
        action="store_true",
        help=(
            "do not force observed training paths into top-k; by default positive-count "
            "observed paths are retained first, then remaining slots are filled by probability"
        ),
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="rebuild compact numeric cache even when the candidate file is unchanged",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete the route-choice output directory before running",
    )
    return parser.parse_args()


def resolve_choice_config(
    model_input: Path,
    config: Mapping[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    choice = dict(config.get("choice_model", {}))
    choice_set = dict(config.get("choice_set", {}))
    spec_path = model_input / "choice_set_spec.json"
    spec: Dict[str, object] = load_json(spec_path) if spec_path.exists() else {}

    top_k = int(
        args.top_k
        or choice_set.get("top_k", 0)
        or choice.get("top_k", 0)
        or spec.get("downstream_choice_set_size", 0)
        or 5
    )
    max_fp = int(
        args.max_fixed_point_iterations
        or choice_set.get("max_fixed_point_iterations", 0)
        or choice.get("max_fixed_point_iterations", 0)
        or 8
    )
    if top_k < 2:
        raise ValueError("top-k must be at least 2")
    if max_fp < 1:
        raise ValueError("max_fixed_point_iterations must be at least 1")

    initial = choice.get("initial_beta", [-0.08, -0.08, -0.8])
    if not isinstance(initial, (list, tuple)) or len(initial) != 3:
        raise ValueError("choice_model.initial_beta must contain three values")
    bounds_cfg = dict(choice.get("bounds", {}))
    bounds = [
        list(map(float, bounds_cfg.get("beta_total_time", [-5.0, -1.0e-6]))),
        list(map(float, bounds_cfg.get("beta_walk_time_extra", [-10.0, 0.0]))),
        list(map(float, bounds_cfg.get("beta_transfer", [-20.0, 0.0]))),
    ]
    for name, values in zip(
        ["beta_total_time", "beta_walk_time_extra", "beta_transfer"], bounds
    ):
        if len(values) != 2 or values[1] <= values[0]:
            raise ValueError(f"invalid bounds for {name}: {values}")

    configured_selection_mode = str(
        choice_set.get("selection_mode", "retain_observed")
    ).strip().lower()
    aliases = {
        "retain-observed": "retain_observed",
        "preserve_observed": "retain_observed",
        "probability": "pure_probability",
        "strict_probability": "pure_probability",
        "strict-probability": "pure_probability",
    }
    configured_selection_mode = aliases.get(
        configured_selection_mode, configured_selection_mode
    )
    if configured_selection_mode not in {"retain_observed", "pure_probability"}:
        raise ValueError(
            "choice_set.selection_mode must be 'retain_observed' or "
            "'pure_probability'"
        )
    if args.strict_probability_top_k:
        configured_selection_mode = "pure_probability"

    return {
        "top_k": top_k,
        "max_fixed_point_iterations": max_fp,
        "initial_beta": [float(value) for value in initial],
        "bounds": bounds,
        "l2_regularization": float(choice.get("l2_regularization", 1.0e-5)),
        "max_iterations": int(choice.get("max_iterations", 300)),
        "ftol": float(choice.get("ftol", 1.0e-11)),
        "gtol": float(choice.get("gtol", 1.0e-7)),
        "max_line_search_steps": int(choice.get("max_line_search_steps", 30)),
        "value_of_time_won_per_hour": float(
            choice.get("value_of_time_won_per_hour", 1101.0)
        ),
        "selection_mode": configured_selection_mode,
        "preserve_observed": configured_selection_mode == "retain_observed",
        "candidate_limit": int(
            choice_set.get(
                "candidate_limit", spec.get("candidate_limit_per_hour_od", 15)
            )
        ),
    }


# -----------------------------------------------------------------------------
# Compact numeric cache
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericCache:
    root: Path
    rows: int
    groups: int
    features: np.memmap
    observed: np.memmap
    group_starts: np.ndarray
    metadata: Dict[str, object]

    @property
    def group_lengths(self) -> np.ndarray:
        return np.diff(self.group_starts)


def cache_paths(root: Path) -> Dict[str, Path]:
    return {
        "features": root / "features.f32",
        "observed": root / "observed.f64",
        "starts": root / "group_starts.npy",
        "groups": root / "groups.csv.gz",
        "meta": root / "cache_meta.json",
    }


def build_numeric_cache(
    candidate_pool: Path,
    cache_root: Path,
    signature: str,
    chunksize: int,
    progress_seconds: float,
) -> NumericCache:
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_root)

    header = pd.read_csv(candidate_pool, compression="infer", nrows=0)
    require_columns(list(header.columns), REQUIRED_COLUMNS, "candidate_pool")

    starts: List[int] = []
    row_count = 0
    group_count = 0
    previous_od: Optional[int] = None
    previous_key: Optional[Tuple[int, str, str]] = None
    last_od = -1

    meter = Progress("choice-cache", None, progress_seconds)
    with open(paths["features"], "wb") as feature_handle, open(
        paths["observed"], "wb"
    ) as observed_handle, gzip.open(
        paths["groups"], "wt", encoding="utf-8-sig", newline=""
    ) as group_handle:
        for chunk in pd.read_csv(
            candidate_pool,
            compression="infer",
            usecols=REQUIRED_COLUMNS,
            dtype={
                "od_index": "int64",
                "hour": "int16",
                "origin_stop_id": "string",
                "destination_stop_id": "string",
                "signature": "string",
                "total_time": "float64",
                "walk_time": "float64",
                "transfers": "float64",
                "observed_passengers_train": "float64",
            },
            chunksize=max(1, int(chunksize)),
            low_memory=False,
        ):
            if chunk.empty:
                continue
            numeric = chunk[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
            observed = pd.to_numeric(
                chunk["observed_passengers_train"], errors="coerce"
            ).fillna(0.0)
            od_values = pd.to_numeric(chunk["od_index"], errors="coerce")
            if numeric.isna().any().any() or od_values.isna().any():
                raise ValueError(
                    "candidate_pool contains invalid total_time/walk_time/transfers/od_index"
                )
            if (numeric["total_time"] < 0).any() or (numeric["walk_time"] < 0).any():
                raise ValueError("candidate_pool contains negative path time")
            if (numeric["transfers"] < 0).any():
                raise ValueError("candidate_pool contains negative transfer count")
            if (observed < 0).any():
                raise ValueError("candidate_pool contains negative observed passenger count")

            values = numeric.to_numpy(dtype=np.float32, copy=False)
            values.tofile(feature_handle)
            observed.to_numpy(dtype=np.float64, copy=False).tofile(observed_handle)

            # The path builder writes every hour-OD contiguously and od_index is globally
            # nondecreasing.  Detect group boundaries vectorially so the cache build
            # scales with the number of OD groups rather than looping in Python over
            # every one of the potentially tens of millions of candidate rows.
            od_array = od_values.to_numpy(dtype=np.int64, copy=False)
            hour_array = pd.to_numeric(
                chunk["hour"], errors="raise"
            ).to_numpy(dtype=np.int16, copy=False)
            origin_array = chunk["origin_stop_id"].astype(str).to_numpy()
            destination_array = chunk["destination_stop_id"].astype(str).to_numpy()
            if int(od_array[0]) < last_od or np.any(od_array[1:] < od_array[:-1]):
                raise ValueError(
                    "candidate_pool od_index is not nondecreasing. Use the unmodified "
                    "candidate_pool produced by build_model_input_purpose_candidates15.py."
                )

            same_as_previous_row = od_array[1:] == od_array[:-1]
            inconsistent = same_as_previous_row & (
                (hour_array[1:] != hour_array[:-1])
                | (origin_array[1:] != origin_array[:-1])
                | (destination_array[1:] != destination_array[:-1])
            )
            if np.any(inconsistent):
                index = int(np.flatnonzero(inconsistent)[0] + 1)
                raise ValueError(
                    f"od_index {int(od_array[index])} maps to multiple hour-OD keys "
                    "within candidate_pool"
                )

            first_key = (
                int(hour_array[0]),
                str(origin_array[0]),
                str(destination_array[0]),
            )
            if previous_od is not None and int(od_array[0]) == previous_od:
                if first_key != previous_key:
                    raise ValueError(
                        f"od_index {previous_od} maps to multiple hour-OD keys: "
                        f"{previous_key} versus {first_key}"
                    )

            new_group = np.empty(len(chunk), dtype=bool)
            new_group[0] = previous_od is None or int(od_array[0]) != previous_od
            new_group[1:] = od_array[1:] != od_array[:-1]
            boundary = np.flatnonzero(new_group)
            if len(boundary):
                absolute_starts = row_count + boundary.astype(np.int64)
                starts.extend(map(int, absolute_starts))
                group_frame = pd.DataFrame(
                    {
                        "group_index": np.arange(
                            group_count,
                            group_count + len(boundary),
                            dtype=np.int64,
                        ),
                        "od_index": od_array[boundary],
                        "hour": hour_array[boundary],
                        "origin_stop_id": origin_array[boundary],
                        "destination_stop_id": destination_array[boundary],
                        "start_row": absolute_starts,
                    }
                )
                group_frame.to_csv(
                    group_handle,
                    header=group_count == 0,
                    index=False,
                )
                group_count += len(boundary)

            previous_od = int(od_array[-1])
            previous_key = (
                int(hour_array[-1]),
                str(origin_array[-1]),
                str(destination_array[-1]),
            )
            last_od = previous_od

            row_count += len(chunk)
            meter.update(row_count, extra=f"groups {group_count:,}")

    if row_count == 0 or group_count == 0:
        raise RuntimeError("candidate_pool has no rows")
    starts_array = np.asarray(starts + [row_count], dtype=np.int64)
    np.save(paths["starts"], starts_array, allow_pickle=False)
    metadata = {
        "script_version": SCRIPT_VERSION,
        "signature": signature,
        "candidate_pool": file_signature(candidate_pool),
        "rows": row_count,
        "groups": group_count,
        "feature_dtype": "float32",
        "observed_dtype": "float64",
        "feature_columns": FEATURE_COLUMNS,
        "grouping": "contiguous global od_index",
    }
    paths["meta"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meter.update(row_count, extra=f"groups {group_count:,}", force=True)
    return load_numeric_cache(cache_root)


def load_numeric_cache(cache_root: Path) -> NumericCache:
    paths = cache_paths(cache_root)
    metadata = load_json(paths["meta"])
    rows = int(metadata["rows"])
    groups = int(metadata["groups"])
    starts = np.load(paths["starts"], mmap_mode="r")
    if len(starts) != groups + 1 or int(starts[-1]) != rows:
        raise RuntimeError("route-choice numeric cache is inconsistent")
    features = np.memmap(
        paths["features"], dtype=np.float32, mode="r", shape=(rows, 3)
    )
    observed = np.memmap(paths["observed"], dtype=np.float64, mode="r", shape=(rows,))
    return NumericCache(
        root=cache_root,
        rows=rows,
        groups=groups,
        features=features,
        observed=observed,
        group_starts=starts,
        metadata=metadata,
    )


def prepare_numeric_cache(
    candidate_pool: Path,
    output: Path,
    choice_cfg: Mapping[str, object],
    args: argparse.Namespace,
) -> Tuple[NumericCache, str]:
    signature_payload = {
        "script_version": SCRIPT_VERSION,
        "candidate_pool": file_signature(candidate_pool),
        "features": FEATURE_COLUMNS,
        "candidate_limit": choice_cfg.get("candidate_limit"),
    }
    signature = hash_payload(signature_payload)
    cache_root = output / "_choice_cache"
    paths = cache_paths(cache_root)
    reusable = False
    if not args.rebuild_cache and paths["meta"].exists():
        try:
            reusable = load_json(paths["meta"]).get("signature") == signature
        except Exception:
            reusable = False
    if reusable:
        cache = load_numeric_cache(cache_root)
        print(
            f"[choice-cache] reused: {cache.rows:,} paths, {cache.groups:,} hour-OD groups",
            flush=True,
        )
        return cache, signature
    return (
        build_numeric_cache(
            candidate_pool,
            cache_root,
            signature,
            args.chunksize,
            args.progress_seconds,
        ),
        signature,
    )


# -----------------------------------------------------------------------------
# Weighted grouped MNL
# -----------------------------------------------------------------------------


@dataclass
class FitResult:
    beta: np.ndarray
    success: bool
    message: str
    iterations: int
    function_evaluations: int
    penalized_objective: float
    log_likelihood: float
    null_log_likelihood: float
    mcfadden_r2: float
    weighted_observations: float
    fitting_groups: int
    fitting_rows: int
    standard_errors: Optional[np.ndarray] = None


class GroupedMNLObjective:
    def __init__(
        self,
        cache: NumericCache,
        selection: Optional[np.memmap],
        regularization: float,
        block_rows: int,
        progress_seconds: float,
        label: str,
    ) -> None:
        self.cache = cache
        self.selection = selection
        self.regularization = max(0.0, float(regularization))
        self.block_rows = max(10_000, int(block_rows))
        self.progress_seconds = max(0.1, float(progress_seconds))
        self.label = label
        self.evaluations = 0
        self.started = time.perf_counter()
        self.last_print = self.started
        self.last_value = float("nan")
        self.last_raw_log_likelihood = float("nan")
        self.last_stats: Dict[str, float] = {}

    def group_blocks(self) -> Iterator[Tuple[int, int]]:
        starts = self.cache.group_starts
        group = 0
        while group < self.cache.groups:
            row_start = int(starts[group])
            target = row_start + self.block_rows
            end = int(np.searchsorted(starts, target, side="right") - 1)
            end = max(group + 1, min(end, self.cache.groups))
            yield group, end
            group = end

    def _selected_block(
        self, row_start: int, row_end: int
    ) -> np.ndarray:
        if self.selection is None:
            return np.ones(row_end - row_start, dtype=bool)
        return np.asarray(self.selection[row_start:row_end], dtype=np.uint8) > 0

    def evaluate(self, beta: np.ndarray) -> Tuple[float, np.ndarray]:
        beta = np.asarray(beta, dtype=np.float64)
        total_nll = 0.0
        total_grad = np.zeros(3, dtype=np.float64)
        total_obs = 0.0
        total_groups = 0
        total_rows = 0
        starts_all = self.cache.group_starts

        for group_start, group_end in self.group_blocks():
            row_start = int(starts_all[group_start])
            row_end = int(starts_all[group_end])
            full_lengths = np.diff(starts_all[group_start : group_end + 1]).astype(
                np.int64, copy=False
            )
            block_mask = self._selected_block(row_start, row_end)
            local_starts = np.r_[0, np.cumsum(full_lengths[:-1])]
            selected_lengths = np.add.reduceat(
                block_mask.astype(np.int32, copy=False), local_starts
            ).astype(np.int64, copy=False)

            counts_full = np.asarray(
                self.cache.observed[row_start:row_end], dtype=np.float64
            )
            selected_counts = counts_full * block_mask
            observed_totals = np.add.reduceat(selected_counts, local_starts)
            valid_groups = (selected_lengths >= 2) & (observed_totals > 0)
            if not np.any(valid_groups):
                continue

            x_selected = np.asarray(
                self.cache.features[row_start:row_end][block_mask], dtype=np.float64
            )
            counts_selected = counts_full[block_mask]
            selected_group_ids = np.repeat(
                np.arange(len(selected_lengths), dtype=np.int32), selected_lengths
            )
            valid_rows = valid_groups[selected_group_ids]
            x = x_selected[valid_rows]
            counts = counts_selected[valid_rows]
            lengths = selected_lengths[valid_groups]
            group_totals = observed_totals[valid_groups]
            starts = np.r_[0, np.cumsum(lengths[:-1])].astype(np.int64, copy=False)
            group_index = np.repeat(
                np.arange(len(lengths), dtype=np.int32), lengths
            )

            utility = x @ beta
            maxima = np.maximum.reduceat(utility, starts)
            shifted = utility - maxima[group_index]
            exponent = np.exp(np.clip(shifted, -700.0, 50.0))
            denominators = np.add.reduceat(exponent, starts)
            probabilities = exponent / np.maximum(denominators[group_index], EPS)
            log_probabilities = shifted - np.log(
                np.maximum(denominators[group_index], EPS)
            )
            total_nll -= float(np.dot(counts, log_probabilities))
            expected = probabilities * group_totals[group_index]
            total_grad -= x.T @ (counts - expected)
            total_obs += float(group_totals.sum())
            total_groups += int(len(lengths))
            total_rows += int(len(x))

        if total_groups == 0 or total_obs <= 0:
            raise RuntimeError(
                "No fitting groups with at least two selected alternatives and positive "
                "observed training passengers"
            )
        raw_nll = total_nll
        if self.regularization > 0:
            total_nll += self.regularization * float(np.dot(beta, beta))
            total_grad += 2.0 * self.regularization * beta

        self.evaluations += 1
        self.last_value = total_nll
        self.last_raw_log_likelihood = -raw_nll
        self.last_stats = {
            "weighted_observations": total_obs,
            "fitting_groups": float(total_groups),
            "fitting_rows": float(total_rows),
        }
        now = time.perf_counter()
        if now - self.last_print >= self.progress_seconds:
            elapsed = now - self.started
            print(
                f"[{self.label}] eval {self.evaluations}, objective {total_nll:,.3f}, "
                f"beta=({beta[0]:.6g}, {beta[1]:.6g}, {beta[2]:.6g}), "
                f"elapsed {format_seconds(elapsed)}, avg {elapsed/self.evaluations:,.2f}s/eval",
                flush=True,
            )
            self.last_print = now
        return total_nll, total_grad

    def fit_statistics(self, beta: np.ndarray) -> Dict[str, float]:
        beta = np.asarray(beta, dtype=np.float64)
        log_likelihood = 0.0
        null_log_likelihood = 0.0
        observations = 0.0
        fitting_groups = 0
        fitting_rows = 0
        starts_all = self.cache.group_starts
        for group_start, group_end in self.group_blocks():
            row_start = int(starts_all[group_start])
            row_end = int(starts_all[group_end])
            full_lengths = np.diff(starts_all[group_start : group_end + 1]).astype(
                np.int64, copy=False
            )
            mask = self._selected_block(row_start, row_end)
            local_starts = np.r_[0, np.cumsum(full_lengths[:-1])]
            selected_lengths = np.add.reduceat(mask.astype(np.int32), local_starts).astype(
                np.int64
            )
            counts_full = np.asarray(
                self.cache.observed[row_start:row_end], dtype=np.float64
            )
            counts_selected = counts_full * mask
            totals = np.add.reduceat(counts_selected, local_starts)
            valid = (selected_lengths >= 2) & (totals > 0)
            if not np.any(valid):
                continue
            x_selected = np.asarray(
                self.cache.features[row_start:row_end][mask], dtype=np.float64
            )
            c_selected = counts_full[mask]
            gids_selected = np.repeat(
                np.arange(len(selected_lengths), dtype=np.int32), selected_lengths
            )
            valid_rows = valid[gids_selected]
            x = x_selected[valid_rows]
            counts = c_selected[valid_rows]
            lengths = selected_lengths[valid]
            group_totals = totals[valid]
            starts = np.r_[0, np.cumsum(lengths[:-1])].astype(np.int64)
            gids = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
            utility = x @ beta
            maxima = np.maximum.reduceat(utility, starts)
            shifted = utility - maxima[gids]
            exp_u = np.exp(np.clip(shifted, -700.0, 50.0))
            den = np.add.reduceat(exp_u, starts)
            logp = shifted - np.log(np.maximum(den[gids], EPS))
            log_likelihood += float(np.dot(counts, logp))
            null_log_likelihood -= float(
                np.dot(group_totals, np.log(np.maximum(lengths, 1)))
            )
            observations += float(group_totals.sum())
            fitting_groups += int(len(lengths))
            fitting_rows += int(len(x))
        rho2 = (
            1.0 - log_likelihood / null_log_likelihood
            if null_log_likelihood < -EPS
            else float("nan")
        )
        return {
            "log_likelihood": log_likelihood,
            "null_log_likelihood": null_log_likelihood,
            "mcfadden_r2": rho2,
            "weighted_observations": observations,
            "fitting_groups": float(fitting_groups),
            "fitting_rows": float(fitting_rows),
        }

    def information_matrix(self, beta: np.ndarray) -> np.ndarray:
        beta = np.asarray(beta, dtype=np.float64)
        information = np.zeros((3, 3), dtype=np.float64)
        starts_all = self.cache.group_starts
        for group_start, group_end in self.group_blocks():
            row_start = int(starts_all[group_start])
            row_end = int(starts_all[group_end])
            full_lengths = np.diff(starts_all[group_start : group_end + 1]).astype(
                np.int64, copy=False
            )
            mask = self._selected_block(row_start, row_end)
            local_starts = np.r_[0, np.cumsum(full_lengths[:-1])]
            selected_lengths = np.add.reduceat(mask.astype(np.int32), local_starts).astype(
                np.int64
            )
            counts_full = np.asarray(
                self.cache.observed[row_start:row_end], dtype=np.float64
            )
            totals = np.add.reduceat(counts_full * mask, local_starts)
            valid = (selected_lengths >= 2) & (totals > 0)
            if not np.any(valid):
                continue
            x_selected = np.asarray(
                self.cache.features[row_start:row_end][mask], dtype=np.float64
            )
            gids_selected = np.repeat(
                np.arange(len(selected_lengths), dtype=np.int32), selected_lengths
            )
            x = x_selected[valid[gids_selected]]
            lengths = selected_lengths[valid]
            totals_valid = totals[valid]
            starts = np.r_[0, np.cumsum(lengths[:-1])].astype(np.int64)
            group_index = np.repeat(
                np.arange(len(lengths), dtype=np.int32), lengths
            )
            utility = x @ beta
            maxima = np.maximum.reduceat(utility, starts)
            shifted = utility - maxima[group_index]
            exponent = np.exp(np.clip(shifted, -700.0, 50.0))
            denominators = np.add.reduceat(exponent, starts)
            probabilities = exponent / np.maximum(
                denominators[group_index], EPS
            )
            row_weights = probabilities * totals_valid[group_index]
            information += x.T @ (row_weights[:, None] * x)
            means = np.column_stack(
                [
                    np.add.reduceat(probabilities * x[:, column], starts)
                    for column in range(x.shape[1])
                ]
            )
            weighted_means = means * np.sqrt(totals_valid)[:, None]
            information -= weighted_means.T @ weighted_means
        if self.regularization > 0:
            information += 2.0 * self.regularization * np.eye(3)
        return information


def open_selection(path: Path, rows: int, mode: str = "r") -> np.memmap:
    return np.memmap(path, dtype=np.uint8, mode=mode, shape=(rows,))


def fit_mnl(
    cache: NumericCache,
    selection_path: Optional[Path],
    initial_beta: np.ndarray,
    cfg: Mapping[str, object],
    args: argparse.Namespace,
    label: str,
    compute_standard_errors: bool = False,
) -> FitResult:
    selection = (
        open_selection(selection_path, cache.rows, "r")
        if selection_path is not None
        else None
    )
    objective = GroupedMNLObjective(
        cache,
        selection,
        float(cfg["l2_regularization"]),
        args.fit_block_rows,
        args.progress_seconds,
        label,
    )
    bounds = [tuple(map(float, value)) for value in cfg["bounds"]]
    max_iterations = int(cfg["max_iterations"])
    callback_state = {"iteration": 0, "started": time.perf_counter()}

    def callback(xk: np.ndarray) -> None:
        callback_state["iteration"] += 1
        elapsed = time.perf_counter() - callback_state["started"]
        rate = callback_state["iteration"] / max(elapsed, EPS)
        eta = (max_iterations - callback_state["iteration"]) / max(rate, EPS)
        print(
            f"[{label}] iteration {callback_state['iteration']}/{max_iterations}, "
            f"objective {objective.last_value:,.3f}, "
            f"beta=({xk[0]:.6g}, {xk[1]:.6g}, {xk[2]:.6g}), "
            f"elapsed {format_seconds(elapsed)}, upper-bound ETA {format_seconds(eta)}",
            flush=True,
        )

    result = minimize(
        objective.evaluate,
        np.asarray(initial_beta, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": max_iterations,
            "ftol": float(cfg["ftol"]),
            "gtol": float(cfg["gtol"]),
            "maxls": int(cfg["max_line_search_steps"]),
        },
    )
    beta = np.asarray(result.x, dtype=np.float64)
    stats = objective.fit_statistics(beta)
    standard_errors: Optional[np.ndarray] = None
    if compute_standard_errors:
        print(f"[{label}] computing observed-information standard errors", flush=True)
        information = objective.information_matrix(beta)
        covariance = np.linalg.pinv(information, hermitian=True)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return FitResult(
        beta=beta,
        success=bool(result.success),
        message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
        function_evaluations=int(getattr(result, "nfev", objective.evaluations)),
        penalized_objective=float(result.fun),
        log_likelihood=float(stats["log_likelihood"]),
        null_log_likelihood=float(stats["null_log_likelihood"]),
        mcfadden_r2=float(stats["mcfadden_r2"]),
        weighted_observations=float(stats["weighted_observations"]),
        fitting_groups=int(stats["fitting_groups"]),
        fitting_rows=int(stats["fitting_rows"]),
        standard_errors=standard_errors,
    )


# -----------------------------------------------------------------------------
# Top-k selection
# -----------------------------------------------------------------------------


@dataclass
class SelectionStats:
    groups: int = 0
    selected_rows: int = 0
    groups_with_fewer_than_k: int = 0
    groups_changed: int = 0
    observed_passengers_total: float = 0.0
    observed_passengers_retained: float = 0.0
    observed_alternatives_overflow_groups: int = 0
    observed_passengers_excluded_by_overflow: float = 0.0

    @property
    def observed_retention_rate(self) -> float:
        return self.observed_passengers_retained / max(
            self.observed_passengers_total, EPS
        )


def select_top_k(
    cache: NumericCache,
    beta: np.ndarray,
    top_k: int,
    destination: Path,
    preserve_observed: bool,
    progress_seconds: float,
    previous_path: Optional[Path] = None,
) -> SelectionStats:
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected = open_selection(destination, cache.rows, "w+")
    selected[:] = 0
    previous = (
        open_selection(previous_path, cache.rows, "r")
        if previous_path is not None
        else None
    )
    stats = SelectionStats(groups=cache.groups)
    meter = Progress("top-k-select", cache.groups, progress_seconds)
    starts = cache.group_starts
    beta = np.asarray(beta, dtype=np.float64)

    for group in range(cache.groups):
        start = int(starts[group])
        end = int(starts[group + 1])
        count = end - start
        if count <= 0:
            continue
        keep_count = min(top_k, count)
        if count < top_k:
            stats.groups_with_fewer_than_k += 1
        x = np.asarray(cache.features[start:end], dtype=np.float64)
        observed = np.asarray(cache.observed[start:end], dtype=np.float64)
        utility = x @ beta
        probability_order = np.argsort(-utility, kind="stable")

        if preserve_observed:
            observed_local = np.flatnonzero(observed > 0)
            if len(observed_local) >= keep_count:
                # Exact five-alternative assumption wins when more than five observed
                # paths exist.  Retain the highest-count observed paths and report the
                # passenger mass excluded by this unavoidable overflow.
                ordering = np.lexsort(
                    (
                        observed_local,
                        -utility[observed_local],
                        -observed[observed_local],
                    )
                )
                chosen = observed_local[ordering[:keep_count]]
                if len(observed_local) > keep_count:
                    stats.observed_alternatives_overflow_groups += 1
                    omitted = np.setdiff1d(
                        observed_local, chosen, assume_unique=False
                    )
                    stats.observed_passengers_excluded_by_overflow += float(
                        observed[omitted].sum()
                    )
            else:
                chosen_list = list(map(int, observed_local))
                chosen_set = set(chosen_list)
                for index in probability_order:
                    index_int = int(index)
                    if index_int in chosen_set:
                        continue
                    chosen_list.append(index_int)
                    chosen_set.add(index_int)
                    if len(chosen_list) >= keep_count:
                        break
                chosen = np.asarray(chosen_list, dtype=np.int64)
        else:
            chosen = probability_order[:keep_count]

        selected[start + chosen] = 1
        stats.selected_rows += int(len(chosen))
        total_observed = float(observed.sum())
        retained_observed = float(observed[chosen].sum())
        stats.observed_passengers_total += total_observed
        stats.observed_passengers_retained += retained_observed
        if previous is not None:
            old = np.asarray(previous[start:end], dtype=np.uint8)
            new = np.asarray(selected[start:end], dtype=np.uint8)
            if not np.array_equal(old, new):
                stats.groups_changed += 1
        meter.update(
            group + 1,
            extra=(
                f"selected {stats.selected_rows:,}, observed retention "
                f"{100.0*stats.observed_retention_rate:,.3f}%"
            ),
        )

    selected.flush()
    meter.update(
        cache.groups,
        extra=(
            f"selected {stats.selected_rows:,}, changed {stats.groups_changed:,}, "
            f"observed retention {100.0*stats.observed_retention_rate:,.3f}%"
        ),
        force=True,
    )
    return stats


def mask_hash(path: Path, rows: int, chunk_rows: int = 4_000_000) -> str:
    values = open_selection(path, rows, "r")
    digest = hashlib.sha256()
    for start in range(0, rows, chunk_rows):
        digest.update(np.asarray(values[start : start + chunk_rows]).tobytes())
    return digest.hexdigest()


# -----------------------------------------------------------------------------
# Final choice-set output
# -----------------------------------------------------------------------------


def create_probability_cache(
    cache: NumericCache,
    selection_path: Path,
    beta: np.ndarray,
    output: Path,
    progress_seconds: float,
) -> Tuple[Path, Path]:
    probability_path = output / "_choice_cache" / "baseline_probability.f32"
    rank_path = output / "_choice_cache" / "probability_rank.i16"
    probabilities = np.memmap(
        probability_path, dtype=np.float32, mode="w+", shape=(cache.rows,)
    )
    ranks = np.memmap(rank_path, dtype=np.int16, mode="w+", shape=(cache.rows,))
    probabilities[:] = np.nan
    ranks[:] = 0
    selected = open_selection(selection_path, cache.rows, "r")
    starts = cache.group_starts
    meter = Progress("top-k-probability", cache.groups, progress_seconds)
    beta = np.asarray(beta, dtype=np.float64)
    for group in range(cache.groups):
        start = int(starts[group])
        end = int(starts[group + 1])
        local = np.flatnonzero(np.asarray(selected[start:end]) > 0)
        if len(local) == 0:
            continue
        x = np.asarray(cache.features[start:end][local], dtype=np.float64)
        utility = x @ beta
        exp_u = np.exp(np.clip(utility - np.max(utility), -700.0, 50.0))
        p = exp_u / max(float(exp_u.sum()), EPS)
        order = np.argsort(-p, kind="stable")
        local_rank = np.empty(len(local), dtype=np.int16)
        local_rank[order] = np.arange(1, len(local) + 1, dtype=np.int16)
        probabilities[start + local] = p.astype(np.float32)
        ranks[start + local] = local_rank
        meter.update(group + 1)
    probabilities.flush()
    ranks.flush()
    meter.update(cache.groups, force=True)
    return probability_path, rank_path


def write_top_k_file(
    candidate_pool: Path,
    destination: Path,
    cache: NumericCache,
    selection_path: Path,
    probability_path: Path,
    rank_path: Path,
    beta: np.ndarray,
    chunksize: int,
    progress_seconds: float,
) -> int:
    selected = open_selection(selection_path, cache.rows, "r")
    probabilities = np.memmap(
        probability_path, dtype=np.float32, mode="r", shape=(cache.rows,)
    )
    ranks = np.memmap(rank_path, dtype=np.int16, mode="r", shape=(cache.rows,))
    selected_group_lengths = np.add.reduceat(
        np.asarray(selected, dtype=np.uint8), cache.group_starts[:-1]
    ).astype(np.int16)
    # Expand group size only while streaming; no permanent n-row array is needed.
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    row_offset = 0
    written = 0
    first = True
    meter = Progress("write-top-k", cache.rows, progress_seconds)
    beta = np.asarray(beta, dtype=np.float64)
    # Open one gzip stream for the entire output.  Reopening gzip for every chunk
    # creates many concatenated members and is substantially slower on a multi-
    # million-row top-5 file.
    with gzip.open(destination, "wt", encoding="utf-8-sig", newline="") as handle:
        for chunk in pd.read_csv(
            candidate_pool,
            compression="infer",
            dtype=str,
            chunksize=max(1, int(chunksize)),
            low_memory=False,
        ):
            start = row_offset
            end = start + len(chunk)
            local_mask = np.asarray(selected[start:end], dtype=np.uint8) > 0
            if np.any(local_mask):
                kept = chunk.loc[local_mask].copy()
                numeric = kept[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
                kept["baseline_utility"] = numeric.to_numpy(float) @ beta
                kept["baseline_probability"] = np.asarray(
                    probabilities[start:end], dtype=np.float32
                )[local_mask]
                kept["probability_rank"] = np.asarray(
                    ranks[start:end], dtype=np.int16
                )[local_mask]
                # od_index is the compact cache group key.  Convert each kept row to
                # its group position by binary search in group starts.
                absolute_rows = np.arange(start, end, dtype=np.int64)[local_mask]
                group_ids = np.searchsorted(
                    cache.group_starts, absolute_rows, side="right"
                ) - 1
                kept["choice_set_size"] = selected_group_lengths[group_ids]
                observed = pd.to_numeric(
                    kept.get("observed_passengers_train", 0), errors="coerce"
                ).fillna(0.0)
                kept["is_observed_training_path"] = (observed > 0).astype(np.int8)

                # Explicit aliases make the stage boundary self-describing and keep
                # compatibility with older versions of the policy optimizer.
                kept.insert(
                    0,
                    "top5_path_index",
                    np.arange(written, written + len(kept), dtype=np.int64),
                )
                kept["baseline_probability_top5"] = kept["baseline_probability"]
                kept["probability_rank_top5"] = kept["probability_rank"]
                kept["choice_set_size_actual"] = kept["choice_set_size"]

                kept.to_csv(handle, header=first, index=False)
                first = False
                written += len(kept)
            row_offset = end
            meter.update(row_offset, extra=f"written {written:,}")

    if row_offset != cache.rows:
        raise RuntimeError(
            f"candidate file row count changed during run: cache={cache.rows:,}, "
            f"current={row_offset:,}"
        )
    if first:
        raise RuntimeError("top-k selection produced no output rows")
    meter.update(row_offset, extra=f"written {written:,}", force=True)
    return written


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def fit_to_dict(fit: FitResult) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "beta": [float(value) for value in fit.beta],
        "success": fit.success,
        "message": fit.message,
        "iterations": fit.iterations,
        "function_evaluations": fit.function_evaluations,
        "penalized_objective": fit.penalized_objective,
        "log_likelihood": fit.log_likelihood,
        "null_log_likelihood": fit.null_log_likelihood,
        "mcfadden_r2": fit.mcfadden_r2,
        "weighted_observations": fit.weighted_observations,
        "fitting_groups": fit.fitting_groups,
        "fitting_rows": fit.fitting_rows,
    }
    if fit.standard_errors is not None:
        payload["standard_errors"] = [float(value) for value in fit.standard_errors]
    return payload


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    model_input = Path(args.model_input).resolve()
    output = Path(args.output).resolve()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    if args.fresh and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    candidate_pool = (
        Path(args.candidate_pool).resolve()
        if args.candidate_pool
        else model_input / "candidate_pool.csv.gz"
    )
    if not candidate_pool.exists():
        raise FileNotFoundError(candidate_pool)
    choice_cfg = resolve_choice_config(model_input, config, args)
    top_k = int(choice_cfg["top_k"])
    max_fp = int(choice_cfg["max_fixed_point_iterations"])
    print("[stage] compact numeric cache", flush=True)
    cache, cache_signature = prepare_numeric_cache(
        candidate_pool, output, choice_cfg, args
    )
    if int(choice_cfg["candidate_limit"]) < top_k:
        raise ValueError(
            f"candidate limit {choice_cfg['candidate_limit']} is smaller than top-k {top_k}"
        )

    history: List[Dict[str, object]] = []
    print("[stage] initial MNL fit on all candidates", flush=True)
    initial_fit = fit_mnl(
        cache,
        None,
        np.asarray(choice_cfg["initial_beta"], dtype=float),
        choice_cfg,
        args,
        "MNL-all-candidates",
    )
    history.append(
        {
            "stage": "all_candidates",
            "fixed_point_iteration": 0,
            "selection_hash": None,
            "groups_changed": None,
            **fit_to_dict(initial_fit),
        }
    )
    beta = initial_fit.beta
    selection_path = output / "_choice_cache" / "selection_01.u1"
    selection_stats = select_top_k(
        cache,
        beta,
        top_k,
        selection_path,
        bool(choice_cfg["preserve_observed"]),
        args.progress_seconds,
    )

    stable = False
    seen_hashes: Dict[str, int] = {}
    final_fit: Optional[FitResult] = None
    final_selection_stats = selection_stats
    for fixed_point_iteration in range(1, max_fp + 1):
        selection_hash = mask_hash(selection_path, cache.rows)
        if selection_hash in seen_hashes:
            print(
                f"WARNING: top-k selection cycle detected; hash first seen at "
                f"iteration {seen_hashes[selection_hash]}",
                flush=True,
            )
            break
        seen_hashes[selection_hash] = fixed_point_iteration
        print(
            f"[stage] top-{top_k} MNL fit, fixed-point iteration "
            f"{fixed_point_iteration}/{max_fp}",
            flush=True,
        )
        fit = fit_mnl(
            cache,
            selection_path,
            beta,
            choice_cfg,
            args,
            f"MNL-top{top_k}-fp{fixed_point_iteration}",
        )
        next_path = output / "_choice_cache" / f"selection_{fixed_point_iteration + 1:02d}.u1"
        next_stats = select_top_k(
            cache,
            fit.beta,
            top_k,
            next_path,
            bool(choice_cfg["preserve_observed"]),
            args.progress_seconds,
            previous_path=selection_path,
        )
        history.append(
            {
                "stage": f"top_{top_k}",
                "fixed_point_iteration": fixed_point_iteration,
                "selection_hash": selection_hash,
                "groups_changed_after_refit": next_stats.groups_changed,
                "selected_rows": selection_stats.selected_rows,
                "observed_retention_rate": selection_stats.observed_retention_rate,
                **fit_to_dict(fit),
            }
        )
        beta = fit.beta
        final_fit = fit
        if next_stats.groups_changed == 0:
            stable = True
            next_path.unlink(missing_ok=True)
            final_selection_stats = selection_stats
            break
        selection_path = next_path
        selection_stats = next_stats
        final_selection_stats = next_stats

    if final_fit is None:
        # This only occurs if a cycle is detected before the first top-k refit.
        final_fit = fit_mnl(
            cache,
            selection_path,
            beta,
            choice_cfg,
            args,
            f"MNL-top{top_k}-final",
        )
        beta = final_fit.beta
    elif not stable:
        # Ensure the final coefficients are fitted on the final retained set even when the
        # fixed-point iteration limit was reached or a cycle was detected.
        final_fit = fit_mnl(
            cache,
            selection_path,
            beta,
            choice_cfg,
            args,
            f"MNL-top{top_k}-final",
            compute_standard_errors=False,
        )
        beta = final_fit.beta

    # Standard errors are computed only once for the final model.
    final_fit = fit_mnl(
        cache,
        selection_path,
        beta,
        choice_cfg,
        args,
        f"MNL-top{top_k}-final-SE",
        compute_standard_errors=True,
    )
    beta = final_fit.beta
    fit_warnings: List[str] = []
    if not final_fit.success:
        fit_warnings.append(
            "Final L-BFGS-B optimizer did not report success: " + final_fit.message
        )
    parameter_names = [
        "beta_total_time",
        "beta_walk_time_extra",
        "beta_transfer",
    ]
    for name, value, bounds in zip(parameter_names, beta, choice_cfg["bounds"]):
        lower, upper = map(float, bounds)
        tolerance = 1.0e-7 * max(1.0, abs(lower), abs(upper))
        if abs(float(value) - lower) <= tolerance:
            fit_warnings.append(f"{name} is at its lower bound ({lower:g})")
        if abs(float(value) - upper) <= tolerance:
            fit_warnings.append(f"{name} is at its upper bound ({upper:g})")
    if not stable:
        fit_warnings.append(
            "The self-consistent top-k set did not become stable within the configured limit"
        )
    for warning in fit_warnings:
        print("WARNING:", warning, flush=True)
    history.append(
        {
            "stage": f"top_{top_k}_final",
            "fixed_point_iteration": len(history),
            "selection_hash": mask_hash(selection_path, cache.rows),
            "selection_stable": stable,
            "selected_rows": final_selection_stats.selected_rows,
            "observed_retention_rate": final_selection_stats.observed_retention_rate,
            **fit_to_dict(final_fit),
        }
    )

    print("[stage] final top-k probabilities and output", flush=True)
    probability_path, rank_path = create_probability_cache(
        cache, selection_path, beta, output, args.progress_seconds
    )
    top_path = output / f"candidate_paths_top{top_k}.csv.gz"
    written = write_top_k_file(
        candidate_pool,
        top_path,
        cache,
        selection_path,
        probability_path,
        rank_path,
        beta,
        args.chunksize,
        args.progress_seconds,
    )

    vot = float(choice_cfg["value_of_time_won_per_hour"])
    if vot <= 0:
        raise ValueError("value_of_time_won_per_hour must be positive")
    beta_discount = -float(beta[0]) / (vot / 60.0)
    names = ["beta_total_time", "beta_walk_time_extra", "beta_transfer"]
    standard_errors = (
        final_fit.standard_errors
        if final_fit.standard_errors is not None
        else np.full(3, np.nan)
    )
    z_values = np.divide(
        beta,
        standard_errors,
        out=np.full(3, np.nan),
        where=standard_errors > EPS,
    )
    parameter_payload: Dict[str, object] = {
        "created_at": pd.Timestamp.now().isoformat(),
        "script_version": SCRIPT_VERSION,
        "model": "weighted grouped multinomial logit",
        "utility_formula": (
            "V = beta_total_time*total_time + beta_walk_time_extra*walk_time "
            "+ beta_transfer*transfers + beta_discount_per_won*discount_won"
        ),
        "beta_total_time": float(beta[0]),
        "beta_walk_time_extra": float(beta[1]),
        "beta_transfer": float(beta[2]),
        "beta_discount_per_won": float(beta_discount),
        "value_of_time_won_per_hour": vot,
        "standard_errors": {
            name: float(value) for name, value in zip(names, standard_errors)
        },
        "z_values": {name: float(value) for name, value in zip(names, z_values)},
        "fit": {
            **fit_to_dict(final_fit),
            "selection_stable": stable,
            "fixed_point_iterations_run": sum(
                1 for row in history if row.get("stage") == f"top_{top_k}"
            ),
            "l2_regularization": float(choice_cfg["l2_regularization"]),
            "bounds": choice_cfg["bounds"],
            "selection_mode": choice_cfg["selection_mode"],
            "observed_path_preservation": bool(choice_cfg["preserve_observed"]),
            "warnings": fit_warnings,
        },
        "choice_set": {
            "candidate_pool_rows": cache.rows,
            "candidate_pool_hour_od_groups": cache.groups,
            "candidate_limit": int(choice_cfg["candidate_limit"]),
            "top_k": top_k,
            "selected_rows": written,
            "selection_method": (
                "self-consistent baseline MNL probability with observed-path retention"
                if bool(choice_cfg["preserve_observed"])
                else "self-consistent pure baseline MNL probability"
            ),
            "selection_mode": choice_cfg["selection_mode"],
            "probabilities_normalized_over_final_top_k": True,
            "observed_passenger_retention_rate": final_selection_stats.observed_retention_rate,
            "observed_overflow_groups": final_selection_stats.observed_alternatives_overflow_groups,
            "observed_passengers_excluded_by_overflow": final_selection_stats.observed_passengers_excluded_by_overflow,
            "top_paths_file": top_path.name,
        },
    }
    (output / "route_choice_parameters.json").write_text(
        json.dumps(parameter_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    history_frame = pd.json_normalize(history, sep=".")
    history_frame.to_csv(
        output / "route_choice_fit_history.csv", index=False, encoding="utf-8-sig"
    )
    lengths = cache.group_lengths
    pool_distribution = {
        str(int(value)): int(count)
        for value, count in zip(*np.unique(lengths, return_counts=True))
    }
    selected = open_selection(selection_path, cache.rows, "r")
    selected_lengths = np.add.reduceat(
        np.asarray(selected, dtype=np.uint8), cache.group_starts[:-1]
    )
    selected_distribution = {
        str(int(value)): int(count)
        for value, count in zip(*np.unique(selected_lengths, return_counts=True))
    }
    qa = {
        "script_version": SCRIPT_VERSION,
        "candidate_pool": file_signature(candidate_pool),
        "numeric_cache_signature": cache_signature,
        "candidate_pool_rows": cache.rows,
        "candidate_pool_hour_od_groups": cache.groups,
        "candidate_count_distribution": pool_distribution,
        "final_choice_count_distribution": selected_distribution,
        "top_k": top_k,
        "selection_stable": stable,
        "selection_mode": choice_cfg["selection_mode"],
        "preserve_observed_paths": bool(choice_cfg["preserve_observed"]),
        "selection_stats": asdict(final_selection_stats),
        "observed_passenger_retention_rate": final_selection_stats.observed_retention_rate,
        "final_fit": fit_to_dict(final_fit),
        "warnings": fit_warnings,
        "output_rows": written,
    }
    (output / "choice_set_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "created_at": pd.Timestamp.now().isoformat(),
        "script_version": SCRIPT_VERSION,
        "stage": "route_choice_parameter_fit_and_top_k_selection",
        "source_model_input": str(model_input),
        "source_candidate_pool": str(candidate_pool),
        "config": str(config_path),
        "outputs": {
            "route_choice_parameters": "route_choice_parameters.json",
            "candidate_paths_top_k": top_path.name,
            "fit_history": "route_choice_fit_history.csv",
            "choice_set_qa": "choice_set_qa.json",
        },
        "downstream_stage": "fit_public_edge_bounty_policy.py",
    }
    (output / "route_choice_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\ncompleted:", output, flush=True)
    print(" -", output / "route_choice_parameters.json", flush=True)
    print(" -", top_path, flush=True)
    print(" -", output / "route_choice_fit_history.csv", flush=True)
    print(" -", output / "choice_set_qa.json", flush=True)
    print(
        "parameters:",
        {
            "beta_total_time": float(beta[0]),
            "beta_walk_time_extra": float(beta[1]),
            "beta_transfer": float(beta[2]),
            "beta_discount_per_won": float(beta_discount),
        },
        flush=True,
    )
    print(
        f"selection stable: {stable}; observed retention: "
        f"{100.0*final_selection_stats.observed_retention_rate:,.3f}%",
        flush=True,
    )
    print("total elapsed:", format_seconds(time.perf_counter() - run_started), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Numeric cache is reusable.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
