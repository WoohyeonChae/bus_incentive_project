from __future__ import annotations

"""Build Seoul-scale purpose-trip candidate-path model inputs only.

This script intentionally stops after candidate-path generation.  Route-choice
model fitting and the final top-five selection are deferred to the downstream
optimization/model script.

Model assumptions encoded here:

* candidate topology is generated once per ``hour x OD`` and reused by date;
* walking from the OD origin to the first boarding stop is allowed within 500 m
  by default;
* walking between transfer stops is allowed within 250 m by default;
* the destination stop must be reached exactly (no destination-side egress
  radius is expanded);
* at most two bus transfers and at most 15 candidate paths are retained per OD;
* the downstream model is expected to fit route choice on the candidate pool,
  retain the top five alternatives by baseline choice probability, and assume
  that a traveller chooses one of those five alternatives;
* work is checkpointed by hour and OD batch, with rate and ETA logging.

The main output is ``candidate_pool.csv.gz``.  It contains path topology,
numeric path attributes, search rank, and observed train/all passenger counts.
No MNL coefficients, utility scores, or top-five path file are created here.
"""

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import math
import os
import pickle
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

EPS = 1.0e-12
DEFAULT_HOURS = (7, 8, 9, 17, 18, 19)
BUILDER_VERSION = "candidate_only_access_transfer_split_v1"
SegmentKey = Tuple[int, str, str, str]
ODPair = Tuple[int, str, str]


# -----------------------------------------------------------------------------
# General helpers and progress reporting
# -----------------------------------------------------------------------------


def normalize_id(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    if text.endswith(".0") and text[:-2].replace("-", "").isdigit():
        text = text[:-2]
    return text


def parse_csv_set(text: str, cast=str) -> List[object]:
    return [cast(token.strip()) for token in text.split(",") if token.strip()]


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


class ProgressETA:
    """Rate and ETA reporter with a short warm-up and smoothed throughput."""

    def __init__(
        self,
        label: str,
        total: Optional[int],
        print_every_seconds: float = 15.0,
        warmup_units: int = 1,
    ) -> None:
        self.label = label
        self.total = int(total) if total is not None and total > 0 else None
        self.print_every_seconds = max(1.0, float(print_every_seconds))
        self.warmup_units = max(1, int(warmup_units))
        self.started = time.perf_counter()
        self.last_print = self.started
        self.last_done = 0
        self.last_time = self.started
        self.smoothed_rate: Optional[float] = None

    def update(self, done: int, extra: str = "", force: bool = False) -> None:
        now = time.perf_counter()
        done = int(done)
        if not force and now - self.last_print < self.print_every_seconds:
            return
        delta_units = done - self.last_done
        delta_time = now - self.last_time
        if delta_units > 0 and delta_time > 0:
            current_rate = delta_units / delta_time
            if self.smoothed_rate is None:
                self.smoothed_rate = current_rate
            else:
                self.smoothed_rate = 0.25 * current_rate + 0.75 * self.smoothed_rate
        elapsed = now - self.started
        average_rate = done / elapsed if done > 0 and elapsed > 0 else 0.0
        rate = self.smoothed_rate if self.smoothed_rate and done >= self.warmup_units else average_rate
        if self.total is None:
            progress = f"{done:,}"
            eta = "ETA --"
        else:
            pct = 100.0 * done / max(self.total, 1)
            progress = f"{done:,}/{self.total:,} ({pct:5.1f}%)"
            remaining = max(0, self.total - done)
            eta_seconds = remaining / rate if rate > 0 else float("inf")
            eta = f"ETA {format_seconds(eta_seconds)}"
        suffix = f", {extra}" if extra else ""
        print(
            f"[{self.label}] {progress}, {rate:,.2f}/s, "
            f"elapsed {format_seconds(elapsed)}, {eta}{suffix}",
            flush=True,
        )
        self.last_print = now
        self.last_done = done
        self.last_time = now


# -----------------------------------------------------------------------------
# Model data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkLink:
    origin: str
    destination: str
    distance_m: float
    time_min: float


@dataclass
class RouteChain:
    uid: int
    hour: int
    route_id: str
    chain_id: int
    stops: Tuple[str, ...]
    edge_keys: Tuple[SegmentKey, ...]
    travel_prefix: Tuple[float, ...]
    distance_prefix: Tuple[float, ...]
    edge_trips: Tuple[float, ...]
    positions: Dict[str, Tuple[int, ...]]


@dataclass(frozen=True)
class RouteLeg:
    chain_uid: int
    route_id: str
    origin: str
    destination: str
    start_pos: int
    end_pos: int
    edge_keys: Tuple[SegmentKey, ...]
    travel_time: float
    distance_m: float
    boarding_edge: SegmentKey
    wait_time: float


@dataclass(frozen=True)
class TransferOption:
    to_chain: int
    from_pos: int
    to_pos: int
    walk: WalkLink


@dataclass(frozen=True)
class PathSpec:
    hour: int
    origin_stop_id: str
    destination_stop_id: str
    signature: str
    route_legs: Tuple[Tuple[str, str, str], ...]
    segment_keys: Tuple[SegmentKey, ...]
    boarding_edges: Tuple[SegmentKey, ...]
    walk_links: Tuple[WalkLink, ...]
    ride_time: float
    wait_time: float
    walk_time: float
    transfers: int
    source: str

    @property
    def total_time(self) -> float:
        return self.ride_time + self.wait_time + self.walk_time


@dataclass(frozen=True)
class ObservedCandidate:
    spec: PathSpec
    passengers_train: float
    passengers_all: float


@dataclass
class SearchStats:
    expanded_states: int = 0
    generated_paths: int = 0
    observed_kept: int = 0
    observed_truncated: int = 0
    hit_state_cap: int = 0
    no_access: int = 0
    no_egress: int = 0
    candidate_rows: int = 0
    ods_with_candidates: int = 0
    ods_without_candidates: int = 0
    ods_reaching_candidate_limit: int = 0

    def add(self, other: "SearchStats") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, int(getattr(self, field)) + int(getattr(other, field)))


@dataclass
class HourNetwork:
    hour: int
    chains: Dict[int, RouteChain]
    chains_by_route: Dict[str, Tuple[int, ...]]
    occurrences: Dict[str, Tuple[Tuple[int, int], ...]]
    transitions: Dict[int, Dict[int, Tuple[TransferOption, ...]]]
    predecessors: Dict[int, frozenset]
    access_walk_neighbors: Dict[str, Tuple[WalkLink, ...]]
    transfer_walk_neighbors: Dict[str, Tuple[WalkLink, ...]]
    max_transfer_options_per_pair: int

    def route_leg(self, chain_uid: int, start: int, end: int) -> RouteLeg:
        chain = self.chains[chain_uid]
        if not 0 <= start < end < len(chain.stops):
            raise ValueError("invalid route-chain range")
        return RouteLeg(
            chain_uid=chain_uid,
            route_id=chain.route_id,
            origin=chain.stops[start],
            destination=chain.stops[end],
            start_pos=start,
            end_pos=end,
            edge_keys=chain.edge_keys[start:end],
            travel_time=chain.travel_prefix[end] - chain.travel_prefix[start],
            distance_m=chain.distance_prefix[end] - chain.distance_prefix[start],
            boarding_edge=chain.edge_keys[start],
            wait_time=30.0 / max(chain.edge_trips[start], EPS),
        )

    def best_route_leg(self, route_id: str, origin: str, destination: str) -> Optional[RouteLeg]:
        best: Optional[RouteLeg] = None
        for uid in self.chains_by_route.get(route_id, ()):
            chain = self.chains[uid]
            for start in chain.positions.get(origin, ()):
                for end in chain.positions.get(destination, ()):
                    if end <= start:
                        continue
                    leg = self.route_leg(uid, start, end)
                    if best is None or (
                        leg.travel_time + leg.wait_time,
                        leg.distance_m,
                        uid,
                    ) < (
                        best.travel_time + best.wait_time,
                        best.distance_m,
                        best.chain_uid,
                    ):
                        best = leg
        return best


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "목적통행 model_input 후보경로 생성 전용: 출발 접근 500m, "
            "환승보행 250m, 목적지 정확 도달, 최대 환승 2회, 후보 15개. "
            "MNL fitting과 최종 top-5 선택은 다음 모델에서 수행"
        )
    )
    parser.add_argument("--model-input", required=True, help="기존 버스 model_input 폴더")
    parser.add_argument("--purpose-detail", required=True, help="purpose_trip_detail.csv.gz")
    parser.add_argument("--output", required=True, help="출력 model_input_purpose 폴더")
    parser.add_argument("--dates", default="", help="전체 분석 날짜. 비우면 manifest/OD에서 읽음")
    parser.add_argument("--train-dates", default="", help="다음 MNL 모델의 학습 날짜. 비우면 앞 2/3")
    parser.add_argument("--hours", default=",".join(map(str, DEFAULT_HOURS)))
    parser.add_argument(
        "--access-radius-m",
        type=float,
        default=500.0,
        help="OD 출발지에서 첫 승차 정류장까지의 최대 접근보행 거리",
    )
    parser.add_argument(
        "--transfer-radius-m",
        type=float,
        default=250.0,
        help="환승 하차 정류장에서 다음 승차 정류장까지의 최대 보행거리",
    )
    parser.add_argument("--walk-speed-kmph", type=float, default=4.0)
    parser.add_argument("--max-transfers", type=int, default=2)
    parser.add_argument("--candidate-limit", type=int, default=15)
    parser.add_argument(
        "--choice-set-size",
        type=int,
        default=5,
        help="다음 모델이 MNL fitting 후 확률순으로 남길 최종 대안 수(여기서는 선택하지 않고 메타데이터에 기록)",
    )
    parser.add_argument("--max-path-edges", type=int, default=120)
    parser.add_argument("--max-search-states-per-od", type=int, default=25_000)
    parser.add_argument(
        "--max-transfer-options-per-pair",
        type=int,
        default=4,
        help=(
            "한 상태에서 동일 다음 노선체인으로 확장할 환승위치 상한. "
            "0이면 모두 사용(정확하지만 느림); 기본 4는 속도용 근사"
        ),
    )
    parser.add_argument("--search-transfer-penalty-min", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 2)))
    parser.add_argument("--od-batch-size", type=int, default=500)
    parser.add_argument(
        "--max-od",
        type=int,
        default=0,
        help="테스트 전용 시간대별 OD 상한. 전체 실행은 반드시 0",
    )
    parser.add_argument("--expected-purpose-rows", type=int, default=0)
    parser.add_argument(
        "--expected-segment-rows",
        type=int,
        default=0,
        help="segments.csv 전체 예상 행 수. 지정하면 읽기 단계 ETA가 더 정확함",
    )
    parser.add_argument(
        "--observed-path-dates",
        choices=["train", "all"],
        default="train",
        help=(
            "관측 경로를 후보 풀에 넣을 날짜 범위. train은 평가일 경로정보 누출을 막고, "
            "all은 모든 분석일 관측경로를 보존"
        ),
    )
    parser.add_argument("--progress-seconds", type=float, default=15.0)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: 필수 컬럼 누락: {missing}")


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def determine_dates(model_input: Path, explicit_dates: str) -> List[str]:
    if explicit_dates:
        return sorted({normalize_id(value) for value in parse_csv_set(explicit_dates) if normalize_id(value)})
    manifest = load_json(model_input / "model_input_manifest.json")
    values = manifest.get("used_dates", []) or manifest.get("target_dates", [])
    dates = sorted({normalize_id(value).replace("-", "") for value in values if normalize_id(value)})
    if dates:
        return dates
    path = model_input / "od_hourly.csv"
    if not path.exists():
        return []
    result: Set[str] = set()
    for chunk in pd.read_csv(path, usecols=["date"], dtype=str, chunksize=500_000):
        result.update(normalize_id(value) for value in chunk["date"].dropna())
    return sorted(value for value in result if value)


def determine_train_dates(dates: Sequence[str], explicit_train: str) -> List[str]:
    if explicit_train:
        wanted = {normalize_id(value) for value in parse_csv_set(explicit_train)}
        return [date for date in dates if date in wanted]
    if not dates:
        return []
    count = max(1, int(math.floor(len(dates) * 2.0 / 3.0)))
    if len(dates) >= 3:
        count = min(count, len(dates) - 1)
    return list(dates[:count])


def load_baseline_segments(
    model_input: Path,
    hours: Set[int],
    expected_rows: int = 0,
    progress_seconds: float = 15.0,
) -> pd.DataFrame:
    baseline = model_input / "segments_baseline.csv.gz"
    if baseline.exists():
        started = time.perf_counter()
        frame = pd.read_csv(baseline, compression="gzip", low_memory=False)
        print(
            f"[segments-baseline] {len(frame):,} rows loaded in "
            f"{format_seconds(time.perf_counter()-started)}",
            flush=True,
        )
    else:
        path = model_input / "segments.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        usecols = [
            "hour",
            "route_id",
            "from_stop_id",
            "to_stop_id",
            "travel_time",
            "distance",
            "trips",
            "avg_onboard",
            "stop_sequence",
        ]
        parts: List[pd.DataFrame] = []
        processed = 0
        meter = ProgressETA(
            "segments",
            int(expected_rows) if int(expected_rows) > 0 else None,
            progress_seconds,
            warmup_units=500_000,
        )
        for chunk in pd.read_csv(
            path,
            usecols=lambda column: column in set(usecols),
            chunksize=500_000,
            low_memory=False,
        ):
            processed += len(chunk)
            chunk["hour"] = pd.to_numeric(chunk["hour"], errors="coerce")
            chunk = chunk[chunk["hour"].isin(hours)].copy()
            if chunk.empty:
                meter.update(processed, extra=f"selected {sum(len(v) for v in parts):,}")
                continue
            for column in ["travel_time", "distance", "trips", "avg_onboard", "stop_sequence"]:
                if column in chunk.columns:
                    chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
            parts.append(chunk)
            meter.update(processed, extra=f"selected {sum(len(v) for v in parts):,}")
        meter.update(
            processed,
            extra=f"selected {sum(len(v) for v in parts):,}",
            force=True,
        )
        if not parts:
            raise RuntimeError("선택 시간대의 segments.csv 행이 없습니다.")
        daily = pd.concat(parts, ignore_index=True)
        aggregation: Dict[str, Tuple[str, str]] = {
            "travel_time": ("travel_time", "mean"),
            "distance": ("distance", "mean"),
            "trips": ("trips", "mean"),
        }
        if "avg_onboard" in daily.columns:
            aggregation["avg_onboard"] = ("avg_onboard", "mean")
        if "stop_sequence" in daily.columns:
            aggregation["stop_sequence"] = ("stop_sequence", "min")
        frame = daily.groupby(
            ["hour", "route_id", "from_stop_id", "to_stop_id"], as_index=False
        ).agg(**aggregation)

    required = [
        "hour",
        "route_id",
        "from_stop_id",
        "to_stop_id",
        "travel_time",
        "trips",
    ]
    require_columns(frame, required, "segments_baseline")
    frame = frame.copy()
    frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce")
    for column in ["travel_time", "distance", "trips", "avg_onboard", "stop_sequence"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "distance" not in frame.columns:
        frame["distance"] = 1.0
    frame["distance"] = frame["distance"].fillna(1.0).clip(lower=EPS)
    if "avg_onboard" not in frame.columns:
        frame["avg_onboard"] = 0.0
    for column in ["route_id", "from_stop_id", "to_stop_id"]:
        frame[column] = frame[column].map(normalize_id)
    frame = frame[
        frame["hour"].isin(hours)
        & frame["travel_time"].gt(0)
        & frame["trips"].gt(0)
        & frame["route_id"].ne("")
        & frame["from_stop_id"].ne("")
        & frame["to_stop_id"].ne("")
    ].copy()
    frame["hour"] = frame["hour"].astype(int)
    key = ["hour", "route_id", "from_stop_id", "to_stop_id"]
    if frame.duplicated(key).any():
        aggregation = {
            "travel_time": ("travel_time", "mean"),
            "distance": ("distance", "mean"),
            "trips": ("trips", "mean"),
            "avg_onboard": ("avg_onboard", "mean"),
        }
        if "stop_sequence" in frame.columns:
            aggregation["stop_sequence"] = ("stop_sequence", "min")
        frame = frame.groupby(key, as_index=False).agg(**aggregation)
    return frame.sort_values(key, kind="stable").reset_index(drop=True)


def _column_by_alias(columns: Iterable[str], aliases: Sequence[str]) -> Optional[str]:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def load_stops(model_input: Path) -> pd.DataFrame:
    path = model_input / "stops.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, low_memory=False)
    stop_col = _column_by_alias(
        frame.columns, ["stop_id", "station_id", "node_id", "crtr_id", "stops_id", "정류장_id", "정류장id"]
    )
    lat_col = _column_by_alias(
        frame.columns, ["latitude", "lat", "y", "ycord", "gps_y", "wgs84_y", "위도"]
    )
    lon_col = _column_by_alias(
        frame.columns, ["longitude", "lon", "lng", "x", "xcord", "gps_x", "wgs84_x", "경도"]
    )
    if not stop_col or not lat_col or not lon_col:
        raise ValueError(f"stops.csv 좌표 컬럼을 찾지 못했습니다: {list(frame.columns)}")
    output = frame[[stop_col, lat_col, lon_col]].rename(
        columns={stop_col: "stop_id", lat_col: "latitude", lon_col: "longitude"}
    )
    output["stop_id"] = output["stop_id"].map(normalize_id)
    output["latitude"] = pd.to_numeric(output["latitude"], errors="coerce")
    output["longitude"] = pd.to_numeric(output["longitude"], errors="coerce")
    output = output.dropna().drop_duplicates("stop_id")
    output = output[
        output["stop_id"].ne("")
        & output["latitude"].between(-90, 90)
        & output["longitude"].between(-180, 180)
    ].copy()
    if output.empty:
        raise RuntimeError("사용 가능한 정류장 좌표가 없습니다.")
    return output.reset_index(drop=True)


def load_od_baseline(
    model_input: Path,
    hours: Set[int],
    max_od_per_hour: int,
) -> pd.DataFrame:
    path = model_input / "od_baseline.csv.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, compression="gzip", low_memory=False)
    require_columns(
        frame,
        ["hour", "origin_stop_id", "destination_stop_id", "avg_passengers"],
        "od_baseline",
    )
    frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce")
    frame["avg_passengers"] = pd.to_numeric(frame["avg_passengers"], errors="coerce")
    for column in ["origin_stop_id", "destination_stop_id"]:
        frame[column] = frame[column].map(normalize_id)
    frame = frame[
        frame["hour"].isin(hours)
        & frame["avg_passengers"].gt(0)
        & frame["origin_stop_id"].ne("")
        & frame["destination_stop_id"].ne("")
        & frame["origin_stop_id"].ne(frame["destination_stop_id"])
    ].copy()
    frame["hour"] = frame["hour"].astype(int)
    frame = frame.sort_values(
        ["hour", "avg_passengers", "origin_stop_id", "destination_stop_id"],
        ascending=[True, False, True, True],
        kind="stable",
    ).drop_duplicates(["hour", "origin_stop_id", "destination_stop_id"])
    if max_od_per_hour > 0:
        frame = frame.groupby("hour", group_keys=False).head(max_od_per_hour)
    frame = frame.sort_values(
        ["hour", "origin_stop_id", "destination_stop_id"], kind="stable"
    ).reset_index(drop=True)
    frame["od_index"] = np.arange(len(frame), dtype=np.int64)
    return frame


# -----------------------------------------------------------------------------
# Walking links and route/transfer index
# -----------------------------------------------------------------------------


def build_walk_neighbors(
    stops: pd.DataFrame,
    radius_m: float,
    speed_kmph: float,
    progress_seconds: float = 15.0,
    label: str = "walk-index",
) -> Dict[str, Tuple[WalkLink, ...]]:
    if radius_m <= 0 or speed_kmph <= 0:
        raise ValueError("walk radius and speed must be positive")
    mean_lat = math.radians(float(stops["latitude"].mean()))
    x = stops["longitude"].to_numpy(float) * 111_320.0 * math.cos(mean_lat)
    y = stops["latitude"].to_numpy(float) * 110_540.0
    ids = stops["stop_id"].astype(str).to_numpy()
    cell = float(radius_m)
    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for index, (xi, yi) in enumerate(zip(x, y)):
        buckets[(math.floor(xi / cell), math.floor(yi / cell))].append(index)
    metres_per_minute = speed_kmph * 1000.0 / 60.0
    result: Dict[str, Tuple[WalkLink, ...]] = {}
    meter = ProgressETA(label, len(ids), progress_seconds, warmup_units=250)
    directed_links = 0
    for index, (xi, yi, stop_id) in enumerate(zip(x, y, ids)):
        cx, cy = math.floor(xi / cell), math.floor(yi / cell)
        candidates: List[Tuple[float, str]] = [(0.0, str(stop_id))]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in buckets.get((cx + dx, cy + dy), []):
                    if other == index:
                        continue
                    distance = math.hypot(float(x[other] - xi), float(y[other] - yi))
                    if distance <= radius_m + 1.0e-9:
                        candidates.append((distance, str(ids[other])))
        candidates.sort(key=lambda value: (value[0], value[1]))
        result[str(stop_id)] = tuple(
            WalkLink(
                origin=str(stop_id),
                destination=destination,
                distance_m=float(distance),
                time_min=float(distance / metres_per_minute),
            )
            for distance, destination in candidates
        )
        directed_links += max(0, len(candidates) - 1)
        if (index + 1) % 100 == 0:
            meter.update(index + 1, extra=f"directed links {directed_links:,}")
    meter.update(len(ids), extra=f"directed links {directed_links:,}", force=True)
    return result


def filter_walk_neighbors(
    neighbors: Mapping[str, Sequence[WalkLink]],
    radius_m: float,
) -> Dict[str, Tuple[WalkLink, ...]]:
    """Return a radius-specific view while always preserving each zero-distance self link."""
    if radius_m < 0:
        raise ValueError("walk radius cannot be negative")
    limit = float(radius_m) + 1.0e-9
    return {
        stop_id: tuple(link for link in links if link.distance_m <= limit)
        for stop_id, links in neighbors.items()
    }


def count_directed_walk_links(
    neighbors: Mapping[str, Sequence[WalkLink]],
) -> int:
    return int(
        sum(1 for links in neighbors.values() for link in links if link.distance_m > EPS)
    )


def write_walk_edges(
    path: Path,
    neighbors: Mapping[str, Sequence[WalkLink]],
    walk_type: str,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "walk_type",
        "from_stop_id",
        "to_stop_id",
        "distance_m",
        "walk_time_min",
    ]
    rows = 0
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for links in neighbors.values():
            for link in links:
                if link.distance_m <= EPS:
                    continue
                writer.writerow(
                    {
                        "walk_type": walk_type,
                        "from_stop_id": link.origin,
                        "to_stop_id": link.destination,
                        "distance_m": f"{link.distance_m:.6f}",
                        "walk_time_min": f"{link.time_min:.6f}",
                    }
                )
                rows += 1
    return rows


def prefix(values: Sequence[float]) -> Tuple[float, ...]:
    output = [0.0]
    total = 0.0
    for value in values:
        total += float(value)
        output.append(total)
    return tuple(output)


def build_route_chains(segments: pd.DataFrame, hour: int) -> Tuple[Dict[int, RouteChain], Dict[str, Tuple[int, ...]], Dict[str, Tuple[Tuple[int, int], ...]]]:
    chains: Dict[int, RouteChain] = {}
    chains_by_route_temp: Dict[str, List[int]] = defaultdict(list)
    occurrences_temp: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    uid = 0
    hour_frame = segments[segments["hour"] == int(hour)]
    for route_id, group in hour_frame.groupby("route_id", sort=False):
        group = group.copy()
        if "stop_sequence" in group.columns and group["stop_sequence"].notna().any():
            group = group.sort_values(
                ["stop_sequence", "from_stop_id", "to_stop_id"], kind="stable"
            )
        else:
            group = group.sort_values(["from_stop_id", "to_stop_id"], kind="stable")
        blocks: List[List[object]] = []
        current: List[object] = []
        previous_to: Optional[str] = None
        for row in group.itertuples(index=False):
            if current and previous_to != str(row.from_stop_id):
                blocks.append(current)
                current = []
            current.append(row)
            previous_to = str(row.to_stop_id)
        if current:
            blocks.append(current)
        for chain_id, rows in enumerate(blocks):
            if not rows:
                continue
            stops_list = [str(rows[0].from_stop_id)]
            edge_keys: List[SegmentKey] = []
            travels: List[float] = []
            distances: List[float] = []
            trips: List[float] = []
            valid = True
            for row in rows:
                if stops_list[-1] != str(row.from_stop_id):
                    valid = False
                    break
                edge_keys.append(
                    (int(hour), str(route_id), str(row.from_stop_id), str(row.to_stop_id))
                )
                travels.append(float(row.travel_time))
                distances.append(float(row.distance))
                trips.append(max(float(row.trips), EPS))
                stops_list.append(str(row.to_stop_id))
            if not valid or not edge_keys:
                continue
            positions_temp: Dict[str, List[int]] = defaultdict(list)
            for pos, stop in enumerate(stops_list):
                positions_temp[stop].append(pos)
            chain = RouteChain(
                uid=uid,
                hour=int(hour),
                route_id=str(route_id),
                chain_id=chain_id,
                stops=tuple(stops_list),
                edge_keys=tuple(edge_keys),
                travel_prefix=prefix(travels),
                distance_prefix=prefix(distances),
                edge_trips=tuple(trips),
                positions={stop: tuple(values) for stop, values in positions_temp.items()},
            )
            chains[uid] = chain
            chains_by_route_temp[str(route_id)].append(uid)
            for pos, stop in enumerate(chain.stops):
                occurrences_temp[stop].append((uid, pos))
            uid += 1
    return (
        chains,
        {route: tuple(values) for route, values in chains_by_route_temp.items()},
        {stop: tuple(values) for stop, values in occurrences_temp.items()},
    )


def build_transfer_index(
    chains: Mapping[int, RouteChain],
    occurrences: Mapping[str, Sequence[Tuple[int, int]]],
    walk_neighbors: Mapping[str, Sequence[WalkLink]],
) -> Tuple[Dict[int, Dict[int, Tuple[TransferOption, ...]]], Dict[int, frozenset], int]:
    pair_options: Dict[int, Dict[int, Dict[Tuple[int, int], TransferOption]]] = defaultdict(lambda: defaultdict(dict))
    raw_count = 0
    for from_uid, chain in chains.items():
        # position zero cannot be an alighting position after a positive ride.
        for from_pos in range(1, len(chain.stops)):
            alight_stop = chain.stops[from_pos]
            for link in walk_neighbors.get(alight_stop, ()):
                for to_uid, to_pos in occurrences.get(link.destination, ()):
                    if to_uid == from_uid:
                        continue
                    to_chain = chains[to_uid]
                    if to_pos >= len(to_chain.stops) - 1:
                        continue
                    raw_count += 1
                    key = (from_pos, to_pos)
                    old = pair_options[from_uid][to_uid].get(key)
                    option = TransferOption(
                        to_chain=to_uid,
                        from_pos=from_pos,
                        to_pos=to_pos,
                        walk=WalkLink(
                            origin=alight_stop,
                            destination=link.destination,
                            distance_m=link.distance_m,
                            time_min=link.time_min,
                        ),
                    )
                    if old is None or option.walk.time_min < old.walk.time_min:
                        pair_options[from_uid][to_uid][key] = option
    transitions: Dict[int, Dict[int, Tuple[TransferOption, ...]]] = {}
    predecessors_temp: Dict[int, Set[int]] = defaultdict(set)
    kept = 0
    for from_uid, by_to in pair_options.items():
        transitions[from_uid] = {}
        for to_uid, options in by_to.items():
            values = tuple(
                sorted(
                    options.values(),
                    key=lambda value: (
                        value.from_pos,
                        value.to_pos,
                        value.walk.time_min,
                    ),
                )
            )
            transitions[from_uid][to_uid] = values
            predecessors_temp[to_uid].add(from_uid)
            kept += len(values)
    predecessors = {uid: frozenset(values) for uid, values in predecessors_temp.items()}
    return transitions, predecessors, kept


def build_hour_network(
    segments: pd.DataFrame,
    hour: int,
    access_walk_neighbors: Dict[str, Tuple[WalkLink, ...]],
    transfer_walk_neighbors: Dict[str, Tuple[WalkLink, ...]],
    max_transfer_options_per_pair: int,
) -> Tuple[HourNetwork, Dict[str, int]]:
    chains, chains_by_route, occurrences = build_route_chains(segments, hour)
    transitions, predecessors, transfer_options = build_transfer_index(
        chains, occurrences, transfer_walk_neighbors
    )
    network = HourNetwork(
        hour=int(hour),
        chains=chains,
        chains_by_route=chains_by_route,
        occurrences=occurrences,
        transitions=transitions,
        predecessors=predecessors,
        access_walk_neighbors=access_walk_neighbors,
        transfer_walk_neighbors=transfer_walk_neighbors,
        max_transfer_options_per_pair=max(0, int(max_transfer_options_per_pair)),
    )
    stats = {
        "route_chains": len(chains),
        "route_ids": len(chains_by_route),
        "route_stop_occurrences": sum(len(values) for values in occurrences.values()),
        "ordered_chain_pairs": sum(len(values) for values in transitions.values()),
        "transfer_options": transfer_options,
    }
    return network, stats


# -----------------------------------------------------------------------------
# Path serialization and observed-path reconstruction
# -----------------------------------------------------------------------------


def path_signature(route_legs: Sequence[Tuple[str, str, str]], walks: Sequence[WalkLink]) -> str:
    route_part = ";".join(f"{route}|{board}|{alight}" for route, board, alight in route_legs)
    walk_part = ";".join(f"{link.origin}|{link.destination}" for link in walks)
    return route_part + "||" + walk_part


def make_path_spec(
    hour: int,
    origin: str,
    destination: str,
    legs: Sequence[RouteLeg],
    walks: Sequence[WalkLink],
    source: str,
) -> Optional[PathSpec]:
    if not legs:
        return None
    route_legs = tuple((leg.route_id, leg.origin, leg.destination) for leg in legs)
    segment_keys = tuple(key for leg in legs for key in leg.edge_keys)
    if not segment_keys:
        return None
    return PathSpec(
        hour=int(hour),
        origin_stop_id=str(origin),
        destination_stop_id=str(destination),
        signature=path_signature(route_legs, walks),
        route_legs=route_legs,
        segment_keys=segment_keys,
        boarding_edges=tuple(leg.boarding_edge for leg in legs),
        walk_links=tuple(walks),
        ride_time=float(sum(leg.travel_time for leg in legs)),
        wait_time=float(sum(leg.wait_time for leg in legs)),
        walk_time=float(sum(link.time_min for link in walks)),
        transfers=max(0, len(legs) - 1),
        source=source,
    )


def split_chain(value: object) -> List[str]:
    text = normalize_id(value)
    if not text:
        return []
    return [normalize_id(token) for token in text.split(">") if normalize_id(token)]


def parse_transfer_pairs(value: object) -> List[Tuple[str, str]]:
    text = normalize_id(value)
    if not text:
        return []
    output: List[Tuple[str, str]] = []
    for token in text.split(";"):
        if ">" not in token:
            continue
        left, right = token.split(">", 1)
        left, right = normalize_id(left), normalize_id(right)
        if left and right:
            output.append((left, right))
    return output


def walk_link_between(
    origin: str,
    destination: str,
    walk_neighbors: Mapping[str, Sequence[WalkLink]],
) -> Optional[WalkLink]:
    if origin == destination:
        return WalkLink(origin, destination, 0.0, 0.0)
    for link in walk_neighbors.get(origin, ()):
        if link.destination == destination:
            return link
    return None


def reconstruct_observed_spec(
    network: HourNetwork,
    origin: str,
    destination: str,
    route_chain_text: str,
    transfer_chain_text: str,
    max_transfers: int,
) -> Tuple[Optional[PathSpec], str]:
    routes = [route for route in split_chain(route_chain_text) if route != "RAIL/OTHER"]
    pairs = parse_transfer_pairs(transfer_chain_text)
    if not routes:
        return None, "missing_routes"
    if len(routes) - 1 > max_transfers:
        return None, "transfer_cap"
    if len(pairs) != max(0, len(routes) - 1):
        return None, "chain_length_mismatch"
    boards = [origin] + [right for _, right in pairs]
    alights = [left for left, _ in pairs] + [destination]
    legs: List[RouteLeg] = []
    walks: List[WalkLink] = []
    for index, route_id in enumerate(routes):
        leg = network.best_route_leg(route_id, boards[index], alights[index])
        if leg is None:
            return None, "unmapped_route_leg"
        legs.append(leg)
        if index < len(routes) - 1:
            link = walk_link_between(alights[index], boards[index + 1], network.transfer_walk_neighbors)
            if link is None:
                return None, "walk_radius"
            if link.distance_m > EPS:
                walks.append(link)
    spec = make_path_spec(
        network.hour,
        origin,
        destination,
        legs,
        walks,
        "observed",
    )
    return spec, "mapped" if spec is not None else "empty"


def serialize_path(
    spec: PathSpec,
    od_index: int,
    observed_train: float,
    observed_all: float,
    candidate_rank_search: int,
) -> Dict[str, object]:
    access_link: Optional[WalkLink] = None
    transfer_links: Sequence[WalkLink] = spec.walk_links
    if spec.walk_links and spec.route_legs:
        first = spec.walk_links[0]
        first_board_stop = spec.route_legs[0][1]
        if (
            first.origin == spec.origin_stop_id
            and first.destination == first_board_stop
        ):
            access_link = first
            transfer_links = spec.walk_links[1:]
    access_distance = float(access_link.distance_m) if access_link is not None else 0.0
    access_time = float(access_link.time_min) if access_link is not None else 0.0
    transfer_distance = float(sum(link.distance_m for link in transfer_links))
    transfer_time = float(sum(link.time_min for link in transfer_links))
    return {
        "od_index": int(od_index),
        "hour": spec.hour,
        "origin_stop_id": spec.origin_stop_id,
        "destination_stop_id": spec.destination_stop_id,
        "signature": spec.signature,
        "route_legs": ";".join("|".join(value) for value in spec.route_legs),
        "segment_keys": ";".join(
            f"{route}|{origin}|{destination}"
            for _, route, origin, destination in spec.segment_keys
        ),
        "boarding_edges": ";".join(
            f"{route}|{origin}|{destination}"
            for _, route, origin, destination in spec.boarding_edges
        ),
        "walk_links": ";".join(
            f"{link.origin}|{link.destination}|{link.distance_m:.6f}|{link.time_min:.6f}"
            for link in spec.walk_links
        ),
        "ride_time": spec.ride_time,
        "wait_time": spec.wait_time,
        "access_walk_distance_m": access_distance,
        "access_walk_time": access_time,
        "transfer_walk_distance_m": transfer_distance,
        "transfer_walk_time": transfer_time,
        "total_walk_distance_m": access_distance + transfer_distance,
        "walk_time": spec.walk_time,
        "total_time": spec.total_time,
        "transfers": spec.transfers,
        "source": spec.source,
        "observed_passengers_train": float(observed_train),
        "observed_passengers_all": float(observed_all),
        "candidate_rank_search": int(candidate_rank_search),
    }


# -----------------------------------------------------------------------------
# Bounded candidate search
# -----------------------------------------------------------------------------


def _access_options(network: HourNetwork, origin: str) -> List[Tuple[int, int, WalkLink]]:
    best: Dict[Tuple[int, int], WalkLink] = {}
    for link in network.access_walk_neighbors.get(origin, ()):
        for chain_uid, board_pos in network.occurrences.get(link.destination, ()):
            chain = network.chains[chain_uid]
            if board_pos >= len(chain.stops) - 1:
                continue
            key = (chain_uid, board_pos)
            old = best.get(key)
            if old is None or link.time_min < old.time_min:
                best[key] = link
    return [
        (chain_uid, board_pos, link)
        for (chain_uid, board_pos), link in sorted(
            best.items(), key=lambda item: (item[1].time_min, item[0][0], item[0][1])
        )
    ]


def _egress_options(
    network: HourNetwork,
    destination: str,
) -> Dict[int, Tuple[Tuple[int, WalkLink], ...]]:
    """Require exact arrival at the OD destination stop; no egress radius is expanded."""
    zero = WalkLink(
        origin=str(destination),
        destination=str(destination),
        distance_m=0.0,
        time_min=0.0,
    )
    best: Dict[int, Dict[int, WalkLink]] = defaultdict(dict)
    for chain_uid, alight_pos in network.occurrences.get(str(destination), ()):
        if alight_pos <= 0:
            continue
        best[chain_uid][alight_pos] = zero
    return {
        chain_uid: tuple(sorted(values.items(), key=lambda item: item[0]))
        for chain_uid, values in best.items()
    }


def _reachable_chain_sets(
    network: HourNetwork,
    egress_chains: Set[int],
    max_transfers: int,
) -> List[Set[int]]:
    reachable: List[Set[int]] = [set(egress_chains)]
    for _ in range(max_transfers):
        previous = reachable[-1]
        current = set(previous)
        for chain_uid in previous:
            current.update(network.predecessors.get(chain_uid, ()))
        reachable.append(current)
    return reachable


def _push_label(
    labels: MutableMapping[Tuple[int, int, Tuple[int, ...]], List[float]],
    key: Tuple[int, int, Tuple[int, ...]],
    value: float,
    keep: int,
) -> bool:
    values = labels.setdefault(key, [])
    if len(values) < keep:
        values.append(value)
        values.sort()
        return True
    if value + 1.0e-9 < values[-1]:
        values[-1] = value
        values.sort()
        return True
    return False


def search_generated_paths(
    network: HourNetwork,
    origin: str,
    destination: str,
    observed_signatures: Set[str],
    needed: int,
    max_transfers: int,
    max_path_edges: int,
    max_states: int,
    transfer_penalty_min: float,
    access_cache: Optional[MutableMapping[str, List[Tuple[int, int, WalkLink]]]] = None,
    egress_cache: Optional[MutableMapping[str, Dict[int, Tuple[Tuple[int, WalkLink], ...]]]] = None,
    reachable_cache: Optional[MutableMapping[Tuple[str, int], List[Set[int]]]] = None,
) -> Tuple[List[PathSpec], SearchStats]:
    stats = SearchStats()
    if needed <= 0:
        return [], stats
    if access_cache is None:
        access = _access_options(network, origin)
    else:
        access = access_cache.get(origin)
        if access is None:
            access = _access_options(network, origin)
            access_cache[origin] = access
    if not access:
        stats.no_access = 1
        return [], stats
    if egress_cache is None:
        egress = _egress_options(network, destination)
    else:
        egress = egress_cache.get(destination)
        if egress is None:
            egress = _egress_options(network, destination)
            egress_cache[destination] = egress
    if not egress:
        stats.no_egress = 1
        return [], stats
    reach_key = (destination, max_transfers)
    if reachable_cache is None:
        reachable = _reachable_chain_sets(network, set(egress), max_transfers)
    else:
        reachable = reachable_cache.get(reach_key)
        if reachable is None:
            reachable = _reachable_chain_sets(network, set(egress), max_transfers)
            reachable_cache[reach_key] = reachable

    # Heap item: priority, serial, kind, payload.
    # kind 0 = expandable state, kind 1 = completed PathSpec.
    heap: List[Tuple[float, int, int, object]] = []
    serial = 0
    labels: Dict[Tuple[int, int, Tuple[int, ...]], List[float]] = {}
    for chain_uid, board_pos, access_walk in access:
        if chain_uid not in reachable[max_transfers]:
            continue
        chain = network.chains[chain_uid]
        actual_before = access_walk.time_min + 30.0 / max(chain.edge_trips[board_pos], EPS)
        score_before = actual_before
        walks = tuple([access_walk] if access_walk.distance_m > EPS else [])
        used_chains = (chain_uid,)
        state = (
            chain_uid,
            board_pos,
            tuple(),
            walks,
            used_chains,
            frozenset((origin, chain.stops[board_pos])),
            actual_before,
            score_before,
            0,
        )
        label_key = (chain_uid, board_pos, used_chains)
        if _push_label(labels, label_key, score_before, max(needed, 2)):
            heapq.heappush(heap, (score_before, serial, 0, state))
            serial += 1

    accepted: List[PathSpec] = []
    seen = set(observed_signatures)
    while heap and len(accepted) < needed and stats.expanded_states < max_states:
        priority, _, kind, payload = heapq.heappop(heap)
        if kind == 1:
            spec = payload  # type: ignore[assignment]
            assert isinstance(spec, PathSpec)
            if spec.signature in seen:
                continue
            seen.add(spec.signature)
            accepted.append(spec)
            stats.generated_paths += 1
            continue

        (
            chain_uid,
            board_pos,
            completed_legs,
            walks,
            used_chains,
            visited_stops,
            actual_before,
            score_before,
            edge_count_completed,
        ) = payload  # type: ignore[misc]
        stats.expanded_states += 1
        chain = network.chains[chain_uid]
        transfers_used = len(completed_legs)

        # Completion on the current chain.  Terminal events share the same heap,
        # so accepted paths are popped in generalized-time order.
        for alight_pos, egress_walk in egress.get(chain_uid, ()):
            if alight_pos <= board_pos:
                continue
            edge_count = edge_count_completed + (alight_pos - board_pos)
            if edge_count > max_path_edges:
                continue
            leg = network.route_leg(chain_uid, board_pos, alight_pos)
            final_walks = list(walks)
            if egress_walk.distance_m > EPS:
                final_walks.append(egress_walk)
            spec = make_path_spec(
                network.hour,
                origin,
                destination,
                list(completed_legs) + [leg],
                final_walks,
                "generated_bounded",
            )
            if spec is None or spec.signature in seen:
                continue
            terminal_score = (
                actual_before
                + leg.travel_time
                + egress_walk.time_min
                + transfer_penalty_min * transfers_used
            )
            heapq.heappush(heap, (terminal_score, serial, 1, spec))
            serial += 1

        if transfers_used >= max_transfers:
            continue
        remaining_after_transfer = max_transfers - (transfers_used + 1)
        allowed_next = reachable[remaining_after_transfer]
        by_next = network.transitions.get(chain_uid, {})
        for to_chain_uid in sorted(set(by_next).intersection(allowed_next)):
            if to_chain_uid in used_chains:
                continue
            next_chain = network.chains[to_chain_uid]
            feasible: List[Tuple[float, TransferOption, RouteLeg]] = []
            for option in by_next[to_chain_uid]:
                if option.from_pos <= board_pos:
                    continue
                if option.to_pos >= len(next_chain.stops) - 1:
                    continue
                if option.walk.origin in visited_stops and option.walk.origin != chain.stops[board_pos]:
                    continue
                if option.walk.destination in visited_stops and option.walk.destination != option.walk.origin:
                    continue
                edge_count = edge_count_completed + (option.from_pos - board_pos)
                if edge_count >= max_path_edges:
                    continue
                leg = network.route_leg(chain_uid, board_pos, option.from_pos)
                actual_new = (
                    actual_before
                    + leg.travel_time
                    + option.walk.time_min
                    + 30.0 / max(next_chain.edge_trips[option.to_pos], EPS)
                )
                score_new = actual_new + transfer_penalty_min * (transfers_used + 1)
                feasible.append((score_new, option, leg))
            feasible.sort(
                key=lambda item: (
                    item[0],
                    item[1].from_pos,
                    item[1].to_pos,
                    item[1].walk.time_min,
                )
            )
            if network.max_transfer_options_per_pair > 0:
                feasible = feasible[: network.max_transfer_options_per_pair]
            for score_new, option, leg in feasible:
                actual_new = (
                    actual_before
                    + leg.travel_time
                    + option.walk.time_min
                    + 30.0 / max(next_chain.edge_trips[option.to_pos], EPS)
                )
                new_walks = list(walks)
                if option.walk.distance_m > EPS:
                    new_walks.append(option.walk)
                new_completed = tuple(completed_legs) + (leg,)
                new_used = tuple(used_chains) + (to_chain_uid,)
                new_visited = frozenset(
                    set(visited_stops)
                    | {option.walk.origin, option.walk.destination}
                )
                new_edge_count = edge_count_completed + (option.from_pos - board_pos)
                state = (
                    to_chain_uid,
                    option.to_pos,
                    new_completed,
                    tuple(new_walks),
                    new_used,
                    new_visited,
                    actual_new,
                    score_new,
                    new_edge_count,
                )
                label_key = (to_chain_uid, option.to_pos, new_used)
                if _push_label(labels, label_key, score_new, max(needed, 2)):
                    heapq.heappush(heap, (score_new, serial, 0, state))
                    serial += 1

    if heap and stats.expanded_states >= max_states and len(accepted) < needed:
        stats.hit_state_cap = 1
    return accepted, stats


def build_candidates_for_od(
    network: HourNetwork,
    od_index: int,
    origin: str,
    destination: str,
    observed: Sequence[ObservedCandidate],
    candidate_limit: int,
    max_transfers: int,
    max_path_edges: int,
    max_states: int,
    transfer_penalty_min: float,
    access_cache: Optional[MutableMapping[str, List[Tuple[int, int, WalkLink]]]] = None,
    egress_cache: Optional[MutableMapping[str, Dict[int, Tuple[Tuple[int, WalkLink], ...]]]] = None,
    reachable_cache: Optional[MutableMapping[Tuple[str, int], List[Set[int]]]] = None,
) -> Tuple[List[Dict[str, object]], SearchStats]:
    stats = SearchStats()
    observed_sorted = sorted(
        observed,
        key=lambda value: (
            -value.passengers_train,
            -value.passengers_all,
            value.spec.total_time,
            value.spec.signature,
        ),
    )
    if len(observed_sorted) > candidate_limit:
        stats.observed_truncated = len(observed_sorted) - candidate_limit
    observed_kept = observed_sorted[:candidate_limit]
    stats.observed_kept = len(observed_kept)
    seen = {value.spec.signature for value in observed_kept}
    generated, generated_stats = search_generated_paths(
        network,
        origin,
        destination,
        seen,
        candidate_limit - len(observed_kept),
        max_transfers,
        max_path_edges,
        max_states,
        transfer_penalty_min,
        access_cache,
        egress_cache,
        reachable_cache,
    )
    stats.add(generated_stats)
    combined: List[Tuple[PathSpec, float, float]] = [
        (value.spec, value.passengers_train, value.passengers_all)
        for value in observed_kept
    ]
    combined.extend((spec, 0.0, 0.0) for spec in generated)
    # Observed paths remain guaranteed in the pool.  Search rank is diagnostic.
    combined.sort(
        key=lambda value: (
            0 if value[0].source == "observed" else 1,
            value[0].total_time + transfer_penalty_min * value[0].transfers,
            value[0].signature,
        )
    )
    rows = [
        serialize_path(spec, od_index, train_count, all_count, rank)
        for rank, (spec, train_count, all_count) in enumerate(combined[:candidate_limit], start=1)
    ]
    return rows, stats


# -----------------------------------------------------------------------------
# Purpose-detail streaming aggregation
# -----------------------------------------------------------------------------


def read_expected_rows(detail_path: Path, explicit: int) -> Optional[int]:
    if explicit > 0:
        return explicit
    for manifest_name in ["purpose_trip_manifest.json", "model_input_manifest.json"]:
        path = detail_path.parent / manifest_name
        if not path.exists():
            continue
        payload = load_json(path)
        value = payload.get("output_rows", {}).get("purpose_trip_detail") if isinstance(payload.get("output_rows"), dict) else None
        if value:
            return int(value)
    return None


def aggregate_observed_patterns(
    detail_path: Path,
    dates: Set[str],
    train_dates: Set[str],
    hours: Set[int],
    expected_rows: Optional[int],
    progress_seconds: float,
    observed_path_dates: str,
) -> Tuple[Dict[int, Dict[Tuple[str, str, str, str], Tuple[float, float]]], Dict[str, int]]:
    required = [
        "service_date",
        "hour",
        "origin_stop_id",
        "destination_stop_id",
        "passengers",
        "route_chain",
        "transfer_stop_chain",
    ]
    totals: Dict[int, Dict[Tuple[str, str, str, str], List[float]]] = defaultdict(dict)
    qa: Dict[str, int] = defaultdict(int)
    source_dates = train_dates if observed_path_dates == "train" else dates
    meter = ProgressETA("purpose-detail", expected_rows, progress_seconds, warmup_units=200_000)
    processed = 0
    reader = pd.read_csv(
        detail_path,
        compression="infer",
        usecols=lambda column: column in set(required),
        dtype=str,
        chunksize=300_000,
        low_memory=False,
    )
    for chunk in reader:
        require_columns(chunk, required, "purpose_trip_detail")
        processed += len(chunk)
        qa["input_rows"] += len(chunk)
        chunk["service_date"] = chunk["service_date"].map(normalize_id)
        chunk["hour"] = pd.to_numeric(chunk["hour"], errors="coerce")
        chunk["passengers"] = pd.to_numeric(chunk["passengers"], errors="coerce")
        for column in ["origin_stop_id", "destination_stop_id", "route_chain", "transfer_stop_chain"]:
            chunk[column] = chunk[column].map(normalize_id)
        chunk = chunk[
            chunk["service_date"].isin(source_dates)
            & chunk["hour"].isin(hours)
            & chunk["passengers"].gt(0)
            & chunk["origin_stop_id"].ne("")
            & chunk["destination_stop_id"].ne("")
            & chunk["route_chain"].ne("")
        ].copy()
        qa["selected_rows"] += len(chunk)
        if chunk.empty:
            meter.update(processed)
            continue
        if observed_path_dates == "train":
            chunk["passengers_train"] = chunk["passengers"]
        else:
            chunk["is_train"] = chunk["service_date"].isin(train_dates)
            chunk["passengers_train"] = (
                chunk["passengers"] * chunk["is_train"].astype(float)
            )
        grouped = chunk.groupby(
            [
                "hour",
                "origin_stop_id",
                "destination_stop_id",
                "route_chain",
                "transfer_stop_chain",
            ],
            as_index=False,
            dropna=False,
        ).agg(
            passengers_all=("passengers", "sum"),
            passengers_train=("passengers_train", "sum"),
        )
        for row in grouped.itertuples(index=False):
            hour = int(row.hour)
            key = (
                str(row.origin_stop_id),
                str(row.destination_stop_id),
                str(row.route_chain),
                str(row.transfer_stop_chain),
            )
            values = totals[hour].setdefault(key, [0.0, 0.0])
            values[0] += float(row.passengers_train)
            values[1] += float(row.passengers_all)
        meter.update(
            processed,
            extra=f"selected {qa['selected_rows']:,}, unique-patterns {sum(len(v) for v in totals.values()):,}",
        )
    meter.update(
        processed,
        extra=f"selected {qa['selected_rows']:,}, unique-patterns {sum(len(v) for v in totals.values()):,}",
        force=True,
    )
    normalized: Dict[int, Dict[Tuple[str, str, str, str], Tuple[float, float]]] = {
        hour: {key: (values[0], values[1]) for key, values in patterns.items()}
        for hour, patterns in totals.items()
    }
    qa["unique_patterns"] = sum(len(values) for values in normalized.values())
    return normalized, dict(qa)


def reconstruct_observed_for_hour(
    network: HourNetwork,
    patterns: Mapping[Tuple[str, str, str, str], Tuple[float, float]],
    max_transfers: int,
    progress_seconds: float,
    observed_path_dates: str,
) -> Tuple[Dict[Tuple[str, str], Tuple[ObservedCandidate, ...]], Dict[str, int]]:
    by_od_temp: Dict[Tuple[str, str], Dict[str, List[object]]] = defaultdict(dict)
    qa: Dict[str, int] = defaultdict(int)
    meter = ProgressETA(
        f"observed-map-{network.hour:02d}", len(patterns), progress_seconds, warmup_units=1000
    )
    for index, (key, counts) in enumerate(patterns.items(), start=1):
        origin, destination, routes, transfers = key
        if observed_path_dates == "train" and float(counts[0]) <= 0:
            qa["test_only_pattern_skipped"] += 1
            meter.update(index, extra=f"mapped {qa['mapped']:,}")
            continue
        spec, reason = reconstruct_observed_spec(
            network, origin, destination, routes, transfers, max_transfers
        )
        qa[reason] += 1
        if spec is not None:
            local = by_od_temp[(origin, destination)].setdefault(
                spec.signature, [spec, 0.0, 0.0]
            )
            local[1] = float(local[1]) + float(counts[0])
            local[2] = float(local[2]) + float(counts[1])
        meter.update(index, extra=f"mapped {qa['mapped']:,}")
    meter.update(len(patterns), extra=f"mapped {qa['mapped']:,}", force=True)
    by_od = {
        od: tuple(
            ObservedCandidate(
                spec=value[0],
                passengers_train=float(value[1]),
                passengers_all=float(value[2]),
            )
            for value in values.values()
        )
        for od, values in by_od_temp.items()
    }
    qa["mapped_od_pairs"] = len(by_od)
    qa["mapped_unique_paths"] = sum(len(values) for values in by_od.values())
    return by_od, dict(qa)


# -----------------------------------------------------------------------------
# Multiprocessing worker and checkpoint files
# -----------------------------------------------------------------------------


_WORKER_NETWORK: Optional[HourNetwork] = None
_WORKER_OBSERVED: Optional[Dict[Tuple[str, str], Tuple[ObservedCandidate, ...]]] = None
_WORKER_CONFIG: Optional[Dict[str, object]] = None
_WORKER_ACCESS_CACHE: Dict[str, List[Tuple[int, int, WalkLink]]] = {}
_WORKER_EGRESS_CACHE: Dict[str, Dict[int, Tuple[Tuple[int, WalkLink], ...]]] = {}
_WORKER_REACHABLE_CACHE: Dict[Tuple[str, int], List[Set[int]]] = {}


def _init_worker(context_path: str) -> None:
    global _WORKER_NETWORK, _WORKER_OBSERVED, _WORKER_CONFIG
    global _WORKER_ACCESS_CACHE, _WORKER_EGRESS_CACHE, _WORKER_REACHABLE_CACHE
    with open(context_path, "rb") as handle:
        payload = pickle.load(handle)
    _WORKER_NETWORK = payload["network"]
    _WORKER_OBSERVED = payload["observed"]
    _WORKER_CONFIG = payload["config"]
    _WORKER_ACCESS_CACHE = {}
    _WORKER_EGRESS_CACHE = {}
    _WORKER_REACHABLE_CACHE = {}


def _process_od_batch(batch: Sequence[Tuple[int, str, str]]) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    if _WORKER_NETWORK is None or _WORKER_OBSERVED is None or _WORKER_CONFIG is None:
        raise RuntimeError("worker context not initialized")
    rows: List[Dict[str, object]] = []
    total_stats = SearchStats()
    config = _WORKER_CONFIG
    for od_index, origin, destination in batch:
        local_rows, stats = build_candidates_for_od(
            _WORKER_NETWORK,
            int(od_index),
            str(origin),
            str(destination),
            _WORKER_OBSERVED.get((str(origin), str(destination)), ()),
            int(config["candidate_limit"]),
            int(config["max_transfers"]),
            int(config["max_path_edges"]),
            int(config["max_states"]),
            float(config["transfer_penalty_min"]),
            _WORKER_ACCESS_CACHE,
            _WORKER_EGRESS_CACHE,
            _WORKER_REACHABLE_CACHE,
        )
        rows.extend(local_rows)
        total_stats.add(stats)
        total_stats.candidate_rows += len(local_rows)
        if local_rows:
            total_stats.ods_with_candidates += 1
            if len(local_rows) >= int(config["candidate_limit"]):
                total_stats.ods_reaching_candidate_limit += 1
        else:
            total_stats.ods_without_candidates += 1
    return rows, asdict(total_stats)


CANDIDATE_COLUMNS = [
    "od_index",
    "hour",
    "origin_stop_id",
    "destination_stop_id",
    "signature",
    "route_legs",
    "segment_keys",
    "boarding_edges",
    "walk_links",
    "ride_time",
    "wait_time",
    "access_walk_distance_m",
    "access_walk_time",
    "transfer_walk_distance_m",
    "transfer_walk_time",
    "total_walk_distance_m",
    "walk_time",
    "total_time",
    "transfers",
    "source",
    "observed_passengers_train",
    "observed_passengers_all",
    "candidate_rank_search",
]


def write_rows_gzip(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A gzip file containing an empty string still has a non-zero on-disk size.
    # Always write the CSV header so zero-path batches are valid empty CSV files.
    columns = list(rows[0].keys()) if rows else CANDIDATE_COLUMNS
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def combine_gzip_csv(
    parts: Sequence[Path],
    destination: Path,
    progress_seconds: float = 15.0,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    wrote_header = False
    meter = ProgressETA(
        "combine-candidates",
        len(parts),
        progress_seconds,
        warmup_units=max(1, min(10, len(parts))),
    )
    with gzip.open(destination, "wt", encoding="utf-8-sig", newline="") as output:
        for part_index, part in enumerate(parts, start=1):
            if not part.exists() or part.stat().st_size == 0:
                meter.update(part_index, extra=f"rows {rows:,}")
                continue
            with gzip.open(part, "rt", encoding="utf-8-sig", newline="") as source:
                header = source.readline()
                if not header:
                    meter.update(part_index, extra=f"rows {rows:,}")
                    continue
                if not wrote_header:
                    output.write(header.lstrip("\ufeff"))
                    wrote_header = True
                for line in source:
                    output.write(line)
                    rows += 1
            meter.update(part_index, extra=f"rows {rows:,}")
        if not wrote_header:
            output.write(",".join(CANDIDATE_COLUMNS) + "\n")
    meter.update(len(parts), extra=f"rows {rows:,}", force=True)
    return rows


def generate_hour_batches(
    network: HourNetwork,
    observed: Dict[Tuple[str, str], Tuple[ObservedCandidate, ...]],
    od_group: pd.DataFrame,
    work_dir: Path,
    args: argparse.Namespace,
    global_meter: ProgressETA,
    global_done_before: int,
) -> Tuple[List[Path], Dict[str, int]]:
    hour = network.hour
    records = [
        (int(row.od_index), str(row.origin_stop_id), str(row.destination_stop_id))
        for row in od_group.itertuples(index=False)
    ]
    batch_size = max(1, int(args.od_batch_size))
    batches = [records[index : index + batch_size] for index in range(0, len(records), batch_size)]
    context_path = work_dir / f"hour_{hour:02d}_context.pkl"
    config = {
        "candidate_limit": args.candidate_limit,
        "max_transfers": args.max_transfers,
        "max_path_edges": args.max_path_edges,
        "max_states": args.max_search_states_per_od,
        "transfer_penalty_min": args.search_transfer_penalty_min,
    }
    with open(context_path, "wb") as handle:
        pickle.dump(
            {"network": network, "observed": observed, "config": config},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    part_paths = [work_dir / f"hour_{hour:02d}_batch_{index:06d}.csv.gz" for index in range(len(batches))]
    stats_paths = [work_dir / f"hour_{hour:02d}_batch_{index:06d}.json" for index in range(len(batches))]
    resume = not args.no_resume
    completed_ods = 0
    total_stats = SearchStats()
    pending: List[int] = []
    for index, batch in enumerate(batches):
        if resume and part_paths[index].exists() and stats_paths[index].exists():
            completed_ods += len(batch)
            payload = load_json(stats_paths[index])
            total_stats.add(SearchStats(**{key: int(payload.get(key, 0)) for key in SearchStats.__dataclass_fields__}))
        else:
            pending.append(index)

    local_meter = ProgressETA(
        f"path-{hour:02d}", len(records), args.progress_seconds, warmup_units=batch_size
    )
    if completed_ods:
        local_meter.update(
            completed_ods,
            extra=f"resume batches {len(batches)-len(pending):,}/{len(batches):,}",
            force=True,
        )
        global_meter.update(global_done_before + completed_ods, force=True)

    def consume_result(batch_index: int, result: Tuple[List[Dict[str, object]], Dict[str, int]]) -> None:
        nonlocal completed_ods
        rows, raw_stats = result
        write_rows_gzip(part_paths[batch_index], rows)
        stats_paths[batch_index].write_text(
            json.dumps(raw_stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        local = SearchStats(**{key: int(raw_stats.get(key, 0)) for key in SearchStats.__dataclass_fields__})
        total_stats.add(local)
        completed_ods += len(batches[batch_index])
        extra = (
            f"paths {len(rows):,} in last batch, "
            f"avg candidates {total_stats.candidate_rows/max(completed_ods,1):,.2f}, "
            f"avg states {total_stats.expanded_states/max(completed_ods,1):,.1f}, "
            f"full-{int(args.candidate_limit)} ODs {total_stats.ods_reaching_candidate_limit:,}, "
            f"state-cap ODs {total_stats.hit_state_cap:,}"
        )
        local_meter.update(completed_ods, extra=extra, force=True)
        global_meter.update(global_done_before + completed_ods, extra=f"hour {hour:02d}")

    if pending:
        if args.workers <= 1:
            _init_worker(str(context_path))
            for batch_index in pending:
                consume_result(batch_index, _process_od_batch(batches[batch_index]))
        else:
            with ProcessPoolExecutor(
                max_workers=int(args.workers),
                initializer=_init_worker,
                initargs=(str(context_path),),
            ) as executor:
                futures = {
                    executor.submit(_process_od_batch, batches[index]): index for index in pending
                }
                for future in as_completed(futures):
                    consume_result(futures[future], future.result())
    local_meter.update(
        len(records),
        extra=(
            f"avg candidates {total_stats.candidate_rows/max(len(records),1):,.2f}, "
            f"avg states {total_stats.expanded_states/max(len(records),1):,.1f}, "
            f"full-{int(args.candidate_limit)} ODs {total_stats.ods_reaching_candidate_limit:,}, "
            f"state-cap ODs {total_stats.hit_state_cap:,}"
        ),
        force=True,
    )
    return part_paths, asdict(total_stats)


# -----------------------------------------------------------------------------
# Output helpers and main
# -----------------------------------------------------------------------------


def copy_model_input_files(model_input: Path, output: Path) -> None:
    for name in [
        "segments.csv",
        "segments_baseline.csv.gz",
        "stops.csv",
        "routes.csv",
        "od_hourly.csv",
        "od_baseline.csv.gz",
    ]:
        source = model_input / name
        if source.exists():
            shutil.copy2(source, output / name)


def run_signature(
    segments: pd.DataFrame,
    stops: pd.DataFrame,
    od: pd.DataFrame,
    args: argparse.Namespace,
    dates: Sequence[str],
    train_dates: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for frame, columns in [
        (
            segments,
            ["hour", "route_id", "from_stop_id", "to_stop_id", "travel_time", "trips"],
        ),
        (stops, ["stop_id", "latitude", "longitude"]),
        (od, ["hour", "origin_stop_id", "destination_stop_id", "avg_passengers"]),
    ]:
        ordered = frame[columns].sort_values(columns, kind="stable")
        digest.update(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes())
    payload = {
        "builder_version": BUILDER_VERSION,
        "dates": list(dates),
        "train_dates": list(train_dates),
        "hours": sorted(parse_csv_set(args.hours, int)),
        "access_radius_m": args.access_radius_m,
        "transfer_radius_m": args.transfer_radius_m,
        "egress_radius_m": 0.0,
        "walk_speed_kmph": args.walk_speed_kmph,
        "max_transfers": args.max_transfers,
        "candidate_limit": args.candidate_limit,
        "choice_set_size": args.choice_set_size,
        "max_path_edges": args.max_path_edges,
        "max_search_states_per_od": args.max_search_states_per_od,
        "max_transfer_options_per_pair": args.max_transfer_options_per_pair,
        "search_transfer_penalty_min": args.search_transfer_penalty_min,
        "max_od": args.max_od,
        "od_batch_size": args.od_batch_size,
        "observed_path_dates": args.observed_path_dates,
        "purpose_detail_path": str(Path(args.purpose_detail).resolve()),
        "purpose_detail_size": Path(args.purpose_detail).stat().st_size,
        "purpose_detail_mtime_ns": Path(args.purpose_detail).stat().st_mtime_ns,
    }
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    if args.max_transfers != 2:
        raise ValueError("이 모델 입력 생성기는 최대 환승 2회를 전제로 합니다: --max-transfers 2")
    if args.candidate_limit <= 0:
        raise ValueError("candidate-limit은 1 이상이어야 합니다.")
    if args.choice_set_size <= 0 or args.choice_set_size > args.candidate_limit:
        raise ValueError("choice-set-size는 1 이상 candidate-limit 이하여야 합니다.")
    if args.access_radius_m <= 0:
        raise ValueError("access-radius-m은 0보다 커야 합니다.")
    if args.transfer_radius_m < 0:
        raise ValueError("transfer-radius-m은 0 이상이어야 합니다.")
    if args.walk_speed_kmph <= 0:
        raise ValueError("walk-speed-kmph은 0보다 커야 합니다.")
    if args.max_od > 0:
        print(
            "*** TEST MODE: --max-od는 시간대별 샘플 상한이며 최종 분석에 사용하면 안 됩니다. ***",
            flush=True,
        )

    model_input = Path(args.model_input).resolve()
    purpose_detail = Path(args.purpose_detail).resolve()
    output = Path(args.output).resolve()
    if not model_input.exists():
        raise FileNotFoundError(model_input)
    if not purpose_detail.exists():
        raise FileNotFoundError(purpose_detail)
    if args.fresh and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    # Remove stale outputs from the earlier integrated fitting version.  The
    # checkpoint directory and candidate parts are deliberately kept for resume.
    stale_names = [
        "route_choice_parameters.json",
        "route_choice_training_table.csv.gz",
        "route_choice_training_top3.csv.gz",
        "route_choice_training_top5.csv.gz",
        "path_choice_sets.csv.gz",
    ]
    for name in stale_names:
        path = output / name
        if path.exists():
            path.unlink()
    for path in output.glob("candidate_paths_top*.csv.gz"):
        path.unlink()

    dates = determine_dates(model_input, args.dates)
    if not dates:
        raise RuntimeError("분석 날짜를 찾지 못했습니다. --dates를 지정하세요.")
    train_dates = determine_train_dates(dates, args.train_dates)
    hours = set(parse_csv_set(args.hours, int))
    print("analysis dates:", ", ".join(dates), flush=True)
    print("training dates:", ", ".join(train_dates), flush=True)
    print("hours:", ", ".join(map(str, sorted(hours))), flush=True)
    print(
        "walking rules: "
        f"origin access <= {args.access_radius_m:g}m, "
        f"transfer <= {args.transfer_radius_m:g}m, "
        "destination egress = 0m (exact destination stop)",
        flush=True,
    )
    print(
        f"candidate rule: up to {args.candidate_limit} paths per hour-OD; "
        f"downstream fitted model retains top {args.choice_set_size}",
        flush=True,
    )

    stage_start = time.perf_counter()
    segments = load_baseline_segments(
        model_input,
        hours,
        expected_rows=args.expected_segment_rows,
        progress_seconds=args.progress_seconds,
    )
    stops = load_stops(model_input)
    od = load_od_baseline(model_input, hours, args.max_od)
    network_stops = set(segments["from_stop_id"]) | set(segments["to_stop_id"])
    od = od[
        od["origin_stop_id"].isin(network_stops)
        & od["destination_stop_id"].isin(network_stops)
    ].copy().reset_index(drop=True)
    od["od_index"] = np.arange(len(od), dtype=np.int64)
    print(
        f"baseline segments {len(segments):,}, stops {len(stops):,}, "
        f"unique hour-OD {len(od):,} "
        f"({format_seconds(time.perf_counter()-stage_start)})",
        flush=True,
    )

    max_walk_radius = max(float(args.access_radius_m), float(args.transfer_radius_m))
    print(
        f"[stage] building one spatial walking index at {max_walk_radius:g}m "
        "and deriving access/transfer views",
        flush=True,
    )
    max_walk_neighbors = build_walk_neighbors(
        stops,
        max_walk_radius,
        args.walk_speed_kmph,
        progress_seconds=args.progress_seconds,
        label="walk-index",
    )
    access_walk_neighbors = filter_walk_neighbors(
        max_walk_neighbors, args.access_radius_m
    )
    transfer_walk_neighbors = filter_walk_neighbors(
        max_walk_neighbors, args.transfer_radius_m
    )
    access_directed_links = count_directed_walk_links(access_walk_neighbors)
    transfer_directed_links = count_directed_walk_links(transfer_walk_neighbors)
    access_written = write_walk_edges(
        output / "walk_access_edges.csv.gz",
        access_walk_neighbors,
        "origin_access",
    )
    transfer_written = write_walk_edges(
        output / "walk_transfer_edges.csv.gz",
        transfer_walk_neighbors,
        "transfer",
    )
    if access_written != access_directed_links or transfer_written != transfer_directed_links:
        raise RuntimeError("walking-link count mismatch while writing model inputs")
    print(
        f"access links <= {args.access_radius_m:g}m: {access_directed_links:,}; "
        f"transfer links <= {args.transfer_radius_m:g}m: {transfer_directed_links:,}",
        flush=True,
    )
    del max_walk_neighbors

    walking_rules = {
        "origin_access_radius_m": float(args.access_radius_m),
        "transfer_radius_m": float(args.transfer_radius_m),
        "destination_egress_radius_m": 0.0,
        "destination_rule": "exact_destination_stop",
        "walk_speed_kmph": float(args.walk_speed_kmph),
        "access_directed_links": access_directed_links,
        "transfer_directed_links": transfer_directed_links,
        "access_edges_file": "walk_access_edges.csv.gz",
        "transfer_edges_file": "walk_transfer_edges.csv.gz",
    }
    (output / "walking_rules.json").write_text(
        json.dumps(walking_rules, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    signature = run_signature(segments, stops, od, args, dates, train_dates)
    work_dir = output / "_path_work" / signature[:16]
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "run_signature.json").write_text(
        json.dumps({"signature": signature}, indent=2), encoding="utf-8"
    )

    expected_rows = read_expected_rows(purpose_detail, args.expected_purpose_rows)
    print("[stage] streaming purpose-trip patterns", flush=True)
    patterns_by_hour, detail_qa = aggregate_observed_patterns(
        purpose_detail,
        set(dates),
        set(train_dates),
        hours,
        expected_rows,
        args.progress_seconds,
        args.observed_path_dates,
    )

    total_od = len(od)
    global_meter = ProgressETA(
        "all-paths",
        total_od,
        args.progress_seconds,
        warmup_units=max(1, args.od_batch_size),
    )
    all_parts: List[Path] = []
    global_done = 0
    hour_qa: Dict[str, object] = {}
    total_search = SearchStats()
    observed_qa_total: Dict[str, int] = defaultdict(int)
    for hour in sorted(hours):
        od_group = od[od["hour"] == hour].copy()
        if od_group.empty:
            continue
        print(f"\n=== hour {hour:02d}: OD {len(od_group):,} ===", flush=True)
        index_started = time.perf_counter()
        network, index_stats = build_hour_network(
            segments,
            hour,
            access_walk_neighbors,
            transfer_walk_neighbors,
            args.max_transfer_options_per_pair,
        )
        print(
            f"route/transfer index: {json.dumps(index_stats, ensure_ascii=False)} "
            f"in {format_seconds(time.perf_counter()-index_started)}",
            flush=True,
        )
        observed, observed_qa = reconstruct_observed_for_hour(
            network,
            patterns_by_hour.get(hour, {}),
            args.max_transfers,
            args.progress_seconds,
            args.observed_path_dates,
        )
        for key, value in observed_qa.items():
            observed_qa_total[key] += int(value)
        parts, search_stats = generate_hour_batches(
            network,
            observed,
            od_group,
            work_dir,
            args,
            global_meter,
            global_done,
        )
        all_parts.extend(parts)
        global_done += len(od_group)
        local_stats = SearchStats(
            **{
                key: int(search_stats.get(key, 0))
                for key in SearchStats.__dataclass_fields__
            }
        )
        total_search.add(local_stats)
        hour_qa[str(hour)] = {
            "od_pairs": len(od_group),
            "index": index_stats,
            "observed": observed_qa,
            "search": search_stats,
        }
        del network
        del observed
    global_meter.update(total_od, force=True)

    candidate_pool_path = output / "candidate_pool.csv.gz"
    candidate_rows = combine_gzip_csv(
        all_parts, candidate_pool_path, args.progress_seconds
    )
    if candidate_rows == 0:
        raise RuntimeError("후보 경로가 한 건도 생성되지 않았습니다.")
    if total_search.candidate_rows and candidate_rows != total_search.candidate_rows:
        print(
            "WARNING: checkpoint candidate-row count differs from combined file: "
            f"stats={total_search.candidate_rows:,}, file={candidate_rows:,}",
            flush=True,
        )
    print(f"candidate pool rows: {candidate_rows:,}", flush=True)

    print("[stage] copying base model_input files", flush=True)
    copy_model_input_files(model_input, output)

    choice_set_spec = {
        "candidate_pool_file": "candidate_pool.csv.gz",
        "candidate_limit_per_hour_od": int(args.candidate_limit),
        "downstream_choice_set_size": int(args.choice_set_size),
        "route_choice_fitting_stage": "downstream_model",
        "route_choice_fitted_here": False,
        "top_choice_file_created_here": False,
        "selection_method": "self_consistent_mnl_top_k",
        "candidate_rank_search_is_final_choice_rank": False,
        "recommended_downstream_sequence": [
            "fit an initial route-choice MNL using all generated candidates and training observed counts",
            f"rank candidates within each hour-OD by fitted baseline choice probability and retain top {int(args.choice_set_size)}",
            f"refit on the retained top {int(args.choice_set_size)} and repeat selection until the retained set is stable",
            "during training, report any observed path excluded from the retained set and preserve it when required for likelihood construction",
            f"normalize probabilities over the final top {int(args.choice_set_size)} alternatives",
            "assume each traveller chooses one retained alternative",
        ],
        "training_observation_columns": [
            "observed_passengers_train",
            "observed_passengers_all",
        ],
        "path_feature_columns": [
            "ride_time",
            "wait_time",
            "access_walk_time",
            "transfer_walk_time",
            "walk_time",
            "total_time",
            "transfers",
        ],
    }
    (output / "choice_set_spec.json").write_text(
        json.dumps(choice_set_spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    routeable_od_pairs = int(total_search.ods_with_candidates)
    avg_candidates = candidate_rows / max(routeable_od_pairs, 1)
    qa = {
        "builder_version": BUILDER_VERSION,
        "run_signature": signature,
        "analysis_dates": dates,
        "training_dates": train_dates,
        "hours": sorted(hours),
        "input_scale": {
            "baseline_segment_rows": len(segments),
            "stops": len(stops),
            "unique_hour_od_pairs": len(od),
            "purpose_detail": detail_qa,
        },
        "walking": walking_rules,
        "constraints": {
            "max_transfers": args.max_transfers,
            "candidate_limit": args.candidate_limit,
            "downstream_choice_set_size": args.choice_set_size,
            "destination_egress_radius_m": 0.0,
            "max_path_edges": args.max_path_edges,
            "max_search_states_per_od": args.max_search_states_per_od,
            "max_transfer_options_per_pair": args.max_transfer_options_per_pair,
        },
        "observed_path_source_dates": args.observed_path_dates,
        "observed_path_qa": dict(observed_qa_total),
        "search_qa": asdict(total_search),
        "outputs": {
            "candidate_pool_rows": candidate_rows,
            "candidate_pool_od_pairs": routeable_od_pairs,
            "average_candidates_per_routeable_od": avg_candidates,
            "od_pairs_without_candidates": int(total_search.ods_without_candidates),
            "od_pairs_reaching_candidate_limit": int(
                total_search.ods_reaching_candidate_limit
            ),
            "mnl_fitted": False,
            "top_choice_rows_created": False,
        },
        "hours_detail": hour_qa,
        "approximation": (
            "For speed, at most max_transfer_options_per_pair feasible transfer "
            "locations are expanded toward the same next route chain at one search "
            "state. Set it to 0 for exhaustive transfer-location expansion. "
            "max_search_states_per_od and max_path_edges are computational guards; "
            "any state-cap hit is counted in search_qa.hit_state_cap. "
            f"The {int(args.max_transfers)}-transfer cap, {int(args.candidate_limit)}-candidate cap, "
            f"{float(args.access_radius_m):g}m origin-access rule, "
            f"{float(args.transfer_radius_m):g}m transfer rule, and exact-destination "
            f"rule are explicit model constraints. The downstream top-{int(args.choice_set_size)} "
            "set is not selected by this script."
        ),
    }
    (output / "path_generation_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = load_json(model_input / "model_input_manifest.json")
    manifest.update(
        {
            "created_at": pd.Timestamp.now().isoformat(),
            "path_builder_version": BUILDER_VERSION,
            "source_model_input": str(model_input),
            "source_purpose_detail": str(purpose_detail),
            "target_dates": dates,
            "training_dates": train_dates,
            "observed_path_dates": args.observed_path_dates,
            "hours": sorted(hours),
            "path_builder": Path(__file__).name,
            "path_builder_stage": "candidate_generation_only",
            "path_topology_dimension": "hour x origin x destination (reused across dates)",
            "origin_access_radius_m": float(args.access_radius_m),
            "transfer_walk_radius_m": float(args.transfer_radius_m),
            "destination_egress_radius_m": 0.0,
            "destination_rule": "exact_destination_stop",
            "walk_speed_kmph": float(args.walk_speed_kmph),
            "transfer_count_cap": int(args.max_transfers),
            "candidate_limit": int(args.candidate_limit),
            "downstream_choice_set_size": int(args.choice_set_size),
            "route_choice_fitting_stage": "downstream_model",
            "candidate_pool_file": "candidate_pool.csv.gz",
            "choice_set_spec_file": "choice_set_spec.json",
            "walking_rules_file": "walking_rules.json",
            "access_walk_edges_file": "walk_access_edges.csv.gz",
            "transfer_walk_edges_file": "walk_transfer_edges.csv.gz",
            "path_generation_qa_file": "path_generation_qa.json",
        }
    )
    (output / "model_input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n완료:", output, flush=True)
    print(f" - candidate_pool.csv.gz: {candidate_rows:,} rows", flush=True)
    print(
        f" - routeable hour-OD: {routeable_od_pairs:,}; "
        f"average candidates: {avg_candidates:,.2f}",
        flush=True,
    )
    print(
        f" - full {args.candidate_limit}-candidate OD: "
        f"{total_search.ods_reaching_candidate_limit:,}",
        flush=True,
    )
    print(" - choice_set_spec.json (downstream top-5 assumption)", flush=True)
    print(" - walking_rules.json", flush=True)
    print(" - path_generation_qa.json", flush=True)
    print(" - MNL fitting/top-5 selection: not run in this script", flush=True)
    print(
        f"total elapsed: {format_seconds(time.perf_counter()-run_started)}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n사용자 중단: 완료된 배치 체크포인트는 다음 실행에서 재사용됩니다.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
