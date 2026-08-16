from __future__ import annotations

"""Fit a selective bus-edge bounty policy with analytical ReliefPotential.

This script reuses the route-choice and daily-state caches from the existing
no-capacity stage-2 model.  It adds an edge feature derived from the exact local
MNL derivative of a congestion metric with respect to a one-won edge bounty.

For a path p in OD group i and edge e,

    d P_ip / d r_e = beta_money * P_ip * (A_pe - sum_k P_ik A_ke)

where A_pe is one when path p traverses edge e.  Combining this derivative with
an edge-level marginal congestion cost gives every candidate edge's local
marginal relief without running one full simulation per edge.

The final selective policy is

    selection_score_e = theta_c * LowCrowding_e
                      + theta_a * AltProb_e
                      + theta_r * ReliefPotential_e

    selected_e = 1 if edge e is in the top target_share among eligible edges

    reward_e = selected_e * Rmax * sigmoid(theta_0 + selection_score_e)

The Optuna objective remains the exact nonlinear full-policy simulation:
maximize total training-date network congestion relief subject to the daily
budget.  ReliefPotential is therefore a policy feature and search guide, not a
replacement for the full simulation.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from scipy.special import expit

EPS = 1.0e-12
SCRIPT_VERSION = "relief-potential-selective-bounty-v1.0.0"
RELIEF_CACHE_VERSION = "analytical-mnl-relief-v1"
POLICY_FORM_VERSION = "top-share-relief-selective-sigmoid-v1"

METRIC_WHOLE_H = "whole_network_h"
METRIC_EXCESS_H2 = "squared_excess_above_reference"
METRIC_EXCESS_PM = "excess_passenger_minutes"
SUPPORTED_METRICS = (METRIC_WHOLE_H, METRIC_EXCESS_H2, METRIC_EXCESS_PM)

TRANSFORM_POSITIVE_LOG_P99 = "positive_log_p99"
TRANSFORM_POSITIVE_LINEAR_P99 = "positive_linear_p99"
TRANSFORM_POSITIVE_RANK = "positive_rank"
SUPPORTED_TRANSFORMS = (
    TRANSFORM_POSITIVE_LOG_P99,
    TRANSFORM_POSITIVE_LINEAR_P99,
    TRANSFORM_POSITIVE_RANK,
)

# Optuna writes harmless experimental warnings to stderr.  Windows PowerShell
# 5.1 may otherwise treat them as native-process failures when output is piped.
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class ReliefSelectiveTheta:
    intercept: float
    low_crowding: float
    altprob: float
    relief_potential: float
    target_share: float


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
        rate = done / elapsed if done > 0 else 0.0
        if self.total is not None and self.total > 0:
            pct = 100.0 * done / self.total
            eta = max(0, self.total - done) / max(rate, EPS)
            message = (
                f"[{self.label}] {done:,}/{self.total:,} ({pct:5.1f}%), "
                f"{rate:,.2f}/s, elapsed {format_seconds(elapsed)}, "
                f"ETA {format_seconds(eta)}"
            )
        else:
            message = (
                f"[{self.label}] {done:,}, {rate:,.2f}/s, "
                f"elapsed {format_seconds(elapsed)}"
            )
        if extra:
            message += f", {extra}"
        print(message, flush=True)
        self.last_print = now


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


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def file_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n - " + "\n - ".join(missing))


def import_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a selective edge-bounty policy using analytical MNL "
            "marginal congestion ReliefPotential"
        )
    )
    parser.add_argument(
        "--stage2-script",
        default=r".\fit_public_edge_bounty_policy.py",
        help="Existing no-capacity stage-2 script used to read caches and simulate policy",
    )
    parser.add_argument("--model-input", required=True)
    parser.add_argument("--route-choice-output", required=True)
    parser.add_argument(
        "--cache-source",
        required=True,
        help="Existing public_edge_bounty_top5 output containing topology/daily caches",
    )
    parser.add_argument(
        "--distributed-policy-output",
        default="",
        help="Optional distributed-policy output for comparison",
    )
    parser.add_argument(
        "--selective-policy-output",
        default="",
        help="Optional prior selective-policy output for comparison and warm start",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy-trials", type=int, default=0)
    parser.add_argument("--progress-seconds", type=float, default=15.0)
    parser.add_argument(
        "--selection-scope",
        choices=["global", "per_hour"],
        default=None,
    )
    parser.add_argument(
        "--relief-metric",
        choices=list(SUPPORTED_METRICS),
        default=None,
        help="Override policy.relief_selective.relief_potential.metric",
    )
    parser.add_argument(
        "--relief-transform",
        choices=list(SUPPORTED_TRANSFORMS),
        default=None,
        help="Override policy.relief_selective.relief_potential.transform",
    )
    parser.add_argument(
        "--gradient-check-edges",
        type=int,
        default=None,
        help="Finite-difference checks on this many edges after analytical cache creation",
    )
    parser.add_argument(
        "--gradient-check-reward-won",
        type=float,
        default=None,
        help="Small one-edge bounty used for finite-difference checks",
    )
    parser.add_argument("--rebuild-relief-cache", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def relief_selective_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    policy = dict(config.get("policy", {}))
    value = dict(policy.get("relief_selective", {}))
    value.setdefault("selection_scope", "global")
    value.setdefault("minimum_selected_edges", 1)
    value.setdefault("reward_rounding_won", 0.0)
    value.setdefault("target_share_log", True)
    value.setdefault("require_positive_relief", True)
    value.setdefault("minimum_positive_date_rate", 0.0)
    value.setdefault("relief_potential", {})
    value.setdefault("bayesian", {})
    relief = dict(value["relief_potential"])
    relief.setdefault("metric", METRIC_WHOLE_H)
    relief.setdefault("transform", TRANSFORM_POSITIVE_LOG_P99)
    relief.setdefault("compute_all_metrics", True)
    relief.setdefault("gradient_check_edges", 6)
    relief.setdefault("gradient_check_reward_won", 0.1)
    value["relief_potential"] = relief
    return value


def resolve_relief_settings(
    config: Mapping[str, Any],
    metric_override: Optional[str],
    transform_override: Optional[str],
    gradient_edges_override: Optional[int],
    gradient_reward_override: Optional[float],
) -> Dict[str, Any]:
    cfg = relief_selective_config(config)
    relief = dict(cfg["relief_potential"])
    metric = metric_override or str(relief["metric"]).strip()
    transform = transform_override or str(relief["transform"]).strip()
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported relief metric: {metric}")
    if transform not in SUPPORTED_TRANSFORMS:
        raise ValueError(f"Unsupported relief transform: {transform}")
    gradient_edges = (
        int(gradient_edges_override)
        if gradient_edges_override is not None
        else int(relief.get("gradient_check_edges", 6))
    )
    gradient_reward = (
        float(gradient_reward_override)
        if gradient_reward_override is not None
        else float(relief.get("gradient_check_reward_won", 0.1))
    )
    if gradient_edges < 0:
        raise ValueError("gradient-check-edges must be >= 0")
    if gradient_reward <= 0:
        raise ValueError("gradient-check-reward-won must be > 0")
    return {
        "metric": metric,
        "transform": transform,
        "compute_all_metrics": bool(relief.get("compute_all_metrics", True)),
        "gradient_check_edges": gradient_edges,
        "gradient_check_reward_won": gradient_reward,
    }


def metric_slug(metric: str) -> str:
    mapping = {
        METRIC_WHOLE_H: "whole_h",
        METRIC_EXCESS_H2: "excess_h2",
        METRIC_EXCESS_PM: "excess_pm",
    }
    return mapping[metric]


def metric_label(metric: str) -> str:
    mapping = {
        METRIC_WHOLE_H: "sum(travel_time * trips * load^2)",
        METRIC_EXCESS_H2: (
            "sum(travel_time * trips * max(load-reference,0)^2)"
        ),
        METRIC_EXCESS_PM: (
            "sum(travel_time * trips * max(load-reference,0))"
        ),
    }
    return mapping[metric]


def marginal_edge_cost(
    metric: str,
    travel_time: np.ndarray,
    load0: np.ndarray,
    reference_load: float,
) -> np.ndarray:
    """Derivative of the selected congestion metric with respect to edge users.

    load = users / trips, so the trips factor cancels in the derivative.
    """
    travel = np.asarray(travel_time, dtype=np.float64)
    load = np.asarray(load0, dtype=np.float64)
    if metric == METRIC_WHOLE_H:
        return 2.0 * travel * load
    if metric == METRIC_EXCESS_H2:
        return 2.0 * travel * np.maximum(load - reference_load, 0.0)
    if metric == METRIC_EXCESS_PM:
        return travel * (load > reference_load + 1.0e-9).astype(np.float64)
    raise ValueError(f"Unsupported metric: {metric}")


def metric_value(
    metric: str,
    travel_time: np.ndarray,
    trips: np.ndarray,
    load: np.ndarray,
    reference_load: float,
) -> float:
    travel = np.asarray(travel_time, dtype=np.float64)
    runs = np.asarray(trips, dtype=np.float64)
    onboard = np.asarray(load, dtype=np.float64)
    if metric == METRIC_WHOLE_H:
        return float(np.sum(travel * runs * onboard * onboard))
    if metric == METRIC_EXCESS_H2:
        excess = np.maximum(onboard - reference_load, 0.0)
        return float(np.sum(travel * runs * excess * excess))
    if metric == METRIC_EXCESS_PM:
        excess = np.maximum(onboard - reference_load, 0.0)
        return float(np.sum(travel * runs * excess))
    raise ValueError(f"Unsupported metric: {metric}")


def analytical_relief_for_state(
    topology: Any,
    state: Any,
    beta_discount_per_won: float,
    metric: str,
    reference_load: float,
) -> np.ndarray:
    """Return d(metric improvement)/d(edge bounty) at the no-bounty baseline."""
    edge_count = int(topology.path_edge.shape[1])
    if topology.groups == 0 or topology.paths == 0:
        return np.zeros(edge_count, dtype=np.float64)

    p0 = np.asarray(state.p0, dtype=np.float64)
    group_demand = np.asarray(state.group_demand, dtype=np.float64)
    travel = np.asarray(state.travel_time, dtype=np.float64)
    load0 = np.asarray(state.load0, dtype=np.float64)

    edge_shadow_cost = marginal_edge_cost(
        metric, travel, load0, reference_load
    )
    path_shadow_cost = np.asarray(
        topology.path_edge @ edge_shadow_cost,
        dtype=np.float64,
    ).reshape(-1)

    weighted_path_cost = p0 * path_shadow_cost
    group_mean_path_cost = np.add.reduceat(
        weighted_path_cost,
        np.asarray(topology.group_starts[:-1], dtype=np.int64),
    )
    centered_path_cost = (
        path_shadow_cost - group_mean_path_cost[topology.group_index]
    )
    demand_path = group_demand[topology.group_index]

    # dH/dr_e = A.T @ [D_i * beta * p_ip * (C_ip - E_i[C])]
    # Relief is -dH/dr_e because improvement = H0 - H1.
    path_weight = (
        float(beta_discount_per_won)
        * demand_path
        * p0
        * centered_path_cost
    )
    derivative = np.asarray(
        topology.path_edge.T @ path_weight,
        dtype=np.float64,
    ).reshape(-1)
    return -derivative


def relief_cache_signature(
    base_module: Any,
    topology_meta: Mapping[str, Any],
    daily_meta: Mapping[str, Any],
    parameters: Any,
    train_dates: Sequence[str],
    reference_load: float,
    metrics: Sequence[str],
) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "relief_cache_version": RELIEF_CACHE_VERSION,
        "base_policy_simulation_version": str(
            getattr(base_module, "POLICY_SIMULATION_VERSION", "")
        ),
        "topology_signature": topology_meta.get("signature"),
        "daily_cache_signature": daily_meta.get("signature"),
        "choice_parameters": asdict(parameters),
        "train_dates": list(train_dates),
        "reference_load": float(reference_load),
        "metrics": list(metrics),
    }
    return hash_payload(payload)


def add_relief_transforms(
    frame: pd.DataFrame,
    metric: str,
) -> Dict[str, float]:
    slug = metric_slug(metric)
    raw_col = f"relief_{slug}_mean_per_won"
    values = pd.to_numeric(frame[raw_col], errors="coerce").fillna(0.0).to_numpy(float)
    positive = np.maximum(values, 0.0)
    positive_mask = positive > 0.0

    rank = np.zeros(len(frame), dtype=np.float64)
    if np.any(positive_mask):
        local = pd.Series(positive[positive_mask])
        rank[positive_mask] = local.rank(method="average", pct=True).to_numpy(float)

    p99 = float(np.quantile(positive[positive_mask], 0.99)) if np.any(positive_mask) else 0.0
    linear = np.divide(
        positive,
        max(p99, EPS),
        out=np.zeros_like(positive),
        where=p99 > EPS,
    )
    linear = np.clip(linear, 0.0, 1.0)

    log_positive = np.log1p(positive)
    log_p99 = (
        float(np.quantile(log_positive[positive_mask], 0.99))
        if np.any(positive_mask)
        else 0.0
    )
    log_scaled = np.divide(
        log_positive,
        max(log_p99, EPS),
        out=np.zeros_like(log_positive),
        where=log_p99 > EPS,
    )
    log_scaled = np.clip(log_scaled, 0.0, 1.0)

    frame[f"relief_{slug}_{TRANSFORM_POSITIVE_RANK}"] = rank
    frame[f"relief_{slug}_{TRANSFORM_POSITIVE_LINEAR_P99}"] = linear
    frame[f"relief_{slug}_{TRANSFORM_POSITIVE_LOG_P99}"] = log_scaled

    return {
        "raw_min": float(np.min(values)) if len(values) else 0.0,
        "raw_max": float(np.max(values)) if len(values) else 0.0,
        "raw_mean": float(np.mean(values)) if len(values) else 0.0,
        "positive_edges": int(np.count_nonzero(positive_mask)),
        "positive_share": float(np.mean(positive_mask)) if len(values) else 0.0,
        "positive_raw_p50": float(np.quantile(positive[positive_mask], 0.50))
        if np.any(positive_mask)
        else 0.0,
        "positive_raw_p90": float(np.quantile(positive[positive_mask], 0.90))
        if np.any(positive_mask)
        else 0.0,
        "positive_raw_p99": p99,
        "positive_log_p99": log_p99,
    }


def build_relief_potential_cache(
    base_module: Any,
    baseline_hours: Mapping[int, pd.DataFrame],
    topologies: Mapping[int, Any],
    simulator: Any,
    parameters: Any,
    train_dates: Sequence[str],
    reference_load: float,
    metrics: Sequence[str],
    cache_root: Path,
    signature: str,
    progress_seconds: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    total_states = len(train_dates) * len(topologies)
    meter = Progress("relief-potential", total_states, progress_seconds)
    done = 0
    hour_frames = []

    for hour in sorted(topologies):
        topology = topologies[hour]
        edge_frame = baseline_hours[hour][
            ["hour", "route_id", "from_stop_id", "to_stop_id"]
        ].copy()
        n_edges = len(edge_frame)
        sums = {metric: np.zeros(n_edges, dtype=np.float64) for metric in metrics}
        sums_sq = {metric: np.zeros(n_edges, dtype=np.float64) for metric in metrics}
        positive_count = {metric: np.zeros(n_edges, dtype=np.int16) for metric in metrics}

        for date in train_dates:
            state = simulator.state(str(date), int(hour))
            for metric in metrics:
                relief = analytical_relief_for_state(
                    topology,
                    state,
                    parameters.beta_discount_per_won,
                    metric,
                    reference_load,
                )
                sums[metric] += relief
                sums_sq[metric] += relief * relief
                positive_count[metric] += (relief > 0.0).astype(np.int16)
            done += 1
            meter.update(
                done,
                extra=f"date {date}, hour {hour:02d}, edges {n_edges:,}",
            )

        n_dates = max(len(train_dates), 1)
        for metric in metrics:
            slug = metric_slug(metric)
            mean = sums[metric] / n_dates
            variance = np.maximum(sums_sq[metric] / n_dates - mean * mean, 0.0)
            edge_frame[f"relief_{slug}_sum_per_won"] = sums[metric]
            edge_frame[f"relief_{slug}_mean_per_won"] = mean
            edge_frame[f"relief_{slug}_std_per_won"] = np.sqrt(variance)
            edge_frame[f"relief_{slug}_positive_date_count"] = positive_count[metric]
            edge_frame[f"relief_{slug}_positive_date_rate"] = (
                positive_count[metric].astype(np.float64) / n_dates
            )
        hour_frames.append(edge_frame)

    meter.update(total_states, force=True)
    relief_frame = pd.concat(hour_frames, ignore_index=True)
    transform_meta: Dict[str, Any] = {}
    for metric in metrics:
        transform_meta[metric] = add_relief_transforms(relief_frame, metric)

    feature_path = cache_root / "relief_potential_features.csv.gz"
    relief_frame.to_csv(
        feature_path,
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )
    metadata = {
        "created_at": pd.Timestamp.now().isoformat(),
        "script_version": SCRIPT_VERSION,
        "relief_cache_version": RELIEF_CACHE_VERSION,
        "signature": signature,
        "train_dates": list(train_dates),
        "reference_load": float(reference_load),
        "metrics": list(metrics),
        "metric_formulas": {metric: metric_label(metric) for metric in metrics},
        "choice_parameters": asdict(parameters),
        "rows": int(len(relief_frame)),
        "transform_metadata": transform_meta,
        "feature_file": feature_path.name,
        "derivative_note": (
            "Each raw value is the analytical local derivative of metric relief "
            "with respect to one won of reward on the indicated hour-route-directed edge, "
            "summed or averaged over training dates only."
        ),
    }
    write_json(cache_root / "relief_potential_meta.json", metadata)
    return relief_frame, metadata


def prepare_relief_potential_cache(
    base_module: Any,
    baseline_hours: Mapping[int, pd.DataFrame],
    topologies: Mapping[int, Any],
    simulator: Any,
    topology_meta: Mapping[str, Any],
    daily_meta: Mapping[str, Any],
    parameters: Any,
    train_dates: Sequence[str],
    reference_load: float,
    metrics: Sequence[str],
    cache_root: Path,
    rebuild: bool,
    progress_seconds: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    signature = relief_cache_signature(
        base_module,
        topology_meta,
        daily_meta,
        parameters,
        train_dates,
        reference_load,
        metrics,
    )
    meta_path = cache_root / "relief_potential_meta.json"
    feature_path = cache_root / "relief_potential_features.csv.gz"
    reusable = False
    if not rebuild and meta_path.exists() and feature_path.exists():
        try:
            reusable = load_json(meta_path).get("signature") == signature
        except Exception:
            reusable = False
    if reusable:
        metadata = load_json(meta_path)
        frame = pd.read_csv(feature_path, compression="gzip", low_memory=False)
        print(
            f"[relief-potential-cache] reused: {len(frame):,} edge rows",
            flush=True,
        )
        return frame, metadata
    return build_relief_potential_cache(
        base_module,
        baseline_hours,
        topologies,
        simulator,
        parameters,
        train_dates,
        reference_load,
        metrics,
        cache_root,
        signature,
        progress_seconds,
    )


def merge_relief_features(
    base_features: pd.DataFrame,
    relief_features: pd.DataFrame,
    metric: str,
    transform: str,
) -> pd.DataFrame:
    key = ["hour", "route_id", "from_stop_id", "to_stop_id"]
    left = base_features.copy()
    right = relief_features.copy()
    for frame in [left, right]:
        frame["hour"] = pd.to_numeric(frame["hour"], errors="raise").astype(int)
        for column in ["route_id", "from_stop_id", "to_stop_id"]:
            frame[column] = frame[column].map(normalize_id)
    if left.duplicated(key).any():
        raise ValueError("Base operational features contain duplicate edge keys")
    if right.duplicated(key).any():
        raise ValueError("ReliefPotential features contain duplicate edge keys")

    left["__base_row_order"] = np.arange(len(left), dtype=np.int64)
    merged = left.merge(
        right,
        on=key,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    merged = (
        merged.sort_values("__base_row_order", kind="stable")
        .drop(columns=["__base_row_order"])
        .reset_index(drop=True)
    )
    slug = metric_slug(metric)
    raw_col = f"relief_{slug}_mean_per_won"
    rate_col = f"relief_{slug}_positive_date_rate"
    feature_col = f"relief_{slug}_{transform}"
    missing = [c for c in [raw_col, rate_col, feature_col] if c not in merged.columns]
    if missing:
        raise KeyError(f"Missing ReliefPotential columns after merge: {missing}")
    if merged[raw_col].isna().any():
        missing_count = int(merged[raw_col].isna().sum())
        raise RuntimeError(f"ReliefPotential merge left {missing_count:,} unmatched edge rows")

    merged["relief_potential_metric"] = metric
    merged["relief_potential_transform"] = transform
    merged["relief_potential_raw_mean_per_won"] = pd.to_numeric(
        merged[raw_col], errors="coerce"
    ).fillna(0.0)
    merged["relief_potential_positive_date_rate"] = pd.to_numeric(
        merged[rate_col], errors="coerce"
    ).fillna(0.0)
    merged["relief_potential_feature"] = pd.to_numeric(
        merged[feature_col], errors="coerce"
    ).fillna(0.0)
    return merged


def _top_k_mask(
    scores: np.ndarray,
    eligible_indices: np.ndarray,
    target_share: float,
    minimum_selected_edges: int,
) -> Tuple[np.ndarray, float]:
    selected = np.zeros(len(scores), dtype=bool)
    n = int(len(eligible_indices))
    if n == 0 or target_share <= 0:
        return selected, math.inf
    k = int(math.ceil(float(target_share) * n))
    k = max(int(minimum_selected_edges), k)
    k = min(n, k)
    if k == n:
        selected[eligible_indices] = True
        return selected, float(np.min(scores[eligible_indices]))
    local_scores = np.asarray(scores[eligible_indices], dtype=np.float64)
    split = n - k
    chosen_local = np.argpartition(local_scores, split)[split:]
    chosen = eligible_indices[chosen_local]
    selected[chosen] = True
    cutoff = float(np.min(local_scores[chosen_local]))
    return selected, cutoff


def select_edges(
    features: pd.DataFrame,
    selection_score: np.ndarray,
    eligible: np.ndarray,
    target_share: float,
    scope: str,
    minimum_selected_edges: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    target_share = float(np.clip(target_share, 0.0, 1.0))
    eligible = np.asarray(eligible, dtype=bool)
    selected = np.zeros(len(features), dtype=bool)
    cutoffs: Dict[str, float] = {}
    if scope == "global":
        indices = np.flatnonzero(eligible)
        selected, cutoff = _top_k_mask(
            selection_score,
            indices,
            target_share,
            minimum_selected_edges,
        )
        cutoffs["global"] = cutoff
    elif scope == "per_hour":
        hours = pd.to_numeric(features["hour"], errors="coerce").to_numpy()
        finite_hours = hours[np.isfinite(hours)]
        for hour in sorted(int(value) for value in np.unique(finite_hours)):
            indices = np.flatnonzero(eligible & (hours == hour))
            local, cutoff = _top_k_mask(
                selection_score,
                indices,
                target_share,
                minimum_selected_edges,
            )
            selected |= local
            cutoffs[str(hour)] = cutoff
    else:
        raise ValueError("selection_scope must be 'global' or 'per_hour'")

    eligible_count = int(np.count_nonzero(eligible))
    selected_count = int(np.count_nonzero(selected))
    return selected, {
        "selection_scope": scope,
        "target_share": target_share,
        "eligible_edges": eligible_count,
        "selected_edges": selected_count,
        "realized_selected_share_of_eligible": (
            selected_count / eligible_count if eligible_count else 0.0
        ),
        "selection_cutoff_by_scope": cutoffs,
    }


def score_relief_selective_features(
    features: pd.DataFrame,
    theta: ReliefSelectiveTheta,
    config: Mapping[str, Any],
    selection_scope_override: str = "",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = relief_selective_config(config)
    scope = selection_scope_override or str(cfg["selection_scope"]).strip().lower()
    minimum_selected_edges = max(1, int(cfg["minimum_selected_edges"]))

    scored = features.copy()
    low = pd.to_numeric(
        scored["low_crowding_train_mean"], errors="coerce"
    ).fillna(0.0).to_numpy(float)
    alt = pd.to_numeric(
        scored["altprob_train_mean"], errors="coerce"
    ).fillna(0.0).to_numpy(float)
    relief = pd.to_numeric(
        scored["relief_potential_feature"], errors="coerce"
    ).fillna(0.0).to_numpy(float)
    relief_raw = pd.to_numeric(
        scored["relief_potential_raw_mean_per_won"], errors="coerce"
    ).fillna(0.0).to_numpy(float)
    positive_rate = pd.to_numeric(
        scored["relief_potential_positive_date_rate"], errors="coerce"
    ).fillna(0.0).to_numpy(float)

    selection_score = (
        theta.low_crowding * low
        + theta.altprob * alt
        + theta.relief_potential * relief
    )
    policy_score = theta.intercept + selection_score

    minimum_altprob = float(config.get("policy", {}).get("minimum_altprob", 0.0))
    eligible = alt > minimum_altprob
    if bool(cfg.get("require_positive_relief", True)):
        eligible &= relief_raw > 0.0
    minimum_positive_rate = float(cfg.get("minimum_positive_date_rate", 0.0))
    if minimum_positive_rate > 0:
        eligible &= positive_rate >= minimum_positive_rate

    selected, selection_meta = select_edges(
        scored,
        selection_score,
        eligible,
        theta.target_share,
        scope,
        minimum_selected_edges,
    )

    maximum = float(config.get("policy", {}).get("max_edge_reward_won", 500.0))
    reward = np.clip(maximum * expit(policy_score), 0.0, maximum)
    reward = np.where(selected, reward, 0.0)

    rounding = float(cfg.get("reward_rounding_won", 0.0) or 0.0)
    if rounding > 0:
        reward = np.round(reward / rounding) * rounding
        reward = np.clip(reward, 0.0, maximum)

    scored["selection_score"] = selection_score
    scored["policy_score"] = policy_score
    scored["eligible"] = eligible.astype(np.int8)
    scored["selected_for_bounty"] = selected.astype(np.int8)
    scored["reward_won"] = reward

    positive = reward > 0.0
    selected_relief_raw = relief_raw[selected]
    selection_meta.update(
        {
            "positive_reward_edges": int(np.count_nonzero(positive)),
            "mean_positive_reward_won": float(np.mean(reward[positive]))
            if np.any(positive)
            else 0.0,
            "median_positive_reward_won": float(np.median(reward[positive]))
            if np.any(positive)
            else 0.0,
            "p90_positive_reward_won": float(np.quantile(reward[positive], 0.90))
            if np.any(positive)
            else 0.0,
            "max_reward_won": float(np.max(reward[positive]))
            if np.any(positive)
            else 0.0,
            "mean_selected_relief_raw_per_won": float(np.mean(selected_relief_raw))
            if len(selected_relief_raw)
            else 0.0,
            "min_selected_relief_raw_per_won": float(np.min(selected_relief_raw))
            if len(selected_relief_raw)
            else 0.0,
            "negative_relief_selected_edges": int(
                np.count_nonzero(selected_relief_raw <= 0.0)
            ),
            "reward_rounding_won": rounding,
        }
    )
    return scored, selection_meta


def _range(
    ranges: Mapping[str, Any],
    name: str,
    default: Tuple[float, float],
) -> Tuple[float, float]:
    values = ranges.get(name, list(default))
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"policy.relief_selective.bayesian.ranges.{name} must be [min,max]")
    low, high = float(values[0]), float(values[1])
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError(f"Invalid range for {name}: {values}")
    return low, high


def policy_search_signature(
    base_module: Any,
    daily_meta: Mapping[str, Any],
    relief_meta: Mapping[str, Any],
    features: pd.DataFrame,
    parameters: Any,
    train_dates: Sequence[str],
    config: Mapping[str, Any],
    selection_scope_override: str,
    metric: str,
    transform: str,
) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "policy_form_version": POLICY_FORM_VERSION,
        "base_policy_simulation_version": str(
            getattr(base_module, "POLICY_SIMULATION_VERSION", "")
        ),
        "daily_cache_signature": daily_meta.get("signature"),
        "relief_cache_signature": relief_meta.get("signature"),
        "parameters": asdict(parameters),
        "train_dates": list(train_dates),
        "budget": config.get("budget_per_day"),
        "policy": config.get("policy", {}),
        "selection_scope_override": selection_scope_override,
        "relief_metric": metric,
        "relief_transform": transform,
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
        "relief_potential_raw_mean_per_won",
        "relief_potential_feature",
    ]
    digest.update(
        pd.util.hash_pandas_object(features[columns], index=False).to_numpy().tobytes()
    )
    return digest.hexdigest()[:20]


def warm_start_points(
    bounds: Mapping[str, Tuple[float, float]],
    bayes_cfg: Mapping[str, Any],
    selective_policy_output: Optional[Path],
) -> list[Dict[str, float]]:
    points = list(
        bayes_cfg.get(
            "enqueue",
            [
                {
                    "intercept": -12.0,
                    "low_crowding": 2.0,
                    "altprob": 1.0,
                    "relief_potential": 12.0,
                    "target_share": 0.01,
                },
                {
                    "intercept": -14.0,
                    "low_crowding": 1.0,
                    "altprob": 1.0,
                    "relief_potential": 16.0,
                    "target_share": 0.05,
                },
                {
                    "intercept": -10.0,
                    "low_crowding": 4.0,
                    "altprob": 2.0,
                    "relief_potential": 8.0,
                    "target_share": 0.005,
                },
            ],
        )
    )
    if selective_policy_output is not None:
        learned_path = selective_policy_output / "learned_bounty_function.json"
        if learned_path.exists():
            learned = load_json(learned_path)
            theta = learned.get("theta") or {}
            needed = ["intercept", "low_crowding", "altprob", "target_share"]
            if all(name in theta for name in needed):
                points.insert(
                    0,
                    {
                        "intercept": float(theta["intercept"]),
                        "low_crowding": float(theta["low_crowding"]),
                        "altprob": float(theta["altprob"]),
                        "relief_potential": 0.0,
                        "target_share": float(theta["target_share"]),
                    },
                )

    output = []
    for point in points:
        candidate = {name: float(point[name]) for name in bounds if name in point}
        if len(candidate) != len(bounds):
            continue
        if all(bounds[name][0] <= value <= bounds[name][1] for name, value in candidate.items()):
            output.append(candidate)
    return output


def fit_relief_selective_policy(
    base_module: Any,
    features: pd.DataFrame,
    baseline_hours: Mapping[int, pd.DataFrame],
    simulator: Any,
    train_dates: Sequence[str],
    parameters: Any,
    config: MutableMapping[str, Any],
    output: Path,
    daily_meta: Mapping[str, Any],
    relief_meta: Mapping[str, Any],
    override_trials: int,
    selection_scope_override: str,
    metric: str,
    transform: str,
    selective_policy_output: Optional[Path],
) -> Tuple[
    Optional[ReliefSelectiveTheta],
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
    bool,
]:
    cfg = relief_selective_config(config)
    bayes_cfg = dict(cfg.get("bayesian", {}))
    ranges = dict(bayes_cfg.get("ranges", {}))
    bounds = {
        "intercept": _range(ranges, "intercept", (-24.0, 1.0)),
        "low_crowding": _range(ranges, "low_crowding", (0.0, 12.0)),
        "altprob": _range(ranges, "altprob", (0.0, 20.0)),
        "relief_potential": _range(ranges, "relief_potential", (0.0, 25.0)),
        "target_share": _range(ranges, "target_share", (0.0001, 1.0)),
    }
    target_share_log = bool(cfg.get("target_share_log", True))
    if target_share_log and bounds["target_share"][0] <= 0:
        raise ValueError("target_share lower bound must be > 0 when target_share_log=true")
    if bounds["target_share"][1] > 1.0:
        raise ValueError("target_share upper bound cannot exceed 1.0")

    signature = policy_search_signature(
        base_module,
        daily_meta,
        relief_meta,
        features,
        parameters,
        train_dates,
        config,
        selection_scope_override,
        metric,
        transform,
    )
    study_name = f"relief_selective_edge_bounty_{signature}"
    storage_path = output / "relief_selective_bounty_study.sqlite3"
    sampler = optuna.samplers.TPESampler(
        seed=int(bayes_cfg.get("seed", 42)),
        n_startup_trials=int(bayes_cfg.get("n_startup_trials", 40)),
        multivariate=bool(bayes_cfg.get("multivariate", True)),
        group=bool(bayes_cfg.get("group", True)),
    )
    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///" + storage_path.as_posix(),
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    if len(study.trials) == 0:
        for point in warm_start_points(bounds, bayes_cfg, selective_policy_output):
            study.enqueue_trial(point)

    target = int(override_trials or bayes_cfg.get("n_trials", 300))
    completed_before = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining = max(0, target - completed_before)
    print(
        f"[Relief-selective bounty] completed {completed_before}/{target}; "
        f"running {remaining}",
        flush=True,
    )
    started = time.perf_counter()
    completed_this_run = 0
    stop_on_infeasible = bool(
        config.get("policy", {}).get("early_stop_infeasible_trials", True)
    )

    def objective(trial: optuna.Trial) -> float:
        nonlocal completed_this_run
        theta = ReliefSelectiveTheta(
            intercept=trial.suggest_float("intercept", *bounds["intercept"]),
            low_crowding=trial.suggest_float(
                "low_crowding", *bounds["low_crowding"]
            ),
            altprob=trial.suggest_float("altprob", *bounds["altprob"]),
            relief_potential=trial.suggest_float(
                "relief_potential", *bounds["relief_potential"]
            ),
            target_share=trial.suggest_float(
                "target_share",
                *bounds["target_share"],
                log=target_share_log,
            ),
        )
        scored, selection_meta = score_relief_selective_features(
            features,
            theta,
            config,
            selection_scope_override,
        )
        rewards = base_module.rewards_by_hour(scored, baseline_hours)
        record, _ = simulator.evaluate(
            rewards,
            train_dates,
            stop_on_infeasible=stop_on_infeasible,
        )
        attrs: Dict[str, float] = {key: float(value) for key, value in record.items()}
        for key in [
            "eligible_edges",
            "selected_edges",
            "realized_selected_share_of_eligible",
            "positive_reward_edges",
            "mean_positive_reward_won",
            "median_positive_reward_won",
            "p90_positive_reward_won",
            "max_reward_won",
            "mean_selected_relief_raw_per_won",
            "min_selected_relief_raw_per_won",
            "negative_relief_selected_edges",
        ]:
            attrs[key] = float(selection_meta[key])
        for key, value in attrs.items():
            trial.set_user_attr(key, value)

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
            "[Relief-selective] {0}/{1}, elapsed {2}, ETA {3}: "
            "relief={4:,.3f}, max daily cost={5:,.0f}, feasible={6}, "
            "selected={7:,} ({8:.3%}), mean reward={9:.2f}, "
            "theta=({10:.4f},{11:.4f},{12:.4f},{13:.4f},q={14:.6f})".format(
                completed_total,
                target,
                format_seconds(elapsed),
                format_seconds(eta),
                record["objective_improvement"],
                record["max_daily_cost"],
                bool(record["feasible"] > 0.5),
                int(selection_meta["selected_edges"]),
                selection_meta["realized_selected_share_of_eligible"],
                selection_meta["mean_positive_reward_won"],
                theta.intercept,
                theta.low_crowding,
                theta.altprob,
                theta.relief_potential,
                theta.target_share,
            ),
            flush=True,
        )
        return value

    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True, n_jobs=1)

    completed = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    rows: list[Dict[str, Any]] = []
    for trial in completed:
        row: Dict[str, Any] = {
            "trial": int(trial.number),
            "policy_type": "RELIEF_SELECTIVE_SIGMOID_BOUNTY",
            "intercept": float(trial.params["intercept"]),
            "low_crowding": float(trial.params["low_crowding"]),
            "altprob": float(trial.params["altprob"]),
            "relief_potential": float(trial.params["relief_potential"]),
            "target_share": float(trial.params["target_share"]),
            "study_objective": float(trial.value)
            if trial.value is not None
            else np.nan,
        }
        row.update(trial.user_attrs)
        rows.append(row)

    zero_rewards = {
        hour: np.zeros(len(edges), dtype=np.float64)
        for hour, edges in baseline_hours.items()
    }
    null_record, _ = simulator.evaluate(
        zero_rewards,
        train_dates,
        stop_on_infeasible=False,
    )
    null_row: Dict[str, Any] = {
        "trial": -1,
        "policy_type": "NO_BOUNTY",
        "intercept": np.nan,
        "low_crowding": np.nan,
        "altprob": np.nan,
        "relief_potential": np.nan,
        "target_share": 0.0,
        "study_objective": float(null_record["objective_improvement"]),
        **null_record,
        "eligible_edges": 0.0,
        "selected_edges": 0.0,
        "realized_selected_share_of_eligible": 0.0,
        "positive_reward_edges": 0.0,
        "mean_positive_reward_won": 0.0,
        "median_positive_reward_won": 0.0,
        "p90_positive_reward_won": 0.0,
        "max_reward_won": 0.0,
        "mean_selected_relief_raw_per_won": 0.0,
        "min_selected_relief_raw_per_won": 0.0,
        "negative_relief_selected_edges": 0.0,
    }
    if bool(config.get("policy", {}).get("allow_no_bounty", True)):
        rows.append(null_row)
    if not rows:
        raise RuntimeError("No completed relief-selective policy trial")

    trials = pd.DataFrame(rows).sort_values(
        ["feasible", "objective_improvement", "max_daily_cost"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    feasible = trials[trials["feasible"] > 0.5]
    if feasible.empty:
        raise RuntimeError("No budget-feasible relief-selective bounty policy")

    minimum_improvement = float(
        config.get("policy", {}).get("minimum_training_improvement", 0.0)
    )
    best = feasible.iloc[0]
    policy_active = bool(
        str(best["policy_type"]) == "RELIEF_SELECTIVE_SIGMOID_BOUNTY"
        and float(best["objective_improvement"]) > minimum_improvement
    )
    if policy_active:
        theta: Optional[ReliefSelectiveTheta] = ReliefSelectiveTheta(
            intercept=float(best["intercept"]),
            low_crowding=float(best["low_crowding"]),
            altprob=float(best["altprob"]),
            relief_potential=float(best["relief_potential"]),
            target_share=float(best["target_share"]),
        )
        scored, best_selection_meta = score_relief_selective_features(
            features,
            theta,
            config,
            selection_scope_override,
        )
    else:
        theta = None
        scored = features.copy()
        scored["selection_score"] = 0.0
        scored["policy_score"] = -np.inf
        scored["eligible"] = 0
        scored["selected_for_bounty"] = 0
        scored["reward_won"] = 0.0
        best_selection_meta = {
            "selection_scope": selection_scope_override
            or relief_selective_config(config)["selection_scope"],
            "target_share": 0.0,
            "eligible_edges": 0,
            "selected_edges": 0,
            "realized_selected_share_of_eligible": 0.0,
            "positive_reward_edges": 0,
            "mean_positive_reward_won": 0.0,
            "median_positive_reward_won": 0.0,
            "p90_positive_reward_won": 0.0,
            "max_reward_won": 0.0,
            "mean_selected_relief_raw_per_won": 0.0,
            "min_selected_relief_raw_per_won": 0.0,
            "negative_relief_selected_edges": 0,
        }

    metadata = {
        "method": (
            "Optuna TPE selective top-share sigmoid bounty with analytical "
            "MNL ReliefPotential and explicit no-bounty alternative"
        ),
        "script_version": SCRIPT_VERSION,
        "policy_form_version": POLICY_FORM_VERSION,
        "study_name": study_name,
        "storage": str(storage_path),
        "completed_trials": len(completed),
        "target_trials": target,
        "selected_policy": "RELIEF_SELECTIVE_SIGMOID_BOUNTY"
        if policy_active
        else "NO_BOUNTY",
        "policy_active": policy_active,
        "selected_trial": int(best["trial"]),
        "best_objective_improvement": float(best["objective_improvement"]),
        "best_max_daily_cost": float(best["max_daily_cost"]),
        "best_efficiency_per_1000": float(best["efficiency_per_1000"]),
        "best_selection": best_selection_meta,
        "ranges": {name: list(value) for name, value in bounds.items()},
        "target_share_log": target_share_log,
        "relief_metric": metric,
        "relief_metric_formula": metric_label(metric),
        "relief_transform": transform,
        "objective": "maximize exact total whole-network congestion relief on training dates",
        "constraints": {
            "budget_per_day": config.get("budget_per_day", 3_000_000.0),
            "capacity_constraint": None,
            "payment_rule": (
                "all post-policy users of each rewarded edge receive the edge bounty"
            ),
        },
        "deployment_dimension": "hour x route x directed edge",
        "date_specific_rewards": False,
        "feature_aggregation": "training dates only",
    }
    return theta, scored, trials, metadata, policy_active


def summarize_policy_daily(
    path: Path,
    policy_name: str,
    budget_per_day: float,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    policy = frame[frame["scenario"] == "PUBLIC_EDGE_BOUNTY"].copy()
    if policy.empty:
        raise ValueError(f"No PUBLIC_EDGE_BOUNTY rows in {path}")
    rows = []
    for split, group in policy.groupby("split", sort=False):
        h0 = float(group["h0"].sum())
        improvement = float(group["improvement"].sum())
        cost = float(group["cost"].sum())
        rows.append(
            {
                "policy": policy_name,
                "split": str(split),
                "dates": int(group["date"].nunique()),
                "H0": h0,
                "H1": float(group["h1"].sum()),
                "improvement": improvement,
                "improvement_pct": 100.0 * improvement / max(h0, EPS),
                "total_cost": cost,
                "mean_daily_cost": float(group["cost"].mean()),
                "max_daily_cost": float(group["cost"].max()),
                "mean_budget_usage_pct": float(group["cost"].mean())
                / max(float(budget_per_day), EPS)
                * 100.0,
                "all_dates_feasible": bool(group["feasible"].all()),
                "efficiency_per_1000": improvement / cost * 1000.0
                if cost > 0
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_reward_diagnostics(scored: pd.DataFrame, output: Path) -> None:
    rows = []
    for scope, group in [("all_hours", scored)] + [
        (f"hour_{int(hour):02d}", group)
        for hour, group in scored.groupby("hour", sort=True)
    ]:
        local = group[group["reward_won"] > 0]
        rows.append(
            {
                "scope": scope,
                "edges": int(len(group)),
                "eligible_edges": int(group["eligible"].sum()),
                "selected_edges": int(group["selected_for_bounty"].sum()),
                "positive_reward_edges": int(len(local)),
                "mean_positive_reward_won": float(local["reward_won"].mean())
                if len(local)
                else 0.0,
                "median_positive_reward_won": float(local["reward_won"].median())
                if len(local)
                else 0.0,
                "p90_positive_reward_won": float(local["reward_won"].quantile(0.90))
                if len(local)
                else 0.0,
                "p95_positive_reward_won": float(local["reward_won"].quantile(0.95))
                if len(local)
                else 0.0,
                "p99_positive_reward_won": float(local["reward_won"].quantile(0.99))
                if len(local)
                else 0.0,
                "max_reward_won": float(local["reward_won"].max())
                if len(local)
                else 0.0,
                "mean_selected_relief_raw_per_won": float(
                    local["relief_potential_raw_mean_per_won"].mean()
                )
                if len(local)
                else 0.0,
                "min_selected_relief_raw_per_won": float(
                    local["relief_potential_raw_mean_per_won"].min()
                )
                if len(local)
                else 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(
        output / "reward_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_relief_distribution(features: pd.DataFrame, output: Path) -> None:
    raw = pd.to_numeric(
        features["relief_potential_raw_mean_per_won"], errors="coerce"
    ).fillna(0.0)
    feature = pd.to_numeric(
        features["relief_potential_feature"], errors="coerce"
    ).fillna(0.0)
    rows = []
    for scope, group in [("all_hours", features)] + [
        (f"hour_{int(hour):02d}", group)
        for hour, group in features.groupby("hour", sort=True)
    ]:
        values = pd.to_numeric(
            group["relief_potential_raw_mean_per_won"], errors="coerce"
        ).fillna(0.0)
        transformed = pd.to_numeric(
            group["relief_potential_feature"], errors="coerce"
        ).fillna(0.0)
        positive = values[values > 0]
        rows.append(
            {
                "scope": scope,
                "edges": int(len(group)),
                "positive_edges": int((values > 0).sum()),
                "positive_share": float((values > 0).mean()) if len(values) else 0.0,
                "raw_mean": float(values.mean()) if len(values) else 0.0,
                "raw_min": float(values.min()) if len(values) else 0.0,
                "raw_max": float(values.max()) if len(values) else 0.0,
                "positive_raw_p50": float(positive.quantile(0.50))
                if len(positive)
                else 0.0,
                "positive_raw_p90": float(positive.quantile(0.90))
                if len(positive)
                else 0.0,
                "positive_raw_p99": float(positive.quantile(0.99))
                if len(positive)
                else 0.0,
                "feature_mean": float(transformed.mean()) if len(transformed) else 0.0,
                "feature_max": float(transformed.max()) if len(transformed) else 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(
        output / "relief_potential_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )


def finite_difference_metric_relief(
    simulator: Any,
    baseline_hours: Mapping[int, pd.DataFrame],
    train_dates: Sequence[str],
    rewards: Mapping[int, np.ndarray],
    metric: str,
    reference_load: float,
) -> float:
    path_discounts = simulator.path_discounts(rewards)
    total_improvement = 0.0
    for date in train_dates:
        _, details = simulator.simulate_day(
            str(date),
            rewards,
            detailed=True,
            path_discount_by_hour=path_discounts,
        )
        for hour in sorted(baseline_hours):
            state = simulator.state(str(date), int(hour))
            load0 = np.asarray(state.load0, dtype=np.float64)
            load1 = np.asarray(details[hour]["load1"], dtype=np.float64)
            travel = np.asarray(state.travel_time, dtype=np.float64)
            trips = np.asarray(state.trips, dtype=np.float64)
            total_improvement += metric_value(
                metric,
                travel,
                trips,
                load0,
                reference_load,
            ) - metric_value(
                metric,
                travel,
                trips,
                load1,
                reference_load,
            )
    return float(total_improvement)


def run_gradient_checks(
    base_module: Any,
    features: pd.DataFrame,
    baseline_hours: Mapping[int, pd.DataFrame],
    simulator: Any,
    train_dates: Sequence[str],
    metric: str,
    reference_load: float,
    n_edges: int,
    reward_won: float,
    output: Path,
) -> pd.DataFrame:
    if n_edges <= 0:
        return pd.DataFrame()

    positive = features[
        features["relief_potential_raw_mean_per_won"] > 0
    ].copy()
    if positive.empty:
        print("[gradient-check] skipped: no positive ReliefPotential edges", flush=True)
        return pd.DataFrame()

    positive = positive.sort_values(
        "relief_potential_raw_mean_per_won",
        ascending=False,
        kind="stable",
    )
    top_n = min(max(1, n_edges // 2), len(positive))
    top = positive.head(top_n)
    remaining = max(0, n_edges - len(top))
    if remaining > 0 and len(positive) > top_n:
        sample = positive.iloc[top_n:].sample(
            n=min(remaining, len(positive) - top_n),
            random_state=42,
        )
        selected = pd.concat([top, sample], ignore_index=True)
    else:
        selected = top.reset_index(drop=True)

    lookup: Dict[int, Dict[Tuple[str, str, str], int]] = {}
    for hour, edges in baseline_hours.items():
        lookup[int(hour)] = {
            (
                normalize_id(row.route_id),
                normalize_id(row.from_stop_id),
                normalize_id(row.to_stop_id),
            ): int(row.edge_index)
            for row in edges.itertuples(index=False)
        }

    rows = []
    meter = Progress("gradient-check", len(selected), 1.0)
    for i, row in enumerate(selected.itertuples(index=False), start=1):
        hour = int(row.hour)
        key = (
            normalize_id(row.route_id),
            normalize_id(row.from_stop_id),
            normalize_id(row.to_stop_id),
        )
        edge_index = lookup[hour][key]
        rewards = {
            h: np.zeros(len(edges), dtype=np.float64)
            for h, edges in baseline_hours.items()
        }
        rewards[hour][edge_index] = float(reward_won)
        actual_improvement = finite_difference_metric_relief(
            simulator,
            baseline_hours,
            train_dates,
            rewards,
            metric,
            reference_load,
        )
        finite_difference = actual_improvement / float(reward_won)
        analytical_sum = float(row.relief_potential_raw_mean_per_won) * len(train_dates)
        absolute_error = finite_difference - analytical_sum
        relative_error = absolute_error / max(abs(analytical_sum), 1.0e-9)
        rows.append(
            {
                "hour": hour,
                "route_id": key[0],
                "from_stop_id": key[1],
                "to_stop_id": key[2],
                "reward_won": float(reward_won),
                "metric": metric,
                "analytical_sum_relief_per_won": analytical_sum,
                "finite_difference_relief_per_won": finite_difference,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
            }
        )
        meter.update(i, extra=f"relative error {relative_error:.3%}")
    meter.update(len(selected), force=True)
    result = pd.DataFrame(rows)
    result.to_csv(
        output / "relief_potential_gradient_check.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()

    stage2_script = Path(args.stage2_script).resolve()
    model_input = Path(args.model_input).resolve()
    route_choice_output = Path(args.route_choice_output).resolve()
    cache_source = Path(args.cache_source).resolve()
    distributed_policy_output = (
        Path(args.distributed_policy_output).resolve()
        if args.distributed_policy_output
        else None
    )
    selective_policy_output = (
        Path(args.selective_policy_output).resolve()
        if args.selective_policy_output
        else None
    )
    output = Path(args.output).resolve()
    config_path = Path(args.config).resolve()

    if args.fresh and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    topology_root = cache_source / "_topology_cache"
    daily_root = cache_source / "_daily_cache"
    require_paths(
        [
            stage2_script,
            model_input,
            route_choice_output / "route_choice_parameters.json",
            config_path,
            topology_root / "topology_meta.json",
            daily_root / "daily_meta.json",
            cache_source / "operational_hour_edge_features.csv.gz",
        ]
    )

    base = import_module_from_path(
        stage2_script,
        "relief_selective_base_stage2",
    )
    simulation_version = str(getattr(base, "POLICY_SIMULATION_VERSION", ""))
    if "no-capacity" not in simulation_version:
        raise RuntimeError(
            "The stage-2 script is not the no-capacity-constraint version."
        )

    config: MutableMapping[str, Any] = load_json(config_path)
    settings = resolve_relief_settings(
        config,
        args.relief_metric,
        args.relief_transform,
        args.gradient_check_edges,
        args.gradient_check_reward_won,
    )
    metric = str(settings["metric"])
    transform = str(settings["transform"])

    parameters = base.load_choice_parameters(
        route_choice_output / "route_choice_parameters.json"
    )
    dates = base.parse_dates(model_input)
    train_dates, test_dates = base.split_dates(dates, config)
    hours = {int(value) for value in config.get("hours", [7, 8, 9, 17, 18, 19])}
    baseline = base.load_baseline_segments(model_input, hours)
    baseline_hours = base.baseline_by_hour(baseline)
    topology_meta = load_json(topology_root / "topology_meta.json")
    daily_meta = load_json(daily_root / "daily_meta.json")
    reference_load = float(base.load_reference_value(config))

    print("analysis dates:", dates, flush=True)
    print("training dates:", train_dates, flush=True)
    print("evaluation dates:", test_dates, flush=True)
    print("hours:", sorted(hours), flush=True)
    print("cache source:", cache_source, flush=True)
    print("ReliefPotential metric:", metric, flush=True)
    print("ReliefPotential transform:", transform, flush=True)

    topologies = {
        hour: base.load_hour_topology(
            topology_root,
            hour,
            baseline_hours[hour],
            include_board=False,
        )
        for hour in sorted(baseline_hours)
    }
    simulator = base.PolicySimulator(
        topologies,
        daily_root,
        dates,
        parameters,
        config,
    )

    metrics = list(SUPPORTED_METRICS) if settings["compute_all_metrics"] else [metric]
    relief_cache_root = output / "_relief_potential_cache"
    print("[stage] analytical ReliefPotential cache", flush=True)
    relief_frame, relief_meta = prepare_relief_potential_cache(
        base,
        baseline_hours,
        topologies,
        simulator,
        topology_meta,
        daily_meta,
        parameters,
        train_dates,
        reference_load,
        metrics,
        relief_cache_root,
        args.rebuild_relief_cache,
        progress_seconds=float(args.progress_seconds),
    )

    base_features = pd.read_csv(
        cache_source / "operational_hour_edge_features.csv.gz",
        compression="gzip",
        low_memory=False,
    )
    features = merge_relief_features(
        base_features,
        relief_frame,
        metric,
        transform,
    )
    features.to_csv(
        output / "operational_hour_edge_features_with_relief.csv.gz",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )
    write_relief_distribution(features, output)

    gradient_check = run_gradient_checks(
        base,
        features,
        baseline_hours,
        simulator,
        train_dates,
        metric,
        reference_load,
        int(settings["gradient_check_edges"]),
        float(settings["gradient_check_reward_won"]),
        output,
    )

    if args.prepare_only:
        prepare_manifest = {
            "created_at": pd.Timestamp.now().isoformat(),
            "script_version": SCRIPT_VERSION,
            "stage": "relief_potential_preparation_only",
            "metric": metric,
            "metric_formula": metric_label(metric),
            "transform": transform,
            "train_dates": train_dates,
            "test_dates_not_used_for_relief_features": test_dates,
            "relief_cache": relief_meta,
            "gradient_check_rows": int(len(gradient_check)),
            "next_step": "rerun without --prepare-only and without --fresh",
        }
        write_json(output / "prepare_only_manifest.json", prepare_manifest)
        print("\nReliefPotential preparation completed:", output, flush=True)
        print(" -", output / "operational_hour_edge_features_with_relief.csv.gz")
        print(" -", output / "relief_potential_distribution.csv")
        if len(gradient_check):
            print(" -", output / "relief_potential_gradient_check.csv")
        print("total elapsed:", format_seconds(time.perf_counter() - run_started))
        return

    print("[stage] ReliefPotential selective bounty fit", flush=True)
    theta, scored, trials, search_meta, policy_active = fit_relief_selective_policy(
        base,
        features,
        baseline_hours,
        simulator,
        train_dates,
        parameters,
        config,
        output,
        daily_meta,
        relief_meta,
        args.policy_trials,
        args.selection_scope,
        metric,
        transform,
        selective_policy_output,
    )
    trials.to_csv(
        output / "policy_bayes_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scored.to_csv(
        output / "operational_hour_edge_rewards.csv.gz",
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )
    write_reward_diagnostics(scored, output)

    if theta is None:
        print("selected policy: NO_BOUNTY", flush=True)
    else:
        print("learned ReliefPotential theta:", asdict(theta), flush=True)

    print("[stage] fixed-policy train/test evaluation", flush=True)
    daily_results, edge_rows = base.write_final_results(
        output,
        None,
        scored,
        baseline_hours,
        simulator,
        train_dates,
        test_dates,
    )

    policy_rows = daily_results[
        daily_results["scenario"] == "PUBLIC_EDGE_BOUNTY"
    ].copy()
    load_qa = {
        "reference_load": reference_load,
        "capacity_constraint_enforced": False,
        "max_policy_avg_onboard": float(policy_rows["max_policy_avg_onboard"].max())
        if len(policy_rows)
        else 0.0,
        "total_overloaded_date_hour_edges_above_reference": int(
            policy_rows["overloaded_edges_above_reference"].sum()
        )
        if len(policy_rows)
        else 0,
        "total_new_overloaded_date_hour_edges_above_reference": int(
            policy_rows["new_overloaded_edges_above_reference"].sum()
        )
        if len(policy_rows)
        else 0,
        "total_increased_existing_overload_edges": int(
            policy_rows["increased_existing_overload_edges"].sum()
        )
        if len(policy_rows)
        else 0,
        "max_load_increase": float(policy_rows["max_load_increase"].max())
        if len(policy_rows)
        else 0.0,
        "note": (
            "These load values are diagnostics only and are not feasibility constraints."
        ),
    }
    write_json(output / "policy_load_qa.json", load_qa)

    selected_count = int(scored["selected_for_bounty"].sum())
    eligible_count = int(scored["eligible"].sum())
    learned = {
        "created_at": pd.Timestamp.now().isoformat(),
        "script_version": SCRIPT_VERSION,
        "policy_form_version": POLICY_FORM_VERSION,
        "policy_active": policy_active,
        "selected_policy": "RELIEF_SELECTIVE_SIGMOID_BOUNTY"
        if policy_active
        else "NO_BOUNTY",
        "theta": asdict(theta) if theta is not None else None,
        "selection_formula": (
            "select top target_share among eligible hour-route-directed-edge rows by "
            "low_crowding*low_crowding_train_mean + "
            "altprob*altprob_train_mean + "
            "relief_potential*relief_potential_feature"
        ),
        "reward_formula": (
            "reward_won = selected * max_edge_reward_won * sigmoid(intercept + "
            "selection_score)"
        ),
        "relief_potential": {
            "metric": metric,
            "metric_formula": metric_label(metric),
            "transform": transform,
            "raw_definition": (
                "analytical local derivative of training-date congestion relief "
                "with respect to one won of edge reward"
            ),
            "uses_test_dates": False,
            "cache_signature": relief_meta.get("signature"),
        },
        "selection_scope": args.selection_scope
        or relief_selective_config(config)["selection_scope"],
        "eligible_edges": eligible_count,
        "selected_edges": selected_count,
        "realized_selected_share_of_eligible": selected_count
        / max(eligible_count, 1),
        "positive_reward_edges": int(
            np.count_nonzero(scored["reward_won"].to_numpy(float) > 0)
        ),
        "payment_rule": (
            "all passengers traversing each rewarded edge after policy receive "
            "that edge bounty"
        ),
        "budget_per_day": config.get("budget_per_day", 3_000_000.0),
        "capacity_constraint_enforced": False,
        "cache_source": str(cache_source),
    }
    write_json(output / "learned_bounty_function.json", learned)
    write_json(output / "policy_search_metadata.json", search_meta)

    comparison_frames = [
        summarize_policy_daily(
            output / "daily_policy_results.csv",
            "relief_selective_bounty",
            float(config.get("budget_per_day", 3_000_000.0)),
        )
    ]
    for name, prior in [
        ("distributed_bounty", distributed_policy_output),
        ("selective_bounty", selective_policy_output),
    ]:
        if prior is not None and (prior / "daily_policy_results.csv").exists():
            comparison_frames.insert(
                0,
                summarize_policy_daily(
                    prior / "daily_policy_results.csv",
                    name,
                    float(config.get("budget_per_day", 3_000_000.0)),
                ),
            )
    comparison = pd.concat(comparison_frames, ignore_index=True)
    comparison.to_csv(
        output / "policy_comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "created_at": pd.Timestamp.now().isoformat(),
        "script_version": SCRIPT_VERSION,
        "stage2_script": str(stage2_script),
        "base_policy_simulation_version": simulation_version,
        "model_input": str(model_input),
        "route_choice_output": str(route_choice_output),
        "cache_source": str(cache_source),
        "distributed_policy_output": str(distributed_policy_output)
        if distributed_policy_output is not None
        else None,
        "selective_policy_output": str(selective_policy_output)
        if selective_policy_output is not None
        else None,
        "output": str(output),
        "config": str(config_path),
        "dates": dates,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "hours": sorted(hours),
        "relief_settings": settings,
        "relief_cache": relief_meta,
        "gradient_check_rows": int(len(gradient_check)),
        "learned_policy": learned,
        "policy_search": search_meta,
        "output_rows": {
            "operational_rewards": int(len(scored)),
            "daily_policy_results": int(len(daily_results)),
            "edge_rewards": int(edge_rows),
            "trials": int(len(trials)),
        },
        "model_notes": [
            "ReliefPotential is computed from training dates only.",
            "The analytical derivative is evaluated at the no-bounty MNL baseline.",
            "The final Optuna objective uses the full nonlinear policy simulation.",
            "Payment rule A is retained: all users of a rewarded edge are paid.",
            "No hard capacity constraint is imposed; load effects are reported as QA.",
        ],
    }
    write_json(output / "relief_selective_bounty_manifest.json", manifest)

    print("\ncompleted:", output, flush=True)
    for name in [
        "learned_bounty_function.json",
        "operational_hour_edge_features_with_relief.csv.gz",
        "operational_hour_edge_rewards.csv.gz",
        "relief_potential_distribution.csv",
        "relief_potential_gradient_check.csv",
        "policy_bayes_results.csv",
        "daily_policy_results.csv",
        "scenario_summary.csv",
        "edge_rewards.csv.gz",
        "reward_distribution.csv",
        "policy_comparison_summary.csv",
        "policy_load_qa.json",
        "policy_search_metadata.json",
        "relief_selective_bounty_manifest.json",
    ]:
        path = output / name
        if path.exists():
            print(" -", path, flush=True)
    print("total elapsed:", format_seconds(time.perf_counter() - run_started), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted. ReliefPotential cache and Optuna SQLite study are reusable; "
            "rerun without --fresh.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
