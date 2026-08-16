from __future__ import annotations

"""Fit a selective (sparse) public edge-bounty policy.

This experiment reuses the topology and daily-state caches produced by the
no-capacity version of ``fit_public_edge_bounty_policy.py``.  It does not
refit the route-choice model and does not rebuild the heavy caches.

The distributed policy pays every eligible edge a positive sigmoid reward.
The selective policy adds one learned parameter, ``target_share``:

    selection_score_e = theta_c * low_crowding_e + theta_a * AltProb_e
    selected_e = 1 if edge e is in the top target_share of eligible edges
    reward_score_e = theta_0 + selection_score_e
    reward_e = selected_e * Rmax * sigmoid(reward_score_e)

``target_share=1`` exactly nests the existing distributed sigmoid policy.
The payment rule remains rule A: every passenger traversing a rewarded edge
after the policy receives that edge's bounty.
"""

import argparse
import copy
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
SCRIPT_VERSION = "selective-edge-bounty-v1.0.0"
POLICY_FORM_VERSION = "top-share-selective-sigmoid-v1"

# PowerShell 5.1 can treat Optuna's harmless experimental warning as a native
# process error when stderr is piped through Tee-Object.
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class SelectiveTheta:
    intercept: float
    low_crowding: float
    altprob: float
    target_share: float


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
        description="Fit a sparse selective edge-bounty policy using existing stage-2 caches"
    )
    parser.add_argument(
        "--stage2-script",
        default=r".\fit_public_edge_bounty_policy.py",
        help="no-capacity fit_public_edge_bounty_policy.py used to build the caches",
    )
    parser.add_argument("--model-input", required=True)
    parser.add_argument("--route-choice-output", required=True)
    parser.add_argument(
        "--cache-source",
        required=True,
        help="existing distributed-policy output containing _topology_cache and _daily_cache",
    )
    parser.add_argument(
        "--baseline-policy-output",
        default="",
        help="optional distributed-policy output for automatic comparison",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy-trials", type=int, default=0)
    parser.add_argument(
        "--selection-scope",
        choices=["global", "per_hour"],
        default=None,
        help="override policy.selective.selection_scope",
    )
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def selective_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    policy = dict(config.get("policy", {}))
    value = dict(policy.get("selective", {}))
    value.setdefault("selection_scope", "global")
    value.setdefault("minimum_selected_edges", 1)
    value.setdefault("reward_rounding_won", 0.0)
    value.setdefault("target_share_log", True)
    value.setdefault("bayesian", {})
    return value


def _top_k_mask(
    scores: np.ndarray,
    eligible_indices: np.ndarray,
    target_share: float,
    minimum_selected_edges: int,
) -> Tuple[np.ndarray, float]:
    """Return a boolean mask selecting the highest-scoring eligible edges."""
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
            selection_score, indices, target_share, minimum_selected_edges
        )
        cutoffs["global"] = cutoff
    elif scope == "per_hour":
        hours = pd.to_numeric(features["hour"], errors="coerce").to_numpy()
        for hour in sorted(int(value) for value in np.unique(hours[np.isfinite(hours)])):
            indices = np.flatnonzero(eligible & (hours == hour))
            local_mask, cutoff = _top_k_mask(
                selection_score, indices, target_share, minimum_selected_edges
            )
            selected |= local_mask
            cutoffs[str(hour)] = cutoff
    else:
        raise ValueError("selection_scope must be 'global' or 'per_hour'")

    eligible_count = int(np.count_nonzero(eligible))
    selected_count = int(np.count_nonzero(selected))
    meta = {
        "selection_scope": scope,
        "target_share": target_share,
        "eligible_edges": eligible_count,
        "selected_edges": selected_count,
        "realized_selected_share_of_eligible": (
            selected_count / eligible_count if eligible_count else 0.0
        ),
        "selection_cutoff_by_scope": cutoffs,
    }
    return selected, meta


def score_selective_features(
    features: pd.DataFrame,
    theta: SelectiveTheta,
    config: Mapping[str, Any],
    selection_scope_override: str = "",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = selective_config(config)
    scope = selection_scope_override or str(cfg["selection_scope"]).strip().lower()
    minimum_selected_edges = max(1, int(cfg["minimum_selected_edges"]))

    scored = features.copy()
    low = pd.to_numeric(
        scored["low_crowding_train_mean"], errors="coerce"
    ).fillna(0.0).to_numpy(float)
    alt = pd.to_numeric(
        scored["altprob_train_mean"], errors="coerce"
    ).fillna(0.0).to_numpy(float)

    selection_score = theta.low_crowding * low + theta.altprob * alt
    policy_score = theta.intercept + selection_score
    minimum_altprob = float(config.get("policy", {}).get("minimum_altprob", 0.0))
    eligible = alt > minimum_altprob

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

    positive = reward > 0
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
            "reward_rounding_won": rounding,
        }
    )
    return scored, selection_meta


def policy_search_signature(
    base_module: Any,
    daily_meta: Mapping[str, Any],
    features: pd.DataFrame,
    parameters: Any,
    train_dates: Sequence[str],
    config: Mapping[str, Any],
    selection_scope_override: str,
) -> str:
    payload = {
        "script_version": SCRIPT_VERSION,
        "policy_form_version": POLICY_FORM_VERSION,
        "base_policy_simulation_version": str(
            getattr(base_module, "POLICY_SIMULATION_VERSION", "")
        ),
        "daily_cache_signature": daily_meta.get("signature"),
        "parameters": asdict(parameters),
        "train_dates": list(train_dates),
        "budget": config.get("budget_per_day"),
        "policy": config.get("policy", {}),
        "selection_scope_override": selection_scope_override,
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


def _range(
    ranges: Mapping[str, Any], name: str, default: Tuple[float, float]
) -> Tuple[float, float]:
    values = ranges.get(name, list(default))
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"policy.selective.bayesian.ranges.{name} must be [min,max]")
    low, high = float(values[0]), float(values[1])
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError(f"invalid range for {name}: {values}")
    return low, high


def fit_selective_policy(
    base_module: Any,
    features: pd.DataFrame,
    baseline_hours: Mapping[int, pd.DataFrame],
    simulator: Any,
    train_dates: Sequence[str],
    parameters: Any,
    config: MutableMapping[str, Any],
    output: Path,
    daily_meta: Mapping[str, Any],
    override_trials: int,
    selection_scope_override: str,
    baseline_policy_output: Optional[Path],
) -> Tuple[Optional[SelectiveTheta], pd.DataFrame, pd.DataFrame, Dict[str, Any], bool]:
    sel_cfg = selective_config(config)
    bayes_cfg = dict(sel_cfg.get("bayesian", {}))
    ranges = dict(bayes_cfg.get("ranges", {}))
    bounds = {
        "intercept": _range(ranges, "intercept", (-24.0, 1.0)),
        "low_crowding": _range(ranges, "low_crowding", (0.0, 12.0)),
        "altprob": _range(ranges, "altprob", (0.0, 20.0)),
        "target_share": _range(ranges, "target_share", (0.0001, 1.0)),
    }
    target_share_log = bool(sel_cfg.get("target_share_log", True))
    if target_share_log and bounds["target_share"][0] <= 0:
        raise ValueError("target_share lower bound must be > 0 when target_share_log=true")
    if bounds["target_share"][1] > 1.0:
        raise ValueError("target_share upper bound cannot exceed 1.0")

    signature = policy_search_signature(
        base_module,
        daily_meta,
        features,
        parameters,
        train_dates,
        config,
        selection_scope_override,
    )
    study_name = f"selective_edge_bounty_{signature}"
    storage_path = output / "selective_bounty_study.sqlite3"
    sampler = optuna.samplers.TPESampler(
        seed=int(bayes_cfg.get("seed", 42)),
        n_startup_trials=int(bayes_cfg.get("n_startup_trials", 32)),
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
        enqueue = list(
            bayes_cfg.get(
                "enqueue",
                [
                    {
                        "intercept": -14.0,
                        "low_crowding": 4.0,
                        "altprob": 8.0,
                        "target_share": 0.01,
                    },
                    {
                        "intercept": -12.0,
                        "low_crowding": 4.0,
                        "altprob": 8.0,
                        "target_share": 0.001,
                    },
                    {
                        "intercept": -18.0,
                        "low_crowding": 6.0,
                        "altprob": 10.0,
                        "target_share": 0.05,
                    },
                    {
                        "intercept": -20.0,
                        "low_crowding": 6.0,
                        "altprob": 10.0,
                        "target_share": 0.10,
                    },
                ],
            )
        )

        # Add the fitted distributed policy as a nested q=1 reference whenever it
        # is available and lies inside the configured ranges.
        if baseline_policy_output is not None:
            learned_path = baseline_policy_output / "learned_bounty_function.json"
            if learned_path.exists():
                learned = load_json(learned_path)
                theta = learned.get("theta") or {}
                if all(name in theta for name in ["intercept", "low_crowding", "altprob"]):
                    enqueue.insert(
                        0,
                        {
                            "intercept": float(theta["intercept"]),
                            "low_crowding": float(theta["low_crowding"]),
                            "altprob": float(theta["altprob"]),
                            "target_share": 1.0,
                        },
                    )

        for point in enqueue:
            candidate = {name: float(point[name]) for name in bounds if name in point}
            if len(candidate) != len(bounds):
                continue
            inside = all(bounds[name][0] <= value <= bounds[name][1] for name, value in candidate.items())
            if inside:
                study.enqueue_trial(candidate)

    target = int(override_trials or bayes_cfg.get("n_trials", 300))
    completed_before = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining = max(0, target - completed_before)
    print(
        f"[Selective bounty] completed {completed_before}/{target}; running {remaining}",
        flush=True,
    )

    started = time.perf_counter()
    completed_this_run = 0
    stop_on_infeasible = bool(
        config.get("policy", {}).get("early_stop_infeasible_trials", True)
    )

    def objective(trial: optuna.Trial) -> float:
        nonlocal completed_this_run
        theta = SelectiveTheta(
            intercept=trial.suggest_float("intercept", *bounds["intercept"]),
            low_crowding=trial.suggest_float(
                "low_crowding", *bounds["low_crowding"]
            ),
            altprob=trial.suggest_float("altprob", *bounds["altprob"]),
            target_share=trial.suggest_float(
                "target_share",
                *bounds["target_share"],
                log=target_share_log,
            ),
        )
        scored, selection_meta = score_selective_features(
            features, theta, config, selection_scope_override
        )
        rewards = base_module.rewards_by_hour(scored, baseline_hours)
        record, _ = simulator.evaluate(
            rewards, train_dates, stop_on_infeasible=stop_on_infeasible
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
            "[Selective] {0}/{1}, elapsed {2}, ETA {3}: relief={4:,.3f}, "
            "max daily cost={5:,.0f}, feasible={6}, selected={7:,} ({8:.3%}), "
            "mean reward={9:.2f}, theta=({10:.4f},{11:.4f},{12:.4f},q={13:.6f})".format(
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
    rows = []
    for trial in completed:
        row: Dict[str, Any] = {
            "trial": int(trial.number),
            "policy_type": "SELECTIVE_SIGMOID_BOUNTY",
            "intercept": float(trial.params["intercept"]),
            "low_crowding": float(trial.params["low_crowding"]),
            "altprob": float(trial.params["altprob"]),
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
        zero_rewards, train_dates, stop_on_infeasible=False
    )
    null_row: Dict[str, Any] = {
        "trial": -1,
        "policy_type": "NO_BOUNTY",
        "intercept": np.nan,
        "low_crowding": np.nan,
        "altprob": np.nan,
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
    }
    if bool(config.get("policy", {}).get("allow_no_bounty", True)):
        rows.append(null_row)
    if not rows:
        raise RuntimeError("No completed selective-policy trial")

    trials = pd.DataFrame(rows).sort_values(
        ["feasible", "objective_improvement", "max_daily_cost"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    feasible = trials[trials["feasible"] > 0.5]
    if feasible.empty:
        raise RuntimeError("No budget-feasible selective bounty policy")

    minimum_improvement = float(
        config.get("policy", {}).get("minimum_training_improvement", 0.0)
    )
    best = feasible.iloc[0]
    policy_active = bool(
        str(best["policy_type"]) == "SELECTIVE_SIGMOID_BOUNTY"
        and float(best["objective_improvement"]) > minimum_improvement
    )
    if policy_active:
        theta: Optional[SelectiveTheta] = SelectiveTheta(
            intercept=float(best["intercept"]),
            low_crowding=float(best["low_crowding"]),
            altprob=float(best["altprob"]),
            target_share=float(best["target_share"]),
        )
        scored, best_selection_meta = score_selective_features(
            features, theta, config, selection_scope_override
        )
    else:
        theta = None
        scored = features.copy()
        scored["selection_score"] = 0.0
        scored["policy_score"] = -np.inf
        alt = pd.to_numeric(scored["altprob_train_mean"], errors="coerce").fillna(0.0)
        minimum_altprob = float(config.get("policy", {}).get("minimum_altprob", 0.0))
        scored["eligible"] = (alt > minimum_altprob).astype(np.int8)
        scored["selected_for_bounty"] = 0
        scored["reward_won"] = 0.0
        best_selection_meta = {
            "selection_scope": selection_scope_override
            or selective_config(config)["selection_scope"],
            "target_share": 0.0,
            "eligible_edges": int(scored["eligible"].sum()),
            "selected_edges": 0,
            "realized_selected_share_of_eligible": 0.0,
            "positive_reward_edges": 0,
            "mean_positive_reward_won": 0.0,
            "median_positive_reward_won": 0.0,
            "p90_positive_reward_won": 0.0,
            "max_reward_won": 0.0,
        }

    metadata = {
        "method": "Optuna TPE selective top-share sigmoid bounty with explicit no-bounty alternative",
        "script_version": SCRIPT_VERSION,
        "policy_form_version": POLICY_FORM_VERSION,
        "study_name": study_name,
        "storage": str(storage_path),
        "completed_trials": len(completed),
        "target_trials": target,
        "selected_policy": "SELECTIVE_SIGMOID_BOUNTY"
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
        "objective": "maximize total congestion relief on training dates",
        "constraints": {
            "budget_per_day": config.get("budget_per_day", 3_000_000.0),
            "capacity_constraint": None,
            "payment_rule": "all post-policy users of each rewarded edge receive the edge bounty",
        },
        "deployment_dimension": "hour x route x directed edge",
        "date_specific_rewards": False,
        "feature_aggregation": "training-date arithmetic mean only",
    }
    return theta, scored, trials, metadata, policy_active


def summarize_policy_daily(path: Path, policy_name: str, budget_per_day: float) -> pd.DataFrame:
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
                "efficiency_per_1000": improvement / cost * 1000.0
                if cost > 0
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_reward_diagnostics(scored: pd.DataFrame, output: Path) -> None:
    positive = scored[scored["reward_won"] > 0].copy()
    overall = {
        "scope": "all_hours",
        "edges": int(len(scored)),
        "eligible_edges": int(scored["eligible"].sum()),
        "selected_edges": int(scored["selected_for_bounty"].sum()),
        "positive_reward_edges": int(len(positive)),
        "mean_positive_reward_won": float(positive["reward_won"].mean())
        if len(positive)
        else 0.0,
        "median_positive_reward_won": float(positive["reward_won"].median())
        if len(positive)
        else 0.0,
        "p90_positive_reward_won": float(positive["reward_won"].quantile(0.90))
        if len(positive)
        else 0.0,
        "p95_positive_reward_won": float(positive["reward_won"].quantile(0.95))
        if len(positive)
        else 0.0,
        "p99_positive_reward_won": float(positive["reward_won"].quantile(0.99))
        if len(positive)
        else 0.0,
        "max_reward_won": float(positive["reward_won"].max())
        if len(positive)
        else 0.0,
    }
    rows = [overall]
    for hour, group in scored.groupby("hour", sort=True):
        local = group[group["reward_won"] > 0]
        rows.append(
            {
                "scope": f"hour_{int(hour):02d}",
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
            }
        )
    pd.DataFrame(rows).to_csv(
        output / "reward_distribution.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    stage2_script = Path(args.stage2_script).resolve()
    model_input = Path(args.model_input).resolve()
    route_choice_output = Path(args.route_choice_output).resolve()
    cache_source = Path(args.cache_source).resolve()
    baseline_policy_output = (
        Path(args.baseline_policy_output).resolve()
        if args.baseline_policy_output
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

    base = import_module_from_path(stage2_script, "selective_bounty_base_stage2")
    simulation_version = str(getattr(base, "POLICY_SIMULATION_VERSION", ""))
    if "no-capacity" not in simulation_version:
        raise RuntimeError(
            "The stage-2 script is not the no-capacity-constraint version."
        )

    config: MutableMapping[str, Any] = load_json(config_path)
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

    print("analysis dates:", dates, flush=True)
    print("training dates:", train_dates, flush=True)
    print("evaluation dates:", test_dates, flush=True)
    print("hours:", sorted(hours), flush=True)
    print("cache source:", cache_source, flush=True)

    topologies = {
        hour: base.load_hour_topology(
            topology_root, hour, baseline_hours[hour], include_board=False
        )
        for hour in sorted(baseline_hours)
    }
    simulator = base.PolicySimulator(
        topologies, daily_root, dates, parameters, config
    )
    features = pd.read_csv(
        cache_source / "operational_hour_edge_features.csv.gz",
        compression="gzip",
        low_memory=False,
    )

    theta, scored, trials, search_meta, policy_active = fit_selective_policy(
        base,
        features,
        baseline_hours,
        simulator,
        train_dates,
        parameters,
        config,
        output,
        daily_meta,
        args.policy_trials,
        args.selection_scope,
        baseline_policy_output,
    )

    trials.to_csv(
        output / "policy_bayes_results.csv", index=False, encoding="utf-8-sig"
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
        print("learned selective theta:", asdict(theta), flush=True)

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
    reference = float(base.load_reference_value(config))
    load_qa = {
        "reference_load": reference,
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
    }
    write_json(output / "policy_load_qa.json", load_qa)

    selected_count = int(scored["selected_for_bounty"].sum())
    eligible_count = int(scored["eligible"].sum())
    learned = {
        "created_at": pd.Timestamp.now().isoformat(),
        "script_version": SCRIPT_VERSION,
        "policy_form_version": POLICY_FORM_VERSION,
        "policy_active": policy_active,
        "selected_policy": "SELECTIVE_SIGMOID_BOUNTY"
        if policy_active
        else "NO_BOUNTY",
        "theta": asdict(theta) if theta is not None else None,
        "selection_formula": (
            "select the top target_share among eligible hour-route-directed-edge rows "
            "by low_crowding*low_crowding_train_mean + altprob*altprob_train_mean"
        ),
        "reward_formula": (
            "reward_won = selected * max_edge_reward_won * sigmoid(" 
            "intercept + low_crowding*low_crowding_train_mean + "
            "altprob*altprob_train_mean)"
        ),
        "selection_scope": args.selection_scope
        or selective_config(config)["selection_scope"],
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
            "selective_bounty",
            float(config.get("budget_per_day", 3_000_000.0)),
        )
    ]
    if baseline_policy_output is not None:
        baseline_daily = baseline_policy_output / "daily_policy_results.csv"
        if baseline_daily.exists():
            comparison_frames.insert(
                0,
                summarize_policy_daily(
                    baseline_daily,
                    "distributed_bounty_baseline",
                    float(config.get("budget_per_day", 3_000_000.0)),
                ),
            )
    comparison = pd.concat(comparison_frames, ignore_index=True)
    comparison.to_csv(
        output / "selective_vs_distributed_summary.csv",
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
        "baseline_policy_output": str(baseline_policy_output)
        if baseline_policy_output is not None
        else None,
        "output": str(output),
        "config": str(config_path),
        "dates": dates,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "hours": sorted(hours),
        "topology_cache": topology_meta,
        "daily_cache_signature": daily_meta.get("signature"),
        "learned_policy": learned,
        "policy_search": search_meta,
        "output_rows": {
            "operational_rewards": int(len(scored)),
            "daily_policy_results": int(len(daily_results)),
            "edge_rewards": int(edge_rows),
            "trials": int(len(trials)),
        },
    }
    write_json(output / "selective_bounty_manifest.json", manifest)

    print("\ncompleted:", output, flush=True)
    for name in [
        "learned_bounty_function.json",
        "operational_hour_edge_rewards.csv.gz",
        "policy_bayes_results.csv",
        "daily_policy_results.csv",
        "scenario_summary.csv",
        "edge_rewards.csv.gz",
        "reward_distribution.csv",
        "selective_vs_distributed_summary.csv",
        "policy_load_qa.json",
        "policy_search_metadata.json",
        "selective_bounty_manifest.json",
    ]:
        print(" -", output / name, flush=True)
    print("total elapsed:", format_seconds(time.perf_counter() - run_started), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted. The Optuna SQLite study is reusable; rerun without --fresh.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
