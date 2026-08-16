from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS = 1.0e-12


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_policy_daily(output: Path, model: str) -> pd.DataFrame:
    path = output / "daily_policy_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"date", "split", "scenario", "h0", "h1", "improvement", "cost"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    frame = frame[frame["scenario"].eq("PUBLIC_EDGE_BOUNTY")].copy()
    if frame.empty:
        raise ValueError(f"{path}: no PUBLIC_EDGE_BOUNTY rows")
    frame["model"] = model
    frame["date"] = frame["date"].astype(str)
    frame["improvement_pct"] = 100.0 * frame["improvement"] / frame["h0"].clip(lower=EPS)
    frame["budget_usage_pct"] = 100.0 * frame["cost"] / 3_000_000.0
    return frame


def performance_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (model, split), group in daily.groupby(["model", "split"], sort=False):
        h0 = float(group["h0"].sum())
        improvement = float(group["improvement"].sum())
        cost = float(group["cost"].sum())
        rows.append(
            {
                "model": model,
                "split": split,
                "dates": int(group["date"].nunique()),
                "H0": h0,
                "H1": float(group["h1"].sum()),
                "improvement": improvement,
                "improvement_pct": 100.0 * improvement / max(h0, EPS),
                "total_cost_won": cost,
                "mean_daily_cost_won": float(group["cost"].mean()),
                "max_daily_cost_won": float(group["cost"].max()),
                "mean_budget_usage_pct": float(group["budget_usage_pct"].mean()),
                "efficiency_per_1000_won": improvement / max(cost, EPS) * 1000.0,
                "min_daily_improvement_pct": float(group["improvement_pct"].min()),
                "max_daily_improvement_pct": float(group["improvement_pct"].max()),
            }
        )
    return pd.DataFrame(rows)


def reward_summary(output: Path, model: str) -> Dict[str, object]:
    path = output / "operational_hour_edge_rewards.csv.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, compression="gzip", low_memory=False)
    reward = pd.to_numeric(frame["reward_won"], errors="coerce").fillna(0.0)
    positive = reward[reward > 0]
    eligible = int(
        pd.to_numeric(frame.get("eligible", 0), errors="coerce").fillna(0).gt(0).sum()
    )
    if "selected_for_bounty" in frame.columns:
        selected = int(
            pd.to_numeric(frame["selected_for_bounty"], errors="coerce")
            .fillna(0)
            .gt(0)
            .sum()
        )
    else:
        selected = int(len(positive))
    row: Dict[str, object] = {
        "model": model,
        "hour_edge_rows": int(len(frame)),
        "eligible_edges": eligible,
        "selected_reward_edges": selected,
        "selected_share_of_eligible": selected / max(eligible, 1),
        "positive_reward_edges": int(len(positive)),
        "mean_positive_reward_won": float(positive.mean()) if len(positive) else 0.0,
        "median_positive_reward_won": float(positive.median()) if len(positive) else 0.0,
        "p90_positive_reward_won": float(positive.quantile(0.90)) if len(positive) else 0.0,
        "p95_positive_reward_won": float(positive.quantile(0.95)) if len(positive) else 0.0,
        "p99_positive_reward_won": float(positive.quantile(0.99)) if len(positive) else 0.0,
        "max_reward_won": float(positive.max()) if len(positive) else 0.0,
    }
    learned_path = output / "learned_bounty_function.json"
    if learned_path.exists():
        learned = load_json(learned_path)
        theta = learned.get("theta") or learned.get("parameters") or {}
        row["selected_policy"] = learned.get("selected_policy")
        for name in ["intercept", "low_crowding", "altprob", "target_share"]:
            row[f"theta_{name}"] = theta.get(name)
    return row


def load_qa(output: Path, model: str) -> Dict[str, object]:
    row: Dict[str, object] = {"model": model}
    path = output / "policy_load_qa.json"
    if not path.exists():
        return row
    qa = load_json(path)
    for key in [
        "reference_load",
        "max_policy_avg_onboard",
        "total_overloaded_date_hour_edges_above_reference",
        "total_new_overloaded_date_hour_edges_above_reference",
        "total_increased_existing_overload_edges",
        "max_load_increase",
    ]:
        row[key] = qa.get(key)
    return row


def plot_daily(daily: pd.DataFrame, path: Path) -> None:
    pivot = daily.pivot_table(
        index="date", columns="model", values="improvement_pct", aggfunc="first"
    ).sort_index()
    ax = pivot.plot(kind="bar", figsize=(13, 6))
    ax.set_title("Daily congestion improvement")
    ax.set_xlabel("Date")
    ax.set_ylabel("Improvement (%)")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_reward_histogram(base: Path, selective: Path, path: Path) -> None:
    arrays = []
    labels = []
    for output, label in [(base, "distributed"), (selective, "selective")]:
        values = pd.read_csv(
            output / "operational_hour_edge_rewards.csv.gz",
            usecols=["reward_won"],
            compression="gzip",
        )["reward_won"]
        values = pd.to_numeric(values, errors="coerce").fillna(0.0)
        arrays.append(values[values > 0].to_numpy(float))
        labels.append(label)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(arrays, bins=60, label=labels, alpha=0.65)
    ax.set_title("Positive edge-bounty distribution")
    ax.set_xlabel("Reward (won)")
    ax.set_ylabel("Hour-edge count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare distributed and selective bounties")
    parser.add_argument("--base-output", required=True)
    parser.add_argument("--selective-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = Path(args.base_output).resolve()
    selective = Path(args.selective_output).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    daily = pd.concat(
        [
            load_policy_daily(base, "distributed"),
            load_policy_daily(selective, "selective"),
        ],
        ignore_index=True,
    )
    daily.to_csv(output / "daily_policy_comparison.csv", index=False, encoding="utf-8-sig")

    performance = performance_summary(daily)
    performance.to_csv(
        output / "policy_performance_comparison.csv", index=False, encoding="utf-8-sig"
    )

    rewards = pd.DataFrame(
        [reward_summary(base, "distributed"), reward_summary(selective, "selective")]
    )
    rewards.to_csv(
        output / "reward_distribution_comparison.csv", index=False, encoding="utf-8-sig"
    )

    qa = pd.DataFrame([load_qa(base, "distributed"), load_qa(selective, "selective")])
    qa.to_csv(output / "load_qa_comparison.csv", index=False, encoding="utf-8-sig")

    plot_daily(daily, output / "daily_improvement_comparison.png")
    plot_reward_histogram(base, selective, output / "reward_distribution_comparison.png")

    print("\nPolicy performance")
    print(performance.to_string(index=False))
    print("\nReward distribution")
    print(rewards.to_string(index=False))
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
