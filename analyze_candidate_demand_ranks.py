from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Progress:
    def __init__(self, label: str, print_every_seconds: float = 15.0):
        self.label = label
        self.print_every_seconds = max(1.0, float(print_every_seconds))
        self.started = time.perf_counter()
        self.last_print = self.started
        self.rows = 0

    def update(self, rows: int, force: bool = False) -> None:
        self.rows += int(rows)
        now = time.perf_counter()
        if not force and now - self.last_print < self.print_every_seconds:
            return
        elapsed = now - self.started
        rate = self.rows / elapsed if elapsed > 0 else 0.0
        print(
            f"[read] rows {self.rows:,}, {rate:,.0f} rows/s, "
            f"elapsed {format_seconds(elapsed)}",
            flush=True,
        )
        self.last_print = now


def process_complete_groups(
    frame: pd.DataFrame,
    demand_col: str,
    max_rank: int,
    accum: dict,
) -> None:
    if frame.empty:
        return

    # One OD is defined by hour + od_index.
    for (_, _), g in frame.groupby(["hour", "od_index"], sort=False):
        demand = pd.to_numeric(g[demand_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        demand = np.maximum(demand, 0.0)

        total = float(demand.sum())
        accum["all_od_count"] += 1

        positive_paths = int(np.count_nonzero(demand > 0))
        accum["positive_path_count_sum"] += positive_paths
        accum["candidate_count_sum"] += len(demand)

        # No matched observed demand -> cannot define within-OD demand shares.
        if total <= 0:
            accum["zero_observed_od_count"] += 1
            continue

        accum["valid_od_count"] += 1
        accum["total_matched_demand"] += total

        ranked = np.sort(demand)[::-1]
        if len(ranked) < max_rank:
            ranked = np.pad(ranked, (0, max_rank - len(ranked)))
        else:
            ranked = ranked[:max_rank]

        shares = ranked / total

        # Unweighted OD-level mean:
        # each OD contributes equally regardless of its total demand.
        accum["share_sum"] += shares
        accum["share_sq_sum"] += shares ** 2

        # Aggregate passenger share:
        # numerator = passengers at each rank, denominator = all matched passengers.
        accum["rank_demand_sum"] += ranked

        accum["rank_present_count"] += (np.arange(max_rank) < min(len(demand), max_rank)).astype(np.int64)
        accum["rank_positive_count"] += (ranked > 0).astype(np.int64)


def finalize_stats(accum: dict, max_rank: int) -> pd.DataFrame:
    n = int(accum["valid_od_count"])
    total_demand = float(accum["total_matched_demand"])

    if n == 0:
        raise RuntimeError(
            "관측 수요가 1명 이상 매칭된 OD가 없습니다. "
            "수요 컬럼과 candidate_pool.csv.gz를 확인하세요."
        )

    mean_share = accum["share_sum"] / n
    variance = np.maximum(accum["share_sq_sum"] / n - mean_share ** 2, 0.0)
    std_share = np.sqrt(variance)

    aggregate_share = (
        accum["rank_demand_sum"] / total_demand
        if total_demand > 0
        else np.zeros(max_rank, dtype=float)
    )

    out = pd.DataFrame(
        {
            "demand_rank": np.arange(1, max_rank + 1),
            "mean_od_share": mean_share,
            "std_od_share": std_share,
            "aggregate_passenger_share": aggregate_share,
            "mean_od_cumulative_share": np.cumsum(mean_share),
            "aggregate_cumulative_share": np.cumsum(aggregate_share),
            "ods_with_candidate_at_rank": accum["rank_present_count"],
            "ods_with_positive_demand_at_rank": accum["rank_positive_count"],
            "positive_demand_od_rate_at_rank": accum["rank_positive_count"] / n,
            "rank_total_observed_demand": accum["rank_demand_sum"],
        }
    )
    return out


def make_plots(stats: pd.DataFrame, output_dir: Path, demand_label: str) -> None:
    ranks = stats["demand_rank"].to_numpy()

    # Chart 1: Average OD-level demand share
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(ranks, stats["mean_od_share"] * 100)
    ax.set_title(f"Average demand share by within-OD rank ({demand_label})")
    ax.set_xlabel("Demand rank within OD")
    ax.set_ylabel("Average share of OD demand (%)")
    ax.set_xticks(ranks)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "demand_share_by_rank.png", dpi=180)
    plt.close(fig)

    # Chart 2: Cumulative share
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        ranks,
        stats["mean_od_cumulative_share"] * 100,
        marker="o",
        label="Mean of OD-level shares",
    )
    ax.plot(
        ranks,
        stats["aggregate_cumulative_share"] * 100,
        marker="s",
        label="Passenger-weighted aggregate",
    )
    ax.set_title(f"Cumulative demand coverage by rank ({demand_label})")
    ax.set_xlabel("Top-N paths retained")
    ax.set_ylabel("Cumulative demand share (%)")
    ax.set_xticks(ranks)
    ax.set_ylim(0, 101)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_demand_share.png", dpi=180)
    plt.close(fig)

    # Chart 3: How often each rank actually has positive observed demand
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(ranks, stats["positive_demand_od_rate_at_rank"] * 100)
    ax.set_title(f"ODs with positive observed demand at each rank ({demand_label})")
    ax.set_xlabel("Demand rank within OD")
    ax.set_ylabel("Share of ODs with positive demand (%)")
    ax.set_xticks(ranks)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "positive_demand_od_rate_by_rank.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank candidate paths by observed demand within each hour-OD and "
            "visualize the average/aggregate demand share of ranks 1..15."
        )
    )
    parser.add_argument(
        "--candidate-pool",
        default=r".\model_input_purpose_candidates15\candidate_pool.csv.gz",
        help="Path to candidate_pool.csv.gz",
    )
    parser.add_argument(
        "--output",
        default=r".\demand_rank_analysis",
        help="Output directory",
    )
    parser.add_argument(
        "--demand-column",
        default="observed_passengers_train",
        choices=["observed_passengers_train", "observed_passengers_all"],
        help=(
            "Use train demand for deciding the stage-1 fitting set. "
            "Use all only for descriptive analysis."
        ),
    )
    parser.add_argument("--max-rank", type=int, default=15)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--progress-seconds", type=float, default=15.0)
    args = parser.parse_args()

    candidate_path = Path(args.candidate_pool)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)

    max_rank = int(args.max_rank)
    if max_rank <= 0:
        raise ValueError("--max-rank must be >= 1")

    usecols = ["hour", "od_index", args.demand_column]

    accum = {
        "all_od_count": 0,
        "valid_od_count": 0,
        "zero_observed_od_count": 0,
        "total_matched_demand": 0.0,
        "positive_path_count_sum": 0,
        "candidate_count_sum": 0,
        "share_sum": np.zeros(max_rank, dtype=float),
        "share_sq_sum": np.zeros(max_rank, dtype=float),
        "rank_demand_sum": np.zeros(max_rank, dtype=float),
        "rank_present_count": np.zeros(max_rank, dtype=np.int64),
        "rank_positive_count": np.zeros(max_rank, dtype=np.int64),
    }

    meter = Progress("candidate demand ranking", args.progress_seconds)
    carry = pd.DataFrame(columns=usecols)

    reader = pd.read_csv(
        candidate_path,
        usecols=usecols,
        chunksize=int(args.chunksize),
        low_memory=False,
    )

    for chunk in reader:
        meter.update(len(chunk))

        chunk["hour"] = pd.to_numeric(chunk["hour"], errors="coerce").astype("Int64")
        chunk["od_index"] = pd.to_numeric(chunk["od_index"], errors="coerce").astype("Int64")
        chunk[args.demand_column] = pd.to_numeric(
            chunk[args.demand_column], errors="coerce"
        ).fillna(0.0)

        chunk = chunk.dropna(subset=["hour", "od_index"])

        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = pd.DataFrame(columns=usecols)

        if chunk.empty:
            continue

        # candidate_pool is generated in hour/OD order. A read chunk can split the
        # final OD, so keep that final group and process it with the next chunk.
        last_hour = chunk.iloc[-1]["hour"]
        last_od = chunk.iloc[-1]["od_index"]
        last_mask = (chunk["hour"] == last_hour) & (chunk["od_index"] == last_od)

        carry = chunk.loc[last_mask].copy()
        complete = chunk.loc[~last_mask]

        process_complete_groups(
            complete,
            demand_col=args.demand_column,
            max_rank=max_rank,
            accum=accum,
        )

    if not carry.empty:
        process_complete_groups(
            carry,
            demand_col=args.demand_column,
            max_rank=max_rank,
            accum=accum,
        )

    meter.update(0, force=True)

    stats = finalize_stats(accum, max_rank)
    stats.to_csv(
        output_dir / "demand_rank_share_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    make_plots(stats, output_dir, args.demand_column)

    valid_od = int(accum["valid_od_count"])
    all_od = int(accum["all_od_count"])
    zero_od = int(accum["zero_observed_od_count"])
    avg_positive_paths = (
        accum["positive_path_count_sum"] / all_od if all_od else 0.0
    )
    avg_candidates = accum["candidate_count_sum"] / all_od if all_od else 0.0

    summary_lines = [
        f"demand_column: {args.demand_column}",
        f"all_ODs: {all_od:,}",
        f"ODs_with_positive_matched_demand: {valid_od:,}",
        f"ODs_with_zero_matched_demand: {zero_od:,}",
        f"positive_matched_OD_rate: {valid_od / all_od:.4%}" if all_od else "positive_matched_OD_rate: --",
        f"total_matched_observed_demand: {accum['total_matched_demand']:,.2f}",
        f"average_candidate_paths_per_OD: {avg_candidates:.3f}",
        f"average_positive-demand_paths_per_OD: {avg_positive_paths:.3f}",
        "",
        "Top-N cumulative demand coverage:",
    ]
    for _, row in stats.iterrows():
        rank = int(row["demand_rank"])
        summary_lines.append(
            f"top {rank:2d}: "
            f"mean-OD {row['mean_od_cumulative_share']:.2%}, "
            f"passenger-weighted {row['aggregate_cumulative_share']:.2%}"
        )

    (output_dir / "analysis_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n".join(summary_lines), flush=True)
    print(f"\nSaved to: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
