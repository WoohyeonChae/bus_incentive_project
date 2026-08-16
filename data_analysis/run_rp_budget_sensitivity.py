from __future__ import annotations

"""
ReliefPotential Selective 현상금 정책의 일일 예산 민감도 분석.

핵심 설계
---------
1. 1단계 경로선택모형과 ReliefPotential은 다시 계산하지 않는다.
2. 기존 topology/daily cache와 이미 계산된 ReliefPotential feature를 한 번만 읽는다.
3. 각 일일 예산 상한마다 현상금 함수 파라미터를 Optuna로 다시 최적화한다.
4. 학습기간과 평가기간을 분리해 혼잡 완화량과 실제 지출액 대비 효율을 산출한다.
5. 중단 후 같은 명령을 다시 실행하면 예산별 SQLite study에서 이어서 실행한다.

기본 예산
---------
50만, 100만, 200만, 300만, 500만, 1,000만원/일

주요 결과
---------
- budget_sensitivity_summary.csv
- budget_sensitivity_test_korean.csv
- budget_sensitivity_daily.csv
- budget_sensitivity_parameters.csv
- budget_sensitivity_marginal_test.csv
- figures/*.png, *.svg
"""

import argparse
import copy
import gc
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd


EPS = 1.0e-12
SCRIPT_VERSION = "rp-budget-sensitivity-v1.0.0"


def import_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("필요한 파일/폴더가 없습니다:\n - " + "\n - ".join(missing))


def parse_number_list(value: str) -> list[float]:
    result = []
    for token in str(value).split(","):
        token = token.strip().replace("_", "")
        if not token:
            continue
        number = float(token)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"잘못된 예산값: {token}")
        result.append(number)
    if not result:
        raise ValueError("예산 목록이 비어 있습니다.")
    return sorted(set(result))


def budget_slug(budget: float) -> str:
    if abs(budget - round(budget)) < 1.0e-9:
        return f"budget_{int(round(budget)):010d}"
    return "budget_" + str(budget).replace(".", "p")


def setup_korean_font() -> Optional[fm.FontProperties]:
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    candidates = [
        windir / "Fonts" / "malgun.ttf",
        windir / "Fonts" / "malgunbd.ttf",
        windir / "Fonts" / "gulim.ttc",
        windir / "Fonts" / "batang.ttc",
    ]
    for path in candidates:
        if path.exists():
            fm.fontManager.addfont(str(path))
            prop = fm.FontProperties(fname=str(path))
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return prop
    plt.rcParams["axes.unicode_minus"] = False
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ReliefPotential Selective 정책의 일일 예산 민감도 분석"
    )
    p.add_argument(
        "--stage2-script",
        default=r".\fit_public_edge_bounty_policy.py",
    )
    p.add_argument(
        "--relief-script",
        default=r".\fit_relief_potential_bounty_policy.py",
    )
    p.add_argument(
        "--model-input",
        default=r"..\model_input_purpose_candidates15",
    )
    p.add_argument(
        "--route-choice-output",
        default=r".\route_choice_top5",
    )
    p.add_argument(
        "--cache-source",
        default=r".\public_edge_bounty_top5",
    )
    p.add_argument(
        "--selective-policy-output",
        default=r".\public_edge_bounty_selective",
        help="기존 Selective 정책 결과. Optuna warm start에 사용.",
    )
    p.add_argument(
        "--reference-relief-output",
        default=r".\public_edge_bounty_relief_selective",
        help="기존 300만원 RP 결과와 ReliefPotential feature/cache의 위치.",
    )
    p.add_argument(
        "--config",
        default=r".\optimization_config_public_edge_relief_selective.json",
    )
    p.add_argument(
        "--output",
        default=r".\rp_budget_sensitivity",
    )
    p.add_argument(
        "--budgets",
        default="500000,1000000,2000000,3000000,5000000,10000000",
        help="쉼표로 구분한 일일 예산 상한(원)",
    )
    p.add_argument(
        "--policy-trials",
        type=int,
        default=300,
        help="각 예산별 총 Optuna trial 수",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--save-operational-rewards",
        action="store_true",
        help="예산별 operational_hour_edge_rewards.csv.gz도 저장(디스크 사용 증가)",
    )
    p.add_argument(
        "--rerun-completed",
        action="store_true",
        help="완료된 예산 결과도 다시 평가/작성",
    )
    p.add_argument(
        "--no-reuse-reference-budget",
        action="store_true",
        help="기존 reference Relief 결과와 같은 예산도 새 sensitivity study로 다시 적합",
    )
    return p.parse_args()


def completed_trials(metadata_path: Path) -> int:
    if not metadata_path.exists():
        return 0
    try:
        return int(load_json(metadata_path).get("completed_trials", 0) or 0)
    except Exception:
        return 0


def is_complete_output(output: Path, trials: int) -> bool:
    required = [
        output / "daily_policy_results.csv",
        output / "scenario_summary.csv",
        output / "policy_search_metadata.json",
        output / "learned_bounty_function.json",
        output / "policy_bayes_results.csv",
    ]
    return all(path.exists() for path in required) and completed_trials(
        output / "policy_search_metadata.json"
    ) >= int(trials)


def inject_reference_warm_starts(
    config: Dict[str, Any],
    reference_output: Path,
) -> None:
    learned_path = reference_output / "learned_bounty_function.json"
    if not learned_path.exists():
        return
    try:
        theta = load_json(learned_path).get("theta") or {}
        required = [
            "intercept",
            "low_crowding",
            "altprob",
            "relief_potential",
            "target_share",
        ]
        if not all(name in theta for name in required):
            return

        policy = config.setdefault("policy", {})
        relief_cfg = policy.setdefault("relief_selective", {})
        bayes = relief_cfg.setdefault("bayesian", {})
        existing = list(bayes.get("enqueue", []))

        base_point = {name: float(theta[name]) for name in required}
        candidates = []
        for shift in [-2.0, -1.0, 0.0, 1.0]:
            point = dict(base_point)
            point["intercept"] = float(base_point["intercept"]) + shift
            candidates.append(point)

        # Keep order stable and remove exact duplicates.
        seen = set()
        merged = []
        for point in candidates + existing:
            try:
                key = tuple(round(float(point[name]), 12) for name in required)
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            merged.append({name: float(point[name]) for name in required})
        bayes["enqueue"] = merged
    except Exception as error:
        print(f"[warm-start] 참고 RP 파라미터를 읽지 못해 건너뜀: {error}", flush=True)



def to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    true_values = {"true", "1", "yes", "y", "t", "예"}
    return series.fillna(False).map(
        lambda value: str(value).strip().lower() in true_values
    )


def summarize_daily_policy(
    daily: pd.DataFrame,
    budget_per_day: float,
) -> pd.DataFrame:
    policy = daily.copy()
    if "scenario" in policy.columns:
        policy = policy[policy["scenario"] == "PUBLIC_EDGE_BOUNTY"].copy()
    if policy.empty:
        raise ValueError("일별 정책 결과에 PUBLIC_EDGE_BOUNTY 행이 없습니다.")

    rows = []
    for split, group in policy.groupby("split", sort=False):
        h0 = float(group["h0"].sum())
        h1 = float(group["h1"].sum())
        improvement = float(group["improvement"].sum())
        total_cost = float(group["cost"].sum())
        dates = int(group["date"].nunique())
        budget_ceiling_total = float(budget_per_day) * dates
        mean_daily_improvement = improvement / max(dates, 1)

        rows.append(
            {
                "split": str(split),
                "dates": dates,
                "budget_per_day": float(budget_per_day),
                "budget_ceiling_total": budget_ceiling_total,
                "H0": h0,
                "H1": h1,
                "improvement": improvement,
                "mean_daily_improvement": mean_daily_improvement,
                "improvement_pct": 100.0 * improvement / max(h0, EPS),
                "total_cost": total_cost,
                "mean_daily_cost": float(group["cost"].mean()),
                "max_daily_cost": float(group["cost"].max()),
                "min_daily_cost": float(group["cost"].min()),
                "actual_budget_usage_pct": total_cost
                / max(budget_ceiling_total, EPS)
                * 100.0,
                "all_dates_feasible": bool(to_bool_series(group["feasible"]).all()),
                "relief_per_actual_won": improvement / total_cost
                if total_cost > 0
                else 0.0,
                "relief_per_actual_1000_won": improvement
                / total_cost
                * 1000.0
                if total_cost > 0
                else 0.0,
                "relief_per_budget_ceiling_1000_won": improvement
                / max(budget_ceiling_total, EPS)
                * 1000.0,
                "min_daily_improvement": float(group["improvement"].min()),
                "max_daily_improvement": float(group["improvement"].max()),
            }
        )
    return pd.DataFrame(rows)


def results_to_daily_frame(
    train_results: Sequence[Any],
    test_results: Sequence[Any],
) -> pd.DataFrame:
    rows = []
    for split, results in [("train", train_results), ("test", test_results)]:
        for result in results:
            rows.append(
                {
                    "date": str(result.date),
                    "split": split,
                    "scenario": "PUBLIC_EDGE_BOUNTY",
                    **asdict(result),
                }
            )
    return pd.DataFrame(rows)


def copy_lightweight_reference_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        "daily_policy_results.csv",
        "scenario_summary.csv",
        "policy_search_metadata.json",
        "learned_bounty_function.json",
        "policy_bayes_results.csv",
        "reward_distribution.csv",
        "policy_load_qa.json",
    ]
    for name in names:
        src = source / name
        if src.exists():
            shutil.copy2(src, destination / name)
    write_json(
        destination / "reused_reference_output.json",
        {
            "source": str(source.resolve()),
            "note": "기존 동일 예산 RP 결과를 민감도 분석에 재사용함",
        },
    )


def reward_parameter_row(
    budget: float,
    learned: Mapping[str, Any],
    search_meta: Mapping[str, Any],
    reward_distribution: Optional[pd.DataFrame],
    source_output: Path,
) -> Dict[str, Any]:
    theta = learned.get("theta") or {}
    row: Dict[str, Any] = {
        "budget_per_day": float(budget),
        "source_output": str(source_output),
        "selected_policy": learned.get("selected_policy"),
        "policy_active": bool(learned.get("policy_active", False)),
        "intercept": theta.get("intercept"),
        "low_crowding": theta.get("low_crowding"),
        "altprob": theta.get("altprob"),
        "relief_potential": theta.get("relief_potential"),
        "target_share": theta.get("target_share"),
        "eligible_edges": learned.get("eligible_edges"),
        "selected_edges": learned.get("selected_edges"),
        "realized_selected_share_of_eligible": learned.get(
            "realized_selected_share_of_eligible"
        ),
        "completed_trials": search_meta.get("completed_trials"),
        "selected_trial": search_meta.get("selected_trial"),
        "best_training_improvement": search_meta.get("best_objective_improvement"),
        "best_training_max_daily_cost": search_meta.get("best_max_daily_cost"),
    }
    if reward_distribution is not None and not reward_distribution.empty:
        all_hours = reward_distribution[
            reward_distribution["scope"].astype(str) == "all_hours"
        ]
        if not all_hours.empty:
            r = all_hours.iloc[0]
            for name in [
                "positive_reward_edges",
                "mean_positive_reward_won",
                "median_positive_reward_won",
                "p90_positive_reward_won",
                "p95_positive_reward_won",
                "p99_positive_reward_won",
                "max_reward_won",
                "mean_selected_relief_raw_per_won",
                "min_selected_relief_raw_per_won",
            ]:
                if name in r.index:
                    row[name] = r[name]
    return row


def load_summary_from_output(
    output: Path,
    budget: float,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    daily = pd.read_csv(output / "daily_policy_results.csv", low_memory=False)
    summary = summarize_daily_policy(daily, budget)
    learned = load_json(output / "learned_bounty_function.json")
    search_meta = load_json(output / "policy_search_metadata.json")
    reward_dist_path = output / "reward_distribution.csv"
    reward_dist = (
        pd.read_csv(reward_dist_path, low_memory=False)
        if reward_dist_path.exists()
        else None
    )
    params = reward_parameter_row(
        budget,
        learned,
        search_meta,
        reward_dist,
        output,
    )
    return summary, daily, params


def save_plot(fig, png: Path, svg: Path) -> None:
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def plot_results(test: pd.DataFrame, figures: Path, font_prop) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    x = test["budget_per_day"] / 1_000_000.0

    # 1. 예산과 일평균 혼잡 완화량
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(x, test["mean_daily_improvement"], marker="o")
    ax.set_xlabel("일일 예산 상한 (백만원)", fontproperties=font_prop)
    ax.set_ylabel("평가기간 일평균 혼잡 완화량 (H₀-H₁)", fontproperties=font_prop)
    ax.set_title("예산별 평가기간 혼잡 완화량", fontproperties=font_prop)
    ax.grid(axis="both", linestyle=":", alpha=0.35)
    for xi, yi in zip(x, test["mean_daily_improvement"]):
        ax.annotate(
            f"{yi:,.0f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontproperties=font_prop,
            fontsize=8,
        )
    save_plot(
        fig,
        figures / "fig_01_budget_vs_mean_daily_relief.png",
        figures / "fig_01_budget_vs_mean_daily_relief.svg",
    )

    # 2. 예산과 개선율
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(x, test["improvement_pct"], marker="o")
    ax.set_xlabel("일일 예산 상한 (백만원)", fontproperties=font_prop)
    ax.set_ylabel("평가기간 혼잡개선율 (%)", fontproperties=font_prop)
    ax.set_title("예산별 평가기간 혼잡개선율", fontproperties=font_prop)
    ax.grid(axis="both", linestyle=":", alpha=0.35)
    for xi, yi in zip(x, test["improvement_pct"]):
        ax.annotate(
            f"{yi:.4f}%",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontproperties=font_prop,
            fontsize=8,
        )
    save_plot(
        fig,
        figures / "fig_02_budget_vs_relief_rate.png",
        figures / "fig_02_budget_vs_relief_rate.svg",
    )

    # 3. 실제 지출 1,000원당 혼잡 완화량
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(x, test["relief_per_actual_1000_won"], marker="o")
    ax.set_xlabel("일일 예산 상한 (백만원)", fontproperties=font_prop)
    ax.set_ylabel("실제 지출 1,000원당 혼잡 완화량", fontproperties=font_prop)
    ax.set_title("예산별 비용효율", fontproperties=font_prop)
    ax.grid(axis="both", linestyle=":", alpha=0.35)
    for xi, yi in zip(x, test["relief_per_actual_1000_won"]):
        ax.annotate(
            f"{yi:.2f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontproperties=font_prop,
            fontsize=8,
        )
    save_plot(
        fig,
        figures / "fig_03_budget_vs_efficiency.png",
        figures / "fig_03_budget_vs_efficiency.svg",
    )

    # 4. 예산 상한과 실제 지출
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(x, test["mean_daily_cost"] / 1_000_000.0, marker="o", label="실제 평균 일일비용")
    ax.plot(x, x, linestyle="--", label="예산 상한")
    ax.set_xlabel("일일 예산 상한 (백만원)", fontproperties=font_prop)
    ax.set_ylabel("평균 일일비용 (백만원)", fontproperties=font_prop)
    ax.set_title("예산 상한과 실제 지출 비교", fontproperties=font_prop)
    ax.grid(axis="both", linestyle=":", alpha=0.35)
    ax.legend(prop=font_prop)
    save_plot(
        fig,
        figures / "fig_04_budget_vs_actual_spending.png",
        figures / "fig_04_budget_vs_actual_spending.svg",
    )

    # 5. 실제 비용-효과 프런티어
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cost_x = test["mean_daily_cost"] / 1_000_000.0
    relief_y = test["mean_daily_improvement"]
    ax.plot(cost_x, relief_y, marker="o")
    ax.set_xlabel("실제 평균 일일비용 (백만원)", fontproperties=font_prop)
    ax.set_ylabel("평가기간 일평균 혼잡 완화량 (H₀-H₁)", fontproperties=font_prop)
    ax.set_title("실제 비용-혼잡 완화 프런티어", fontproperties=font_prop)
    ax.grid(axis="both", linestyle=":", alpha=0.35)
    for cx, ry, budget in zip(cost_x, relief_y, test["budget_per_day"]):
        ax.annotate(
            f"예산 {budget/1_000_000:g}백만",
            (cx, ry),
            textcoords="offset points",
            xytext=(5, 6),
            fontproperties=font_prop,
            fontsize=8,
        )
    save_plot(
        fig,
        figures / "fig_05_actual_cost_relief_frontier.png",
        figures / "fig_05_actual_cost_relief_frontier.svg",
    )


def make_marginal_table(test: pd.DataFrame) -> pd.DataFrame:
    x = test.sort_values("budget_per_day").copy()
    x["additional_budget_limit"] = x["budget_per_day"].diff()
    x["additional_actual_mean_daily_cost"] = x["mean_daily_cost"].diff()
    x["additional_mean_daily_improvement"] = x["mean_daily_improvement"].diff()
    x["additional_improvement_pct"] = x["improvement_pct"].diff()
    x["marginal_relief_per_additional_actual_1000_won"] = np.where(
        x["additional_actual_mean_daily_cost"] > EPS,
        x["additional_mean_daily_improvement"]
        / x["additional_actual_mean_daily_cost"]
        * 1000.0,
        np.nan,
    )
    return x[
        [
            "budget_per_day",
            "additional_budget_limit",
            "additional_actual_mean_daily_cost",
            "additional_mean_daily_improvement",
            "additional_improvement_pct",
            "marginal_relief_per_additional_actual_1000_won",
        ]
    ]


def write_korean_test_table(test: pd.DataFrame, path: Path) -> None:
    out = pd.DataFrame(
        {
            "일일 예산상한(원)": test["budget_per_day"],
            "평가일수": test["dates"],
            "평가기간 총 혼잡완화량": test["improvement"],
            "일평균 혼잡완화량": test["mean_daily_improvement"],
            "혼잡개선율(%)": test["improvement_pct"],
            "실제 평균 일일비용(원)": test["mean_daily_cost"],
            "최대 일일비용(원)": test["max_daily_cost"],
            "예산 사용률(%)": test["actual_budget_usage_pct"],
            "실제 지출 1,000원당 혼잡완화량": test[
                "relief_per_actual_1000_won"
            ],
            "예산상한 1,000원당 혼잡완화량": test[
                "relief_per_budget_ceiling_1000_won"
            ],
            "모든 평가일 예산준수": test["all_dates_feasible"].map(
                lambda value: "예" if bool(value) else "아니오"
            ),
        }
    )
    out.to_csv(path, index=False, encoding="utf-8-sig")


def write_summary_text(test: pd.DataFrame, path: Path) -> None:
    if test.empty:
        path.write_text("평가기간 결과가 없습니다.", encoding="utf-8")
        return

    max_relief = test.loc[test["mean_daily_improvement"].idxmax()]
    max_eff = test.loc[test["relief_per_actual_1000_won"].idxmax()]
    max_rate = test.loc[test["improvement_pct"].idxmax()]

    lines = [
        "ReliefPotential Selective 일일 예산 민감도 분석 요약",
        "",
        f"- 분석 예산 수: {len(test)}개",
        f"- 최대 일평균 혼잡 완화량 예산: {max_relief['budget_per_day']:,.0f}원/일",
        f"  · 일평균 혼잡 완화량: {max_relief['mean_daily_improvement']:,.3f}",
        f"  · 혼잡개선율: {max_relief['improvement_pct']:.6f}%",
        f"- 최고 비용효율 예산: {max_eff['budget_per_day']:,.0f}원/일",
        f"  · 실제 지출 1,000원당 혼잡 완화량: {max_eff['relief_per_actual_1000_won']:.6f}",
        f"- 최고 혼잡개선율 예산: {max_rate['budget_per_day']:,.0f}원/일",
        f"  · 혼잡개선율: {max_rate['improvement_pct']:.6f}%",
        "",
        "주의:",
        "- 혼잡 완화량은 목적함수 H의 감소량(H0-H1)이다.",
        "- 비용효율은 명목 예산상한이 아니라 실제 정책지출을 분모로 계산한 값을 우선 사용한다.",
        "- 각 예산에서 현상금 파라미터를 독립적으로 재최적화하므로 단순 비례확대 결과가 아니다.",
        "- 최종 정책예산 선택 시 혼잡 완화량, 비용효율, 모든 평가일 예산준수 여부를 함께 본다.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    stage2_script = Path(args.stage2_script).resolve()
    relief_script = Path(args.relief_script).resolve()
    model_input = Path(args.model_input).resolve()
    route_choice_output = Path(args.route_choice_output).resolve()
    cache_source = Path(args.cache_source).resolve()
    selective_policy_output = Path(args.selective_policy_output).resolve()
    reference_relief_output = Path(args.reference_relief_output).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    runs_root = output / "runs"
    figures = output / "figures"

    output.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    budgets = parse_number_list(args.budgets)
    if args.policy_trials <= 0:
        raise ValueError("--policy-trials는 1 이상이어야 합니다.")

    topology_root = cache_source / "_topology_cache"
    daily_root = cache_source / "_daily_cache"
    relief_feature_path = (
        reference_relief_output / "operational_hour_edge_features_with_relief.csv.gz"
    )
    relief_meta_path = (
        reference_relief_output
        / "_relief_potential_cache"
        / "relief_potential_meta.json"
    )

    require_paths(
        [
            stage2_script,
            relief_script,
            model_input,
            route_choice_output / "route_choice_parameters.json",
            topology_root / "topology_meta.json",
            daily_root / "daily_meta.json",
            relief_feature_path,
            relief_meta_path,
            config_path,
        ]
    )

    base = import_module_from_path(stage2_script, "rp_budget_base_stage2")
    relief = import_module_from_path(relief_script, "rp_budget_relief_model")

    simulation_version = str(getattr(base, "POLICY_SIMULATION_VERSION", ""))
    if "no-capacity" not in simulation_version:
        raise RuntimeError("fit_public_edge_bounty_policy.py가 no-capacity 버전이 아닙니다.")

    base_config = load_json(config_path)
    base_config.setdefault("policy", {}).setdefault(
        "relief_selective", {}
    ).setdefault("bayesian", {})["seed"] = int(args.seed)

    settings = relief.resolve_relief_settings(
        base_config,
        None,
        None,
        0,
        0.1,
    )
    metric = str(settings["metric"])
    transform = str(settings["transform"])

    parameters = base.load_choice_parameters(
        route_choice_output / "route_choice_parameters.json"
    )
    dates = base.parse_dates(model_input)
    train_dates, test_dates = base.split_dates(dates, base_config)
    hours = {
        int(value)
        for value in base_config.get("hours", [7, 8, 9, 17, 18, 19])
    }

    baseline = base.load_baseline_segments(model_input, hours)
    baseline_hours = base.baseline_by_hour(baseline)
    daily_meta = load_json(daily_root / "daily_meta.json")
    relief_meta = load_json(relief_meta_path)

    print("[stage] topology 로드", flush=True)
    topologies = {
        hour: base.load_hour_topology(
            topology_root,
            hour,
            baseline_hours[hour],
            include_board=False,
        )
        for hour in sorted(baseline_hours)
    }

    print("[stage] 기존 ReliefPotential feature 로드", flush=True)
    features = pd.read_csv(
        relief_feature_path,
        compression="gzip",
        low_memory=False,
    )
    print(f"ReliefPotential edge rows: {len(features):,}", flush=True)
    print("budgets:", [f"{v:,.0f}" for v in budgets], flush=True)
    print("trials per budget:", args.policy_trials, flush=True)
    print("train dates:", train_dates, flush=True)
    print("test dates:", test_dates, flush=True)

    reference_budget = float(base_config.get("budget_per_day", 3_000_000.0))
    all_summary = []
    all_daily = []
    parameter_rows = []

    for index, budget in enumerate(budgets, start=1):
        run_dir = runs_root / budget_slug(budget)
        run_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n[budget {index}/{len(budgets)}] {budget:,.0f}원/일",
            flush=True,
        )

        # 1) 기존 sensitivity 결과 재사용
        if (
            not args.rerun_completed
            and is_complete_output(run_dir, args.policy_trials)
        ):
            print("[reuse] 완료된 예산 결과 재사용:", run_dir, flush=True)
            summary, daily, params = load_summary_from_output(run_dir, budget)

        # 2) 기존 본모형과 동일한 예산은 본모형 결과 재사용 가능
        elif (
            not args.no_reuse_reference_budget
            and abs(budget - reference_budget) <= 1.0e-6
            and is_complete_output(reference_relief_output, args.policy_trials)
        ):
            print(
                "[reuse] 기존 RP 본모형 결과 재사용:",
                reference_relief_output,
                flush=True,
            )
            copy_lightweight_reference_files(reference_relief_output, run_dir)
            summary, daily, params = load_summary_from_output(run_dir, budget)

        # 3) 예산별 독립 재최적화
        else:
            budget_config = copy.deepcopy(base_config)
            budget_config["budget_per_day"] = float(budget)
            budget_config.setdefault("policy", {}).setdefault(
                "relief_selective", {}
            ).setdefault("bayesian", {})["seed"] = int(args.seed)
            inject_reference_warm_starts(
                budget_config,
                reference_relief_output,
            )
            write_json(run_dir / "config_budget.json", budget_config)

            simulator = base.PolicySimulator(
                topologies,
                daily_root,
                dates,
                parameters,
                budget_config,
            )

            theta, scored, trials, search_meta, policy_active = (
                relief.fit_relief_selective_policy(
                    base,
                    features,
                    baseline_hours,
                    simulator,
                    train_dates,
                    parameters,
                    budget_config,
                    run_dir,
                    daily_meta,
                    relief_meta,
                    int(args.policy_trials),
                    None,
                    metric,
                    transform,
                    selective_policy_output,
                )
            )

            trials.to_csv(
                run_dir / "policy_bayes_results.csv",
                index=False,
                encoding="utf-8-sig",
            )
            relief.write_reward_diagnostics(scored, run_dir)

            if args.save_operational_rewards:
                scored.to_csv(
                    run_dir / "operational_hour_edge_rewards.csv.gz",
                    index=False,
                    encoding="utf-8-sig",
                    compression="gzip",
                )

            rewards = base.rewards_by_hour(scored, baseline_hours)
            _, train_results = simulator.evaluate(
                rewards,
                train_dates,
                stop_on_infeasible=False,
            )
            _, test_results = simulator.evaluate(
                rewards,
                test_dates,
                stop_on_infeasible=False,
            )
            daily = results_to_daily_frame(train_results, test_results)
            daily.to_csv(
                run_dir / "daily_policy_results.csv",
                index=False,
                encoding="utf-8-sig",
            )
            summary = summarize_daily_policy(daily, budget)
            summary.to_csv(
                run_dir / "scenario_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )

            selected_count = int(scored["selected_for_bounty"].sum())
            eligible_count = int(scored["eligible"].sum())
            learned = {
                "created_at": pd.Timestamp.now().isoformat(),
                "script_version": SCRIPT_VERSION,
                "policy_active": bool(policy_active),
                "selected_policy": (
                    "RELIEF_SELECTIVE_SIGMOID_BOUNTY"
                    if policy_active
                    else "NO_BOUNTY"
                ),
                "theta": asdict(theta) if theta is not None else None,
                "budget_per_day": float(budget),
                "relief_metric": metric,
                "relief_transform": transform,
                "eligible_edges": eligible_count,
                "selected_edges": selected_count,
                "realized_selected_share_of_eligible": (
                    selected_count / max(eligible_count, 1)
                ),
                "payment_rule": (
                    "all passengers traversing each rewarded edge after policy "
                    "receive that edge bounty"
                ),
                "capacity_constraint_enforced": False,
            }
            write_json(run_dir / "learned_bounty_function.json", learned)
            write_json(run_dir / "policy_search_metadata.json", search_meta)

            reward_dist = pd.read_csv(
                run_dir / "reward_distribution.csv",
                low_memory=False,
            )
            params = reward_parameter_row(
                budget,
                learned,
                search_meta,
                reward_dist,
                run_dir,
            )

            del simulator, scored, rewards, train_results, test_results
            gc.collect()

        summary = summary.copy()
        summary.insert(0, "budget_run", budget_slug(budget))
        summary.insert(1, "source_output", str(run_dir))
        all_summary.append(summary)

        policy_daily = daily.copy()
        if "scenario" in policy_daily.columns:
            policy_daily = policy_daily[
                policy_daily["scenario"] == "PUBLIC_EDGE_BOUNTY"
            ].copy()
        policy_daily.insert(0, "budget_per_day", float(budget))
        policy_daily.insert(1, "budget_run", budget_slug(budget))
        policy_daily["budget_usage_pct"] = (
            pd.to_numeric(policy_daily["cost"], errors="coerce")
            / float(budget)
            * 100.0
        )
        all_daily.append(policy_daily)
        parameter_rows.append(params)

    summary_all = pd.concat(all_summary, ignore_index=True)
    daily_all = pd.concat(all_daily, ignore_index=True)
    parameters_all = pd.DataFrame(parameter_rows).sort_values("budget_per_day")

    summary_all.to_csv(
        output / "budget_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily_all.to_csv(
        output / "budget_sensitivity_daily.csv",
        index=False,
        encoding="utf-8-sig",
    )
    parameters_all.to_csv(
        output / "budget_sensitivity_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test = (
        summary_all[summary_all["split"].astype(str).str.lower() == "test"]
        .sort_values("budget_per_day")
        .reset_index(drop=True)
    )
    train = (
        summary_all[summary_all["split"].astype(str).str.lower() == "train"]
        .sort_values("budget_per_day")
        .reset_index(drop=True)
    )

    test.to_csv(
        output / "budget_sensitivity_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    train.to_csv(
        output / "budget_sensitivity_train.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_korean_test_table(
        test,
        output / "budget_sensitivity_test_korean.csv",
    )

    marginal = make_marginal_table(test)
    marginal.to_csv(
        output / "budget_sensitivity_marginal_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    font_prop = setup_korean_font()
    plot_results(test, figures, font_prop)

    if len(marginal.dropna(subset=["marginal_relief_per_additional_actual_1000_won"])):
        plot = marginal.dropna(
            subset=["marginal_relief_per_additional_actual_1000_won"]
        )
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.plot(
            plot["budget_per_day"] / 1_000_000.0,
            plot["marginal_relief_per_additional_actual_1000_won"],
            marker="o",
        )
        ax.set_xlabel("상향 조정된 일일 예산 (백만원)", fontproperties=font_prop)
        ax.set_ylabel(
            "추가 실제지출 1,000원당 추가 혼잡 완화량",
            fontproperties=font_prop,
        )
        ax.set_title("예산 증가의 한계 비용효율", fontproperties=font_prop)
        ax.grid(axis="both", linestyle=":", alpha=0.35)
        save_plot(
            fig,
            figures / "fig_06_marginal_efficiency.png",
            figures / "fig_06_marginal_efficiency.svg",
        )

    write_summary_text(
        test,
        output / "budget_sensitivity_summary_ko.txt",
    )
    write_json(
        output / "budget_sensitivity_manifest.json",
        {
            "created_at": pd.Timestamp.now().isoformat(),
            "script_version": SCRIPT_VERSION,
            "budgets": budgets,
            "policy_trials_per_budget": int(args.policy_trials),
            "seed": int(args.seed),
            "model_input": str(model_input),
            "route_choice_output": str(route_choice_output),
            "cache_source": str(cache_source),
            "reference_relief_output": str(reference_relief_output),
            "relief_metric": metric,
            "relief_transform": transform,
            "train_dates": train_dates,
            "test_dates": test_dates,
            "method_note": (
                "각 예산마다 RP 현상금 파라미터를 독립 재최적화하되, "
                "경로선택모형·topology/daily cache·ReliefPotential feature는 재사용함."
            ),
        },
    )

    elapsed = time.perf_counter() - started
    print("\n완료:", output, flush=True)
    print(" -", output / "budget_sensitivity_test_korean.csv")
    print(" -", output / "budget_sensitivity_summary.csv")
    print(" -", output / "budget_sensitivity_marginal_test.csv")
    print(" -", figures)
    print(f"총 실행시간: {elapsed/60.0:.1f}분", flush=True)


if __name__ == "__main__":
    main()
