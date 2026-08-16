from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "fit_relief_potential_bounty_policy.py"
spec = importlib.util.spec_from_file_location("relief_model", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import model")
model = importlib.util.module_from_spec(spec)
sys.modules["relief_model"] = model
spec.loader.exec_module(model)


@dataclass
class Topology:
    path_edge: csr_matrix
    group_starts: np.ndarray
    group_index: np.ndarray
    groups: int
    paths: int


@dataclass
class State:
    p0: np.ndarray
    group_demand: np.ndarray
    travel_time: np.ndarray
    trips: np.ndarray
    load0: np.ndarray


def grouped_softmax(utility: np.ndarray, starts: np.ndarray, group_index: np.ndarray) -> np.ndarray:
    maxima = np.maximum.reduceat(utility, starts[:-1])
    expv = np.exp(utility - maxima[group_index])
    denom = np.add.reduceat(expv, starts[:-1])
    return expv / denom[group_index]


def numerical_relief(
    topology: Topology,
    state: State,
    beta: float,
    metric: str,
    reference: float,
    edge: int,
    epsilon: float,
) -> float:
    baseline_u = np.log(state.p0)
    reward = np.zeros(topology.path_edge.shape[1], dtype=float)
    reward[edge] = epsilon
    path_discount = np.asarray(topology.path_edge @ reward).reshape(-1)
    p1 = grouped_softmax(
        baseline_u + beta * path_discount,
        topology.group_starts,
        topology.group_index,
    )
    demand_path = state.group_demand[topology.group_index]
    delta_users = np.asarray(
        topology.path_edge.T @ (demand_path * (p1 - state.p0))
    ).reshape(-1)
    load1 = state.load0 + delta_users / state.trips
    h0 = model.metric_value(
        metric, state.travel_time, state.trips, state.load0, reference
    )
    h1 = model.metric_value(
        metric, state.travel_time, state.trips, load1, reference
    )
    return (h0 - h1) / epsilon


def test_analytical_derivative() -> None:
    # Two OD groups and four directed edges.
    # Group 0: paths 0,1,2. Group 1: paths 3,4.
    dense = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ],
        dtype=float,
    )
    path_edge = csr_matrix(dense)
    starts = np.array([0, 3, 5], dtype=np.int64)
    group_index = np.array([0, 0, 0, 1, 1], dtype=np.int32)
    topology = Topology(
        path_edge=path_edge,
        group_starts=starts,
        group_index=group_index,
        groups=2,
        paths=5,
    )
    state = State(
        p0=np.array([0.50, 0.30, 0.20, 0.65, 0.35], dtype=float),
        group_demand=np.array([120.0, 80.0], dtype=float),
        travel_time=np.array([5.0, 8.0, 4.0, 7.0], dtype=float),
        trips=np.array([10.0, 12.0, 9.0, 11.0], dtype=float),
        load0=np.array([60.0, 30.0, 70.0, 20.0], dtype=float),
    )
    beta = 0.009
    reference = 55.0
    epsilon = 1.0e-4

    for metric in model.SUPPORTED_METRICS:
        analytical = model.analytical_relief_for_state(
            topology, state, beta, metric, reference
        )
        numerical = np.array(
            [
                numerical_relief(
                    topology,
                    state,
                    beta,
                    metric,
                    reference,
                    edge,
                    epsilon,
                )
                for edge in range(path_edge.shape[1])
            ]
        )
        abs_error = np.max(np.abs(analytical - numerical))
        scale = max(1.0, float(np.max(np.abs(numerical))))
        rel_error = abs_error / scale
        assert rel_error < 2.0e-5, (
            metric,
            analytical,
            numerical,
            rel_error,
        )
        print(
            f"PASS derivative {metric}: max relative error={rel_error:.3e}"
        )


def test_transforms_and_selection() -> None:
    frame = pd.DataFrame(
        {
            "hour": [7] * 6,
            "route_id": ["R"] * 6,
            "from_stop_id": [str(i) for i in range(6)],
            "to_stop_id": [str(i + 1) for i in range(6)],
            "relief_whole_h_mean_per_won": [-2.0, 0.0, 1.0, 2.0, 4.0, 8.0],
        }
    )
    meta = model.add_relief_transforms(frame, model.METRIC_WHOLE_H)
    assert meta["positive_edges"] == 4
    assert frame.loc[0, "relief_whole_h_positive_rank"] == 0.0
    assert frame.loc[5, "relief_whole_h_positive_rank"] == 1.0

    features = pd.DataFrame(
        {
            "hour": [7] * 6,
            "low_crowding_train_mean": [0.2] * 6,
            "altprob_train_mean": [0.5] * 6,
            "relief_potential_feature": [0.0, 0.0, 0.25, 0.50, 0.75, 1.0],
            "relief_potential_raw_mean_per_won": [-2.0, 0.0, 1.0, 2.0, 4.0, 8.0],
            "relief_potential_positive_date_rate": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
        }
    )
    config = {
        "policy": {
            "minimum_altprob": 0.0,
            "max_edge_reward_won": 500.0,
            "relief_selective": {
                "selection_scope": "global",
                "minimum_selected_edges": 1,
                "require_positive_relief": True,
                "minimum_positive_date_rate": 0.0,
            },
        }
    }
    theta = model.ReliefSelectiveTheta(
        intercept=-5.0,
        low_crowding=0.0,
        altprob=0.0,
        relief_potential=10.0,
        target_share=0.5,
    )
    scored, meta2 = model.score_relief_selective_features(
        features, theta, config
    )
    assert meta2["eligible_edges"] == 4
    assert meta2["selected_edges"] == 2
    selected = np.flatnonzero(scored["selected_for_bounty"].to_numpy() > 0)
    assert selected.tolist() == [4, 5]
    assert meta2["negative_relief_selected_edges"] == 0
    print("PASS transforms and positive-relief top-share selection")


if __name__ == "__main__":
    test_analytical_derivative()
    test_transforms_and_selection()
    print("ALL VALIDATION TESTS PASSED")
