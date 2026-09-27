"""Regression tests for the original algorithm under the current paper model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from project_paths import INPUTS

CORE_DIR = ROOT / "src" / "tsb_parl"
MAIN_CONFIG = (
    INPUTS / "configs"
    / "C6_HIGHER_ROUTE_VALUE_path1_train20269601.json"
)
MAIN_DATA = INPUTS / "data"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from allocator import StructuredAllocator
from belief import JointParticleBelief
from environment import (
    InventoryEnvironment,
    TruthModel,
    load_primary_demand,
    load_substitution_matrix,
    truth_from_config,
)
from utils import ACTION_NAMES, N_ACTIONS, load_config


def test_final_lost_sales_attribution_and_flow_conservation() -> None:
    config = load_config(MAIN_CONFIG)
    base = truth_from_config(config)
    substitution = np.zeros((3, 4), dtype=float)
    substitution[0, 1] = 1.0
    substitution[1, 3] = 1.0
    substitution[2, 3] = 1.0
    truth = TruthModel(
        base.means, base.dist_types, base.dist_params, substitution
    )
    fixed = np.zeros((300, 3), dtype=int)
    fixed[0] = [5, 0, 0]
    env = InventoryEnvironment(config, truth, seed=1, fixed_demand=fixed)
    public, private = env.step(np.array([0, 3, 0]))
    np.testing.assert_array_equal(public["sales"], [0, 3, 0])
    np.testing.assert_array_equal(private["final_lost"], [0, 2, 0])
    price_p2 = float(config["products"][1]["price"])
    shortage_p2 = float(config["products"][1]["shortage_cost"])
    assert private["true_profit"] == 3 * price_p2 - 2 * shortage_p2
    assert int(np.sum(public["sales"]) + np.sum(private["final_lost"])) == 5


def test_policy_observation_does_not_expose_latent_flows() -> None:
    config = load_config(MAIN_CONFIG)
    fixed = np.zeros((300, 3), dtype=int)
    env = InventoryEnvironment(
        config, truth_from_config(config), seed=3, fixed_demand=fixed
    )
    public, _ = env.step(np.zeros(3, dtype=int))
    assert set(public) == {
        "period", "year", "sales", "leftover", "sold_out", "revenue", "holding_cost"
    }


def test_current_csv_inputs_are_complete_and_config_ordered() -> None:
    config = load_config(MAIN_CONFIG)
    pids = [item["id"] for item in config["products"]]
    demand = load_primary_demand(
        MAIN_DATA / "C6_HIGHER_ROUTE_VALUE_path1_demand.csv", pids
    )
    substitution = load_substitution_matrix(
        config, MAIN_DATA / "C6_HIGHER_ROUTE_VALUE_path1_substitution.csv"
    )
    assert demand.shape == (300, 3)
    # Regression total for the locked Path 1 demand file used by the paper.
    assert int(demand.sum()) == 137_085
    np.testing.assert_allclose(substitution.sum(axis=1), 1.0)
    np.testing.assert_allclose(np.diag(substitution[:, :3]), 0.0)


def test_all_programmable_actions_respect_carryover_and_capacity() -> None:
    config = load_config(MAIN_CONFIG)
    price = np.array([item["price"] for item in config["products"]], dtype=float)
    hold = np.array([item["holding_cost"] for item in config["products"]], dtype=float)
    lost = np.array([item["shortage_cost"] for item in config["products"]], dtype=float)
    allocator = StructuredAllocator(450, price, hold, lost, grid_units=4)
    belief = JointParticleBelief(3, 32, seed=9)
    belief.start_year(0)
    x = np.array([10, 20, 30], dtype=int)
    for action in range(N_ACTIONS):
        y = allocator.choose(action, x, belief)
        assert np.issubdtype(y.dtype, np.integer)
        assert np.all(y >= x)
        assert int(y.sum()) <= 450


def test_five_action_library_excludes_inactive_options() -> None:
    assert N_ACTIONS == 5
    assert ACTION_NAMES == (
        "exploit", "demand_reveal", "sub_probe_P1", "sub_probe_P2", "sub_probe_P3"
    )


def test_full_fixed_path_preserves_profit_and_customer_identities() -> None:
    config = load_config(MAIN_CONFIG)
    pids = [item["id"] for item in config["products"]]
    demand = load_primary_demand(
        MAIN_DATA / "C6_HIGHER_ROUTE_VALUE_path1_demand.csv", pids
    )
    truth = truth_from_config(
        config, MAIN_DATA / "C6_HIGHER_ROUTE_VALUE_path1_substitution.csv"
    )
    env = InventoryEnvironment(config, truth, seed=17, fixed_demand=demand)
    rng = np.random.default_rng(18)
    for period in range(env.horizon):
        slack = env.capacity - int(env.x.sum())
        addition = rng.multinomial(slack, np.full(3, 1.0 / 3.0))
        y = env.x + addition
        public, private = env.step(y)
        assert int(np.sum(public["sales"]) + np.sum(private["final_lost"])) == int(
            np.sum(private["base_demand"])
        )
        expected = (
            float(np.dot(env.price, public["sales"]))
            - float(np.dot(env.hold, public["leftover"]))
            - float(np.dot(env.short, private["final_lost"]))
        )
        assert private["true_profit"] == expected
        if period + 1 < env.horizon:
            np.testing.assert_array_equal(env.x, public["leftover"])


def test_symmetric_sparse_substitution_prior_is_exchangeable() -> None:
    belief = JointParticleBelief(
        3, 4096, seed=31, substitution_prior="symmetric_sparse"
    )
    for source in range(3):
        valid = [destination for destination in range(4) if destination != source]
        np.testing.assert_allclose(
            belief.a[:, source, valid].mean(axis=0),
            np.full(3, 1.0 / 3.0),
            atol=0.025,
        )
        assert np.all(belief.a[:, source, source] == 0.0)
