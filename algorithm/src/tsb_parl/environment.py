"""Truth model, demand sampling, and the private inventory simulator."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils import DIST_POISSON, DIST_UNIFORM, DIST_NB


# ── Truth model ──

@dataclass
class TruthModel:
    """Ground-truth demand & substitution parameters (inaccessible to the policy)."""
    means: np.ndarray          # (B, K) annual demand means
    dist_types: np.ndarray     # (B, K) distribution type codes
    dist_params: np.ndarray    # half-width for U, dispersion for NB
    substitution: np.ndarray   # (K, K+1), final column = EXIT probability


def truth_from_config(
    config: dict[str, Any], substitution_path: Path | None = None
) -> TruthModel:
    """Parse the test-environment truth from config.json.

    When ``substitution_path`` is provided, the substitution matrix is loaded from the
    observed ``substitution_matrix.csv`` instead of the config declaration.
    """
    products = config["products"]
    b_count = int(config["calendar"]["num_years"])
    k_count = len(products)
    means = np.zeros((b_count, k_count), dtype=float)
    types = np.zeros((b_count, k_count), dtype=int)
    params = np.zeros((b_count, k_count), dtype=float)
    type_map = {
        "poisson": DIST_POISSON,
        "discrete_uniform": DIST_UNIFORM,
        "negative_binomial": DIST_NB,
    }
    for k, product in enumerate(products):
        for regime in product["yearly_regimes"]:
            b = int(regime["year"]) - 1
            means[b, k] = float(regime["target_mean"])
            types[b, k] = type_map[regime["distribution"]]
            p = regime.get("distribution_parameters", {})
            if types[b, k] == DIST_UNIFORM:
                params[b, k] = float(p.get("half_width", 20.0))
            elif types[b, k] == DIST_NB:
                params[b, k] = float(p.get("dispersion", 200.0))
    pids = [p["id"] for p in products]
    outside = config["substitution"]["outside_option"]
    if substitution_path is not None:
        substitution = load_substitution_matrix(config, substitution_path)
    else:
        matrix = config["substitution"]["matrix"]
        order = pids + [outside]
        substitution = np.array(
            [[float(matrix[pid][dest]) for dest in order] for pid in pids],
            dtype=float,
        )
    return TruthModel(means, types, params, substitution)


def load_primary_demand(path: Path, product_ids: list[str] | tuple[str, ...]) -> np.ndarray:
    """Load per-period primary demand from the long-format CSV.

    Returns an array of shape (T, K) indexed by (period-1, product_index), using the
    explicit product order supplied by the configuration.
    """
    rows: list[tuple[int, str, int]] = []
    pids = list(product_ids)
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pid = r["product_id"]
            if pid not in pids:
                raise ValueError(f"unknown product {pid!r} in {path}")
            period = int(r["period"])
            demand_value = int(round(float(r["primary_demand"])))
            if period < 1:
                raise ValueError(f"invalid period {period} in {path}")
            if demand_value < 0:
                raise ValueError(f"negative demand in period {period}, product {pid}")
            rows.append(
                (period, pid, demand_value)
            )
    if not rows:
        raise ValueError(f"empty demand CSV: {path}")
    k_count = len(pids)
    T = max(p[0] for p in rows)
    demand = np.full((T, k_count), -1, dtype=int)
    pid_index = {pid: i for i, pid in enumerate(pids)}
    for period, pid, d in rows:
        if demand[period - 1, pid_index[pid]] >= 0:
            raise ValueError(f"duplicate demand row: period={period}, product={pid}")
        demand[period - 1, pid_index[pid]] = d
    if np.any(demand < 0):
        period, product = np.argwhere(demand < 0)[0]
        raise ValueError(
            f"missing demand row: period={period + 1}, product={pids[product]}"
        )
    return demand


def load_substitution_matrix(config: dict[str, Any], path: Path) -> np.ndarray:
    """Load the observed substitution matrix from the wide-format CSV.

    The CSV header is ``source, P1, P2, P3, EXIT``; rows are parsed into the
    (K, K+1) substitution array in the product+EXIT order declared by config.
    """
    pids = [p["id"] for p in config["products"]]
    outside = config["substitution"]["outside_option"]
    data: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = ["source", *pids, outside]
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise ValueError(f"invalid substitution CSV header in {path}")
        for row in reader:
            source = row.get("source", "")
            if not source:
                continue
            if source not in pids:
                raise ValueError(f"unknown substitution source {source!r}")
            if source in data:
                raise ValueError(f"duplicate substitution row {source!r}")
            data[source] = {name: float(row[name]) for name in [*pids, outside]}
    order = [*pids, outside]
    try:
        matrix = np.array(
            [[data[source][destination] for destination in order] for source in pids],
            dtype=float,
        )
    except KeyError as error:
        raise ValueError(f"missing substitution row or column: {error}") from error
    if np.any(matrix < 0.0) or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("substitution rows must be nonnegative and sum to one")
    if not np.allclose(np.diag(matrix[:, : len(pids)]), 0.0, atol=1e-12):
        raise ValueError("self-substitution probabilities must be zero")
    declared = config["substitution"]["matrix"]
    expected = np.array(
        [
            [float(declared[source][destination]) for destination in order]
            for source in pids
        ],
        dtype=float,
    )
    if not np.allclose(matrix, expected, atol=5e-5):
        raise ValueError("substitution CSV is inconsistent with config.json")
    return matrix


def randomized_truth(
    config: dict[str, Any], rng: np.random.Generator
) -> TruthModel:
    """Broad meta-training instance that does not reveal the test truth.

    Prices, costs, capacity, and the candidate distribution family are public model
    primitives.  Annual means and substitution directions are sampled from broad
    exchangeable priors rather than perturbing the test instance; this prevents the
    pretrained RL policy from memorizing that P1 happens to substitute strongly to P2.
    """
    base = truth_from_config(config)
    b_count, k_count = base.means.shape
    means = np.zeros_like(base.means)
    means[0] = rng.uniform(80.0, 220.0, size=k_count)
    for b in range(1, b_count):
        if rng.random() < 0.55:
            means[b] = means[b - 1] + rng.normal(0.0, 38.0, size=k_count)
        else:
            means[b] = rng.uniform(80.0, 220.0, size=k_count)
    means = np.clip(means, 55.0, 280.0)
    types = rng.integers(0, 3, size=base.dist_types.shape, dtype=np.int8)
    params = np.zeros_like(base.dist_params)
    for idx in np.ndindex(types.shape):
        if types[idx] == DIST_POISSON:
            params[idx] = 0.0
        elif types[idx] == DIST_UNIFORM:
            params[idx] = rng.uniform(12.0, 32.0)
        else:
            params[idx] = rng.uniform(140.0, 360.0)

    subst = np.zeros_like(base.substitution)
    for i in range(k_count):
        valid = [j for j in range(k_count + 1) if j != i]
        alpha = rng.uniform(0.7, 2.2, size=len(valid))
        subst[i, valid] = rng.dirichlet(alpha)
    return TruthModel(means, types, params, subst)


def sample_base_demand(
    rng: np.random.Generator, mean: float, dist_type: int, param: float,
    legacy_uniform_center_truncation: bool = False,
) -> int:
    """Draw a single base-demand observation from the specified distribution."""
    if dist_type == DIST_POISSON:
        return int(rng.poisson(mean))
    if dist_type == DIST_UNIFORM:
        width = max(1, int(round(param)))
        if legacy_uniform_center_truncation:
            lo = max(0, int(mean) - width)
            hi = int(mean) + width
        else:
            lo = max(0, int(np.floor(mean - width)))
            hi = int(np.ceil(mean + width))
        return int(rng.integers(lo, hi + 1))
    dispersion = max(5.0, float(param))
    success_p = dispersion / (dispersion + mean)
    return int(rng.negative_binomial(dispersion, success_p))


# ── Private simulator ──

class InventoryEnvironment:
    """Private simulator with a public aggregate-sales observation boundary.

    The policy sees only inventory, its own action, aggregate sales, and calendar
    position.  Latent variables (base demand, substitution flows, true profit) remain
    private to this class and are only accessible to the evaluator, not the policy.
    """

    def __init__(
        self,
        config: dict[str, Any],
        truth: TruthModel,
        seed: int,
        fixed_demand: np.ndarray | None = None,
    ):
        self.config = config
        self.truth = truth
        self.rng = np.random.default_rng(seed)
        self.counter_substitution_crn = bool(config.get("evaluation", {}).get("counter_substitution_crn", False))
        self.crn_seed = int(seed)
        self.k_count = len(config["products"])
        self.b_count = int(config["calendar"]["num_years"])
        self.days_per_year = int(config["calendar"]["days_per_year"])
        self.horizon = self.b_count * self.days_per_year
        self.capacity = int(config["simulation"]["shared_capacity"])
        self.legacy_uniform_center_truncation = bool(
            config.get("simulation", {}).get(
                "legacy_uniform_center_truncation", False
            )
        )
        sim = config.get("simulation", {})
        self.price = np.array(
            [float(p.get("price", sim.get("price", 5.0))) for p in config["products"]]
        )
        self.hold = np.array(
            [float(p.get("holding_cost", sim.get("holding_cost", 1.0)))
             for p in config["products"]]
        )
        self.short = np.array(
            [float(p.get("shortage_cost", sim.get("shortage_cost", 3.0)))
             for p in config["products"]]
        )
        self.t = 0
        initial = config["simulation"].get("initial_inventory")
        self.x = (
            np.zeros(self.k_count, dtype=int)
            if initial is None
            else np.asarray(initial, dtype=int).copy()
        )
        if (
            self.price.shape != (self.k_count,)
            or self.hold.shape != (self.k_count,)
            or self.short.shape != (self.k_count,)
            or np.any(self.price < 0)
            or np.any(self.hold < 0)
            or np.any(self.short < 0)
        ):
            raise ValueError("invalid product economic parameters")
        if truth.substitution.shape != (self.k_count, self.k_count + 1):
            raise ValueError("truth substitution matrix has the wrong shape")
        if np.any(truth.substitution < 0) or not np.allclose(
            truth.substitution.sum(axis=1), 1.0, atol=1e-9
        ):
            raise ValueError("truth substitution rows must sum to one")
        if not np.allclose(
            np.diag(truth.substitution[:, : self.k_count]), 0.0, atol=1e-12
        ):
            raise ValueError("truth self-substitution probabilities must be zero")
        if self.x.shape != (self.k_count,) or np.any(self.x < 0):
            raise ValueError("invalid initial inventory")
        if int(self.x.sum()) > self.capacity:
            raise ValueError("initial inventory violates shared capacity")
        self.fixed_demand = None
        if fixed_demand is not None:
            self.fixed_demand = np.asarray(fixed_demand, dtype=int)
            if self.fixed_demand.shape[0] != self.horizon:
                raise ValueError(
                    f"fixed_demand has {self.fixed_demand.shape[0]} periods, "
                    f"expected horizon {self.horizon}"
                )
            if self.fixed_demand.shape[1] != self.k_count:
                raise ValueError(
                    f"fixed_demand has {self.fixed_demand.shape[1]} products, "
                    f"expected {self.k_count}"
                )

    def step(self, y: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_y = np.asarray(y)
        if not np.all(np.isfinite(raw_y)) or not np.allclose(raw_y, np.round(raw_y)):
            raise ValueError("inventory action must be an integer vector")
        y = np.asarray(np.round(raw_y), dtype=int)
        if y.shape != (self.k_count,):
            raise ValueError("inventory action has wrong shape")
        if np.any(y < self.x) or np.any(y < 0) or int(y.sum()) > self.capacity:
            raise ValueError(f"infeasible inventory action x={self.x}, y={y}")
        if self.t >= self.horizon:
            raise RuntimeError("episode already complete")

        year = self.t // self.days_per_year
        if self.fixed_demand is not None:
            d0 = self.fixed_demand[self.t].astype(int).copy()
        else:
            d0 = np.array(
                [
                    sample_base_demand(
                        self.rng,
                        self.truth.means[year, k],
                        int(self.truth.dist_types[year, k]),
                        self.truth.dist_params[year, k],
                        self.legacy_uniform_center_truncation,
                    )
                    for k in range(self.k_count)
                ],
                dtype=int,
            )
        primary_short = np.maximum(d0 - y, 0)
        flows = np.zeros((self.k_count, self.k_count + 1), dtype=int)
        for i in range(self.k_count):
            if primary_short[i] > 0:
                if self.counter_substitution_crn:
                    # Per-customer uniforms are shared across policies regardless
                    # of action-dependent stockout counts; old runs keep the
                    # historical sequential generator when this flag is absent.
                    route_rng = np.random.default_rng(np.random.SeedSequence(
                        [self.crn_seed, self.t, i, 9142026]))
                    cumulative = np.cumsum(self.truth.substitution[i])
                    cumulative[-1] = 1.0
                    destinations = np.searchsorted(cumulative,
                        route_rng.random(int(primary_short[i])), side="right")
                    flows[i] = np.bincount(destinations, minlength=self.k_count + 1)
                else:
                    flows[i] = self.rng.multinomial(
                        int(primary_short[i]), self.truth.substitution[i]
                    )
        inbound = flows[:, : self.k_count].sum(axis=0)
        residual = np.maximum(y - d0, 0)
        primary_sales = np.minimum(d0, y)
        substitution_sales = np.minimum(inbound, residual)
        sales = primary_sales + substitution_sales
        leftover = y - sales
        # Final lost sales are attributed to the product whose demand leaves the
        # system: direct exits retain their source-product label, while an inbound
        # substitution that cannot be served is charged to its target product.
        final_lost = flows[:, -1] + inbound - substitution_sales
        if int(final_lost.sum() + sales.sum()) != int(d0.sum()):
            raise RuntimeError("customer-flow conservation failed")
        revenue = float(np.dot(self.price, sales))
        holding = float(np.dot(self.hold, leftover))
        lost_sales_cost = float(np.dot(self.short, final_lost))
        profit = revenue - holding - lost_sales_cost

        public = {
            "period": self.t + 1,
            "year": year + 1,
            "sales": sales.copy(),
            "leftover": leftover.copy(),
            "sold_out": (sales == y),
            "revenue": revenue,
            "holding_cost": holding,
        }
        private = {
            "base_demand": d0,
            "flows": flows,
            "inbound_substitution": inbound,
            "primary_sales": primary_sales,
            "substitution_sales": substitution_sales,
            "final_lost": final_lost,
            "lost_sales_cost": lost_sales_cost,
            "true_profit": profit,
        }
        self.x = leftover.astype(int)
        self.t += 1
        return public, private
