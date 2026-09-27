"""Programmable actor that maps each RL option to a concrete inventory vector.

The allocator translates exploit, demand_reveal, and one substitution probe per
source product into capacity-feasible inventory targets.  Three-product runs keep
the original exact simplex grid; larger product sets use a documented finite
candidate approximation.  Both score targets using the public posterior only.
"""

from __future__ import annotations

import math

import numpy as np

from utils import action_names, normal_cdf, positive_part_moments

# forward reference for type hint
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from belief import JointParticleBelief


class StructuredAllocator:
    """Maps RL options to feasible inventory vectors using posterior-guided scoring."""

    def __init__(
        self,
        capacity: int,
        price: np.ndarray,
        hold: np.ndarray,
        short: np.ndarray,
        grid_units: int = 28,
        reveal_profit_weight: float = 0.05,
        probe_profit_weight: float = 0.10,
        information_weight: float = 520.0,
        candidate_budget: int = 256,
        candidate_seed: int = 1729,
        refinement_rounds: int = 2,
    ):
        self.capacity = int(capacity)
        self.price = np.asarray(price, dtype=float)
        self.hold = np.asarray(hold, dtype=float)
        self.short = np.asarray(short, dtype=float)
        self.k_count = len(price)
        self.action_names = action_names(self.k_count)
        self.n_actions = len(self.action_names)
        self.candidate_budget = int(candidate_budget)
        self.candidate_seed = int(candidate_seed)
        self.refinement_rounds = int(refinement_rounds)
        if self.candidate_budget < 64 or self.refinement_rounds < 0:
            raise ValueError("candidate_budget must be >=64 and refinement_rounds >=0")
        self.reveal_profit_weight = float(reveal_profit_weight)
        self.probe_profit_weight = float(probe_profit_weight)
        self.information_weight = float(information_weight)
        self._public_context = None
        self._target_cache = {}
        self._general_cache = {}
        self._profit_cache = {}
        if self.k_count == 3:
            # Preserve the exact candidate ordering and arithmetic of locked runs.
            shares = []
            for i in range(grid_units + 1):
                for j in range(grid_units + 1 - i):
                    shares.append((i, j, grid_units - i - j))
            self.shares = np.asarray(shares, dtype=float) / grid_units
        else:
            # Finite numerical approximation, never a K-dimensional grid search.
            # A private, fixed RNG cannot perturb training or simulator randomness.
            rng = np.random.default_rng(self.candidate_seed)
            share_budget = (self.candidate_budget - 1) // 2
            anchors = [np.ones(self.k_count) / self.k_count, *np.eye(self.k_count)]
            while len(anchors) < share_budget:
                alpha = (0.25, 1.0, 4.0)[len(anchors) % 3]
                anchors.append(rng.dirichlet(np.full(self.k_count, alpha)))
            self.shares = np.asarray(anchors[:share_budget], dtype=float)

    def _use_public_context(self, summary: dict[str, np.ndarray]) -> None:
        """Retain exact repeated calculations only while public inputs agree.

        Byte tuples avoid hash collisions and do not rely on a belief version
        counter: resets, fixed-parameter variants and in-place posterior changes
        all invalidate the cache whenever any relevant summary changes.
        """
        context = (
            *(np.asarray(summary[name]).tobytes() for name in
              ("mu_mean", "predictive_var", "a_mean", "a_sd", "mu_sd")),
            self.price.tobytes(), self.hold.tobytes(), self.short.tobytes(),
            self.capacity, self.reveal_profit_weight, self.probe_profit_weight,
            self.information_weight, self.candidate_budget,
            self.candidate_seed, self.refinement_rounds,
        )
        if context != self._public_context:
            self._public_context = context
            self._target_cache.clear()
            self._general_cache.clear()
            self._profit_cache.clear()

    def numerical_settings(self) -> dict[str, int | str]:
        """Expose the approximation for result provenance and equal-budget checks."""
        return {
            "candidate_method": (
                "legacy_three_product_grid" if self.k_count == 3
                else "fixed_simplex_candidates_with_local_transfers"
            ),
            "candidate_budget": self.candidate_budget,
            "candidate_seed": self.candidate_seed,
            "refinement_rounds": self.refinement_rounds,
            "n_actions": self.n_actions,
        }

    def _general_candidates(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=int)
        slack = max(0, self.capacity - int(x.sum()))
        blocks = []
        for usage in (0.88, 1.0):
            q = np.floor(slack * usage * self.shares).astype(int)
            blocks.append(x[None, :] + q)
        blocks.append(x[None, :])
        candidates = np.vstack(blocks)
        candidates = candidates[candidates.sum(axis=1) <= self.capacity]
        return np.unique(candidates, axis=0)

    def _local_candidates(self, x: np.ndarray, center: np.ndarray, radius: int) -> np.ndarray:
        """Bounded coordinate transfers, additions and removals around one target."""
        rows = [np.asarray(center, dtype=int).copy()]
        for source in range(self.k_count):
            available = int(center[source] - x[source])
            if available > 0:
                shift = min(available, radius)
                reduced = center.copy()
                reduced[source] -= shift
                rows.append(reduced)
                for dest in range(self.k_count):
                    if source != dest:
                        moved = reduced.copy()
                        moved[dest] += shift
                        rows.append(moved)
            slack = self.capacity - int(center.sum())
            if slack > 0:
                added = center.copy()
                added[source] += min(slack, radius)
                rows.append(added)
        candidates = np.unique(np.asarray(rows, dtype=int), axis=0)
        if len(candidates) > self.candidate_budget:
            rng = np.random.default_rng(self.candidate_seed + int(radius))
            selected = np.sort(rng.choice(
                len(candidates), self.candidate_budget - 1, replace=False
            ))
            candidates = np.unique(np.vstack((center, candidates[selected])), axis=0)
        return candidates

    def _scaled_candidates(self, x, mu, base_var, a):
        """Share candidates plus posterior targets and bounded profit improvement.

        All K>3 policies use this same numerical actor and fixed budgets.  The
        decision uses posterior summaries only; it never sees simulated truth.
        """
        candidates = self._general_candidates(x)
        anchors = []
        slack = max(0, self.capacity - int(np.sum(x)))
        for z in (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
            extra = np.maximum(mu + z * np.sqrt(base_var) - x, 0.0)
            if float(extra.sum()) > slack:
                extra *= slack / max(float(extra.sum()), 1e-12)
            anchors.append(x + np.floor(extra).astype(int))
        # Reserve six positions for posterior targets within the same hard budget.
        if len(candidates) + len(anchors) > self.candidate_budget:
            keep = np.linspace(0, len(candidates) - 1,
                               self.candidate_budget - len(anchors), dtype=int)
            candidates = candidates[keep]
        candidates = np.unique(np.vstack((candidates, anchors, x)), axis=0)
        score = self._metrics(candidates, mu, base_var, a)["profit"]
        center = candidates[int(np.argmax(score))]
        for level in range(self.refinement_rounds):
            radius = max(1, int(round(self.capacity / self.k_count * 0.25 / (2 ** level))))
            local = self._local_candidates(x, center, radius)
            local_score = self._metrics(local, mu, base_var, a)["profit"]
            center = local[int(np.argmax(local_score))]
            # Retain every general candidate and each locally improved optimum.
            candidates = np.unique(np.vstack((candidates, center)), axis=0)
        return candidates

    def _probe_candidates(
        self, x: np.ndarray, source: int, exploit_target: np.ndarray
    ) -> np.ndarray:
        """Build local, capacity-neutral probe perturbations around exploitation.

        A probe with a radically different stocking vector confounds information
        acquisition with a new operating policy.  The paper candidate therefore
        withholds only a bounded amount of replenishment from the probed source
        and reallocates the same units among potential receivers.
        """
        x = np.asarray(x, dtype=int)
        exploit_target = np.asarray(exploit_target, dtype=int)
        receivers = [j for j in range(self.k_count) if j != source]
        source_replenishment = max(0, int(exploit_target[source] - x[source]))
        max_shift = min(source_replenishment, max(4, int(0.08 * self.capacity)))
        shifts = np.unique(
            np.clip(
                np.rint(max_shift * np.asarray([0.25, 0.50, 0.75, 1.0])),
                1,
                max(max_shift, 1),
            ).astype(int)
        )
        rows = [exploit_target.copy()]
        for shift in shifts:
            if shift <= 0 or shift > source_replenishment:
                continue
            if self.k_count != 3:
                # Multi-receiver allocations: equal shares, directed vertices and
                # fixed simplex weights, all conserving the withheld stock units.
                receiver_shares = [np.ones(len(receivers)) / len(receivers)]
                receiver_shares.extend(np.eye(len(receivers)))
                rng = np.random.default_rng(self.candidate_seed + source)
                draws = max(0, min(24, self.candidate_budget // 4) - len(receiver_shares))
                receiver_shares.extend(rng.dirichlet(np.ones(len(receivers)), size=draws))
                for weights in receiver_shares:
                    candidate = exploit_target.copy()
                    candidate[source] -= int(shift)
                    amounts = np.floor(int(shift) * weights).astype(int)
                    residual = int(shift) - int(amounts.sum())
                    order = np.argsort(-(int(shift) * weights - amounts), kind="stable")
                    amounts[order[:residual]] += 1
                    candidate[receivers] += amounts
                    rows.append(candidate)
                continue
            for fraction in np.linspace(0.0, 1.0, 13):
                candidate = exploit_target.copy()
                candidate[source] -= int(shift)
                first = int(math.floor(float(shift) * float(fraction)))
                candidate[receivers[0]] += first
                candidate[receivers[1]] += int(shift) - first
                rows.append(candidate)
        candidates = np.asarray(rows, dtype=int)
        candidates = candidates[candidates.sum(axis=1) <= self.capacity]
        candidates = candidates[np.all(candidates >= x[None, :], axis=1)]
        return np.unique(candidates, axis=0)

    def _metrics(
        self,
        candidates: np.ndarray,
        mu: np.ndarray,
        base_var: np.ndarray,
        a: np.ndarray,
    ) -> dict[str, np.ndarray]:
        y = candidates.astype(float)
        mu2 = np.broadcast_to(np.asarray(mu, dtype=float), y.shape)
        var2 = np.broadcast_to(np.asarray(base_var, dtype=float), y.shape)
        overflow_mean, overflow_var = positive_part_moments(mu2, var2, y)
        a_prod = np.asarray(a, dtype=float)[:, : self.k_count]
        agg_mean = mu2 + overflow_mean @ a_prod
        agg_var = var2.copy()
        if self.k_count == 3:
            for source in range(self.k_count):
                for dest in range(self.k_count):
                    prob = a_prod[source, dest]
                    agg_var[:, dest] += (
                        prob * (1.0 - prob) * overflow_mean[:, source]
                        + prob * prob * overflow_var[:, source]
                    )
        else:
            agg_var += overflow_mean @ (a_prod * (1.0 - a_prod))
            agg_var += overflow_var @ (a_prod * a_prod)
        aggregate_overflow, _ = positive_part_moments(agg_mean, agg_var, y)
        sales = np.clip(agg_mean - aggregate_overflow, 0.0, y)
        leftover = np.maximum(y - sales, 0.0)
        primary_sales = np.maximum(mu2 - overflow_mean, 0.0)
        substitution_sales = np.maximum(sales - primary_sales, 0.0)
        inbound_mean = overflow_mean @ a_prod
        direct_exit = overflow_mean * np.asarray(a, dtype=float)[:, -1]
        final_lost = np.maximum(
            direct_exit + inbound_mean - substitution_sales, 0.0
        )
        profit = (
            sales @ self.price
            - leftover @ self.hold
            - final_lost @ self.short
        )
        revenue = sales @ self.price
        z = (y - 0.5 - agg_mean) / np.sqrt(np.maximum(agg_var, 4.0))
        nonstock_prob = np.clip(normal_cdf(z), 0.0, 1.0)
        return {
            "profit": profit,
            "revenue": revenue,
            "sales": sales,
            "final_lost": final_lost,
            "nonstock_prob": nonstock_prob,
        }

    def choose(
        self, action: int, x: np.ndarray, belief: "JointParticleBelief"
    ) -> np.ndarray:
        """Select the best feasible inventory vector for the given action."""
        if not 0 <= int(action) < self.n_actions:
            raise ValueError(f"action must lie in [0, {self.n_actions - 1}]")
        summary = belief.summaries()
        self._use_public_context(summary)
        inventory_key = np.asarray(x, dtype=int).tobytes()
        target_key = (int(action), inventory_key)
        if target_key in self._target_cache:
            return self._target_cache[target_key].copy()
        mu = summary["mu_mean"]
        base_var = summary["predictive_var"]
        a = summary["a_mean"]

        if inventory_key in self._general_cache:
            general_candidates, general_metrics = self._general_cache[inventory_key]
        else:
            general_candidates = (
                self._general_candidates(x) if self.k_count == 3
                else self._scaled_candidates(np.asarray(x, dtype=int), mu, base_var, a)
            )
            general_metrics = self._metrics(general_candidates, mu, base_var, a)
            self._general_cache[inventory_key] = (general_candidates, general_metrics)
        exploit_target = general_candidates[
            int(np.argmax(general_metrics["profit"]))
        ].astype(int)

        if 2 <= action < self.n_actions:
            source = action - 2
            candidates = self._probe_candidates(x, source, exploit_target)
            metrics = self._metrics(candidates, mu, base_var, a)
            receivers = [j for j in range(self.k_count) if j != source]
            protection = np.clip(
                metrics["nonstock_prob"][:, receivers], 1e-4, 1.0
            )
            source_stockout = np.clip(
                1.0 - metrics["nonstock_prob"][:, source], 1e-4, 1.0
            )
            route_weight = np.asarray(a[source, receivers], dtype=float)
            receiver_value = self.price[receivers] + self.short[receivers]
            route_weight *= receiver_value / max(float(np.mean(receiver_value)), 1e-6)
            if float(route_weight.sum()) <= 1e-12:
                route_weight = np.ones(len(receivers), dtype=float)
            route_weight /= route_weight.sum()
            row_sd = summary["a_sd"][source]
            valid_sd = [row_sd[j] for j in range(self.k_count + 1) if j != source]
            uncertainty_scale = np.clip(float(np.mean(valid_sd)) / 0.10, 0.25, 1.75)
            information_quality = (
                1.25 * np.log(source_stockout)
                + np.log(protection) @ route_weight
            )
            score = (
                self.probe_profit_weight * metrics["profit"]
                + self.information_weight * uncertainty_scale * information_quality
            )
        else:
            candidates = general_candidates
            metrics = general_metrics
            if action == 1:  # demand reveal
                reveal = np.clip(metrics["nonstock_prob"], 1e-4, 1.0)
                reveal_weight = (
                    (self.price + self.short) * np.maximum(summary["mu_sd"], 1e-6)
                )
                reveal_weight /= max(float(reveal_weight.sum()), 1e-6)
                score = (
                    self.reveal_profit_weight * metrics["profit"]
                    + self.information_weight
                    * self.k_count
                    * (np.log(reveal) @ reveal_weight)
                )
            else:
                score = metrics["profit"]
        y = candidates[int(np.argmax(score))].astype(int)
        if np.any(y < x) or y.sum() > self.capacity:
            raise AssertionError("programmable actor emitted infeasible action")
        self._target_cache[target_key] = y.copy()
        return y

    def expected_profit(
        self, y: np.ndarray, belief: "JointParticleBelief"
    ) -> float:
        summary = belief.summaries()
        self._use_public_context(summary)
        profit_key = np.asarray(y, dtype=int).tobytes()
        if profit_key in self._profit_cache:
            return self._profit_cache[profit_key]
        metrics = self._metrics(
            np.asarray(y, dtype=int)[None, :],
            summary["mu_mean"],
            summary["predictive_var"],
            summary["a_mean"],
        )
        result = float(metrics["profit"][0])
        self._profit_cache[profit_key] = result
        return result

    def information_bonus(
        self,
        x: np.ndarray,
        belief: "JointParticleBelief",
        remaining_fraction: float,
    ) -> np.ndarray:
        """Posterior optimism bonus for the DQN option values.

        The bonus uses only posterior uncertainty and the modeled one-period
        opportunity cost.  It vanishes as a row is identified or the horizon ends.
        """
        bonus = np.zeros(self.n_actions, dtype=float)
        summary = belief.summaries()
        exploit = self.choose(0, x, belief)
        exploit_profit = self.expected_profit(exploit, belief)
        remaining = max(float(remaining_fraction), 0.0)
        demand_unc = float(np.mean(np.minimum(summary["mu_sd"] / 60.0, 2.0)))
        if demand_unc >= 0.40:
            reveal = self.choose(1, x, belief)
            reveal_cost = max(exploit_profit - self.expected_profit(reveal, belief), 0.0)
            # Demand must be reasonably calibrated before excess receiver sales
            # can be attributed to substitution.  Give an early reveal enough
            # priority to precede route probes when both layers are uncertain.
            bonus[1] = 18.0 * demand_unc * remaining * math.exp(-reveal_cost / 80.0)
        for source in range(self.k_count):
            action = 2 + source
            directed = getattr(belief, "directed_rows", np.zeros(self.k_count))
            if bool(directed[source]):
                continue
            valid = [j for j in range(self.k_count + 1) if j != source]
            uncertainty = float(
                np.mean(np.minimum(summary["a_sd"][source, valid] / 0.20, 2.0))
            )
            probe = self.choose(action, x, belief)
            cost = max(exploit_profit - self.expected_profit(probe, belief), 0.0)
            receivers = [j for j in range(self.k_count) if j != source]
            upside = max(float(np.max(self.price[receivers]) - self.price[source]), 0.0)
            max_upside = max(float(np.max(self.price) - np.min(self.price)), 1e-6)
            bonus[action] = (
                12.0
                * uncertainty
                * remaining
                * (upside / max_upside)
                * math.exp(-cost / 80.0)
            )
        return bonus

    def choose_with_economic_safety(
        self,
        action: int,
        x: np.ndarray,
        belief: "JointParticleBelief",
        remaining_fraction: float,
        forced: bool = False,
    ) -> tuple[int, np.ndarray]:
        """Reject options whose modeled opportunity cost exceeds information value.

        The shield uses the same public posterior available to the actor. It is not
        an oracle and does not inspect demand or realized substitution flows.
        """
        proposed = self.choose(action, x, belief)
        if action == 0:
            return int(action), proposed
        exploit = self.choose(0, x, belief)
        exploit_profit = self.expected_profit(exploit, belief)
        proposed_profit = self.expected_profit(proposed, belief)
        opportunity_cost = exploit_profit - proposed_profit
        inventory_shift = float(np.abs(proposed - exploit).sum())
        summary = belief.summaries()
        if action == 1:
            uncertainty = float(np.mean(np.minimum(summary["mu_sd"] / 60.0, 2.0)))
        elif 2 <= action < self.n_actions:
            source = action - 2
            valid = [j for j in range(self.k_count + 1) if j != source]
            uncertainty = float(
                np.mean(np.minimum(summary["a_sd"][source, valid] / 0.20, 2.0))
            )
            receivers = [j for j in range(self.k_count) if j != source]
            economic_upside = max(
                float(np.max(self.price[receivers]) - self.price[source]), 0.0
            )
            # If no receiver is more valuable than the source, learning this row
            # cannot justify a deliberately distorted stocking vector.  A probe
            # remains admissible only when it is already better under the public
            # posterior mean, in which case it is operational improvement rather
            # than costly information acquisition.
            if economic_upside <= 1e-12 and opportunity_cost >= 0.0:
                return 0, exploit
        if 2 <= action < self.n_actions:
            # A local probe can affect all subsequent allocations.  Its admissible
            # one-period cost therefore scales with uncertainty and remaining life.
            recoverable_cost = (
                25.0
                + 200.0
                * uncertainty
                * max(float(remaining_fraction), 0.05)
            )
        else:
            recoverable_cost = (
                5.0
                + 35.0
                * uncertainty
                * max(float(remaining_fraction), 0.05)
            )
        shift_limit = 0.18 if 2 <= action < self.n_actions else 0.10
        excessive_shift = (
            inventory_shift > shift_limit * self.capacity
            and opportunity_cost > -25.0
        )
        if opportunity_cost > recoverable_cost or excessive_shift:
            return 0, exploit
        return int(action), proposed
