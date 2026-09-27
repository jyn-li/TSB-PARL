"""Joint particle filter for annual demand distributions and the persistent substitution matrix.

Uses a Rao-Blackwellized particle approximation: each particle carries a full set of
annual demand parameters (mean, distribution type, parameters) and a substitution
probability row for each product.  Particles are weighted via a censored
aggregate-demand likelihood and resampled when effective sample size drops.
"""

from __future__ import annotations

import math

import numpy as np

from utils import (
    DIST_POISSON,
    DIST_UNIFORM,
    DIST_NB,
    action_names,
    normal_cdf,
    positive_part_moments,
    systematic_resample,
)


class JointParticleBelief:
    """Joint annual-demand/persistent-substitution particle posterior.

    Each particle i stores:
      - mu[i, k]:         mean base demand for product k in the current year
      - dist_types[i, k]: distribution family (Poisson/Uniform/NB)
      - dist_params[i, k]: extra parameter (half-width or dispersion)
      - a[i, k, :]:       substitution probability vector (K targets + EXIT)
    """

    def __init__(
        self,
        k_count: int,
        num_particles: int,
        seed: int,
        likelihood_temperature: float = 0.45,
        substitution_prior: str = "symmetric_sparse",
    ):
        self.k_count = k_count
        self.action_names = action_names(k_count)
        self.n_actions = len(self.action_names)
        self.n = int(num_particles)
        self.rng = np.random.default_rng(seed)
        self.temperature = float(likelihood_temperature)
        self.substitution_prior = str(substitution_prior)
        if self.substitution_prior not in {
            "symmetric_sparse", "exit_conservative"
        }:
            raise ValueError(
                "substitution_prior must be symmetric_sparse or exit_conservative"
            )
        self.weights = np.full(self.n, 1.0 / self.n)
        self.current_year = -1
        self.total_updates = 0
        self.mu = np.zeros((self.n, k_count), dtype=float)
        self.dist_types = np.zeros((self.n, k_count), dtype=np.int8)
        self.dist_params = np.zeros((self.n, k_count), dtype=float)
        self.a = np.zeros((self.n, k_count, k_count + 1), dtype=float)
        self._init_substitution_prior()
        self.route_counts = np.zeros((k_count, k_count + 1), dtype=float)
        self.directed_rows = np.zeros(k_count, dtype=bool)
        self.explicit_directed_updates = np.zeros(k_count, dtype=int)
        for i in range(k_count):
            valid = [j for j in range(k_count + 1) if j != i]
            self.route_counts[i, valid] = 0.25

    def _init_substitution_prior(self) -> None:
        """Conservative prior: most stockout customers leave until data say otherwise."""
        for p in range(self.n):
            for i in range(self.k_count):
                valid = [j for j in range(self.k_count + 1) if j != i]
                if self.substitution_prior == "symmetric_sparse":
                    # Exchangeable and U-shaped: no destination is privileged,
                    # while particles may represent one dominant route.
                    alpha = np.full(len(valid), 0.5, dtype=float)
                else:
                    alpha = np.ones(len(valid), dtype=float)
                    alpha[-1] = 3.0  # EXIT is the final valid destination
                self.a[p, i, valid] = self.rng.dirichlet(alpha)

    def start_year(self, year: int) -> None:
        """Propagate demand particles to a new annual block."""
        if year == self.current_year:
            return
        if self.current_year < 0:
            self.mu = self.rng.lognormal(
                mean=math.log(150.0), sigma=0.32, size=(self.n, self.k_count)
            )
            self.dist_types = self.rng.integers(
                0, 3, size=(self.n, self.k_count), dtype=np.int8
            )
        else:
            self.mu *= self.rng.lognormal(
                mean=0.0, sigma=0.24, size=(self.n, self.k_count)
            )
            mutate = self.rng.random((self.n, self.k_count)) < 0.25
            self.dist_types[mutate] = self.rng.integers(
                0, 3, size=int(mutate.sum()), dtype=np.int8
            )
        self.mu = np.clip(self.mu, 35.0, 320.0)
        uniform = self.dist_types == DIST_UNIFORM
        nb = self.dist_types == DIST_NB
        self.dist_params[uniform] = self.rng.uniform(10.0, 35.0, size=int(uniform.sum()))
        self.dist_params[nb] = self.rng.uniform(120.0, 400.0, size=int(nb.sum()))
        self.dist_params[self.dist_types == DIST_POISSON] = 0.0
        self.current_year = int(year)

    def base_variance(self) -> np.ndarray:
        """Per-particle variance of base demand (before substitution)."""
        var = self.mu.copy()
        uniform = self.dist_types == DIST_UNIFORM
        width = self.dist_params
        var[uniform] = width[uniform] * (width[uniform] + 1.0) / 3.0
        nb = self.dist_types == DIST_NB
        disp = np.maximum(self.dist_params[nb], 5.0)
        var[nb] = self.mu[nb] + self.mu[nb] ** 2 / disp
        return np.maximum(var, 4.0)

    def particle_aggregate_moments(
        self, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-particle mean and variance of aggregate demand after substitution."""
        y2 = np.broadcast_to(np.asarray(y, dtype=float), self.mu.shape)
        base_var = self.base_variance()
        overflow_mean, overflow_var = positive_part_moments(self.mu, base_var, y2)
        a_products = self.a[:, :, : self.k_count]
        agg_mean = self.mu + np.einsum("ni,nij->nj", overflow_mean, a_products)
        flow_var = (
            a_products * (1.0 - a_products) * overflow_mean[:, :, None]
            + a_products**2 * overflow_var[:, :, None]
        )
        agg_var = base_var + flow_var.sum(axis=1)
        return agg_mean, np.maximum(agg_var, 4.0)

    def update(
        self, y: np.ndarray, sales: np.ndarray, action: int | None = None
    ) -> dict[str, float]:
        """Weight particles by censored aggregate-demand likelihood."""
        agg_mean, agg_var = self.particle_aggregate_moments(y)
        sigma = np.sqrt(agg_var)
        sales = np.asarray(sales, dtype=float)
        y = np.asarray(y, dtype=float)
        sold = sales >= y - 1e-9
        loglik = np.zeros(self.n, dtype=float)
        for j in range(self.k_count):
            if sold[j]:
                z = (y[j] - 0.5 - agg_mean[:, j]) / sigma[:, j]
                sf = np.maximum(1.0 - normal_cdf(z), 1e-12)
                loglik += np.log(sf)
            else:
                z = (sales[j] - agg_mean[:, j]) / sigma[:, j]
                loglik += -0.5 * z * z - np.log(sigma[:, j])
        logw = np.log(np.maximum(self.weights, 1e-300)) + self.temperature * loglik
        logw -= np.max(logw)
        w = np.exp(logw)
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0.0:
            self.weights.fill(1.0 / self.n)
        else:
            self.weights = w / total
        ess = 1.0 / float(np.sum(self.weights**2))
        resampled = False
        if ess < 0.45 * self.n:
            idx = systematic_resample(self.weights, self.rng)
            self.mu = self.mu[idx].copy()
            self.dist_types = self.dist_types[idx].copy()
            self.dist_params = self.dist_params[idx].copy()
            self.a = self.a[idx].copy()
            self.weights.fill(1.0 / self.n)
            self._liu_west_rejuvenate()
            resampled = True
        self._reanchor_directed_rows()
        directed_update = self._directed_route_update(y, sales, sold, action=action)
        self.total_updates += 1
        return {
            "ess": ess,
            "resampled": float(resampled),
            "directed_route_update": float(directed_update),
        }

    def _directed_route_update(
        self,
        y: np.ndarray,
        sales: np.ndarray,
        sold: np.ndarray,
        action: int | None = None,
    ) -> bool:
        """Use clean single-source exposures to sharpen a substitution row.

        When exactly one product stocks out and at least one receiver remains
        uncensored, excess receiver sales contain directional routing evidence.
        The update uses posterior demand predictions and public sales only; latent
        demand and simulator flows remain inaccessible.
        """
        summary = self.summaries()
        mu = summary["mu_mean"]
        var = summary["predictive_var"]
        overflow, _ = positive_part_moments(
            mu[None, :], var[None, :], np.asarray(y, dtype=float)[None, :]
        )
        overflow = overflow[0]
        if action is not None and 2 <= int(action) < self.n_actions:
            source = int(action) - 2
            if not bool(sold[source]) or overflow[source] < 4.0:
                return False
        else:
            candidates = [
                i for i in range(self.k_count)
                if bool(sold[i]) and overflow[i] >= 4.0
            ]
            if len(candidates) != 1:
                return False
            source = candidates[0]
            if self.directed_rows[source]:
                return False
        receivers = [j for j in range(self.k_count) if j != source]
        open_receivers = [j for j in receivers if not bool(sold[j])]
        if len(open_receivers) != len(receivers):
            return False

        evidence = np.zeros(self.k_count + 1, dtype=float)
        for destination in open_receivers:
            predicted_primary_sales = min(float(mu[destination]), float(y[destination]))
            evidence[destination] = max(
                float(sales[destination]) - predicted_primary_sales, 0.0
            )
        attributed = min(float(evidence.sum()), float(overflow[source]))
        evidence[-1] = max(float(overflow[source]) - attributed, 0.0)
        if float(evidence.sum()) < 2.0:
            return False

        valid = [j for j in range(self.k_count + 1) if j != source]
        # Designed exposures carry more identifying information than incidental
        # stockouts, while both updates use public sales and the current posterior.
        explicit_probe = action is not None and 2 <= int(action) < self.n_actions
        evidence_weight = 1.00 if explicit_probe else 0.12
        self.route_counts[source, valid] += evidence_weight * evidence[valid]
        count_mean = self.route_counts[source, valid]
        count_mean /= count_mean.sum()
        if explicit_probe:
            # The intervention identifies one substitution row.  Represent its
            # empirical Dirichlet posterior directly instead of letting the
            # generic censored likelihood immediately dilute that evidence.
            alpha = np.maximum(self.route_counts[source, valid], 0.25)
            rows = self.rng.dirichlet(alpha, size=self.n)
            self.a[:, source, :] = 0.0
            self.a[:, source, valid] = rows
            self.directed_rows[source] = True
            self.explicit_directed_updates[source] += 1
            return True
        max_blend = 0.14
        blend = min(max_blend, 0.05 + 0.0040 * float(evidence.sum()))
        rows = self.a[:, source, valid]
        rows = (1.0 - blend) * rows + blend * count_mean[None, :]
        rows = np.maximum(rows, 1e-5)
        rows /= rows.sum(axis=1, keepdims=True)
        self.a[:, source, :] = 0.0
        self.a[:, source, valid] = rows
        return True

    def _reanchor_directed_rows(self) -> None:
        """Protect intervention-identified rows from confounded passive updates."""
        for source in np.flatnonzero(self.directed_rows):
            valid = [j for j in range(self.k_count + 1) if j != source]
            target = self.route_counts[source, valid].copy()
            target /= target.sum()
            rows = self.a[:, source, valid]
            rows = 0.15 * rows + 0.85 * target[None, :]
            rows = np.maximum(rows, 1e-6)
            rows /= rows.sum(axis=1, keepdims=True)
            self.a[:, source, :] = 0.0
            self.a[:, source, valid] = rows

    def _liu_west_rejuvenate(self) -> None:
        """Liu-West shrinkage rejuvenation to prevent particle depletion."""
        shrink = 0.985
        noise_scale = math.sqrt(max(1.0 - shrink * shrink, 1e-6))
        mean_mu = self.mu.mean(axis=0)
        sd_mu = np.maximum(self.mu.std(axis=0), 1.0)
        self.mu = (
            shrink * self.mu
            + (1.0 - shrink) * mean_mu
            + self.rng.normal(size=self.mu.shape) * noise_scale * sd_mu
        )
        self.mu = np.clip(self.mu, 30.0, 330.0)
        for i in range(self.k_count):
            valid = [j for j in range(self.k_count + 1) if j != i]
            rows = self.a[:, i, valid]
            mean_row = rows.mean(axis=0)
            sd_row = np.maximum(rows.std(axis=0), 0.003)
            rows = (
                shrink * rows
                + (1.0 - shrink) * mean_row
                + self.rng.normal(size=rows.shape) * noise_scale * sd_row
            )
            rows = np.maximum(rows, 1e-4)
            rows /= rows.sum(axis=1, keepdims=True)
            self.a[:, i, :] = 0.0
            self.a[:, i, valid] = rows

    def summaries(self) -> dict[str, np.ndarray]:
        """Posterior summaries weighted by particle importance."""
        w = self.weights
        mu_mean = np.einsum("n,nk->k", w, self.mu)
        mu_var_ep = np.einsum("n,nk->k", w, (self.mu - mu_mean) ** 2)
        base_var = self.base_variance()
        predictive_var = np.einsum(
            "n,nk->k", w, base_var + (self.mu - mu_mean) ** 2
        )
        a_mean = np.einsum("n,nij->ij", w, self.a)
        a_var = np.einsum("n,nij->ij", w, (self.a - a_mean) ** 2)
        type_prob = np.zeros((self.k_count, 3), dtype=float)
        for d in range(3):
            type_prob[:, d] = np.einsum(
                "n,nk->k", w, (self.dist_types == d).astype(float)
            )
        return {
            "mu_mean": mu_mean,
            "mu_sd": np.sqrt(np.maximum(mu_var_ep, 1e-9)),
            "predictive_var": np.maximum(predictive_var, 4.0),
            "a_mean": a_mean,
            "a_sd": np.sqrt(np.maximum(a_var, 1e-9)),
            "type_prob": type_prob,
        }

    def collapsed_aggregate_moments(
        self, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Weighted aggregate moments collapsed across particles."""
        pm, pv = self.particle_aggregate_moments(y)
        w = self.weights
        mean = np.einsum("n,nk->k", w, pm)
        var = np.einsum("n,nk->k", w, pv + (pm - mean) ** 2)
        return mean, np.maximum(var, 4.0)

    def collapsed_final_lost_mean(self, y: np.ndarray) -> np.ndarray:
        """Posterior predictive mean final lost sales under target ``y``.

        This is a decision-time quantity: it uses only the pre-observation particle
        posterior, never the simulator's latent demand or realized substitution flow.
        """
        y2 = np.broadcast_to(np.asarray(y, dtype=float), self.mu.shape)
        base_var = self.base_variance()
        overflow, _ = positive_part_moments(self.mu, base_var, y2)
        aggregate_mean, aggregate_var = self.particle_aggregate_moments(y)
        aggregate_overflow, _ = positive_part_moments(
            aggregate_mean, aggregate_var, y2
        )
        total_sales = np.maximum(aggregate_mean - aggregate_overflow, 0.0)
        primary_sales = np.maximum(self.mu - overflow, 0.0)
        substitution_sales = np.maximum(total_sales - primary_sales, 0.0)
        inbound = np.einsum(
            "ni,nij->nj", overflow, self.a[:, :, : self.k_count]
        )
        direct_exit = overflow * self.a[:, :, -1]
        particle_final_lost = np.maximum(
            direct_exit + inbound - substitution_sales, 0.0
        )
        return np.einsum("n,nk->k", self.weights, particle_final_lost)

    def sample_model(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw a single model (mu, variance, a) from the posterior."""
        idx = int(self.rng.choice(self.n, p=self.weights))
        return self.mu[idx].copy(), self.base_variance()[idx].copy(), self.a[idx].copy()

    def uncertainty_potential(self) -> float:
        """Scalar exploration potential based on remaining posterior uncertainty."""
        s = self.summaries()
        valid_a = []
        for i in range(self.k_count):
            valid_a.extend([s["a_sd"][i, j] for j in range(self.k_count + 1) if j != i])
        demand_unc = float(np.mean(np.minimum(s["mu_sd"] / 60.0, 2.0)))
        subst_unc = float(np.mean(np.minimum(np.asarray(valid_a) / 0.20, 2.0)))
        # Potential-based shaping improves credit assignment without changing
        # the undiscounted finite-horizon objective (the terminal potential is
        # fixed at zero in run_experiment.py).
        return -0.50 * demand_unc - 3.00 * subst_unc

    def state_vector(
        self,
        x: np.ndarray,
        capacity: int,
        year: int,
        day_in_year: int,
        days_per_year: int,
        period: int,
        horizon: int,
        last_sold_out: np.ndarray,
        demand_exposure: int,
        row_exposure: np.ndarray,
        last_action: int,
        active_source: int,
        option_age: int,
    ) -> np.ndarray:
        """Build the full RL state vector from posterior summaries + context."""
        s = self.summaries()
        valid_a_mean: list[float] = []
        valid_a_sd: list[float] = []
        for i in range(self.k_count):
            for j in range(self.k_count + 1):
                if j != i:
                    valid_a_mean.append(float(s["a_mean"][i, j]))
                    valid_a_sd.append(float(s["a_sd"][i, j]))
        last_onehot = np.zeros(self.n_actions, dtype=float)
        if 0 <= last_action < self.n_actions:
            last_onehot[last_action] = 1.0
        active_onehot = np.zeros(self.k_count + 1, dtype=float)
        active_onehot[active_source + 1 if active_source >= 0 else 0] = 1.0
        vec = np.concatenate(
            [
                np.asarray(x, dtype=float) / max(capacity, 1),
                s["mu_mean"] / 250.0,
                np.minimum(s["mu_sd"] / 80.0, 2.0),
                np.asarray(valid_a_mean),
                np.minimum(np.asarray(valid_a_sd) / 0.25, 2.0),
                s["type_prob"].reshape(-1),
                np.array(
                    [
                        year / 4.0,
                        day_in_year / max(days_per_year - 1, 1),
                        (horizon - period) / max(horizon, 1),
                    ]
                ),
                np.asarray(last_sold_out, dtype=float),
                np.array([min(demand_exposure / 15.0, 2.0)]),
                np.minimum(np.asarray(row_exposure, dtype=float) / 8.0, 2.0),
                last_onehot,
                active_onehot,
                np.array([option_age / 3.0]),
            ]
        )
        return vec.astype(np.float32)
