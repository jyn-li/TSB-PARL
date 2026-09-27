"""Demand-only belief used by the no-substitution-learning ablation.

The data-generating environment is unchanged and may contain substitution.  The
decision maker deliberately uses the misspecified model in which every unmet
primary customer exits.  Consequently there is no substitution parameter to
learn and no substitution uncertainty in the exploration potential.
"""

from __future__ import annotations

import numpy as np

from belief import JointParticleBelief


class DemandOnlyBelief(JointParticleBelief):
    """Particle demand belief with a fixed no-cross-substitution matrix."""

    def __init__(self, *args, **kwargs):
        # Keep the constructor signature accepted by the shared experiment driver.
        kwargs.pop("substitution_prior", None)
        super().__init__(*args, substitution_prior="symmetric_sparse", **kwargs)
        self.substitution_prior = "none_all_unmet_demand_exits"
        self._force_no_substitution()

    def _init_substitution_prior(self) -> None:
        self._force_no_substitution()

    def _force_no_substitution(self) -> None:
        self.a.fill(0.0)
        self.a[:, :, -1] = 1.0

    def _liu_west_rejuvenate(self) -> None:
        # Reuse the tested demand rejuvenation, then remove the substitution
        # perturbation performed by the joint-belief implementation.
        super()._liu_west_rejuvenate()
        self._force_no_substitution()

    def _directed_route_update(self, y, sales, sold, action=None) -> bool:
        del y, sales, sold, action
        self._force_no_substitution()
        return False

    def uncertainty_potential(self) -> float:
        """Exploration potential contains demand uncertainty only."""
        summary = self.summaries()
        demand_uncertainty = float(
            np.mean(np.minimum(summary["mu_sd"] / 60.0, 2.0))
        )
        return -0.20 * demand_uncertainty
