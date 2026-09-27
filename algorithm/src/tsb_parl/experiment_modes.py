"""Controlled policy/belief variants used by the paper experiment suite."""

from __future__ import annotations

import numpy as np

from belief import JointParticleBelief


BELIEF_MODES = {"full", "frozen_a", "true_a", "reset_all", "no_reset"}
POLICY_MODES = {
    "full",
    "exploit_only",
    "fixed_schedule",
    "uncertainty_rule",
    "no_online_dqn",
}


def make_belief(
    k_count, particles, seed, prior, mode, truth_substitution=None,
    belief_cls=JointParticleBelief,
):
    belief = belief_cls(
        k_count, particles, seed, substitution_prior=prior
    )
    if mode == "true_a":
        if truth_substitution is None:
            raise ValueError("true_a requires the true substitution matrix")
        belief.a[...] = np.asarray(truth_substitution, dtype=float)[None, :, :]
        belief._fixed_a = np.asarray(truth_substitution, dtype=float).copy()
    elif mode == "frozen_a":
        frozen = belief.a.mean(axis=0)
        belief.a[...] = frozen[None, :, :]
        belief._fixed_a = frozen.copy()
    else:
        belief._fixed_a = None
    return belief


def update_belief(belief, y, sales, action=None):
    result = belief.update(y, sales, action=action)
    fixed = getattr(belief, "_fixed_a", None)
    if fixed is not None:
        belief.a[...] = fixed[None, :, :]
    return result


def scheduled_action(
    day, k_count, template="legacy", budget_limit=None, block_length=60
):
    """Return a transparent, calendar-only exploration action.

    ``legacy`` preserves the schedule used by the original paper experiments.
    The budgeted templates spread a fixed number of exploration opportunities
    across the block; they never inspect inventory, sales, or the posterior.
    """
    if template == "legacy" or budget_limit is None:
        if day == 0:
            return 1
        if k_count == 3:
            probe_days = {5: 0, 20: 1, 35: 2}
        else:
            # Same public calendar-only rule, extended to every source product.
            event_days = np.linspace(5, max(5, block_length - 10), k_count)
            probe_days = {int(day): source for source, day in enumerate(event_days)}
        if day in probe_days and probe_days[day] < k_count:
            return 2 + probe_days[day]
        return 0

    limit = max(0, min(int(budget_limit), int(block_length)))
    if limit == 0:
        return 0
    if template == "front_loaded":
        event_days = np.arange(limit, dtype=int)
    elif template == "spread":
        event_days = np.floor(
            np.arange(limit, dtype=float) * block_length / limit
        ).astype(int)
    else:
        raise ValueError(f"unknown fixed exploration template: {template}")
    matches = np.flatnonzero(event_days == int(day))
    if len(matches) == 0:
        return 0
    event_index = int(matches[0])
    cycle = [1, *[2 + i for i in range(k_count)]]
    return int(cycle[event_index % len(cycle)])


def apply_exploration_budget(
    mask, used, limit, remaining_periods=None, exact=False
):
    """Apply a cap, or an exact quota, to explicit information actions."""
    bounded = np.asarray(mask, dtype=bool).copy()
    if limit is not None and int(used) >= int(limit):
        bounded[1:] = False
        # An active multi-period probe may have produced an explicit-only mask.
        # Once it is cancelled, exploitation is valid.
        bounded[0] = True
    elif (
        exact
        and limit is not None
        and remaining_periods is not None
        and int(remaining_periods) <= int(limit) - int(used)
    ):
        bounded[0] = False
        # If uncertainty masks removed every information action, demand reveal
        # is the least structurally disruptive way to satisfy an exact quota.
        if not bool(np.any(bounded[1:])):
            bounded[1] = True
    return bounded


def rule_action(belief):
    """Interpretable uncertainty-threshold selector."""
    summary = belief.summaries()
    if float(np.mean(summary["mu_sd"])) > 28.0:
        return 1
    row_uncertainty = np.mean(summary["a_sd"], axis=1)
    source = int(np.argmax(row_uncertainty))
    if float(row_uncertainty[source]) > 0.10:
        return 2 + source
    return 0


def choose_action(
    mode, agent, state, mask, epsilon, forced, day, belief,
    fixed_template="legacy", budget_limit=None, block_length=60,
    action_bonus=None,
):
    if forced is not None:
        return int(forced)
    if mode == "exploit_only":
        proposed = 0
    elif mode == "fixed_schedule":
        proposed = scheduled_action(
            day,
            belief.k_count,
            template=fixed_template,
            budget_limit=budget_limit,
            block_length=block_length,
        )
    elif mode == "uncertainty_rule":
        proposed = rule_action(belief)
    else:
        return int(agent.select(state, mask, epsilon, action_bonus=action_bonus))
    if 0 <= proposed < len(mask) and mask[proposed]:
        return int(proposed)
    valid = np.flatnonzero(mask)
    return int(valid[0]) if len(valid) else 0
