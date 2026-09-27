"""Train and evaluate TSB-PARL on the current Paper-1 data configuration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAIN_DIR = ROOT / "experiments" / "main"
CORE_DIR = ROOT / "src" / "tsb_parl"
TOOLS_DIR = ROOT / "tools"
for directory in (ROOT, CORE_DIR, TOOLS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from project_paths import INPUTS, BENCHMARK_RESULTS, resolve_path
from utils import action_names, load_config
from environment import (
    InventoryEnvironment,
    randomized_truth,
    truth_from_config,
    load_primary_demand,
    load_substitution_matrix,
)
from belief import JointParticleBelief
from experiment_modes import (
    apply_exploration_budget,
    make_belief,
    update_belief,
    choose_action,
)
from allocator import StructuredAllocator
from rl_agent import (
    DoubleDQNAgent,
    ProbeOptionController,
    posterior_proxy_reward,
    valid_action_mask,
)
try:
    from plot_training_loss import plot_training_loss
    from plot_learning_diagnostics import plot_demand_learning
except (ImportError, OSError):
    plot_training_loss = None
    plot_demand_learning = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matched_benchmark_values(config, demand_path: Path, sub_path: Path):
    """Read matched per-path evidence only after validating inputs and model settings."""
    try:
        path_id = int(config["development_metadata"]["path_id"])
        expected = {
            "demand_sha256": file_sha256(demand_path),
            "substitution_sha256": file_sha256(sub_path),
        }
        summaries = {}
        for method in ("oracle", "upper_bound"):
            summary_path = BENCHMARK_RESULTS / f"path{path_id}" / method / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            recorded = summary["inputs"]
            if any(recorded.get(key) != value for key, value in expected.items()):
                return None, f"Skipped: {method} demand or substitution hash does not match the current input."
            original_config_path = resolve_path(recorded["config"])
            if file_sha256(original_config_path) != recorded["config_sha256"]:
                return None, f"Skipped: the recorded {method} configuration hash cannot be verified."
            original = load_config(original_config_path)
            if int(original["development_metadata"]["path_id"]) != path_id:
                return None, f"Skipped: the recorded {method} path does not match the current path."
            # Training seeds may differ; the underlying inventory model must agree.
            for key in ("calendar", "products", "substitution", "simulation"):
                if original.get(key) != config.get(key):
                    return None, f"Skipped: {method} model field {key!r} differs from the current configuration."
            summaries[method] = summary
        return {
            "approximate_oracle_fixed_csv": float(summaries["oracle"]["fixed_csv_path_policy_evaluation"]["mean_total_profit"]),
            "approximate_pi_fixed_csv": float(summaries["upper_bound"]["approximate_pi_policy_evaluation"]["mean_total_profit"]),
            "certified_pathwise_upper_bound": float(summaries["upper_bound"]["certified_pathwise_relaxation_upper_bound"]),
        }, f"Matched Path {path_id}: demand/substitution hashes, recorded configuration hash, and model settings verified."
    except (OSError, KeyError, TypeError, ValueError) as error:
        return None, f"Skipped: matched benchmark evidence could not be verified ({error})."


def product_params(config):
    sim = config.get("simulation", {})
    price = np.array([float(p.get("price", sim.get("price", 5))) for p in config["products"]])
    hold = np.array(
        [float(p.get("holding_cost", sim.get("holding_cost", 1))) for p in config["products"]]
    )
    short = np.array(
        [float(p.get("shortage_cost", sim.get("shortage_cost", 3))) for p in config["products"]]
    )
    return price, hold, short


def exploration_limit(args, block_length):
    """Translate an optional budget share into an integer per-block limit."""
    if args.exploration_budget_share is None:
        return None
    return int(round(float(args.exploration_budget_share) * int(block_length)))


def budgeted_mask_and_forced(
    mask, option, used, limit, remaining_periods, exact
):
    """Apply the hard explicit-exploration cap and stop over-budget probes."""
    if limit is not None and int(used) >= int(limit):
        option.cancel()
    return (
        apply_exploration_budget(
            mask, used, limit, remaining_periods=remaining_periods, exact=exact
        ),
        option.force_action(),
    )


def state_for(
    belief, env, year, day, period, last_sold, demand_exposure, row_exposure,
    last_action, option,
):
    return belief.state_vector(
        env.x,
        env.capacity,
        year,
        day,
        env.days_per_year,
        period,
        env.horizon,
        last_sold,
        demand_exposure,
        row_exposure,
        last_action,
        option.active_source,
        option.age,
    )


def train(config, args, out_dir: Path):
    seed = int(config["random_seed"]) + 8100
    rng = np.random.default_rng(seed)
    exact_truth = truth_from_config(config)
    price, hold, short = product_params(config)
    k_count = len(price)
    ACTION_NAMES = action_names(k_count)
    N_ACTIONS = len(ACTION_NAMES)
    capacity = int(config["simulation"]["shared_capacity"])
    block_budget = exploration_limit(args, int(config["calendar"]["days_per_year"]))

    probe_belief = JointParticleBelief(
        k_count,
        args.train_particles,
        seed + 1,
        substitution_prior=args.substitution_prior,
    )
    probe_belief.start_year(0)
    dummy_env = InventoryEnvironment(config, exact_truth, seed + 2)
    dummy_option = ProbeOptionController(k_count)
    state_dim = state_for(
        probe_belief, dummy_env, 0, 0, 0, np.zeros(k_count, bool), 0,
        np.zeros(k_count, int), -1, dummy_option,
    ).shape[0]
    agent = DoubleDQNAgent(
        state_dim, seed=seed + 3, hidden=args.hidden, gamma=args.gamma, n_actions=N_ACTIONS
    )
    allocator = StructuredAllocator(
        capacity,
        price,
        hold,
        short,
        grid_units=args.grid_units,
        reveal_profit_weight=args.reveal_profit_weight,
        probe_profit_weight=args.probe_profit_weight,
        information_weight=args.information_weight,
    )

    history = []
    global_step = 0
    start_time = time.time()
    for episode in range(args.train_episodes):
        truth = randomized_truth(config, rng)
        env = InventoryEnvironment(config, truth, seed + 1000 + episode)
        belief_mode = args.mode if args.mode in {"frozen_a", "true_a", "reset_all", "no_reset"} else "full"
        belief = make_belief(
            k_count, args.train_particles, seed + 2000 + episode,
            args.substitution_prior, belief_mode, truth.substitution,
            belief_cls=JointParticleBelief,
        )
        option = ProbeOptionController(k_count)
        last_sold = np.zeros(k_count, dtype=bool)
        last_action = -1
        demand_exposure = 0
        row_exposure = np.zeros(k_count, dtype=int)
        demand_probe_count_year = 0
        sub_probe_starts = np.zeros(k_count, dtype=int)
        explicit_exploration_used = 0
        ep_profit = 0.0
        ep_revenue = 0.0
        ep_actions = Counter()
        losses = []

        for t in range(env.horizon):
            year = t // env.days_per_year
            day = t % env.days_per_year
            if day == 0:
                if args.mode == "reset_all" and t > 0:
                    belief = make_belief(
                        k_count, args.train_particles, seed + 2000 + episode + 10000 * year,
                        args.substitution_prior, "reset_all", truth.substitution,
                        belief_cls=JointParticleBelief,
                    )
                    option = ProbeOptionController(k_count)
                    row_exposure = np.zeros(k_count, dtype=int)
                if args.mode != "no_reset" or belief.current_year < 0:
                    belief.start_year(year)
                demand_exposure = 0
                demand_probe_count_year = 0
                explicit_exploration_used = 0
            state = state_for(
                belief, env, year, day, t, last_sold, demand_exposure,
                row_exposure, last_action, option,
            )
            mask = valid_action_mask(
                belief, option.active_source, demand_probe_count_year, sub_probe_starts
            )
            if args.mode == "no_exploration":
                mask[1:] = False
            mask, forced = budgeted_mask_and_forced(
                mask,
                option,
                explicit_exploration_used,
                block_budget,
                env.days_per_year - day,
                args.exploration_budget_mode == "exact",
            )
            epsilon = max(0.08, 0.95 - 0.91 * global_step / max(args.train_episodes * env.horizon * 0.75, 1))
            policy_mode = args.mode if args.mode in {"exploit_only", "fixed_schedule", "uncertainty_rule"} else "full"
            action = choose_action(
                policy_mode, agent, state, mask, epsilon, forced, day, belief,
                fixed_template=args.fixed_template,
                budget_limit=block_budget,
                block_length=env.days_per_year,
                action_bonus=(
                    allocator.information_bonus(
                        env.x, belief, (env.horizon - t) / max(env.horizon, 1)
                    )
                    if (
                        policy_mode == "full"
                        and args.mode != "no_exploration"
                        and belief.__class__.__name__ != "DemandOnlyBelief"
                        and forced is None
                    )
                    else None
                ),
            )
            action, y = allocator.choose_with_economic_safety(
                action,
                env.x,
                belief,
                remaining_fraction=(env.horizon - t) / max(env.horizon, 1),
                forced=forced is not None,
            )
            if forced is not None and action != forced:
                option.cancel()
            if 1 <= action < N_ACTIONS:
                explicit_exploration_used += 1
            started = option.start_if_probe(action)
            if started:
                sub_probe_starts[action - 2] += 1
            if action == 1:
                demand_probe_count_year += 1
            phi_before = belief.uncertainty_potential()
            public, private = env.step(y)
            update_belief(belief, y, public["sales"], action=action)
            conditional_final_lost = belief.collapsed_final_lost_mean(y)
            proxy = posterior_proxy_reward(public, conditional_final_lost, short)
            sold = np.asarray(public["sold_out"], dtype=bool)
            if np.all(~sold):
                demand_exposure += 1
            for i in range(k_count):
                receivers = [j for j in range(k_count) if j != i]
                if sold[i] and np.all(~sold[receivers]):
                    row_exposure[i] += 1
            option.after_observation(sold)
            next_year = min((t + 1) // env.days_per_year, int(config["calendar"]["num_years"]) - 1)
            next_day = (t + 1) % env.days_per_year
            done = t == env.horizon - 1
            if not done and next_day == 0 and args.mode not in {"reset_all", "no_reset"}:
                belief.start_year(next_year)
                demand_exposure = 0
                demand_probe_count_year = 0
                explicit_exploration_used = 0
            phi_after = 0.0 if done else belief.uncertainty_potential()
            next_state = state_for(
                belief, env, next_year, next_day, t + 1, sold, demand_exposure,
                row_exposure, action, option,
            )
            next_mask = valid_action_mask(
                belief, option.active_source, demand_probe_count_year, sub_probe_starts
            )
            if args.mode == "no_exploration":
                next_mask[1:] = False
            next_mask = apply_exploration_budget(
                next_mask,
                explicit_exploration_used,
                block_budget,
                remaining_periods=env.days_per_year - next_day,
                exact=args.exploration_budget_mode == "exact",
            )
            shaped_reward = proxy / 3000.0 + agent.gamma * phi_after - phi_before
            agent.observe(state, action, shaped_reward, next_state, done, next_mask)
            if global_step % args.learn_every == 0:
                loss = agent.learn(args.batch_size)
                if loss is not None:
                    losses.append(loss)
            ep_profit += private["true_profit"]
            ep_revenue += public["revenue"]
            ep_actions[ACTION_NAMES[action]] += 1
            last_sold = sold
            last_action = action
            global_step += 1

        history.append(
            {
                "episode": episode + 1,
                "profit": ep_profit,
                "revenue": ep_revenue,
                "epsilon_end": epsilon,
                "mean_loss": float(np.mean(losses)) if losses else "",
                **{f"count_{name}": ep_actions[name] for name in ACTION_NAMES},
            }
        )
        if (episode + 1) % max(1, args.train_episodes // 10) == 0:
            recent = history[-min(20, len(history)):]
            print(
                f"[train] episode {episode+1:4d}/{args.train_episodes} "
                f"profit={np.mean([r['profit'] for r in recent]):,.0f} "
                f"revenue={np.mean([r['revenue'] for r in recent]):,.0f} "
                f"eps={epsilon:.3f}"
            )

    with (out_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    agent.online.save(out_dir / "critic_model.npz")
    print(f"[train] completed in {time.time()-start_time:.1f}s")
    return agent, allocator, state_dim


def evaluate(config, args, out_dir: Path, pretrained, allocator, state_dim):
    # Input paths in the configuration are relative to the project root.
    demand_path = resolve_path(config["output"]["demand_long_file"]).resolve()
    sub_path = resolve_path(config["output"]["substitution_wide_file"]).resolve()
    pids = [p["id"] for p in config["products"]]
    fixed_demand = load_primary_demand(demand_path, pids)
    truth = truth_from_config(config, substitution_path=sub_path)
    price, hold, short = product_params(config)
    k_count = len(price)
    ACTION_NAMES = action_names(k_count)
    N_ACTIONS = len(ACTION_NAMES)
    base_seed = int(config["random_seed"])
    b_count = int(config["calendar"]["num_years"])
    days = int(config["calendar"]["days_per_year"])
    horizon = b_count * days
    block_budget = exploration_limit(args, days)

    rep_rows = []
    detail_acc = {
        key: np.zeros((horizon, k_count), dtype=float)
        for key in ("y", "sales", "base", "final_lost", "leftover", "mu_hat")
    }
    period_profit_acc = np.zeros(horizon, dtype=float)
    period_substitution_mae_acc = np.zeros(horizon, dtype=float)
    action_totals = Counter()
    action_period_totals = np.zeros((horizon, len(ACTION_NAMES)), dtype=int)
    final_a_estimates = []

    for rep in range(args.replications):
        env = InventoryEnvironment(
            config, truth, base_seed + 3000 + rep, fixed_demand=fixed_demand
        )
        belief_mode = args.mode if args.mode in {"frozen_a", "true_a", "reset_all", "no_reset"} else "full"
        belief = make_belief(
            k_count, args.eval_particles, base_seed + 9000 + rep,
            args.substitution_prior, belief_mode, truth.substitution,
            belief_cls=JointParticleBelief,
        )
        option = ProbeOptionController(k_count)
        agent = pretrained.clone_for_online(base_seed + 12000 + rep, lr=args.online_lr)
        last_sold = np.zeros(k_count, dtype=bool)
        last_action = -1
        demand_exposure = 0
        row_exposure = np.zeros(k_count, dtype=int)
        demand_probe_count_year = 0
        sub_probe_starts = np.zeros(k_count, dtype=int)
        explicit_exploration_used = 0
        totals = Counter()
        vtot = {
            "sales": np.zeros(k_count),
            "base": np.zeros(k_count),
            "final_lost": np.zeros(k_count),
            "leftover": np.zeros(k_count),
        }
        rep_actions = Counter()

        for t in range(horizon):
            year = t // days
            day = t % days
            if day == 0:
                if args.mode == "reset_all" and t > 0:
                    belief = make_belief(
                        k_count, args.eval_particles,
                        base_seed + 9000 + rep + 10000 * year,
                        args.substitution_prior, "reset_all", truth.substitution,
                        belief_cls=JointParticleBelief,
                    )
                    option = ProbeOptionController(k_count)
                    row_exposure = np.zeros(k_count, dtype=int)
                if args.mode != "no_reset" or belief.current_year < 0:
                    belief.start_year(year)
                demand_exposure = 0
                demand_probe_count_year = 0
                explicit_exploration_used = 0
            state = state_for(
                belief, env, year, day, t, last_sold, demand_exposure,
                row_exposure, last_action, option,
            )
            mask = valid_action_mask(
                belief, option.active_source, demand_probe_count_year, sub_probe_starts
            )
            if args.mode == "no_exploration":
                mask[1:] = False
            mask, forced = budgeted_mask_and_forced(
                mask,
                option,
                explicit_exploration_used,
                block_budget,
                days - day,
                args.exploration_budget_mode == "exact",
            )
            online_epsilon = args.online_epsilon if t < int(0.65 * horizon) else 0.0
            policy_mode = args.mode if args.mode in {"exploit_only", "fixed_schedule", "uncertainty_rule"} else "full"
            action = choose_action(
                policy_mode, agent, state, mask, online_epsilon, forced, day, belief,
                fixed_template=args.fixed_template,
                budget_limit=block_budget,
                block_length=days,
                action_bonus=(
                    allocator.information_bonus(
                        env.x, belief, (horizon - t) / max(horizon, 1)
                    )
                    if (
                        policy_mode == "full"
                        and args.mode != "no_exploration"
                        and belief.__class__.__name__ != "DemandOnlyBelief"
                        and forced is None
                    )
                    else None
                ),
            )
            action, y = allocator.choose_with_economic_safety(
                action,
                env.x,
                belief,
                remaining_fraction=(horizon - t) / max(horizon, 1),
                forced=forced is not None,
            )
            if forced is not None and action != forced:
                option.cancel()
            if 1 <= action < N_ACTIONS:
                explicit_exploration_used += 1
            started = option.start_if_probe(action)
            if started:
                sub_probe_starts[action - 2] += 1
            if action == 1:
                demand_probe_count_year += 1
            pre_summary = belief.summaries()
            valid_substitution = np.ones_like(truth.substitution, dtype=bool)
            for source in range(k_count):
                valid_substitution[source, source] = False
            period_substitution_mae_acc[t] += float(
                np.mean(
                    np.abs(pre_summary["a_mean"] - truth.substitution)[
                        valid_substitution
                    ]
                )
            )
            phi_before = belief.uncertainty_potential()
            public, private = env.step(y)
            update_belief(belief, y, public["sales"], action=action)
            conditional_final_lost = belief.collapsed_final_lost_mean(y)
            proxy = posterior_proxy_reward(public, conditional_final_lost, short)
            sold = np.asarray(public["sold_out"], dtype=bool)
            if np.all(~sold):
                demand_exposure += 1
            for i in range(k_count):
                receivers = [j for j in range(k_count) if j != i]
                if sold[i] and np.all(~sold[receivers]):
                    row_exposure[i] += 1
            option.after_observation(sold)
            next_year = min((t + 1) // days, b_count - 1)
            next_day = (t + 1) % days
            done = t == horizon - 1
            if not done and next_day == 0 and args.mode not in {"reset_all", "no_reset"}:
                belief.start_year(next_year)
                demand_exposure = 0
                demand_probe_count_year = 0
                explicit_exploration_used = 0
            phi_after = 0.0 if done else belief.uncertainty_potential()
            next_state = state_for(
                belief, env, next_year, next_day, t + 1, sold, demand_exposure,
                row_exposure, action, option,
            )
            next_mask = valid_action_mask(
                belief, option.active_source, demand_probe_count_year, sub_probe_starts
            )
            if args.mode == "no_exploration":
                next_mask[1:] = False
            next_mask = apply_exploration_budget(
                next_mask,
                explicit_exploration_used,
                block_budget,
                remaining_periods=days - next_day,
                exact=args.exploration_budget_mode == "exact",
            )
            online_reward = proxy / 3000.0 + agent.gamma * phi_after - phi_before
            if args.mode != "no_online_dqn":
                agent.observe(state, action, online_reward, next_state, done, next_mask)
                if args.online_updates and t % args.learn_every == 0:
                    agent.learn(args.batch_size)

            totals["revenue"] += public["revenue"]
            totals["profit"] += private["true_profit"]
            totals["stockout_cells"] += int(sold.sum())
            totals["cells"] += k_count
            vtot["sales"] += public["sales"]
            vtot["base"] += private["base_demand"]
            vtot["final_lost"] += private["final_lost"]
            vtot["leftover"] += public["leftover"]
            detail_acc["y"][t] += y
            detail_acc["sales"][t] += public["sales"]
            detail_acc["base"][t] += private["base_demand"]
            detail_acc["final_lost"][t] += private["final_lost"]
            detail_acc["leftover"][t] += public["leftover"]
            # A forecast must be measurable before the current observation.
            detail_acc["mu_hat"][t] += pre_summary["mu_mean"]
            period_profit_acc[t] += private["true_profit"]
            last_sold = sold
            last_action = action
            rep_actions[ACTION_NAMES[action]] += 1
            action_period_totals[t, action] += 1

        final_a_estimates.append(belief.summaries()["a_mean"].copy())

        fill_rate = (
            float(vtot["sales"].sum() / vtot["base"].sum())
            if vtot["base"].sum() > 0
            else 0.0
        )
        type1 = (
            1.0 - totals["stockout_cells"] / totals["cells"]
            if totals["cells"] > 0
            else 0.0
        )
        holding_cost = float(vtot["leftover"] @ hold)
        lost_sales_cost = float(vtot["final_lost"] @ short)
        rep_rows.append({
            "replication": rep,
            "total_revenue": totals["revenue"],
            "total_profit": totals["profit"],
            "total_sales": float(vtot["sales"].sum()),
            "total_final_lost": float(vtot["final_lost"].sum()),
            "fill_rate": fill_rate,
            "service_level_type1": type1,
            "holding_cost": holding_cost,
            "lost_sales_cost": lost_sales_cost,
            **{
                f"directed_updates_{pids[i]}": int(
                    belief.explicit_directed_updates[i]
                )
                for i in range(k_count)
            },
            **{f"count_{name}": rep_actions[name] for name in ACTION_NAMES},
        })
        action_totals.update(rep_actions)
        if (rep + 1) % max(1, args.replications // 10) == 0:
            print(f"[eval] replication {rep+1:4d}/{args.replications}")

    reps = max(args.replications, 1)

    # --- Output 1: per-replication evaluation results ---
    with (out_dir / "evaluation_replications.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rep_rows)

    # --- Output 2: decision_detail.csv (per-period per-product averages) ---
    with (out_dir / "decision_detail.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "period", "year", "product_id", "primary_demand_mean",
            "inventory_target_mean", "sales_mean", "final_lost_mean", "leftover_mean",
            "posterior_demand_mean",
        ])
        for t in range(horizon):
            for k in range(k_count):
                writer.writerow([
                    t + 1, t // days + 1, pids[k],
                    round(detail_acc["base"][t, k] / reps, 4),
                    round(detail_acc["y"][t, k] / reps, 4),
                    round(detail_acc["sales"][t, k] / reps, 4),
                    round(detail_acc["final_lost"][t, k] / reps, 4),
                    round(detail_acc["leftover"][t, k] / reps, 4),
                    round(detail_acc["mu_hat"][t, k] / reps, 4),
                ])

    # --- Learning diagnostics ---
    actual_demand = detail_acc["base"] / reps
    predicted_demand = detail_acc["mu_hat"] / reps
    demand_absolute_error = np.abs(actual_demand - predicted_demand)
    true_regional_mean = np.zeros_like(predicted_demand)
    for k, product in enumerate(config["products"]):
        for regime in product["yearly_regimes"]:
            region = int(regime["year"]) - 1
            start = region * days
            true_regional_mean[start : start + days, k] = float(
                regime["target_mean"]
            )
    demand_mean_error = predicted_demand - true_regional_mean
    with (out_dir / "demand_learning.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "period", "region", "product_id", "realized_primary_demand",
                "predicted_demand_pre_observation", "absolute_error",
                "true_regional_demand_mean", "signed_error_to_true_mean",
                "absolute_error_to_true_mean",
            ]
        )
        for t in range(horizon):
            for k, product_id in enumerate(pids):
                writer.writerow(
                    [
                        t + 1,
                        t // days + 1,
                        product_id,
                        round(float(actual_demand[t, k]), 6),
                        round(float(predicted_demand[t, k]), 6),
                        round(float(demand_absolute_error[t, k]), 6),
                        round(float(true_regional_mean[t, k]), 6),
                        round(float(demand_mean_error[t, k]), 6),
                        round(float(abs(demand_mean_error[t, k])), 6),
                    ]
                )

    # Boundary diagnostics are written explicitly so the two-timescale
    # experiment can aggregate by event time without treating periods as
    # independent statistical observations.
    with (out_dir / "boundary_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "period", "region", "day_in_region", "mean_true_profit",
                "demand_mean_mae_to_regime",
                "substitution_mae_pre_observation",
            ]
        )
        for t in range(horizon):
            writer.writerow(
                [
                    t + 1,
                    t // days + 1,
                    t % days + 1,
                    round(float(period_profit_acc[t] / reps), 6),
                    round(float(np.mean(np.abs(demand_mean_error[t]))), 6),
                    round(float(period_substitution_mae_acc[t] / reps), 6),
                ]
            )
    if plot_demand_learning is not None:
        plot_demand_learning(
            actual_demand,
            predicted_demand,
            pids,
            days,
            out_dir / "demand_learning.png",
        )

    final_a_estimates_array = np.asarray(final_a_estimates, dtype=float)
    mean_a = final_a_estimates_array.mean(axis=0)
    std_a = final_a_estimates_array.std(axis=0, ddof=1)
    valid_a = np.ones_like(truth.substitution, dtype=bool)
    for source in range(k_count):
        valid_a[source, source] = False
    replication_a_mae = np.mean(
        np.abs(final_a_estimates_array - truth.substitution[None, :, :])[:, valid_a],
        axis=1,
    )
    destinations = [*pids, config["substitution"]["outside_option"]]
    with (out_dir / "substitution_learning.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source", "destination", "true_rate", "posterior_mean_rate",
                "posterior_rate_std_across_replications", "absolute_error",
            ]
        )
        for source, source_id in enumerate(pids):
            for destination, destination_id in enumerate(destinations):
                if destination == source:
                    continue
                writer.writerow(
                    [
                        source_id,
                        destination_id,
                        float(truth.substitution[source, destination]),
                        float(mean_a[source, destination]),
                        float(std_a[source, destination]),
                        float(
                            abs(
                                mean_a[source, destination]
                                - truth.substitution[source, destination]
                            )
                        ),
                    ]
                )

    with (out_dir / "action_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "action", "total_count_all_replications", "mean_periods_per_300",
                "share_of_periods", "replication_0_count",
            ]
        )
        for action_name in ACTION_NAMES:
            total_count = int(action_totals[action_name])
            writer.writerow(
                [
                    action_name,
                    total_count,
                    total_count / reps,
                    total_count / (reps * horizon),
                    int(rep_rows[0][f"count_{action_name}"]),
                ]
            )

    with (out_dir / "action_timing.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["period", "region", "day_in_region", "action", "count", "share"])
        for t in range(horizon):
            for action_index, action_name in enumerate(ACTION_NAMES):
                count = int(action_period_totals[t, action_index])
                writer.writerow(
                    [t + 1, t // days + 1, t % days + 1, action_name, count, count / reps]
                )

    # --- Aggregate summary ---
    metric_labels = {
        "total_revenue": "Total revenue",
        "total_profit": "Total profit",
        "total_sales": "Total units sold",
        "total_final_lost": "Total final lost sales",
        "fill_rate": "Fill rate (type-2 service level)",
        "service_level_type1": "Cycle service level (type-1)",
        "holding_cost": "Total holding cost",
        "lost_sales_cost": "Total final-lost-sales cost",
    }
    sales_p = detail_acc["sales"].sum(axis=0) / reps
    lost_p = detail_acc["final_lost"].sum(axis=0) / reps
    service_p = np.where(sales_p + lost_p > 0, sales_p / (sales_p + lost_p), 0.0)

    # --- Console summary ---
    metrics = {k: np.array([float(r[k]) for r in rep_rows]) for k in metric_labels}
    profit = metrics["total_profit"]
    profit_std = float(profit.std(ddof=1)) if len(profit) > 1 else 0.0
    profit_se = profit_std / np.sqrt(max(len(profit), 1))
    rl_mean = float(profit.mean())
    result = {
        "algorithm": "TSB-PARL experiment mode: " + args.mode,
        "environment": {
            "reward": "p*sales - h*ending_inventory - b*final_lost",
            "fixed_primary_demand_csv": str(demand_path.resolve()),
            "fixed_substitution_csv": str(sub_path.resolve()),
            "policy_observation": "aggregate sales and ending inventory only",
            "online_updates": bool(args.online_updates),
        },
        "input_hashes": {
            "config_sha256": file_sha256(args.config.resolve()),
            "demand_sha256": file_sha256(demand_path),
            "substitution_sha256": file_sha256(sub_path),
        },
        "settings": {
            "train_episodes": args.train_episodes,
            "replications": args.replications,
            "train_particles": args.train_particles,
            "eval_particles": args.eval_particles,
            "grid_units": args.grid_units,
            "hidden": args.hidden,
            "gamma": float(pretrained.gamma),
            "substitution_prior": args.substitution_prior,
            "experiment_mode": args.mode,
            "exploration_budget_share": args.exploration_budget_share,
            "exploration_budget_periods_per_block": block_budget,
            "exploration_budget_mode": args.exploration_budget_mode,
            "fixed_template": args.fixed_template,
            "reveal_profit_weight": args.reveal_profit_weight,
            "probe_profit_weight": args.probe_profit_weight,
            "information_weight": args.information_weight,
        },
        "profit": {
            "mean": rl_mean,
            "std": profit_std,
            "se": float(profit_se),
            "ci95_low": float(rl_mean - 1.96 * profit_se),
            "ci95_high": float(rl_mean + 1.96 * profit_se),
        },
        "action_counts": dict(action_totals),
        "learning_diagnostics": {
            "demand_mae_overall": float(demand_absolute_error.mean()),
            "demand_mae_by_product": {
                pids[k]: float(demand_absolute_error[:, k].mean())
                for k in range(k_count)
            },
            "demand_mean_mae_overall": float(
                np.mean(np.abs(demand_mean_error))
            ),
            "demand_mean_mae_by_product": {
                pids[k]: float(np.mean(np.abs(demand_mean_error[:, k])))
                for k in range(k_count)
            },
            "demand_mean_signed_bias_by_product": {
                pids[k]: float(np.mean(demand_mean_error[:, k]))
                for k in range(k_count)
            },
            "substitution_final_mae_mean_across_replications": float(
                replication_a_mae.mean()
            ),
            "substitution_final_mae_std_across_replications": float(
                replication_a_mae.std(ddof=1)
            ),
            "substitution_mae_of_mean_posterior_matrix": float(
                np.mean(np.abs(mean_a - truth.substitution)[valid_a])
            ),
        },
    }
    result["benchmarks"] = None
    result["gaps"] = None
    result["benchmark_note"] = "Skipped: --attach-benchmarks was not requested."
    if args.attach_benchmarks:
        result["benchmarks"], result["benchmark_note"] = matched_benchmark_values(
            config, demand_path, sub_path
        )
        if result["benchmarks"] is None:
            print(f"[benchmarks] {result['benchmark_note']}")
    if result["benchmarks"] is not None:
        oracle = result["benchmarks"]["approximate_oracle_fixed_csv"]
        pi = result["benchmarks"]["approximate_pi_fixed_csv"]
        certified = result["benchmarks"]["certified_pathwise_upper_bound"]
        result["gaps"] = {
            "oracle_minus_rl": oracle - rl_mean,
            "oracle_gap_percent": 100.0 * (oracle - rl_mean) / oracle,
            "pi_minus_rl": pi - rl_mean,
            "pi_gap_percent": 100.0 * (pi - rl_mean) / pi,
            "certified_upper_minus_rl": certified - rl_mean,
            "certified_upper_gap_percent": 100.0 * (certified - rl_mean) / certified,
        }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    benchmark_console = ""
    if result["gaps"] is not None:
        benchmark_console = (
            f"\n  Oracle gap         = {result['gaps']['oracle_minus_rl']:,.2f} "
            f"({result['gaps']['oracle_gap_percent']:.2f}%)"
            f"\n  PI gap             = {result['gaps']['pi_minus_rl']:,.2f} "
            f"({result['gaps']['pi_gap_percent']:.2f}%)"
        )
    print(
        f"\n[eval] {args.replications} replications summary:\n"
        f"  total revenue      = {metrics['total_revenue'].mean():,.2f} "
        f"(std {metrics['total_revenue'].std(ddof=0):,.0f})\n"
        f"  total profit       = {metrics['total_profit'].mean():,.2f} "
        f"(std {metrics['total_profit'].std(ddof=0):,.0f})\n"
        f"  fill rate          = {metrics['fill_rate'].mean():.4f}\n"
        f"  cycle service (T1) = {metrics['service_level_type1'].mean():.4f}\n"
        f"  holding cost       = {metrics['holding_cost'].mean():,.2f}\n"
        f"  final-loss cost    = {metrics['lost_sales_cost'].mean():,.2f}\n"
        f"  target service     = " + ", ".join(
            f"{pid}={service_p[i]:.3f}" for i, pid in enumerate(pids)
        )
        + benchmark_console
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=INPUTS / "configs"
        / "C6_HIGHER_ROUTE_VALUE_path1_train20269601.json",
    )
    parser.add_argument("--train-episodes", type=int, default=240)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--train-particles", type=int, default=160)
    parser.add_argument("--eval-particles", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=56)
    parser.add_argument("--grid-units", type=int, default=28)
    parser.add_argument("--learn-every", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--mode",
        choices=(
            "full", "exploit_only", "fixed_schedule", "uncertainty_rule",
            "frozen_a", "true_a", "reset_all", "no_reset",
            "no_online_dqn",
            "no_exploration",
        ),
        default="full",
        help="Controlled paper-experiment variant; full preserves the original algorithm.",
    )
    parser.add_argument(
        "--substitution-prior",
        choices=("symmetric_sparse", "exit_conservative"),
        default="symmetric_sparse",
    )
    parser.add_argument("--online-updates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--online-lr", type=float, default=3e-5)
    parser.add_argument("--online-epsilon", type=float, default=0.015)
    parser.add_argument(
        "--exploration-budget-share",
        type=float,
        default=None,
        help=(
            "Optional hard per-block cap on actions 1-5, expressed as a share "
            "of block length. Omit to preserve the original algorithm."
        ),
    )
    parser.add_argument(
        "--fixed-template",
        choices=("legacy", "spread", "front_loaded"),
        default="legacy",
        help="Calendar-only timing template used by fixed_schedule.",
    )
    parser.add_argument(
        "--exploration-budget-mode",
        choices=("cap", "exact"),
        default="cap",
        help=(
            "Use the budget as an upper cap, or require exactly that many "
            "explicit exploration periods in every block."
        ),
    )
    parser.add_argument("--reveal-profit-weight", type=float, default=0.05)
    parser.add_argument("--probe-profit-weight", type=float, default=0.10)
    parser.add_argument("--information-weight", type=float, default=520.0)
    parser.add_argument("--output", type=Path, default=ROOT / "运行结果" / "单次完整算法")
    parser.add_argument(
        "--attach-benchmarks",
        action="store_true",
        help=(
            "Attach recorded per-path Oracle/PI values only when demand, "
            "substitution, configuration hashes, and model settings can be verified."
        ),
    )
    args = parser.parse_args()

    if not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must lie in [0, 1]")
    if (
        args.exploration_budget_share is not None
        and not 0.0 <= args.exploration_budget_share <= 1.0
    ):
        parser.error("--exploration-budget-share must lie in [0, 1]")
    for name in (
        "reveal_profit_weight",
        "probe_profit_weight",
        "information_weight",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")

    args.output.mkdir(parents=True, exist_ok=True)
    args.config = resolve_path(args.config).resolve()
    config = load_config(args.config)
    agent, allocator, state_dim = train(config, args, args.output)
    if plot_training_loss is not None:
        loss_png = plot_training_loss(args.output)
        print(f"[plot] training loss figure: {loss_png.resolve()}")
    else:
        print("[plot] skipped: matplotlib is unavailable in this runtime")
    evaluate(config, args, args.output, agent, allocator, state_dim)
    print(f"\nOutputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
