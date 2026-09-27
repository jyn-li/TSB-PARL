"""Auditable D2AS2 adaptation and PPO controller for the section 5.1 extension.

Policy constructors receive public primitives only. Hidden demand, routes and true
profit are confined to evaluation accounting (and training diagnostic logging).
PPO learns from the same posterior proxy reward as Full, never private profit.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "algorithm" / "project_paths.py").is_file():
        sys.path.insert(0, str(_parent / "algorithm"))
        break
from project_paths import ALGORITHM, resolve_path
sys.path.insert(0, str(ALGORITHM / "src" / "tsb_parl"))
from allocator import StructuredAllocator
from belief import JointParticleBelief
from environment import InventoryEnvironment, randomized_truth, truth_from_config, load_primary_demand
from rl_agent import posterior_proxy_reward


def option(args, name, default):
    value = getattr(args, name, default)
    return default if value is None else value


def public_config(config):
    """Deliberately omit yearly means, distribution parameters and true routes."""
    return {
        "products": [{key: p[key] for key in ("id", "price", "holding_cost", "shortage_cost")}
                     for p in config["products"]],
        "calendar": copy.deepcopy(config["calendar"]),
        "simulation": {"shared_capacity": int(config["simulation"]["shared_capacity"]),
                       "initial_inventory": copy.deepcopy(config["simulation"].get("initial_inventory"))},
    }


def economics(config):
    return tuple(np.asarray([p[name] for p in config["products"]], dtype=float)
                 for name in ("price", "holding_cost", "shortage_cost"))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def integer_allocation(weights, inventory, capacity):
    """N+1 nonnegative weights: final category is unspent capacity.

    Largest remainder apportions every available integer unit across the N+1
    categories; discarding the last category ensures y>=x and sum(y)<=capacity.
    """
    x = np.asarray(inventory, dtype=int)
    slack = int(capacity - x.sum())
    if slack < 0:
        raise ValueError("starting inventory exceeds capacity")
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(x) + 1,) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("invalid N+1 allocation weights")
    raw = slack * w / w.sum()
    units = np.floor(raw).astype(int)
    remainder = slack - int(units.sum())
    order = np.argsort(-(raw - units), kind="stable")
    units[order[:remainder]] += 1
    return x + units[:-1]


class D2AS2Adapt:
    """Scheduled focal stockouts, matched benchmark contrasts, and exploitation.

    Shared capacity can censor the high-stock benchmark; only uncensored matched
    observations identify a route. Nonidentifiable rows retain their prior rather
    than receiving hidden simulator estimates. Moment scoring uses the same
    feasible candidate generator as Full. See accompanying adaptation note.
    """
    name = "D2AS2-adapt"

    def __init__(self, config, args, seed):
        self.cfg = public_config(config)
        self.n = len(self.cfg["products"])
        self.capacity = self.cfg["simulation"]["shared_capacity"]
        self.days = int(self.cfg["calendar"]["days_per_year"])
        self.price, self.hold, self.short = economics(self.cfg)
        self.allocator = StructuredAllocator(self.capacity, self.price, self.hold, self.short,
                                              grid_units=option(args, "grid_units", 28))
        self.beta = float(option(args, "d2as_beta", 0.25))
        self.i0 = float(option(args, "d2as_i0", 2.0))
        self.v = float(option(args, "d2as_v", 2.0))
        self.a = np.zeros((self.n, self.n + 1))
        for source in range(self.n):
            self.a[source, [j for j in range(self.n + 1) if j != source]] = 1.0 / self.n
        self.route_samples = np.zeros((self.n, self.n), dtype=int)
        self.route_estimate_numerator = np.zeros((self.n, self.n))
        self.route_estimate_denominator = np.zeros((self.n, self.n))
        self.current_block = -1
        self.block_resets = 0
        self.begin_block(0)

    def begin_block(self, block):
        self.current_block = int(block)
        self.block_resets += int(block > 0)
        # Retain persistent route contrasts, refresh nonstationary demand data.
        self.samples = [[] for _ in range(self.n)]
        self.mu = np.full(self.n, 150.0)
        self.var = np.full(self.n, 1600.0)
        self.stage = 0
        self._begin_stage()

    def _begin_stage(self):
        # Log-domain form avoids overflow in v ** (v ** stage).
        log_i = math.log(self.i0) + (self.v ** min(self.stage, 8)) * math.log(self.v)
        self.exploit_length = max(1, min(self.days, int(math.ceil(math.exp(min(log_i, math.log(self.days)))))))
        self.length = max(1, int(math.ceil(self.beta * log_i * math.log(max(log_i, math.e)))))
        self.length = min(self.length, max(1, self.days // (2 * (self.n + 1))))
        self.phase = "benchmark"
        self.source = -1
        self.phase_age = 0
        self.benchmark = []
        self.focal = []
        self.last_label = "benchmark"

    def summaries(self):
        return {"mu_mean": self.mu, "predictive_var": self.var, "a_mean": self.a,
                "mu_sd": np.sqrt(self.var), "a_sd": np.zeros_like(self.a)}

    def target(self, t, inventory):
        block = t // self.days
        if block != self.current_block:
            self.begin_block(block)
        x = np.asarray(inventory, dtype=int)
        if self.phase == "exploit":
            self.last_label = "exploit"
            return self.allocator.choose(0, x, self)
        # Equal service protection adjusted to currently estimated demand.
        desired = self.mu + 2.0 * np.sqrt(np.maximum(self.var, 4.0))
        increments = np.maximum(desired - x, 0.0)
        if self.phase == "focal":
            increments[self.source] = 0.0
            self.last_label = f"focal_{self.source + 1}"
        else:
            self.last_label = "benchmark"
        if increments.sum() <= 0:
            return x.copy()
        # High-inventory experiments use all capacity; never discard carryover.
        return integer_allocation(np.r_[increments, 0.0], x, self.capacity)

    def _refresh_demand(self):
        for k, rows in enumerate(self.samples):
            if rows:
                values = np.asarray(rows, dtype=float)
                self.mu[k] = float(np.mean(values))
                self.var[k] = max(4.0, float(np.var(values, ddof=1))) if len(values) > 1 else max(4.0, self.mu[k])

    def _update_route(self):
        source = self.source
        # Focal source must start at zero; benchmark source and receiver must
        # both be uncensored. Match equal sample counts; never use lost sales.
        for receiver in range(self.n):
            if receiver == source:
                continue
            b = [r for r in self.benchmark if np.all(~r[1])]
            f = [r for r in self.focal if r[2][source] == 0 and not r[1][receiver]]
            count = min(len(b), len(f))
            if count == 0:
                continue
            denominator = float(sum(r[0][source] for r in b[:count]))
            if denominator <= 0:
                continue
            difference = float(sum(r[0][receiver] for r in f[:count]) - sum(r[0][receiver] for r in b[:count]))
            self.route_samples[source, receiver] += count
            self.route_estimate_numerator[source, receiver] += difference
            self.route_estimate_denominator[source, receiver] += denominator
        valid = self.route_estimate_denominator[source] > 0
        row = self.a[source, :self.n].copy()
        row[valid] = np.maximum(self.route_estimate_numerator[source, valid], 0.0) / self.route_estimate_denominator[source, valid]
        row[source] = 0.0
        if row.sum() > 1:
            row /= row.sum()
        self.a[source, :self.n] = row
        self.a[source, -1] = max(0.0, 1.0 - row.sum())

    def observe(self, t, target, inventory_before, public):
        sales = np.asarray(public["sales"], dtype=float)
        sold = np.asarray(public["sold_out"], dtype=bool)
        record = (sales.copy(), sold.copy(), np.asarray(inventory_before).copy())
        self.phase_age += 1
        if self.phase == "benchmark":
            self.benchmark.append(record)
            # All-product uncensored observations guarantee absence of
            # stockout substitution; partial observations cannot identify d0.
            if np.all(~sold):
                for k in range(self.n):
                    self.samples[k].append(float(sales[k]))
                self._refresh_demand()
            if self.phase_age >= self.length:
                self.phase, self.source, self.phase_age = "focal", 0, 0
        elif self.phase == "focal":
            self.focal.append(record)
            depleted = sum(r[2][self.source] == 0 for r in self.focal)
            if depleted >= self.length or self.phase_age >= 3 * self.length + 3:
                self._update_route()
                self.source += 1
                self.phase_age = 0
                self.focal = []
                if self.source == self.n:
                    self.phase = "exploit"
        elif self.phase_age >= self.exploit_length:
            self.stage += 1
            self._begin_stage()
        # Public, clean full-availability observations during exploitation also
        # improve the demand empirical moments without route labels.
        if self.last_label == "exploit" and np.all(~sold):
            for k in range(self.n):
                self.samples[k].append(float(sales[k]))
            self._refresh_demand()


def torch_runtime():
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch


def make_network(state_dim, n, hidden, seed):
    torch = torch_runtime()
    torch.manual_seed(seed)
    nn = torch.nn

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                       nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n + 1))
            self.critic = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                        nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            self.log_std = nn.Parameter(torch.full((n + 1,), -0.7))
            for branch in (self.actor, self.critic):
                for layer in branch:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, np.sqrt(2.0))
                        nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(self.actor[-1].weight, 0.01)
            nn.init.orthogonal_(self.critic[-1].weight, 1.0)
            with torch.no_grad():
                self.actor[-1].bias[-1] = -2.0

        def distribution_value(self, states):
            mean = self.actor(states)
            distribution = torch.distributions.Normal(mean, torch.exp(self.log_std.clamp(-3, 1)))
            return distribution, self.critic(states).squeeze(-1)

    return ActorCritic()


class PPOPolicy:
    name = "PPO (common belief filter)"

    def __init__(self, config, args, seed, bundle):
        self.cfg = public_config(config)
        self.n = len(self.cfg["products"])
        self.capacity = self.cfg["simulation"]["shared_capacity"]
        self.days = int(self.cfg["calendar"]["days_per_year"])
        self.horizon = self.days * int(self.cfg["calendar"]["num_years"])
        self.price, self.hold, self.short = economics(self.cfg)
        self.args = args
        self.network = bundle["network"]
        self.belief = JointParticleBelief(self.n, int(option(args, "eval_particles", 384)), seed,
                                          substitution_prior=option(args, "substitution_prior", "symmetric_sparse"))
        self.last_sales = np.zeros(self.n)
        self.last_sold = np.zeros(self.n)
        self.last_label = "ppo"

    def state(self, t, inventory):
        self.belief.start_year(min(t // self.days, self.horizon // self.days - 1))
        s = self.belief.summaries()
        return np.concatenate([
            np.asarray(inventory) / self.capacity,
            self.last_sales / 300.0, self.last_sold,
            s["mu_mean"] / 300.0, s["mu_sd"] / 100.0,
            np.sqrt(s["predictive_var"]) / 100.0,
            s["a_mean"].ravel(), s["a_sd"].ravel(), s["type_prob"].ravel(),
            self.price / max(float(self.price.max()), 1),
            self.hold / np.maximum(self.price, 1), self.short / np.maximum(self.price, 1),
            [t / max(self.horizon, 1), (t % self.days) / self.days,
             self.capacity / (150.0 * self.n), (self.horizon - t) / max(self.horizon, 1)]
        ]).astype(np.float32)

    def action_from_latent(self, latent, inventory):
        z = np.asarray(latent, dtype=float)
        w = np.exp(z - z.max())
        return integer_allocation(w, inventory, self.capacity)

    def target(self, t, inventory):
        torch = torch_runtime()
        with torch.no_grad():
            distribution, _ = self.network.distribution_value(torch.as_tensor(self.state(t, inventory)))
            latent = distribution.mean.cpu().numpy()
        return self.action_from_latent(latent, inventory)

    def observe(self, t, target, inventory_before, public):
        self.belief.update(target, public["sales"], action=0)
        self.last_sales = np.asarray(public["sales"], dtype=float)
        self.last_sold = np.asarray(public["sold_out"], dtype=float)
        return posterior_proxy_reward(public, self.belief.collapsed_final_lost_mean(target), self.short)


def _ppo_update(network, optimizer, buffer, args, rng):
    torch = torch_runtime()
    states = torch.as_tensor(np.asarray(buffer["states"]), dtype=torch.float32)
    latents = torch.as_tensor(np.asarray(buffer["latents"]), dtype=torch.float32)
    log_old = torch.as_tensor(buffer["logp"], dtype=torch.float32)
    values = np.asarray(buffer["values"])
    rewards, dones = np.asarray(buffer["rewards"]), np.asarray(buffer["dones"])
    next_values = np.asarray(buffer["next_values"])
    gamma, lam = float(option(args, "gamma", 1.0)), float(option(args, "ppo_gae_lambda", 0.95))
    advantages = np.zeros(len(rewards))
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        delta = rewards[i] + gamma * (1 - dones[i]) * next_values[i] - values[i]
        running = delta + gamma * lam * (1 - dones[i]) * running
        advantages[i] = running
    returns = torch.as_tensor(advantages + values, dtype=torch.float32)
    advantages = torch.as_tensor((advantages - advantages.mean()) / max(advantages.std(), 1e-8), dtype=torch.float32)
    clip = float(option(args, "ppo_clip", 0.2))
    losses, kls = [], []
    for epoch in range(int(option(args, "ppo_epochs", 10))):
        indices = rng.permutation(len(rewards))
        for offset in range(0, len(indices), int(option(args, "ppo_batch_size", 64))):
            ix = torch.as_tensor(indices[offset:offset + int(option(args, "ppo_batch_size", 64))])
            dist, value = network.distribution_value(states[ix])
            logp = dist.log_prob(latents[ix]).sum(-1)
            log_ratio = logp - log_old[ix]
            ratio = log_ratio.exp()
            actor_loss = -torch.minimum(ratio * advantages[ix], ratio.clamp(1 - clip, 1 + clip) * advantages[ix]).mean()
            critic_loss = 0.5 * (value - returns[ix]).pow(2).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = actor_loss + critic_loss - float(option(args, "ppo_entropy_coef", 0.0)) * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), float(option(args, "ppo_max_grad_norm", 0.5)))
            optimizer.step()
            losses.append(float(loss.detach()))
            kls.append(float(((ratio - 1) - log_ratio).mean().detach()))
        if np.mean(kls[-max(1, math.ceil(len(rewards) / int(option(args, "ppo_batch_size", 64)))):]) > float(option(args, "ppo_target_kl", 0.03)):
            break
    return float(np.mean(losses)), float(np.mean(kls))


def save_baseline(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if bundle["method"] == "ppo":
        torch = torch_runtime()
        payload = {k: v for k, v in bundle.items() if k != "network"}
        payload["state_dict"] = bundle["network"].state_dict()
        torch.save(payload, path)
    else:
        path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")


def load_baseline(path):
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    torch = torch_runtime()
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    network = make_network(bundle["state_dim"], bundle["n"], bundle["hidden"], bundle["seed"])
    network.load_state_dict(bundle.pop("state_dict"))
    bundle["network"] = network
    return bundle


def train_baseline(config, args, out):
    """One offline model per scenario/training seed; safe to reuse across paths."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    method = option(args, "method", "ppo")
    seed = int(option(args, "train_seed", config["random_seed"])) + 8100
    start = time.perf_counter()
    if method == "d2as2":
        bundle = {"method": method, "seed": seed, "train_steps": 0,
                  "training_seconds": 0.0, "offline_learning": False,
                  "validation_hyperparameter_trials": 0}
        save_baseline(bundle, out / "d2as2_policy.json")
        return bundle
    if method != "ppo":
        raise ValueError("method must be d2as2 or ppo")
    torch = torch_runtime()
    rng = np.random.default_rng(seed)
    cfg = public_config(config)
    n = len(cfg["products"])
    horizon = int(cfg["calendar"]["days_per_year"]) * int(cfg["calendar"]["num_years"])
    steps = int(option(args, "train_steps", int(option(args, "train_episodes", 80)) * horizon))
    if steps < 1:
        raise ValueError("PPO needs positive train_steps")
    hidden = int(option(args, "ppo_hidden", option(args, "hidden", 64)))
    train_args = copy.copy(args)
    train_args.eval_particles = int(option(args, "train_particles", 160))
    dummy = PPOPolicy(cfg, train_args, seed + 1, {"network": None})
    state_dim = len(dummy.state(0, np.zeros(n, dtype=int)))
    network = make_network(state_dim, n, hidden, seed + 3)
    optimizer = torch.optim.Adam(network.parameters(), lr=float(option(args, "ppo_lr", 3e-4)), eps=1e-5)
    bundle = {"method": method, "seed": seed, "state_dim": state_dim, "n": n,
              "hidden": hidden, "network": network, "train_steps": steps,
              "offline_learning": True, "validation_hyperparameter_trials": 0,
              "training_reward": "posterior proxy plus same potential shaping as Full; no latent rewards",
              "test_adaptation": "belief only; PPO weights frozen"}
    buffer = {key: [] for key in ("states", "latents", "logp", "values", "rewards", "dones", "next_values")}
    rows, updates = [], []
    global_step, episode = 0, 0
    while global_step < steps:
        truth = randomized_truth(config, rng)
        env = InventoryEnvironment(config, truth, seed + 1000 + episode)
        policy = PPOPolicy(cfg, train_args, seed + 2000 + episode, bundle)
        totals = {"proxy_reward": 0.0, "diagnostic_true_profit": 0.0}
        for t in range(env.horizon):
            state = policy.state(t, env.x)
            before = env.x.copy()
            phi_before = policy.belief.uncertainty_potential()
            with torch.no_grad():
                dist, value = network.distribution_value(torch.as_tensor(state))
                z = dist.sample()
                logp = dist.log_prob(z).sum()
            target = policy.action_from_latent(z.numpy(), before)
            public, private = env.step(target)
            proxy = policy.observe(t, target, before, public)
            done = t == env.horizon - 1
            next_state = policy.state(t + 1, env.x) if not done else state
            phi_after = 0.0 if done else policy.belief.uncertainty_potential()
            shaped = proxy / 3000.0 + float(option(args, "gamma", 1.0)) * phi_after - phi_before
            with torch.no_grad():
                next_value = 0.0 if done else float(network.distribution_value(torch.as_tensor(next_state))[1])
            for key, val in zip(buffer, (state, z.numpy(), float(logp), float(value), shaped, done, next_value)):
                buffer[key].append(val)
            totals["proxy_reward"] += proxy
            totals["diagnostic_true_profit"] += float(private["true_profit"])
            global_step += 1
            if len(buffer["rewards"]) >= int(option(args, "ppo_rollout_steps", 512)) or global_step == steps:
                loss, kl = _ppo_update(network, optimizer, buffer, args, rng)
                updates.append({"training_step": global_step, "loss": loss, "approximate_kl": kl})
                buffer = {key: [] for key in buffer}
            if global_step >= steps:
                break
        episode += 1
        rows.append({"episode": episode, "training_steps": global_step, **totals})
    bundle["training_seconds"] = time.perf_counter() - start
    bundle["public_config_sha256"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    save_baseline(bundle, out / "ppo_model.pt")
    write_csv(out / "training_history.csv", rows)
    write_csv(out / "ppo_updates.csv", updates)
    metadata = {key: value for key, value in bundle.items() if key != "network"}
    metadata["settings"] = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))}
    (out / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle


def evaluate(config, args, output, pretrained=None):
    """Read test truth only here, after training; policy objects never receive it."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    method = option(args, "method", "ppo")
    if pretrained is None:
        pretrained = train_baseline(config, args, output / "training")
    elif isinstance(pretrained, (str, Path)):
        pretrained = load_baseline(pretrained)
    if pretrained["method"] != method:
        raise ValueError("pretrained baseline method mismatch")
    fixed = load_primary_demand(resolve_path(config["output"]["demand_long_file"]), [p["id"] for p in config["products"]])
    truth = truth_from_config(config, resolve_path(config["output"]["substitution_wide_file"]))
    cfg = public_config(config)
    n, horizon = len(config["products"]), len(fixed)
    if method == "ppo":
        public_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        if public_hash != pretrained.get("public_config_sha256"):
            raise ValueError("PPO checkpoint public scenario primitives differ from evaluation")
    seed = int(option(args, "eval_seed", config["random_seed"]))
    reps = int(option(args, "replications", 4))
    if reps < 1:
        raise ValueError("replications must be positive")
    rows, detail = [], []
    begin = time.perf_counter()
    for rep in range(reps):
        policy = (PPOPolicy(cfg, args, seed + 9000 + rep, pretrained) if method == "ppo"
                  else D2AS2Adapt(cfg, args, seed + 9000 + rep))
        env = InventoryEnvironment(config, truth, seed + 3000 + rep, fixed_demand=fixed)
        totals = dict(total_profit=0.0, total_revenue=0.0, total_holding_cost=0.0,
                      total_lost_penalty=0.0, total_sales=0, total_final_lost=0,
                      stockout_cells=0, decision_seconds=0.0, update_seconds=0.0)
        rep_begin = time.perf_counter()
        for t in range(horizon):
            before = env.x.copy()
            tick = time.perf_counter()
            y = np.asarray(policy.target(t, before), dtype=int)
            decision_time = time.perf_counter() - tick
            public, private = env.step(y)
            tick = time.perf_counter()
            policy.observe(t, y, before, public)
            update_time = time.perf_counter() - tick
            totals["total_profit"] += float(private["true_profit"])
            totals["total_revenue"] += float(public["revenue"])
            totals["total_holding_cost"] += float(public["holding_cost"])
            totals["total_lost_penalty"] += float(private["lost_sales_cost"])
            totals["total_sales"] += int(np.sum(public["sales"]))
            totals["total_final_lost"] += int(np.sum(private["final_lost"]))
            totals["stockout_cells"] += int(np.sum(public["sold_out"]))
            totals["decision_seconds"] += decision_time
            totals["update_seconds"] += update_time
            for k, product in enumerate(config["products"]):
                detail.append({"replication": rep, "period": t + 1, "product_id": product["id"],
                               "action": policy.last_label, "inventory_before": int(before[k]),
                               "inventory_target": int(y[k]), "sales": int(public["sales"][k]),
                               "leftover": int(public["leftover"][k]), "primary_demand": int(private["base_demand"][k]),
                               "final_lost": int(private["final_lost"][k]),
                               "revenue": float(product["price"] * public["sales"][k]),
                               "holding_cost": float(product["holding_cost"] * public["leftover"][k]),
                               "lost_sales_cost": float(product["shortage_cost"] * private["final_lost"][k]),
                               "profit": float(product["price"] * public["sales"][k]
                                               - product["holding_cost"] * public["leftover"][k]
                                               - product["shortage_cost"] * private["final_lost"][k]),
                               "decision_seconds_all_products": decision_time, "update_seconds_all_products": update_time})
        identity = totals["total_revenue"] - totals["total_holding_cost"] - totals["total_lost_penalty"]
        if not np.isclose(identity, totals["total_profit"], rtol=1e-10):
            raise RuntimeError("profit accounting identity failed")
        rows.append({"replication": rep, **totals,
                     "fill_rate": totals["total_sales"] / max(float(fixed.sum()), 1.0),
                     "service_level_type1": 1 - totals["stockout_cells"] / (n * horizon),
                     "holding_cost": totals["total_holding_cost"], "lost_sales_cost": totals["total_lost_penalty"],
                     "profit_per_product_period": totals["total_profit"] / (n * horizon),
                     "elapsed_seconds": time.perf_counter() - rep_begin,
                     "identified_route_pairs": int(np.count_nonzero(policy.route_samples)) if method == "d2as2" else -1,
                     "matched_route_samples": int(policy.route_samples.sum()) if method == "d2as2" else -1})
    write_csv(output / "evaluation_replications.csv", rows)
    write_csv(output / "period_product_detail.csv", detail)
    # Compatible mean detail file for the manuscript mechanism collector.
    averaged = []
    for t in range(horizon):
        for k, product in enumerate(config["products"]):
            selected = [detail[(rep * horizon + t) * n + k] for rep in range(reps)]
            averaged.append({"period": t + 1, "product_id": product["id"],
                             "inventory_target_mean": float(np.mean([r["inventory_target"] for r in selected])),
                             "sales_mean": float(np.mean([r["sales"] for r in selected])),
                             "leftover_mean": float(np.mean([r["leftover"] for r in selected])),
                             "final_lost_mean": float(np.mean([r["final_lost"] for r in selected])),
                             "primary_demand_mean": float(np.mean([r["primary_demand"] for r in selected]))})
    write_csv(output / "decision_detail.csv", averaged)
    profits = np.asarray([r["total_profit"] for r in rows])
    std = float(np.std(profits, ddof=1)) if reps > 1 else 0.0
    se = std / np.sqrt(max(reps, 1))
    summary = {"algorithm": policy.name, "method": method,
               "profit": {"mean": float(profits.mean()), "std": std, "se": float(se),
                          "ci95_low": float(profits.mean() - 1.96 * se), "ci95_high": float(profits.mean() + 1.96 * se)},
               "mean_profit": float(profits.mean()), "evaluation_seconds": time.perf_counter() - begin,
               "training_seconds": float(pretrained["training_seconds"]),
               "settings": {"replications": reps, "train_steps": pretrained["train_steps"],
                            "seed": seed, "training_seed": pretrained["seed"], "n": n, "horizon": horizon,
                            "grid_units": option(args, "grid_units", 28)},
               "information": {"policy_observation": "calendar, inventory, public aggregate sales only",
                               "training_truth": "randomized_truth, independent of evaluation means and routes",
                               "evaluation_latent_usage": "accounting only",
                               "online_weight_updates": False},
               "means": {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != "replication"}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=("d2as2", "ppo"), required=True)
    p.add_argument("--pretrained", type=Path)
    p.add_argument("--train-steps", type=int, default=24000)
    p.add_argument("--train-particles", type=int, default=160)
    p.add_argument("--eval-particles", type=int, default=384)
    p.add_argument("--grid-units", type=int, default=28)
    p.add_argument("--replications", type=int, default=4)
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--train-seed", type=int)
    p.add_argument("--eval-seed", type=int)
    p.add_argument("--gamma", type=float, default=1.0)
    args = p.parse_args()
    config = json.loads(resolve_path(args.config).read_text(encoding="utf-8"))
    for name in ("train_seed", "eval_seed"):
        if getattr(args, name) is None:
            delattr(args, name)
    if args.train_only:
        bundle = train_baseline(config, args, args.output)
        print(json.dumps({k: v for k, v in bundle.items() if k != "network"}, ensure_ascii=False))
    else:
        print(json.dumps(evaluate(config, args, args.output, args.pretrained), ensure_ascii=False))


if __name__ == "__main__":
    main()
