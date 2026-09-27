"""Double-DQN agent, replay buffer, probe controller, and auxiliary RL helpers."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np

from utils import N_ACTIONS, action_names

# forward reference for type hint
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from belief import JointParticleBelief


# ── probe option controller ──

class ProbeOptionController:
    """Makes substitution probes multi-period when carryover prevents immediate stockout."""

    def __init__(
        self, k_count: int, max_age: int = 4, min_informative: int = 2
    ):
        self.k_count = k_count
        self.action_names = action_names(k_count)
        self.n_actions = len(self.action_names)
        self.max_age = max_age
        self.min_informative = min_informative
        self.active_source = -1
        self.age = 0
        self.informative_count = 0

    def force_action(self) -> int | None:
        if self.active_source >= 0:
            return 2 + self.active_source
        return None

    def cancel(self) -> None:
        """End an active probe when an external action budget is exhausted."""
        self.active_source = -1
        self.age = 0
        self.informative_count = 0

    def start_if_probe(self, action: int) -> bool:
        if 2 <= action < self.n_actions and self.active_source < 0:
            self.active_source = action - 2
            self.age = 0
            self.informative_count = 0
            return True
        return False

    def after_observation(self, sold_out: np.ndarray) -> None:
        if self.active_source < 0:
            return
        self.age += 1
        src = self.active_source
        receivers = [j for j in range(self.k_count) if j != src]
        informative = bool(sold_out[src] and np.all(~sold_out[receivers]))
        if informative:
            self.informative_count += 1
        if self.informative_count >= self.min_informative or self.age >= self.max_age:
            self.active_source = -1
            self.age = 0
            self.informative_count = 0


# ── replay buffer ──

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, n_actions: int):
        self.capacity = int(capacity)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int16)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.ones((capacity, n_actions), dtype=bool)
        self.size = 0
        self.pos = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        p = self.pos
        self.state[p] = state
        self.action[p] = action
        self.reward[p] = reward
        self.next_state[p] = next_state
        self.done[p] = float(done)
        self.next_mask[p] = next_mask
        self.pos = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        uniform_n = batch_size // 2
        chosen = list(rng.integers(0, self.size, size=uniform_n))
        present = [
            action for action in range(self.next_mask.shape[1])
            if np.any(self.action[: self.size] == action)
        ]
        for draw in range(batch_size - uniform_n):
            action = present[draw % len(present)] if present else 0
            pool = np.flatnonzero(self.action[: self.size] == action)
            chosen.append(
                int(rng.choice(pool)) if len(pool) else int(rng.integers(0, self.size))
            )
        idx = np.asarray(chosen, dtype=int)
        rng.shuffle(idx)
        return (
            self.state[idx],
            self.action[idx],
            self.reward[idx],
            self.next_state[idx],
            self.done[idx],
            self.next_mask[idx],
        )


# ── Q-network with Adam ──

class NumpyQNetwork:
    """One-hidden-layer Q network with Adam; adequate for the small option space."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int, seed: int):
        rng = np.random.default_rng(seed)
        self.params = {
            "w1": rng.normal(scale=math.sqrt(2.0 / state_dim), size=(state_dim, hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(scale=math.sqrt(2.0 / hidden), size=(hidden, n_actions)),
            "b2": np.zeros(n_actions),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.adam_t = 0

    def copy_from(self, other: "NumpyQNetwork") -> None:
        for k in self.params:
            self.params[k][...] = other.params[k]

    def clone(self) -> "NumpyQNetwork":
        return copy.deepcopy(self)

    def predict(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        h = np.maximum(x @ self.params["w1"] + self.params["b1"], 0.0)
        return h @ self.params["w2"] + self.params["b2"]

    def train_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        lr: float,
    ) -> float:
        x = np.asarray(states, dtype=float)
        z = x @ self.params["w1"] + self.params["b1"]
        h = np.maximum(z, 0.0)
        q = h @ self.params["w2"] + self.params["b2"]
        pred = q[np.arange(len(x)), actions]
        error = pred - targets
        abs_err = np.abs(error)
        loss = np.where(abs_err <= 1.0, 0.5 * error**2, abs_err - 0.5).mean()
        grad_pred = np.where(abs_err <= 1.0, error, np.sign(error)) / len(x)
        dq = np.zeros_like(q)
        dq[np.arange(len(x)), actions] = grad_pred
        grads = {}
        grads["w2"] = h.T @ dq
        grads["b2"] = dq.sum(axis=0)
        dh = dq @ self.params["w2"].T
        dz = dh * (z > 0.0)
        grads["w1"] = x.T @ dz
        grads["b1"] = dz.sum(axis=0)
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        if norm > 10.0:
            scale = 10.0 / norm
            grads = {k: g * scale for k, g in grads.items()}
        self.adam_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for k in self.params:
            self.m[k] = beta1 * self.m[k] + (1.0 - beta1) * grads[k]
            self.v[k] = beta2 * self.v[k] + (1.0 - beta2) * (grads[k] ** 2)
            mhat = self.m[k] / (1.0 - beta1**self.adam_t)
            vhat = self.v[k] / (1.0 - beta2**self.adam_t)
            self.params[k] -= lr * mhat / (np.sqrt(vhat) + eps)
        return float(loss)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **self.params)

    def load(self, path: Path) -> None:
        data = np.load(path)
        for k in self.params:
            self.params[k][...] = data[k]


# ── Double-DQN agent ──

class DoubleDQNAgent:
    def __init__(
        self,
        state_dim: int,
        seed: int,
        hidden: int = 56,
        gamma: float = 1.0,
        lr: float = 2e-4,
        buffer_capacity: int = 40000,
        n_actions: int = N_ACTIONS,
    ):
        self.state_dim = state_dim
        self.n_actions = int(n_actions)
        self.action_names = action_names(self.n_actions - 2)
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.rng = np.random.default_rng(seed)
        self.online = NumpyQNetwork(state_dim, self.n_actions, hidden, seed + 1)
        self.target = NumpyQNetwork(state_dim, self.n_actions, hidden, seed + 2)
        self.target.copy_from(self.online)
        self.buffer = ReplayBuffer(buffer_capacity, state_dim, self.n_actions)
        self.gradient_steps = 0

    def clone_for_online(self, seed: int, lr: float = 3e-5) -> "DoubleDQNAgent":
        hidden = self.online.params["b1"].shape[0]
        clone = DoubleDQNAgent(
            self.state_dim, seed=seed, hidden=hidden, gamma=self.gamma,
            lr=lr, buffer_capacity=3000, n_actions=self.n_actions,
        )
        clone.online.copy_from(self.online)
        clone.target.copy_from(self.target)
        n = min(self.buffer.size, clone.buffer.capacity, 1500)
        if n > 0:
            start = (self.buffer.pos - n) % self.buffer.capacity
            indices = (start + np.arange(n)) % self.buffer.capacity
            for idx in indices:
                clone.buffer.add(
                    self.buffer.state[idx],
                    int(self.buffer.action[idx]),
                    float(self.buffer.reward[idx]),
                    self.buffer.next_state[idx],
                    bool(self.buffer.done[idx]),
                    self.buffer.next_mask[idx],
                )
        return clone

    def select(
        self,
        state: np.ndarray,
        valid_mask: np.ndarray,
        epsilon: float,
        action_bonus: np.ndarray | None = None,
    ) -> int:
        valid = np.flatnonzero(valid_mask)
        if len(valid) == 0:
            return 0
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        q = self.online.predict(state)
        if action_bonus is not None:
            q = q + np.asarray(action_bonus, dtype=float)
        q = np.where(valid_mask, q, -1e30)
        return int(np.argmax(q))

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        self.buffer.add(state, action, reward, next_state, done, next_mask)

    def learn(self, batch_size: int = 48) -> float | None:
        if self.buffer.size < max(256, batch_size):
            return None
        states, actions, rewards, next_states, dones, next_masks = self.buffer.sample(
            batch_size, self.rng
        )
        online_next = self.online.predict(next_states)
        online_next = np.where(next_masks, online_next, -1e30)
        best_next = np.argmax(online_next, axis=1)
        target_next = self.target.predict(next_states)
        bootstrap = target_next[np.arange(batch_size), best_next]
        targets = rewards + self.gamma * (1.0 - dones) * bootstrap
        loss = self.online.train_batch(states, actions, targets, self.lr)
        self.gradient_steps += 1
        if self.gradient_steps % 300 == 0:
            self.target.copy_from(self.online)
        return loss


# ── auxiliary RL helpers ──

def valid_action_mask(
    belief: "JointParticleBelief",
    active_source: int,
    demand_probe_count_year: int,
    sub_probe_starts: np.ndarray,
) -> np.ndarray:
    """Return a boolean mask of permissible actions given the current probe state."""
    if active_source >= 0:
        mask = np.zeros(belief.n_actions, dtype=bool)
        mask[2 + active_source] = True
        return mask
    s = belief.summaries()
    mask = np.ones(belief.n_actions, dtype=bool)
    if demand_probe_count_year >= 8 or float(np.mean(s["mu_sd"])) < 7.0:
        mask[1] = False
    for i in range(belief.k_count):
        valid_sd = [s["a_sd"][i, j] for j in range(belief.k_count + 1) if j != i]
        if sub_probe_starts[i] >= 6 or float(np.mean(valid_sd)) < 0.025:
            mask[2 + i] = False
    return mask


def posterior_proxy_reward(
    public_obs: dict[str, Any],
    predicted_final_lost: np.ndarray,
    short_cost: np.ndarray,
) -> float:
    """Online reward proxy with posterior-predicted final lost sales."""
    return float(
        public_obs["revenue"]
        - public_obs["holding_cost"]
        - np.dot(short_cost, predicted_final_lost)
    )
