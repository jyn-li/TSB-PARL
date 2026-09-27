"""Shared constants, math utilities, and config loading."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# -- demand distribution type identifiers --
DIST_POISSON = 0
DIST_UNIFORM = 1
DIST_NB = 2

# -- RL action space --
ACTION_NAMES = (
    "exploit",
    "demand_reveal",
    "sub_probe_P1",
    "sub_probe_P2",
    "sub_probe_P3",
)
N_ACTIONS = len(ACTION_NAMES)


def action_names(k_count: int = 3) -> tuple[str, ...]:
    """Instance-local option library; legacy constants remain three-product only."""
    if int(k_count) < 2:
        raise ValueError("substitution control requires at least two products")
    return ("exploit", "demand_reveal", *(
        f"sub_probe_P{i + 1}" for i in range(int(k_count))
    ))


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -- normal distribution helpers --

def normal_pdf(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x_arr * x_arr) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: np.ndarray | float) -> np.ndarray:
    """Fast vectorized normal-CDF approximation (GELU/tanh form)."""
    x_arr = np.asarray(x, dtype=float)
    q = math.sqrt(2.0 / math.pi) * (x_arr + 0.044715 * x_arr ** 3)
    return 0.5 * (1.0 + np.tanh(q))


def positive_part_moments(
    mean: np.ndarray,
    variance: np.ndarray,
    threshold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return E[(X-threshold)+] and Var[(X-threshold)+] for normal X."""
    variance = np.maximum(np.asarray(variance, dtype=float), 1e-6)
    sigma = np.sqrt(variance)
    a = np.asarray(mean, dtype=float) - np.asarray(threshold, dtype=float)
    z = -a / sigma
    sf = np.maximum(1.0 - normal_cdf(z), 1e-12)
    pdf = normal_pdf(z)
    first = a * sf + sigma * pdf
    second = (a * a + variance) * sf + a * sigma * pdf
    first = np.maximum(first, 0.0)
    var = np.maximum(second - first * first, 1e-6)
    return first, var


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="left")
