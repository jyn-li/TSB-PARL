"""Portable locations for the public synthetic-only research package."""
from pathlib import Path

ALGORITHM = Path(__file__).resolve().parent
WORKSPACE = ALGORITHM.parent
INPUTS = WORKSPACE / 'inputs'
COMMON = ALGORITHM / 'experiments' / 'common'
BENCHMARK_RESULTS = WORKSPACE / 'outputs' / 'unbundled_historical_benchmarks'

def resolve_path(value, base=None):
    path = Path(value)
    return path if path.is_absolute() else (Path(base) if base else WORKSPACE) / path
