"""Run the dependency-free semantic regression suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Some portable/embedded Python builds omit the script directory from
# ``sys.path``. Keep this dependency-free test runner directly executable.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import test_environment


def main() -> None:
    tests = sorted(
        name for name in dir(test_environment) if name.startswith("test_")
    )
    for name in tests:
        getattr(test_environment, name)()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
