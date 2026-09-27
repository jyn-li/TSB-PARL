"""Run the demand-only TSB-PARL ablation without changing ``algo_rl``.

This adapter reuses the validated environment, allocator, critic, diagnostics,
and experiment seeds from ``algo_rl`` while replacing only its joint belief by
``DemandOnlyBelief``.  Outputs default to this directory's ``output`` folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAIN_DIR = ROOT / "experiments" / "Nosub"
CORE_DIR = ROOT / "src" / "tsb_parl"
for directory in (CORE_DIR, HERE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from demand_only_belief import DemandOnlyBelief  # noqa: E402
import execute_method as shared_experiment  # noqa: E402


def _output_argument() -> Path:
    if "--output" in sys.argv:
        index = sys.argv.index("--output")
        if index + 1 >= len(sys.argv):
            raise SystemExit("--output requires a path")
        return Path(sys.argv[index + 1]).resolve()
    output = ROOT / "运行结果" / "单次无替代学习"
    sys.argv.extend(["--output", str(output)])
    return output


def _label_summary(output: Path) -> None:
    summary_path = output / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    diagnostics = summary["learning_diagnostics"]
    assumed_matrix_mae = diagnostics.pop(
        "substitution_mae_of_mean_posterior_matrix"
    )
    diagnostics.pop("substitution_final_mae_mean_across_replications")
    diagnostics.pop("substitution_final_mae_std_across_replications")
    diagnostics["no_substitution_assumption_matrix_mae"] = assumed_matrix_mae
    summary["algorithm"] = "TSB-PARL comparative-v3 noSub ablation"
    summary["ablation"] = {
        "removed_module": "substitution learning",
        "decision_model": (
            "all unmet primary demand exits; cross-product substitution is zero"
        ),
        "true_environment_substitution_unchanged": True,
        "rl_inventory_decision_unchanged": True,
    }
    summary["settings"]["substitution_prior"] = None
    summary["settings"]["assumed_substitution"] = "EXIT probability 1 for every source"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    old_matrix = output / "substitution_learning.csv"
    new_matrix = output / "assumed_substitution_matrix.csv"
    if old_matrix.exists():
        old_matrix.replace(new_matrix)


def main() -> None:
    output = _output_argument()
    shared_experiment.JointParticleBelief = DemandOnlyBelief
    shared_experiment.main()
    _label_summary(output)
    print(f"[ablation] labelled noSub outputs: {output}")


if __name__ == "__main__":
    main()
