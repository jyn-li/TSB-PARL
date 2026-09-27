# TSB-PARL: synthetic inventory experiments

Research code and synthetic simulation summaries for an inventory policy that learns demand and substitution from censored sales feedback. The release includes the shared TSB-PARL implementation, policy-component comparisons, D2AS2 adaptation, PPO baseline, synthetic input generation, and the numerical summaries used by the manuscript.

**Scope:** this is a compact public code and synthetic-data release. It does not contain the enterprise case study, company sales or budgets, spreadsheets, personal files, trained model archives, or the complete period-level evidence archive. It is not a complete archival reproduction bundle for every manuscript experiment.

## Manuscript map

The current manuscript orders the simulation discussion as follows. Some original experiment names and comments retain the earlier section numbering.

| Manuscript section | Public files | Evidence and interpretation |
|---|---|---|
| 5.1 Inventory diagnostics | `results/inventory_diagnostics/` | Central setting: 3 products, 300 periods, 5 paths, 5 training seeds and 4 evaluation repetitions; inventory, sales, accounting and information-action diagnostics. |
| 5.2 Operating conditions | `experiments/operating/`, `results/operating_conditions/` | 28 original scenarios plus a separately added 15-stage scenario. All 29 conditions and all 9 policies are retained in the released summaries. |
| 5.3 Economic mechanisms | `results/economic_mechanism/`, `protocols/economic_grid.json` | All 36 cells of the substitution-intensity × contribution-heterogeneity grid, with all 108 paired seed-level summaries. |

The known-parameter reference (`informed`) is a feasible **myopic policy** using known demand moments and substitution probabilities. It uses the same candidate construction and moment scoring procedure, but different inventories and information can generate different candidate points. It is neither an optimal oracle nor a certified upper bound.

The 6 × 6 economic grid is an **exploratory single-path study** with three training seeds and four paired evaluations per seed, for both Full and Demand-only. It is not independent five-path/five-seed validation. It was expanded after inspecting earlier grids. All settings, including unfavorable outcomes, are released. Operating cost changes include allocation and service consequences; they are not all information-acquisition costs.

## Install and check

Use Python 3.12. The algorithm and data checks require NumPy only; PPO additionally requires PyTorch. Commands below work from the repository directory in a terminal on Windows, Linux or macOS.

```text
python -m pip install -r requirements.txt
python verify_release.py
python -B algorithm/tests/run_tests.py
python -B reproduce.py smoke
```

For PPO, install its additional dependency and run an optional smoke check:

```text
python -m pip install -r requirements-ppo.txt
python -B reproduce.py smoke --methods ppo
```

`smoke` runs one training episode and one evaluation on one path with reduced particle counts. It checks execution and accounting, and does not estimate paper performance. All new runs go to `outputs/` by default. Released summaries in `results/` are never used as writable run outputs.

## Re-run operating-condition experiments

The original numerical worker, allocator, beliefs, environment, DQN, and baseline implementations are included. The shared core and common executors match the latest numerical source lock byte for byte. Public adapters change repository paths, output locations and batch selection; they do not change the core numerical algorithms. `SOURCE_PROVENANCE.json` records identities before and after adaptation.

```text
python -B reproduce.py prepare --batch original28
python -B reproduce.py run --batch original28 --workers 4
python -B reproduce.py analyze --batch original28

python -B reproduce.py prepare --batch extension15
python -B reproduce.py run --batch extension15 --workers 4
python -B reproduce.py analyze --batch extension15
```

Preparing a batch regenerates its synthetic inputs and checks every demand/substitution CSV against the original hashes: 168 CSVs for the original 28 conditions and 6 for the stage-count extension. Run configurations use the same numerical parameters with portable locations. Both batches remain separate. The 15-stage condition was added after the original results were known and fixed before its own run; the 29 conditions were not one jointly prespecified batch.

Full runs are substantial: the original batch has 25,200 evaluation episodes and the extension has 900. Each scenario uses 5 independent demand-path seeds, 5 training seeds and 4 paired repetitions per path–seed cell. Learned policies use 24,000 offline interactions per model; rules without policy training use zero. Models are shared only when the public economics, calendar, training law, method and seed agree. Full updates its critic online; PPO freezes its trained network during evaluation and continues belief updates.

For a selected starting scenario or policy:

```text
python -B reproduce.py run --cases n3_t300 --methods full d2as2 --workers 2
python -B reproduce.py analyze --allow-partial
```

`--cases` selects scheduling starting points. Scenarios with identical training signatures are evaluated together, so it is not a strict one-row filter. The analyzer refuses to produce a complete table when evidence is missing. A custom `--output PATH` allows separate experiments. Do not reuse an output directory after changing the numerical source; source hashes are checked.

Reproduction on another software/hardware stack may have floating-point differences. The release was checked with Python 3.12, NumPy 2.3.4 and PyTorch 2.9.1; no full 26,100-episode retraining was performed for this export. The archived summaries are exact copies, not replacement values from the smoke runs.

## Files and statistical conventions

- `algorithm/src/tsb_parl/`: environment, joint and demand-only beliefs, allocator, action rules, DQN and utilities.
- `algorithm/experiments/common/`: shared training and evaluation executors.
- `experiments/operating/`: original design, 9-policy worker, literature baselines and audited aggregation, with portable paths.
- `inputs/`: synthetic parameter template, frozen illustrative Path 1 input and hashes for regenerating both operating-condition batches. The illustrative Path 1 is the older economic-grid path; operating-condition paths use separate seeds.
- `results/operating_conditions/`: 29-row full table, 261 policy means, 232 paired intervals and seed comparisons. Fixed schedule is retained even where omitted from the printed table.
- `results/inventory_diagnostics/`: central inventory, product-sales and profit decompositions from the same five-path/five-seed center as the operating table.
- `results/economic_mechanism/`: complete 36-cell economic accounting and three-seed paired details. The compact release omits the historical grid runner/checkpoints; these CSVs support reanalysis, not a claim of exact end-to-end historical grid retraining.
- `MANIFEST.json`: hashes of all release files except the manifest itself.

Operating comparisons first average four repetitions in each path–seed cell and then average the 5 × 5 cells equally. Percentage gains divide the difference of grand means by the baseline grand mean. Intervals use 20,000 crossed resamples of path and training-seed indices; the four repetitions are not treated as extra independent training seeds. These are pointwise intervals without a multiplicity correction. Inventory diagnostics are accounting descriptions, not identified causal mediation effects.

The 5- and 10-product settings use structured extensions of the original three product types. Stage-count comparisons change the learning window, calendar and integer information budget jointly. These experiments do not establish universal performance or fully tuned limits of competing methods. D2AS2 is an adaptation to this inventory setting, not an assertion that the original authors supplied this implementation.

## License and data provenance

No open-source license has been selected by the authors for this release. Public visibility allows inspection; it does not by itself grant an open-source redistribution or reuse license. Licensing terms remain to be decided by the authors. No third-party source code is vendored here; NumPy and PyTorch are external dependencies with their own licenses.

Every released numerical data file comes from synthetic experiments. Company-calibrated or real sales/budget data are excluded. Source-to-release checksums are retained without publishing private filesystem paths. The public summaries preserve original values, signs, seed identifiers and complete grids.

中文操作说明：先执行环境安装、`verify_release.py` 与快速检查；正式复跑请分别运行 `original28` 和 `extension15`。公开汇总为原记录的逐字节副本，试跑不能作为论文正式数值。企业案例及企业原始数据不在本仓库中。
