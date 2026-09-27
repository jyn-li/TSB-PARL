# Released synthetic data

All CSVs are exact copies of archived synthetic analysis outputs. Some files have a UTF-8 byte-order mark; Python readers can use `encoding='utf-8-sig'`. IDs, seeds, decimal precision and negative results have been preserved.

## Operating conditions

`total_table.csv` has 29 rows and 12 fields. `case_label`, `n` and `t` identify the condition, number of products and decision horizon. `full_unit_profit` is total profit divided by products × periods. `gain_METHOD` is `100 × (mean Full profit − mean METHOD profit) / mean METHOD profit`. `informed_gap` reverses the numerator and uses the known-parameter reference mean in the denominator. A negative reference gap means Full exceeded that feasible myopic policy; it is not a violation of an upper bound.

`all_policy_means.csv` contains 29 × 9 policy rows. Monetary totals are per episode, before division by products or periods; `unit_profit` uses that division. Sales and final lost demand are customer/unit counts. `fill_rate` is a proportion. These means average four replications within each path–seed cell, followed by equal averaging over five paths and five training seeds.

`paired_intervals.csv` contains 29 × 8 comparisons. `crossed95_*` fields are pointwise 95% intervals from 20,000 crossed path/seed bootstrap draws. Fields with `percent` are percentage points of the ratio, not fractional units. `seed_t95_*` uses five seed-level differences averaged over paths; `seed_ratio_*` separately describes the seed-level ratios. The latter is not the ratio-of-grand-means point estimator used in the main table.

`seed_level_comparisons.csv` preserves each training seed's path-averaged paired comparison. The `seasons_15` condition belongs to the separate extension; the other 28 belong to the original batch. Original dates and design history are explained in the README, not inferred from merged row order.

Policy codes: `full` = TSB-PARL; `fixed_schedule` = Fixed schedule; `d2as2` = D2AS2 adapted to this environment; `ppo` = PPO; `no_exploration` = Exploit-only; `no_substitution` = Demand-only; `reset_all` = Reset all; `no_reset` = No reset; `informed` = known-parameter myopic reference.

## Inventory diagnostics

These summaries use the same `n3_t300` central evidence as the operating-condition table. `method_path_seed_metrics.csv` contains path–seed averages of episode-level totals. `product_path_seed_metrics.csv` adds product-level inventory targets, sales, primary demand, leftovers, final lost demand and prediction diagnostics. Inventory, sales and lost-demand fields in this product file are averaged per period; `revenue` is the episode total for that product. `product_mean_and_range.csv` aggregates those product cells.

`paired_path_seed_decomposition.csv` preserves paired policy differences used to reconcile profit, revenue and costs. `information_action_cells.csv` contains information-action use and diagnostics; `boundary_substitution_transitions.csv` concerns inferred substitution behavior around demand-stage boundaries. These diagnostic associations and accounting decompositions do not identify causal mediation effects.

## Economic mechanisms

`grid36_cost_benefit.csv` has one row for every `rho` × `eta` pair. `grid36_cost_benefit_seed_details.csv` has three paired seed-level rows per cell. The baseline is Demand-only, and `delta_*` means Full minus Demand-only. Monetary quantities are episode totals for 300 periods and three products.

The identity is `delta_total_profit = delta_total_revenue − delta_holding_cost − delta_lost_sales_cost`. Thus `holding_contribution` and `lost_contribution` are the negatives of the corresponding cost changes. A negative contribution worsens profit. `volume_effect` and `mix_effect` partition the revenue change. The grid is a fixed-path exploratory comparison, so its three-seed intervals do not include variation across independent demand paths.

`rho` is the common non-exit substitution probability. `eta` scales deviations of product price from the original across-product mean; see the exact formula in `protocols/economic_grid.json`. Costs retain their original values. Every cell is retained, including negative profit increments.
