"""Audited aggregation for the prespecified 28-scenario single performance table.

This script never trains or edits inputs/results. Incomplete evidence produces
only a progress report under --allow-partial, never a partial final table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from design_suite import ROOT, scenarios
N_CASES = len(scenarios())
OUT = ROOT / 'analysis'
METHODS = ['full', 'fixed_schedule', 'd2as2', 'ppo', 'no_exploration', 'no_substitution', 'reset_all', 'no_reset', 'informed']
INTERNAL = set(METHODS) - {'d2as2', 'ppo', 'informed'}
BASELINES = METHODS[1:-1]
METRICS = ['total_profit', 'total_revenue', 'holding_cost', 'lost_sales_cost', 'total_sales', 'total_final_lost', 'fill_rate']
TABLE_FIELDS = ['case_label', 'n', 't', 'full_unit_profit', 'gain_fixed_schedule', 'gain_d2as2', 'gain_ppo', 'gain_no_exploration', 'gain_no_substitution', 'gain_reset_all', 'gain_no_reset', 'informed_gap']
N_BOOTSTRAP = 20000
HASHES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_read(path):
    data = path.read_bytes()
    HASHES[str(path)] = hashlib.sha256(data).hexdigest()
    return data.decode('utf-8-sig')


def read_json(path):
    return json.loads(recorded_read(path))


def read_csv(path):
    return list(csv.DictReader(recorded_read(path).splitlines()))


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows, fields=None):
    assert rows
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def public_signature(c, method):
    return {'calendar': c['calendar'], 'economics': [[p[k] for k in ('price', 'holding_cost', 'shortage_cost')] for p in c['products']],
            'capacity': c['simulation']['shared_capacity'], 'initial_inventory': c['simulation']['initial_inventory'],
            'counter_crn': c['evaluation']['counter_substitution_crn'], 'method': method,
            'training_law': 'broad_exchangeable_v1', 'training_steps': 24000}


def group_id(signature):
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]


def public_baseline_config(c):
    return {'products': [{k: p[k] for k in ('id', 'price', 'holding_cost', 'shortage_cost')} for p in c['products']],
            'calendar': c['calendar'], 'simulation': {'shared_capacity': int(c['simulation']['shared_capacity']),
                                                     'initial_inventory': c['simulation'].get('initial_inventory')}}


def case_label(case):
    key, val = case['parameter'], case['level']
    if key == '规模':
        return f"K = {case['n']}, T = {case['t']}" + (' (central)' if case['id'] == 'n3_t300' else '')
    return {
        'seasons': f'{int(val)} demand stages',
        'capacity': f'Capacity: {val:g} x baseline',
        'noise': f'Demand noise: {val:g} x baseline',
        'drift': f'Mean-drift amplitude: {val:g} x baseline',
        'holding': f'Holding cost: {val:g} x baseline',
        'shortage': f'Lost-sales penalty: {val:g} x baseline',
    }[key]


def verify_protocol(protocol):
    assert protocol['scenarios'] == scenarios()
    assert protocol['methods'] == METHODS
    assert protocol['replications'] == 4 and protocol['training_steps'] == 24000
    assert len(protocol['training_seeds']) == 5 and len(protocol['path_seeds']) == 5
    mismatches = []
    for relative, expected in protocol['input_sha256'].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(f'Input manifest mismatch: {mismatches[:10]}')
    lock = read_json(ROOT / 'source_lock.json')
    for filename, expected in lock.items():
        if sha(Path(filename)) != expected:
            raise ValueError(f'Changed numerical source: {filename}')
    return {'input_files_verified': len(protocol['input_sha256']), 'numerical_sources_verified': len(lock)}


def progress_report(protocol):
    missing, completed, groups = [], [], []
    for case in protocol['scenarios']:
        for method in METHODS:
            count = 0
            for path in range(1, 6):
                for seed in protocol['training_seeds']:
                    folder = ROOT / 'results' / case['id'] / method / f'path{path}' / f'seed{seed}'
                    ident = {'case': case['id'], 'method': method, 'path': path, 'seed': seed}
                    if (folder / 'COMPLETE.json').is_file():
                        count += 1
                        completed.append(ident)
                    else:
                        missing.append(ident)
            groups.append({'case': case['id'], 'method': method, 'complete_cells': count, 'required_cells': 25})
    return {'generated_utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete' if not missing else 'incomplete',
            'expected_cells': N_CASES*225, 'completed_cells': len(completed), 'missing_cells': len(missing),
            'expected_evaluation_replications': N_CASES*900, 'groups': groups, 'missing': missing,
            'final_outputs_created': False}


def validate_training(folder, method, seed, c, complete, group_records):
    signature = public_signature(c, method)
    gid = group_id(signature)
    training = ROOT / 'results' / 'training' / method / gid / f'seed{seed}'
    assert Path(complete['training_directory']).resolve() == training.resolve(), (folder, complete['training_directory'], training)
    key = (method, gid, seed)
    if key in group_records:
        return
    record = {'method': method, 'group_id': gid, 'seed': seed, 'directory': str(training),
              'public_signature': signature, 'training_steps': 0, 'model_hash': None}
    if method in INTERNAL:
        provenance = read_json(training / 'training_provenance.json')
        assert provenance['method'] == method and provenance['seed'] == seed
        assert provenance['public_signature'] == signature
        expected_steps = 0 if method in ('fixed_schedule', 'no_exploration') else 24000
        assert provenance['training_steps'] == expected_steps
        model = training / 'evaluation_checkpoint.npz'
        model_hash = sha(model)
        HASHES[str(model)] = model_hash
        assert model_hash == provenance['checkpoint_sha256']
        with np.load(model, allow_pickle=False) as values:
            assert int(values['n_actions']) == len(c['products']) + 2
            assert all('online_' + k in values and 'target_' + k in values for k in ('w1', 'b1', 'w2', 'b2'))
            assert len(values['buffer_action']) == (0 if expected_steps == 0 else 1500)
        record.update(training_steps=expected_steps, model_hash=model_hash, shared_cases=provenance['shared_cases'])
    elif method == 'ppo':
        metadata = read_json(training / 'training_metadata.json')
        assert metadata['method'] == 'ppo' and metadata['train_steps'] == 24000
        assert metadata['seed'] == seed + 8100
        expected_public = hashlib.sha256(json.dumps(public_baseline_config(c), sort_keys=True).encode()).hexdigest()
        assert metadata['public_config_sha256'] == expected_public
        model = training / 'ppo_model.pt'
        model_hash = sha(model)
        HASHES[str(model)] = model_hash
        record.update(training_steps=24000, model_hash=model_hash)
    elif method == 'd2as2':
        model = training / 'd2as2_policy.json'
        metadata = read_json(model)
        assert metadata['method'] == method and metadata['seed'] == seed + 8100 and metadata['train_steps'] == 0
        record.update(model_hash=sha(model))
    elif method == 'informed':
        record.update(model_hash='not_applicable_known_parameter_rule')
    group_records[key] = record


def collect(protocol):
    arrays, cell_rows, rep_rows, group_records = {}, [], [], {}
    for case in protocol['scenarios']:
        for method in METHODS:
            values = {metric: np.zeros((5, 5), dtype=float) for metric in METRICS}
            for path in range(1, 6):
                for si, seed in enumerate(protocol['training_seeds']):
                    folder = ROOT / 'results' / case['id'] / method / f'path{path}' / f'seed{seed}'
                    complete = read_json(folder / 'COMPLETE.json')
                    expected = {'case': case['id'], 'method': method, 'path': path, 'seed': seed, 'replications': 4}
                    assert all(complete[k] == v for k, v in expected.items()), folder
                    config_file = ROOT / 'inputs' / case['id'] / f'path{path}_seed{seed}.json'
                    c = read_json(config_file)
                    assert c['random_seed'] == seed and c['development_metadata']['path_id'] == path
                    assert complete['input_sha256'] == sha(config_file)
                    csv_file = folder / 'evaluation_replications.csv'
                    assert complete['evaluation_sha256'] == sha(csv_file)
                    reps = read_csv(csv_file)
                    assert len(reps) == 4 and {int(r['replication']) for r in reps} == {0, 1, 2, 3}
                    for r in reps:
                        numbers = {k: float(r[k]) for k in METRICS}
                        assert all(np.isfinite(v) for v in numbers.values()), folder
                        residual = numbers['total_revenue'] - numbers['holding_cost'] - numbers['lost_sales_cost'] - numbers['total_profit']
                        assert abs(residual) < 1e-6, (folder, r['replication'], residual)
                        assert numbers['holding_cost'] >= 0 and numbers['lost_sales_cost'] >= 0
                        rep_rows.append(dict(case_id=case['id'], method=method, path=path, seed=seed, replication=int(r['replication']), **numbers))
                    averages = {k: float(np.mean([float(r[k]) for r in reps])) for k in METRICS}
                    summary = read_json(folder / 'summary.json')
                    assert abs(averages['total_profit'] - summary['profit']['mean']) < 1e-6, folder
                    for k in METRICS:
                        values[k][path-1, si] = averages[k]
                    cell_rows.append(dict(case_id=case['id'], method=method, path=path, seed=seed, replications=4, **averages))
                    validate_training(folder, method, seed, c, complete, group_records)
            arrays[case['id'], method] = values
    # Same method/public-signature/seed must point to exactly one checkpoint across paths and cases.
    expected_groups = {(method, group_id(public_signature(read_json(ROOT/'inputs'/case['id']/f'path1_seed{seed}.json'), method)), seed)
                       for case in protocol['scenarios'] for method in METHODS for seed in protocol['training_seeds']}
    assert set(group_records) == expected_groups
    for key, record in group_records.items():
        if record['method'] in INTERNAL:
            expected_cases = {case['id'] for case in protocol['scenarios']
                              if group_id(public_signature(read_json(ROOT/'inputs'/case['id']/f"path1_seed{record['seed']}.json"), record['method'])) == record['group_id']}
            assert set(record['shared_cases']) == expected_cases, (record['directory'], record['shared_cases'], expected_cases)
    return arrays, cell_rows, rep_rows, list(group_records.values())


def percent_difference(full, baseline, informed=False):
    if np.any(np.abs(baseline) < 1e-12):
        raise ValueError('Zero baseline profit makes a percentage undefined; report absolute values before revising presentation.')
    return 100. * ((baseline - full) if informed else (full - baseline)) / baseline


def build_tables(protocol, arrays):
    rng = np.random.default_rng(2026091428)
    pd = rng.integers(0, 5, (N_BOOTSTRAP, 5))
    sd = rng.integers(0, 5, (N_BOOTSTRAP, 5))
    table, means, intervals, seed_details = [], [], [], []
    for case in protocol['scenarios']:
        full = arrays[case['id'], 'full']['total_profit']
        full_boot = full[pd[:, :, None], sd[:, None, :]].mean(axis=(1, 2))
        row = dict(case_label=case_label(case), n=case['n'], t=case['t'], full_unit_profit=float(full.mean() / (case['n'] * case['t'])))
        for method in METHODS:
            val = arrays[case['id'], method]
            means.append(dict(case_id=case['id'], case_label=case_label(case), group=case['group'], parameter=case['parameter'], level=case['level'], n=case['n'], t=case['t'], method=method,
                              n_paths=5, n_training_seeds=5, n_replications_per_cell=4,
                              **{k: float(v.mean()) for k, v in val.items()}, unit_profit=float(val['total_profit'].mean() / (case['n'] * case['t'])),
                              min_path_seed_profit=float(val['total_profit'].min()), max_path_seed_profit=float(val['total_profit'].max())))
            if method == 'full':
                continue
            base = val['total_profit']
            informed = method == 'informed'
            boot_base = base[pd[:, :, None], sd[:, None, :]].mean(axis=(1, 2))
            boot_gain = percent_difference(full_boot, boot_base, informed)
            delta = (base-full) if informed else (full-base)
            delta_boot = delta[pd[:, :, None], sd[:, None, :]].mean(axis=(1, 2))
            seed_delta = delta.mean(axis=0)
            seed_gain = percent_difference(full.mean(axis=0), base.mean(axis=0), informed)
            seed_half = 2.7764451051977987 * seed_delta.std(ddof=1) / math.sqrt(5)
            ratio_half = 2.7764451051977987 * seed_gain.std(ddof=1) / math.sqrt(5)
            gain = float(percent_difference(full.mean(), base.mean(), informed))
            field = 'informed_gap' if informed else f'gain_{method}'
            row[field] = gain
            intervals.append(dict(case_id=case['id'], case_label=case_label(case), baseline=method, statistic='informed_minus_full_over_informed' if informed else 'full_minus_baseline_over_baseline',
                                  full_mean_profit=float(full.mean()), baseline_mean_profit=float(base.mean()), difference=float(delta.mean()), percent=gain,
                                  crossed95_low=float(np.quantile(delta_boot, .025)), crossed95_high=float(np.quantile(delta_boot, .975)),
                                  crossed95_percent_low=float(np.quantile(boot_gain, .025)), crossed95_percent_high=float(np.quantile(boot_gain, .975)),
                                  seed_t95_low=float(seed_delta.mean()-seed_half), seed_t95_high=float(seed_delta.mean()+seed_half),
                                  seed_ratio_mean_percent=float(seed_gain.mean()), seed_ratio_t95_percent_low=float(seed_gain.mean()-ratio_half), seed_ratio_t95_percent_high=float(seed_gain.mean()+ratio_half),
                                  min_paired_path_seed_difference=float(delta.min()), max_paired_path_seed_difference=float(delta.max()),
                                  n_paths=5, n_training_seeds=5, replications_per_cell=4, bootstrap_draws=N_BOOTSTRAP))
            for si, seed in enumerate(protocol['training_seeds']):
                seed_details.append(dict(case_id=case['id'], baseline=method, seed=seed, full_mean_over_paths=float(full[:,si].mean()), baseline_mean_over_paths=float(base[:,si].mean()),
                                         paired_difference=float(seed_delta[si]), percent=float(seed_gain[si])))
        assert list(row) == TABLE_FIELDS
        table.append(row)
    return table, means, intervals, seed_details



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-partial', action='store_true', help='Report completion counts only while data are missing.')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = read_json(ROOT / 'protocol.json')
    checks = verify_protocol(protocol)
    progress = progress_report(protocol)
    progress.update(checks)
    dump(OUT / 'progress.json', progress)
    if progress['missing_cells']:
        message = f"Incomplete: {progress['completed_cells']}/{N_CASES*225} cells; {progress['missing_cells']} missing. No final tables generated."
        print(message)
        if args.allow_partial:
            return
        raise SystemExit(2)
    arrays, cells, reps, models = collect(protocol)
    table, means, intervals, seeds = build_tables(protocol, arrays)
    assert len(table) == N_CASES and len(means) == N_CASES*9 and len(intervals) == N_CASES*8 and len(cells) == N_CASES*225 and len(reps) == N_CASES*900
    # Verify both immutable input manifest and every read evidence file again before publishing output.
    verify_protocol(protocol)
    for filename, expected in HASHES.items():
        assert sha(Path(filename)) == expected, f'Evidence changed during analysis: {filename}'
    write_csv(OUT / 'total_table.csv', table, TABLE_FIELDS)
    write_csv(OUT / 'all_policy_means.csv', means)
    write_csv(OUT / 'paired_intervals.csv', intervals)
    write_csv(OUT / 'seed_level_comparisons.csv', seeds)
    write_csv(OUT / 'path_seed_metrics.csv', cells)
    write_csv(OUT / 'replication_accounting.csv', reps)
    dump(OUT / 'model_sharing_audit.json', {'model_groups': len(models), 'groups': models})
    # Narrative manuscript generation is excluded from the public package.
    progress.update(status='complete_and_verified', final_outputs_created=True, model_groups=len(models), evidence_files_verified=len(HASHES))
    dump(OUT / 'progress.json', progress)
    dump(OUT / 'validation.json', {'status': 'passed', **checks, 'expected_cells': N_CASES*225, 'verified_replications': N_CASES*900, 'bootstrap_draws': N_BOOTSTRAP,
                                 'analysis_script_sha256': sha(Path(__file__)),
                                 'percentage': '100*(mean TSB-PARL - mean baseline)/mean baseline; informed gap reverses difference and divides by mean informed',
                                 'aggregation': '4 replications -> each path-seed mean -> equally weighted 5x5 crossed mean', 'input_files_unchanged': True,
                                 'source_and_evidence_hashes': HASHES})
    print(json.dumps({'status': 'passed', 'table_rows': N_CASES, 'table_columns': 12, 'policy_means_rows': N_CASES*9, 'interval_rows': N_CASES*8, 'model_groups': len(models), 'output': str(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
