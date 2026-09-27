"""Verify the public file manifest and numerical identities of released summaries."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def rows(relative):
    with (ROOT/relative).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

def close(a,b):
    assert abs(float(a)-float(b))<1e-6,(a,b)

def main():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        path=ROOT/record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],record['path']
    table=rows('results/operating_conditions/total_table.csv')
    means=rows('results/operating_conditions/all_policy_means.csv')
    intervals=rows('results/operating_conditions/paired_intervals.csv')
    assert (len(table),len(means),len(intervals))==(29,261,232)
    by_case_method={(r['case_id'],r['method']):r for r in means}
    for row in means:
        close(row['total_profit'],float(row['total_revenue'])-float(row['holding_cost'])-float(row['lost_sales_cost']))
        close(row['unit_profit'],float(row['total_profit'])/(int(row['n'])*int(row['t'])))
    for row in table:
        candidates=[r for r in means if r['case_label']==row['case_label'] and r['method']=='full']
        assert len(candidates)==1
        full=candidates[0]
        close(row['full_unit_profit'],full['unit_profit'])
        for field,value in row.items():
            if field.startswith('gain_'):
                baseline=by_case_method[full['case_id'],field.removeprefix('gain_')]
                close(value,100*(float(full['total_profit'])-float(baseline['total_profit']))/float(baseline['total_profit']))
    mechanism=rows('results/inventory_diagnostics/method_path_seed_metrics.csv')
    for row in mechanism:
        close(row['total_profit'],float(row['total_revenue'])-float(row['holding_cost'])-float(row['lost_sales_cost']))
    for method in {r['method'] for r in mechanism}:
        items=[float(r['total_profit']) for r in mechanism if r['method']==method]
        assert len(items)==25
        close(sum(items)/len(items),by_case_method['n3_t300',method]['total_profit'])
    grid=rows('results/economic_mechanism/grid36_cost_benefit.csv')
    details=rows('results/economic_mechanism/grid36_cost_benefit_seed_details.csv')
    assert len(grid)==36 and len(details)==108
    assert len({(r['rho'],r['eta']) for r in grid})==36
    for row in [*grid,*details]:
        close(row['delta_total_profit'],float(row['delta_total_revenue'])-float(row['delta_holding_cost'])-float(row['delta_lost_sales_cost']))
    for cell in grid:
        seeds=[r for r in details if (r['rho'],r['eta'])==(cell['rho'],cell['eta'])]
        assert len(seeds)==3
        for key in ['full_profit','baseline_profit','delta_total_profit','delta_total_revenue','delta_holding_cost','delta_lost_sales_cost']:
            close(cell[key],sum(float(r[key]) for r in seeds)/3)
    print(f"Verified {len(manifest['files'])} file hashes; 29 scenarios, 261 policy means, 232 intervals; matched central diagnostics; all 36 economic cells and 108 seed pairs.")

if __name__=='__main__':
    main()
