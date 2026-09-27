"""Portable entry point for new synthetic runs; archived summaries are read-only."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
METHODS = ['full','fixed_schedule','d2as2','ppo','no_exploration','no_substitution','reset_all','no_reset','informed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','smoke','run','analyze'])
    parser.add_argument('--batch', choices=['original28','extension15'], default='original28')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--case', help='One case for smoke only; default is the central case or B=15.')
    parser.add_argument('--allow-partial', action='store_true', help='For analyze: report missing evidence without generating final tables.')
    args = parser.parse_args()
    output = (args.output or PACKAGE/'outputs'/args.batch).resolve()
    if output == PACKAGE or output in PACKAGE.parents:
        parser.error('Output must not be the package directory or any of its ancestors.')
    for protected in ['algorithm','experiments','inputs','results']:
        root=(PACKAGE/protected).resolve()
        if output == root or root in output.parents or output in root.parents:
            parser.error('Output overlaps a released source/input/result directory.')
    os.environ['TSB_OUTPUT']=str(output)
    os.environ['TSB_BATCH']=args.batch
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['PYTHONUTF8']='1'
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key]='1'
    sys.dont_write_bytecode=True
    scripts=PACKAGE/'experiments'/'operating'
    sys.path.insert(0,str(scripts))
    import design_suite as design
    import run_suite as runner
    design.prepare()
    hashes=json.loads((PACKAGE/'inputs'/'generated_input_hashes.json').read_text(encoding='utf-8'))
    checked=0
    for key, expected in hashes.items():
        batch,relative=key.split('/',1)
        if batch == args.batch:
            actual=hashlib.sha256((output/'inputs'/relative).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f'Generated synthetic input differs from archived evidence: {key}')
            checked+=1
    expected_inputs=168 if args.batch=='original28' else 6
    assert checked==expected_inputs,(checked,expected_inputs)
    lock=output/'source_lock.json'
    current=runner.sources()
    if lock.exists():
        if json.loads(lock.read_text(encoding='utf-8')) != current:
            raise RuntimeError('Source changed since this run was prepared. Choose a new output directory.')
    else:
        design.dump(lock,current)
    print(f'Verified {checked} generated CSV files against original synthetic input hashes.',flush=True)
    if args.action=='prepare':
        return
    prefix=[sys.executable,'-B','-X','utf8']
    if args.action=='smoke':
        case=args.case or ('seasons_15' if args.batch=='extension15' else 'n3_t300')
        if case not in {c['id'] for c in design.scenarios()}:
            parser.error('Unknown case for this batch.')
        for method in args.methods or ['full']:
            subprocess.run([*prefix,str(scripts/'run_suite.py'),'--worker','--quick','--case',case,'--method',method,'--seed',str(design.SEEDS[0])],check=True,cwd=PACKAGE)
        print('Smoke completed. Reduced budgets are a process check, not paper performance.',flush=True)
    elif args.action=='run':
        command=[*prefix,str(scripts/'run_suite.py'),'--workers',str(max(1,args.workers))]
        if args.methods: command+=['--methods',*args.methods]
        if args.cases: command+=['--cases',*args.cases]
        subprocess.run(command,check=True,cwd=PACKAGE)
    else:
        command=[*prefix,str(scripts/'analyze_total_table.py')]
        if args.allow_partial: command.append('--allow-partial')
        subprocess.run(command,check=True,cwd=PACKAGE)


if __name__=='__main__':
    main()
