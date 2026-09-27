"""Prespecified operating-condition design; no outcome-dependent selection."""
from pathlib import Path
import sys, json, csv, copy, hashlib, os
from datetime import datetime, timezone
import numpy as np

WORKSPACE = next(p for p in Path(__file__).resolve().parents if (p/'algorithm'/'project_paths.py').exists())
sys.path.insert(0, str(WORKSPACE/'algorithm'))
from project_paths import ALGORITHM, INPUTS
sys.path.insert(0, str(ALGORITHM/'src'/'tsb_parl'))
from environment import truth_from_config, sample_base_demand
BATCH = os.environ.get('TSB_BATCH', 'original28')
if BATCH not in ('original28', 'extension15'):
    raise ValueError('Unknown batch')
ROOT = Path(os.environ.get('TSB_OUTPUT', str(WORKSPACE/'outputs'/BATCH))).resolve()
SEEDS = [202610001,203610001,204610001,205610001,206610001]
PATH_SEEDS = [390000001,391000001,392000001,393000001,394000001]
METHODS = ['full','fixed_schedule','d2as2','ppo','no_exploration','no_substitution','reset_all','no_reset','informed']

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2),encoding='utf-8')

def scenarios():
    if BATCH == 'extension15':
        return [dict(id='seasons_15',group='阶段数量与日历结构',n=3,t=300,seasons=15,parameter='seasons',level=15)]
    rows=[]
    for n in (3,5,10):
        for t in (150,300,600):
            rows.append(dict(id=f'n{n}_t{t}',group='产品数量与期限',n=n,t=t,seasons=5,parameter='规模',level=1.0))
    dimensions=[('seasons','阶段数量与日历结构',[3,10]),('capacity','容量松紧',[.8,.9,1.1,1.2]),
                ('noise','需求波动',[.5,1.5,2.]),('drift','需求变化幅度',[0.,.5,1.5,2.]),
                ('holding','持有成本',[.5,1.5,2.]),('shortage','失销惩罚',[.5,1.5,2.])]
    for key,group,values in dimensions:
        for value in values:
            rows.append(dict(id=f'{key}_{value:g}',group=group,n=3,t=300,
                             seasons=int(value) if key=='seasons' else 5,parameter=key,level=value))
    assert len(rows)==28
    return rows

def make_config(case):
    base=json.loads((INPUTS/'configs'/'C6_HIGHER_ROUTE_VALUE_path1_train20269601.json').read_text(encoding='utf-8'))
    n,t,b=case['n'],case['t'],case['seasons']
    c=copy.deepcopy(base); c['calendar']={'num_years':b,'days_per_year':t//b}
    old=base['products']; types=np.arange(n)%3
    products=[copy.deepcopy(old[int(j)]) for j in types]
    for i,p in enumerate(products): p.update(id=f'P{i+1}',name=f'Product {i+1}')
    # Preserve mean unit economics across the structured assortment extension.
    for key in ('price','holding_cost','shortage_cost'):
        scale=np.mean([p[key] for p in old])/np.mean([p[key] for p in products])
        for p in products:p[key]*=float(scale)
    old_means=np.array([[p['yearly_regimes'][s]['target_mean'] for p in old] for s in range(5)])
    source_pos=np.linspace(0,1,5); target_pos=np.linspace(0,1,b)
    means=np.column_stack([np.interp(target_pos,source_pos,old_means[:,j]) for j in types])
    if b != 5:
        # Match the original product-specific temporal mean and standard deviation
        # so regime-count changes do not mechanically reduce drift amplitude.
        original=old_means[:,types]
        means=(means-means.mean(axis=0))*original.std(axis=0)/np.maximum(means.std(axis=0),1e-9)+original.mean(axis=0)
    # Center every product's regime sequence, then preserve aggregate per-SKU scale.
    means*=old_means.mean()/means.mean()
    if case['parameter']=='drift':means=means.mean(axis=0)+(means-means.mean(axis=0))*case['level']
    for i,p in enumerate(products):
        p['yearly_regimes']=[]
        for s in range(b):
            original=copy.deepcopy(old[int(types[i])]['yearly_regimes'][int(round(target_pos[s]*4))])
            original.update(year=s+1,target_mean=float(means[s,i]));p['yearly_regimes'].append(original)
    matrix=np.array([[base['substitution']['matrix'][f'P{i+1}'][d] for d in ['P1','P2','P3','EXIT']] for i in range(3)])
    if n==3: a=matrix.copy()
    else:
        retention=1-matrix[types,-1];retention*=np.mean(1-matrix[:,-1])/retention.mean()
        a=np.zeros((n,n+1))
        for i in range(n):
            weights=matrix[types[i],types].copy();weights[i]=0
            for cls in range(3):weights[types==cls]/=max(np.sum(types==cls),1)
            assert weights.sum()>0 and 0<retention[i]<1
            a[i,:n]=retention[i]*weights/weights.sum();a[i,-1]=1-retention[i]
    order=[f'P{i+1}' for i in range(n)]+['EXIT']
    c['products']=products
    c['substitution']['matrix']={order[i]:dict(zip(order,a[i].tolist())) for i in range(n)}
    capacity=round(490*n/3)
    if case['parameter']=='capacity':capacity=round(capacity*case['level'])
    c['simulation']['shared_capacity']=capacity
    for key,field in [('holding','holding_cost'),('shortage','shortage_cost')]:
        if case['parameter']==key:
            for p in products:p[field]*=case['level']
    c['evaluation']['counter_substitution_crn']=True
    c['evaluation']['demand_noise_scale']=case['level'] if case['parameter']=='noise' else 1.0
    c['development_metadata']={'scenario':case['id'],'protocol':'total_table_20260914','case':case,
        'status':('separately_added_after_original28' if BATCH == 'extension15' else 'new_prespecified_test_paths'),'structured_extension':n>3}
    return c

def prepare():
    if (ROOT/'protocol.json').exists():
        protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
        assert protocol['scenarios']==scenarios(), 'Do not change a frozen protocol silently'
        print('Protocol already prepared');return
    for case in scenarios():
        config=make_config(case);truth=truth_from_config(config);n=case['n'];t=case['t'];days=t//case['seasons']
        target=ROOT/'inputs'/case['id'];target.mkdir(parents=True,exist_ok=True)
        subfile=target/'substitution.csv'
        with subfile.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['source',*[p['id'] for p in config['products']],'EXIT'])
            for p,row in zip(config['products'],truth.substitution):w.writerow([p['id'],*row])
        for path_id,path_seed in enumerate(PATH_SEEDS,1):
            rng=np.random.default_rng(path_seed)
            rounding=np.random.default_rng(path_seed+999)
            rows=[];scale=config['evaluation']['demand_noise_scale']
            for period in range(t):
                s=period//days
                for i,p in enumerate(config['products']):
                    mu=truth.means[s,i]
                    raw=sample_base_demand(rng,mu,int(truth.dist_types[s,i]),truth.dist_params[s,i])
                    # A count-valued location-scale perturbation of the same raw path;
                    # unbiased randomized rounding preserves its conditional mean.
                    value=max(0.,mu+scale*(raw-mu))
                    demand=int(np.floor(value)+(rounding.random() < value-np.floor(value))) if scale!=1 else raw
                    rows.append([period+1,s+1,period%days+1,p['id'],demand])
            demandfile=target/f'path{path_id}.csv'
            with demandfile.open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f);w.writerow(['period','year','day_of_year','product_id','primary_demand']);w.writerows(rows)
            for seed in SEEDS:
                c=copy.deepcopy(config);c['random_seed']=seed
                c['output']['demand_long_file']=str(demandfile);c['output']['substitution_wide_file']=str(subfile)
                c['development_metadata'].update(path_id=path_id,path_seed=path_seed,training_seed=seed)
                dump(target/f'path{path_id}_seed{seed}.json',c)
    inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'inputs').rglob('*')) if p.is_file()}
    dump(ROOT/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),scenarios=scenarios(),
        training_seeds=SEEDS,path_seeds=PATH_SEEDS,methods=METHODS,training_steps=24000,replications=4,
        train_particles=96,eval_particles=192,hidden=56,grid_units=20,baseline_policy='fixed_schedule',
        information_budget_share=.12,interval='crossed path/seed bootstrap of paired mean profit ratios; 20000 resamples',
        input_sha256=inputs,table_rows=len(scenarios()),public_export_batch=BATCH,table_columns=12,
        table_columns_names=['情景','N','T','Full单位利润','vs固定日程%','vsD2AS2%','vsPPO%','vs无探索%','vs无替代%','vs全重置%','vs不重置%','知分布差距%'],
        selection='All rows and seeds retained. No test-based tuning or selection.',
        independent_test=True,training_reuse='Allowed only for exactly identical public economics, calendar, training law, method and seed; complete replay and both networks saved.',
        noise='test demand is a location-scale transformation with randomized integer rounding; training remains broad and common; achieved moments audited',
        informed='Known-distribution myopic candidate policy; not an optimum or certified bound; not the historical grid Oracle.'))
    print(f'Prepared {len(scenarios())} cases, 5 paths, 5 seeds in {ROOT}')

if __name__=='__main__': prepare()
