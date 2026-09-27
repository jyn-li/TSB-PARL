"""Resumable, source-locked training and paired evaluation for the single table."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import sys, json, copy, csv, time, argparse, hashlib, subprocess, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
import numpy as np
from design_suite import ROOT, WORKSPACE, ALGORITHM, SEEDS, METHODS, scenarios, dump
sys.path[:0]=[str(ALGORITHM/'experiments'/'common'),str(ALGORITHM/'src'/'tsb_parl')]
import execute_method as ex
from demand_only_belief import DemandOnlyBelief

INTERNAL=['full','fixed_schedule','no_exploration','no_substitution','reset_all','no_reset']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def config_path(case,seed,path=1):return ROOT/'inputs'/case/f'path{path}_seed{seed}.json'
def read_config(case,seed,path=1):return json.loads(config_path(case,seed,path).read_text(encoding='utf-8'))
def public_signature(config,method):
    return {'calendar':config['calendar'],'economics':[[p[k] for k in ('price','holding_cost','shortage_cost')] for p in config['products']],
        'capacity':config['simulation']['shared_capacity'],'initial_inventory':config['simulation']['initial_inventory'],
        'counter_crn':config['evaluation']['counter_substitution_crn'],'method':method,
        'training_law':'broad_exchangeable_v1','training_steps':24000}
def group_id(config,method):return hashlib.sha256(json.dumps(public_signature(config,method),sort_keys=True).encode()).hexdigest()[:16]

def args_for(config,method,out,quick=False):
    t=config['calendar']['num_years']*config['calendar']['days_per_year']
    return SimpleNamespace(config=Path(config['__config_path']),output=out,train_episodes=max(1,24000//t) if not quick else 1,
        train_steps=24000 if not quick else t,replications=4 if not quick else 1,
        train_particles=96 if not quick else 24,eval_particles=192 if not quick else 32,hidden=56,grid_units=20,
        learn_every=4,batch_size=48,gamma=1.,mode='full' if method=='no_substitution' else method,
        method=method,seed=int(config['random_seed']),substitution_prior='symmetric_sparse',online_updates=method not in ('fixed_schedule','no_exploration'),
        online_lr=3e-5,online_epsilon=0.,exploration_budget_share=.12,exploration_budget_mode='cap',fixed_template='spread',
        reveal_profit_weight=.05,probe_profit_weight=.10,information_weight=520.,attach_benchmarks=False)

def save_checkpoint(path,agent):
    n=min(agent.buffer.size,1500);idx=(agent.buffer.pos-n+np.arange(n))%agent.buffer.capacity
    payload={'state_dim':agent.state_dim,'gamma':agent.gamma,'n_actions':agent.n_actions}
    payload.update({'online_'+k:v for k,v in agent.online.params.items()});payload.update({'target_'+k:v for k,v in agent.target.params.items()})
    payload.update({'buffer_'+k:getattr(agent.buffer,k)[idx] for k in ('state','action','reward','next_state','done','next_mask')})
    np.savez_compressed(path,**payload)

def load_checkpoint(path,config,args):
    with np.load(path,allow_pickle=False) as data:
        dim=int(data['state_dim']);agent=ex.DoubleDQNAgent(dim,seed=int(config['random_seed'])+8103,hidden=len(data['online_b1']),gamma=float(data['gamma']),n_actions=int(data['n_actions']))
        for k in agent.online.params:
            agent.online.params[k][...]=data['online_'+k];agent.target.params[k][...]=data['target_'+k]
        for i in range(len(data['buffer_action'])):
            agent.buffer.add(data['buffer_state'][i],int(data['buffer_action'][i]),float(data['buffer_reward'][i]),data['buffer_next_state'][i],bool(data['buffer_done'][i]),data['buffer_next_mask'][i])
    p,h,b=ex.product_params(config)
    allocator=ex.StructuredAllocator(config['simulation']['shared_capacity'],p,h,b,grid_units=args.grid_units,
        reveal_profit_weight=args.reveal_profit_weight,probe_profit_weight=args.probe_profit_weight,information_weight=args.information_weight)
    return agent,allocator,dim

def initialize_rule(config,args):
    n=len(config['products']);seed=config['random_seed']+8100
    belief=ex.JointParticleBelief(n,args.train_particles,seed+1,substitution_prior=args.substitution_prior);belief.start_year(0)
    env=ex.InventoryEnvironment(config,ex.truth_from_config(config),seed+2);option=ex.ProbeOptionController(n)
    dim=ex.state_for(belief,env,0,0,0,np.zeros(n,bool),0,np.zeros(n,int),-1,option).shape[0]
    agent=ex.DoubleDQNAgent(dim,seed=seed+3,hidden=args.hidden,gamma=args.gamma,n_actions=n+2)
    return agent

def evaluate_informed(config,args,out):
    from environment import InventoryEnvironment,truth_from_config,load_primary_demand
    truth=truth_from_config(config);p,h,b=ex.product_params(config);n=len(p);t=np.prod(list(config['calendar'].values()))
    fixed=load_primary_demand(Path(config['output']['demand_long_file']),[x['id'] for x in config['products']])
    allocator=ex.StructuredAllocator(config['simulation']['shared_capacity'],p,h,b,grid_units=20)
    class Known:
        def summaries(self):
            mu=truth.means[self.year];types=truth.dist_types[self.year];params=truth.dist_params[self.year]
            var=np.where(types==0,mu,np.where(types==1,params*(params+1)/3,mu+mu**2/np.maximum(params,5)))
            var=var*config['evaluation']['demand_noise_scale']**2
            return dict(mu_mean=mu,predictive_var=var,a_mean=truth.substitution,mu_sd=np.zeros(n),a_sd=np.zeros((n,n+1)))
    rows=[];decision=[];start=time.perf_counter()
    for rep in range(args.replications):
        env=InventoryEnvironment(config,truth,config['random_seed']+3000+rep,fixed_demand=fixed);known=Known()
        total=dict(total_profit=0.,total_revenue=0.,total_sales=0.,total_final_lost=0.,holding_cost=0.,lost_sales_cost=0.)
        for period in range(int(t)):
            known.year=period//env.days_per_year;y=allocator.choose(0,env.x,known);public,private=env.step(y)
            for key,value in [('total_profit',private['true_profit']),('total_revenue',public['revenue']),('total_sales',public['sales'].sum()),('total_final_lost',private['final_lost'].sum()),('holding_cost',public['holding_cost']),('lost_sales_cost',private['lost_sales_cost'])]:total[key]+=float(value)
            decision.append(dict(replication=rep,period=period+1,profit=private['true_profit'],sales=float(public['sales'].sum()),inventory=float(y.sum())))
        total.update(replication=rep,fill_rate=total['total_sales']/(total['total_sales']+total['total_final_lost']));rows.append(total)
    for name,data in [('evaluation_replications.csv',rows),('period_audit.csv',decision)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    dump(out/'summary.json',dict(algorithm='Known-distribution myopic moment reference',profit={'mean':float(np.mean([r['total_profit'] for r in rows]))},evaluation_seconds=time.perf_counter()-start,numerical_actor=allocator.numerical_settings(),certified_optimum=False))

def sources():
    paths=list((ALGORITHM/'src'/'tsb_parl').glob('*.py'))+[ALGORITHM/'experiments'/'common'/'execute_method.py',ALGORITHM/'experiments'/'common'/'execute_nosub.py']+[Path(__file__).resolve().parent/name for name in ('design_suite.py','run_suite.py','literature_baselines.py')]
    return {str(p):sha(p) for p in sorted(paths)}

def worker(case_id,method,seed,quick=False):
    cases=[case_id]
    config=read_config(case_id,seed);config['__config_path']=str(config_path(case_id,seed))
    if not quick:
        signature=public_signature(config,method)
        cases=[c['id'] for c in scenarios() if public_signature(read_config(c['id'],seed),method)==signature]
    base=ROOT/('smoke' if quick else 'results');training=base/'training'/method/group_id(config,method)/f'seed{seed}'
    training.mkdir(parents=True,exist_ok=True);args=args_for(config,method,training,quick)
    if method=='no_substitution':ex.JointParticleBelief=DemandOnlyBelief
    ex.plot_training_loss=None;ex.plot_demand_learning=None
    trained=None;train_start=time.perf_counter()
    if method in INTERNAL:
        checkpoint=training/'evaluation_checkpoint.npz'
        if not checkpoint.exists():
            if method in ('fixed_schedule','no_exploration'):
                agent=initialize_rule(config,args)
            else:agent,_,_=ex.train(config,args,training)
            save_checkpoint(checkpoint,agent)
            dump(training/'training_provenance.json',dict(method=method,seed=seed,training_steps=0 if method in ('fixed_schedule','no_exploration') else args.train_episodes*(config['calendar']['num_years']*config['calendar']['days_per_year']),
                seconds=time.perf_counter()-train_start,public_signature=public_signature(config,method),shared_cases=cases,checkpoint_sha256=sha(checkpoint)))
        trained=load_checkpoint(checkpoint,config,args)
    elif method in ('d2as2','ppo'):
        import literature_baselines as lb
        model=training/('ppo_model.pt' if method=='ppo' else 'd2as2_policy.json')
        trained=lb.load_baseline(model) if model.exists() else lb.train_baseline(config,args,training)
    for case in cases:
        for path in (range(1,2) if quick else range(1,6)):
            out=base/case/method/f'path{path}'/f'seed{seed}'
            if (out/'COMPLETE.json').exists():continue
            out.mkdir(parents=True,exist_ok=True);c=read_config(case,seed,path);c['__config_path']=str(config_path(case,seed,path));a=args_for(c,method,out,quick)
            started=time.perf_counter()
            if method in INTERNAL:
                _,allocator,dim=load_checkpoint(training/'evaluation_checkpoint.npz',c,a)
                ex.evaluate(c,a,out,trained[0],allocator,dim)
            elif method in ('d2as2','ppo'):
                lb.evaluate(c,a,out,pretrained=trained)
            else:evaluate_informed(c,a,out)
            reps=list(csv.DictReader((out/'evaluation_replications.csv').open(encoding='utf-8-sig')))
            assert len(reps)==a.replications
            for row in reps:
                assert abs(float(row['total_profit'])-(float(row['total_revenue'])-float(row['holding_cost'])-float(row['lost_sales_cost'])))<1e-6
            dump(out/'COMPLETE.json',dict(case=case,method=method,seed=seed,path=path,replications=len(reps),seconds=time.perf_counter()-started,
                input_sha256=sha(config_path(case,seed,path)),training_directory=str(training),evaluation_sha256=sha(out/'evaluation_replications.csv')))
            print(f'[complete] {case} {method} seed={seed} path={path}',flush=True)

def orchestrate(workers,selected_methods=None,selected_cases=None):
    lock=ROOT/'source_lock.json'
    if lock.exists():
        recorded=json.loads(lock.read_text(encoding='utf-8'))
        # Additional analyses are allowed; executable numerical sources stay locked.
        for path,expected in recorded.items():assert sha(Path(path))==expected, f'Changed locked code: {path}'
    else:dump(lock,sources())
    tasks=[];seen=set()
    for case in scenarios():
        if selected_cases and case['id'] not in selected_cases:continue
        for method in selected_methods or METHODS:
            for seed in SEEDS:
                key=(group_id(read_config(case['id'],seed),method),seed)
                if key in seen:continue
                seen.add(key);tasks.append((case['id'],method,seed))
    logdir=ROOT/'logs';logdir.mkdir(exist_ok=True)
    status=dict(total_groups=len(tasks),completed=0,failed=[],started=time.strftime('%Y-%m-%d %H:%M:%S'),workers=workers)
    dump(ROOT/'status.json',status)
    def run(task):
        case,method,seed=task;log=logdir/f'{case}_{method}_{seed}.log'
        with log.open('w',encoding='utf-8') as f:
            done=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__)), '--worker','--case',case,'--method',method,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,cwd=WORKSPACE)
        return task,done.returncode,str(log)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in as_completed([pool.submit(run,task) for task in tasks]):
            task,code,log=result.result();status['completed']+=1
            if code:status['failed'].append(dict(task=task,log=log))
            status['updated']=time.strftime('%Y-%m-%d %H:%M:%S');dump(ROOT/'status.json',status)
            print(f'[{status["completed"]}/{len(tasks)}] {task} exit={code}',flush=True)
    if status['failed']:raise RuntimeError(f'{len(status["failed"])} failed groups; inspect logs and resume')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--quick',action='store_true')
    parser.add_argument('--case',default='n3_t300');parser.add_argument('--method',default='full');parser.add_argument('--seed',type=int,default=SEEDS[0]);parser.add_argument('--workers',type=int,default=10)
    parser.add_argument('--methods',nargs='+');parser.add_argument('--cases',nargs='+')
    args=parser.parse_args()
    if args.worker:worker(args.case,args.method,args.seed,args.quick)
    else:orchestrate(args.workers,args.methods,args.cases)
