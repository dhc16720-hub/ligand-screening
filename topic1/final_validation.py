"""Post-freeze validation. Historical benchmark scores are never overwritten."""
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from matbench.task import MatbenchTask
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
from topic1.run import ROOT,OUT as V1,TASKS,sha,write_json,scores
from topic1.physics_features import TARGETS
from topic1.physics import read,V2,V3,OUT as V4
from topic1.refine_models import combine,select
from topic1.advanced import inner_folds
from topic1.final_models import fit_procedure,estimator,FinalBundle,specifications

OUT=ROOT/'outputs/topic1_final'
SEEDS=[7,2026]


def register():
    sources=['topic1/final_validation.py','topic1/final_models.py','topic1/final_external.py',
             'topic1/advanced.py','topic1/refine_models.py','topic1/physics_models.py','topic1/physics_features.py']
    p=dict(version='Final validation, V4 frozen',tasks=TASKS,conditional_refit_seeds=SEEDS,full_selection_seeds=SEEDS,
           full_selection_tasks=TASKS[:2],inner_partition='same three-fold splitting algorithm; fixed split seed, model seeds perturbed',
           conformal=dict(tasks=TASKS[:3],nominal=.9,proper_fraction=.8,split_seed=20261110,
                          rule='Composition-group-disjoint proper/calibration split inside each official outer train. Full selection uses proper set only; no reuse of old hyperparameters selected with calibration labels.'),
           common_pool='RF, ET, CatBoost on old and D representations, identical candidate budgets and selection recipes; reuse original OOF and outer predictions',
           external=dict(dataset='glass_ternary_hipt',url='https://ndownloader.figshare.com/files/13298018',
                         sha256='4a4c60ef227891203570dbc3decb606649b877f6b75289afd4e78fe8e8095e63',
                         training='Full matbench_glass only; frozen V4 procedure; no external-label tuning',
                         exclusion='Remove exact fractional composition matches (8 decimals). Primary also removes min L1 distance <=0.02. Keep duplicate measurement rows; bootstrap by composition.',
                         evaluation='Primary separated composition subset; secondary all nonexact-overlap; per-system results. Threshold .5, no external calibration.',
                         caveat='Sputtering versus bulk/melt-based source conditions: external domain-shift stress test, not identical-condition replication. Does not validate all five tasks externally.'),
           caveat='All five official datasets have been inspected in earlier versions. No claim of fresh blind validation for these five. SE gate is a heuristic.',
           sources={p:sha(ROOT/p) for p in sources},historical_manifests={str(d.relative_to(ROOT)):sha(d/'delivery_manifest.json') for d in [V1,V2,V3,V4]})
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p,'Final protocol changed'
    else:write_json(path,p)


def data(name):
    task=MatbenchTask(name);rep='C_structure' if name=='matbench_jdft2d' else 'C'
    matrices={'C':np.load(V1/name/f'features_{rep}.npy')}
    if name in TARGETS:matrices['D']=np.load(V4/name/'features_D.npy')
    info=read(V1/name/'samples.csv').set_index('id');assert list(info.index)==list(task.df.index)
    return task,matrices,info,rep


def take(matrices,indices):return {k:x[indices] for k,x in matrices.items()}


def store_fit(folder,bundle,record,oofs,ids,y):
    folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,folder/'model.joblib',compress=3)
    assignment=np.empty(len(ids),int)
    for j,s in enumerate(record['inner']):assignment[s['validation']]=j
    pd.DataFrame(dict(mbid=ids,target=np.asarray(y,float),inner_fold=assignment,**oofs)).to_csv(folder/'inner_oof.csv',index=False)
    write_json(folder/'selection.json',dict(train_ids=list(ids),**record))


def record_prediction(folder,ids,y,p,cl,extra):
    p=np.asarray(p,float);assert np.isfinite(p).all()
    folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(dict(mbid=ids,target=np.asarray(y,float),prediction=p)).to_csv(folder/'predictions.csv',index=False)
    write_json(folder/'result.json',dict(metrics=scores(y,p,cl),**extra))


def historical_recipe(name,rep,fold):
    r=json.loads((V4/name/'runs/V4/Selected'/f'fold_{fold}.json').read_text())
    physics=name in TARGETS and r['gate']['use_new']
    original=json.loads(((V4/name/'runs/D/Selected' if physics else V3/name/'runs'/rep/'Selected')/f'fold_{fold}.json').read_text())['recipe']
    recipe={**original,'members':[('D:' if physics else 'C:')+f for f in original['members']]}
    params={}
    for key in recipe['members']:
        letter,f=key.split(':')
        folder=V4/name/'runs/D'/f if letter=='D' else (V2 if f in ['Linear','RF','ET','XGB'] else V3)/name/'runs'/rep/f
        params[key]=json.loads((folder/f'fold_{fold}.json').read_text())['selected_parameters']
    return recipe,params


def conditional(name):
    task,matrices,info,rep=data(name);cl=task.metadata.task_type=='classification'
    for fold in task.folds:
        train,y=task.get_train_and_val_data(fold);test=task.get_test_data(fold)
        ai=task.df.index.get_indexer(train.index);bi=task.df.index.get_indexer(test.index)
        recipe,params=historical_recipe(name,rep,fold)
        for seed in SEEDS:
            folder=OUT/name/'conditional'/f'seed_{seed}_fold_{fold}'
            if (folder/'result.json').exists():continue
            fitted={}
            for key in recipe['members']:
                model=estimator(key,params[key],cl,seed)
                with threadpool_limits(limits=1):model.fit(matrices[key.split(':')[0]][ai],y)
                fitted[key]=model
            bundle=FinalBundle(fitted,recipe,cl);p=bundle.predict(take(matrices,bi))
            _,target=task.get_test_data(fold,include_target=True)
            folder.mkdir(parents=True,exist_ok=True);joblib.dump(bundle,folder/'model.joblib',compress=3)
            record_prediction(folder,test.index,target,p,cl,dict(seed=seed,fold=fold,recipe=recipe,parameters=params,train_ids=list(train.index),refits=len(fitted),scope='Conditional on V4 parameters, recipe and gate; no retuning'))
            print('CONDITIONAL',name,seed,fold,flush=True)


def reselection(name):
    task,matrices,info,rep=data(name);cl=False
    for fold in task.folds:
        train,y=task.get_train_and_val_data(fold);test=task.get_test_data(fold)
        ai=task.df.index.get_indexer(train.index);bi=task.df.index.get_indexer(test.index)
        for seed in SEEDS:
            folder=OUT/name/'reselection'/f'seed_{seed}_fold_{fold}'
            if (folder/'result.json').exists():continue
            bundle,record,oofs=fit_procedure(take(matrices,ai),y,info.loc[train.index,'formula'],cl,fold,seed)
            p=bundle.predict(take(matrices,bi));_,target=task.get_test_data(fold,include_target=True)
            store_fit(folder,bundle,record,oofs,train.index,y)
            record_prediction(folder,test.index,target,p,cl,dict(seed=seed,fold=fold,inner_fits=record['inner_fits'],refits=record['refits'],scope='Full parameter, recipe and gate reselection; fixed inner partitions'))
            print('RESELECTION',name,seed,fold,flush=True)


def conformal(name):
    task,matrices,info,rep=data(name)
    for fold in task.folds:
        folder=OUT/name/'conformal'/f'fold_{fold}'
        if (folder/'result.json').exists():continue
        train,ytrain=task.get_train_and_val_data(fold);test=task.get_test_data(fold)
        groups=info.loc[train.index,'formula'].to_numpy()
        proper,cal=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20261110+fold).split(np.zeros(len(train)),groups=groups))
        assert set(groups[proper]).isdisjoint(groups[cal])
        ai=task.df.index.get_indexer(train.index);bi=task.df.index.get_indexer(test.index);y=np.asarray(ytrain)
        bundle,record,oofs=fit_procedure(take(matrices,ai[proper]),y[proper],groups[proper],False,fold,42)
        cal_pred=bundle.predict(take(matrices,ai[cal]));p=bundle.predict(take(matrices,bi))
        residual=np.abs(y[cal]-cal_pred);rank=int(np.ceil((len(cal)+1)*.9));assert rank<=len(cal)
        radius=float(np.sort(residual)[rank-1])
        _,target=task.get_test_data(fold,include_target=True)
        store_fit(folder,bundle,record,oofs,train.index[proper],y[proper])
        pd.DataFrame(dict(mbid=train.index[cal],target=y[cal],prediction=cal_pred,residual=residual)).to_csv(folder/'calibration.csv',index=False)
        pd.DataFrame(dict(mbid=test.index,lower=p-radius,upper=p+radius)).to_csv(folder/'intervals.csv',index=False)
        record_prediction(folder,test.index,target,p,False,dict(fold=fold,nominal=.9,radius=radius,rank=rank,
                  proper_train_ids=list(train.index[proper]),calibration_ids=list(train.index[cal]),
                  coverage=float(np.mean((np.asarray(target)>=p-radius)&(np.asarray(target)<=p+radius))),mean_width=2*radius,
                  inner_fits=record['inner_fits'],refits=record['refits'],scope='Entire frozen selection procedure refitted on proper subset; calibration never enters selection'))
        print('CONFORMAL',name,fold,flush=True)


def common_pool():
    for name in TARGETS:
        task,matrices,info,rep=data(name);cl=task.metadata.task_type=='classification'
        for fold in task.folds:
            train,y=task.get_train_and_val_data(fold);test,target=task.get_test_data(fold,include_target=True)
            inner,_=inner_folds(np.asarray(y),info.loc[train.index,'formula'].to_numpy(),cl,fold)
            for representation in ['C','D']:
                preds={};oofs={}
                for f in ['RF','ET','CatBoost']:
                    folder=V4/name/'runs/D'/f if representation=='D' else (V3 if f=='CatBoost' else V2)/name/'runs'/rep/f
                    a=read(folder/f'inner_oof_{fold}.csv');b=read(folder/f'fold_{fold}.csv')
                    assert list(a.mbid)==list(train.index) and list(b.mbid)==list(test.index)
                    oofs[f]=a.prediction.to_numpy();preds[f]=b.prediction.to_numpy()
                recipe,candidates=select(np.asarray(y),oofs,inner,cl)
                record_prediction(OUT/name/'common_pool'/f'{representation}_fold_{fold}',test.index,target,combine(preds,recipe),cl,
                                  dict(fold=fold,representation=representation,recipe=recipe,candidates=candidates,new_fits=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['conditional','reselection','conformal','common_pool']);p.add_argument('--tasks',nargs='+',choices=TASKS);args=p.parse_args()
    register()
    if args.stage=='common_pool':common_pool()
    else:
        names=args.tasks or (TASKS[:2] if args.stage=='reselection' else TASKS[:3] if args.stage=='conformal' else TASKS)
        for name in names:
            print('START',args.stage,name,flush=True);globals()[args.stage](name)
    print('COMPLETE',flush=True)
