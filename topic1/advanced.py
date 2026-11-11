"""V2: nested model selection with frozen outer Matbench folds.

V1 test results motivated this exploratory revision; V2 is not a new blind test.
All V2 hyperparameters and blend choices are selected using outer-training data.
"""
import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matbench.task import MatbenchTask
from matbench.constants import MBV01_VALIDATION_DATA_PATH
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from topic1.run import ROOT, OUT as V1, TASKS, sha, write_json, scores

OUT=ROOT/'outputs/topic1_v2'
FAMILIES=['Linear','RF','ET','XGB']
SEED=42
CONFIGS={
    'Linear':[{'regularization':.1},{'regularization':10.},{'regularization':100.}],
    'RF':[{'max_features':1.,'min_samples_leaf':1},{'max_features':.5,'min_samples_leaf':1},{'max_features':'sqrt','min_samples_leaf':2}],
    'ET':[{'max_features':1.,'min_samples_leaf':1},{'max_features':.7,'min_samples_leaf':1},{'max_features':.4,'min_samples_leaf':2}],
    'XGB':[{'max_depth':3,'learning_rate':.05,'min_child_weight':1},
           {'max_depth':5,'learning_rate':.05,'min_child_weight':1},
           {'max_depth':4,'learning_rate':.08,'min_child_weight':3}],
}


def make_model(family,params,classification,seed=SEED):
    common=dict(n_estimators=128,random_state=seed,n_jobs=4)
    if family=='Linear':
        reg=params['regularization']
        model=LogisticRegression(C=1/reg,max_iter=5000,solver='lbfgs',random_state=seed) if classification else Ridge(alpha=reg)
        return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),VarianceThreshold(),
                             RobustScaler(quantile_range=(10.,90.)),model)
    if family in ['RF','ET']:
        cls={'RF':(RandomForestRegressor,RandomForestClassifier),'ET':(ExtraTreesRegressor,ExtraTreesClassifier)}[family][int(classification)]
        model=cls(**common,**params)
    else:
        from xgboost import XGBRegressor,XGBClassifier
        cls=XGBClassifier if classification else XGBRegressor
        model=cls(**common,**params,tree_method='hist',subsample=.9,colsample_bytree=.9,reg_lambda=1.,verbosity=0)
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),model)


def predict(model,x,classification):
    return model.predict_proba(x)[:,list(model[-1].classes_).index(True)] if classification else model.predict(x)


def loss(y,p,classification):
    s=scores(y,p,classification)
    return 1-s['roc_auc'] if classification else s['mae']


def inner_folds(y,groups,classification,fold):
    grouped=len(set(groups))<len(groups)
    if grouped:
        splitter=StratifiedGroupKFold(3,shuffle=True,random_state=SEED+fold) if classification else GroupKFold(3,shuffle=True,random_state=SEED+fold)
        splits=list(splitter.split(np.zeros(len(y)),y,groups))
    else:
        splitter=StratifiedKFold(3,shuffle=True,random_state=SEED+fold) if classification else KFold(3,shuffle=True,random_state=SEED+fold)
        splits=list(splitter.split(np.zeros(len(y)),y))
    for a,b in splits:
        assert set(a).isdisjoint(b)
        if grouped:assert set(groups[a]).isdisjoint(groups[b])
        if classification:assert len(set(y[a]))==len(set(y[b]))==2
    return splits,grouped


def mean_inner_loss(y,p,inner,classification):
    return float(np.mean([loss(y[b],p[b],classification) for _,b in inner]))


def select_blend(y,oofs,inner,classification):
    # Same inner partitions for all families. No outer target is an argument.
    ranked=sorted(oofs,key=lambda f:mean_inner_loss(y,oofs[f],inner,classification))
    recipes=[([ranked[0]],oofs[ranked[0]]), (ranked[:2],np.mean([oofs[f] for f in ranked[:2]],axis=0))]
    evaluated=[dict(members=members,inner_loss=mean_inner_loss(y,p,inner,classification)) for members,p in recipes]
    chosen=min(evaluated,key=lambda r:r['inner_loss'])
    return chosen,evaluated


def save_prediction(task,rep,family,fold,ids,y,p,extra):
    folder=OUT/task.dataset_name/'runs'/rep/family;folder.mkdir(parents=True,exist_ok=True)
    assert len(p)==len(ids) and np.isfinite(p).all()
    is_cls=task.metadata.task_type=='classification'
    if is_cls:assert ((p>=0)&(p<=1)).all()
    pd.DataFrame(dict(mbid=ids,fold=fold,target=np.asarray(y,float),prediction=p)).to_csv(folder/f'fold_{fold}.csv',index=False)
    result=dict(task=task.dataset_name,representation=rep,model=family,fold=fold,metrics=scores(y,p,is_cls),**extra)
    write_json(folder/f'fold_{fold}.json',result)
    return result


def tune_fit(task,rep,family,fold,matrix,info,inner):
    folder=OUT/task.dataset_name/'runs'/rep/family
    path=folder/f'fold_{fold}.json'
    if path.exists():
        result=json.loads(path.read_text())
        pred=pd.read_csv(path.with_suffix('.csv'),float_precision='round_trip').prediction.to_numpy()
        oof=pd.read_csv(folder/f'inner_oof_{fold}.csv',float_precision='round_trip').prediction.to_numpy()
        return result,pred,oof
    train,ytrain=task.get_train_and_val_data(fold);test=task.get_test_data(fold)
    ai=task.df.index.get_indexer(train.index);bi=task.df.index.get_indexer(test.index)
    x,y=matrix[ai],np.asarray(ytrain);cl=task.metadata.task_type=='classification'
    traces=[];oofs=[];start=time.monotonic()
    for config in CONFIGS[family]:
        oof=np.full(len(y),np.nan);fold_scores=[]
        for j,(a,b) in enumerate(inner):
            model=make_model(family,config,cl)
            with threadpool_limits(limits=1):
                model.fit(x[a],y[a]);oof[b]=predict(model,x[b],cl)
            fold_scores.append(loss(y[b],oof[b],cl))
        assert np.isfinite(oof).all()
        traces.append(dict(parameters=config,inner_loss_mean=float(np.mean(fold_scores)),inner_losses=fold_scores))
        oofs.append(oof)
    winner=int(np.argmin([r['inner_loss_mean'] for r in traces]))
    params=CONFIGS[family][winner]
    model=make_model(family,params,cl)
    with threadpool_limits(limits=1):
        model.fit(x,y);pred=predict(model,matrix[bi],cl)
    # Held-out targets are fetched only after final predictions.
    _,ytest=task.get_test_data(fold,include_target=True)
    folder.mkdir(parents=True,exist_ok=True)
    inner_assignment=np.empty(len(y),int)
    for j,(_,b) in enumerate(inner):inner_assignment[b]=j
    pd.DataFrame(dict(mbid=train.index,inner_fold=inner_assignment,target=y,prediction=oofs[winner])).to_csv(folder/f'inner_oof_{fold}.csv',index=False)
    if rep in ['C','C_structure']:
        joblib.dump(model,folder/f'model_{fold}.joblib',compress=3)
    result=save_prediction(task,rep,family,fold,test.index,ytest,pred,
                          dict(n_train=len(ai),n_test=len(bi),selected_parameters=params,selected_inner_loss=traces[winner]['inner_loss_mean'],
                               search=traces,seed=SEED,n_inner_fits=9,n_refits=1,seconds=time.monotonic()-start))
    print(f'{task.dataset_name} {rep}/{family} fold={fold}: {result["metrics"]} selected={params}',flush=True)
    return result,pred,oofs[winner]


def export(task,rep,families):
    for family in families:
        official=MatbenchTask(task.dataset_name,autoload=False);official.df=task.df
        for fold in task.folds:
            folder=OUT/task.dataset_name/'runs'/rep/family
            p=pd.read_csv(folder/f'fold_{fold}.csv',float_precision='round_trip')
            result=json.loads((folder/f'fold_{fold}.json').read_text())
            official.record(fold,p.prediction.to_numpy(),params={k:v for k,v in result.items() if k!='metrics'})
        official.validate()
        dest=OUT/task.dataset_name/'official_results'/f'{rep}_{family}.json.gz';dest.parent.mkdir(exist_ok=True)
        official.to_file(str(dest))
        MatbenchTask.from_file(str(dest)).validate()


def register():
    protocol=dict(tasks=TASKS,families=FAMILIES,configurations=CONFIGS,trees=128,seed=SEED,
                  outer='official Matbench 5 folds',inner='3 folds; group compositions if duplicated',
                  primary_representation='C',ablation='ET on A/B/C, identical search budgets; C_structure on jdft2d',
                  selection='inner CV chooses family or equally weighted top-two blend',
                  robustness_seeds=[7,2026],robustness_model='ET, reuse inner-selected parameters; no seed selection',
                  caveat='exploratory V2 following inspection of V1 outer results; not a new untouched benchmark',
                  source_sha256=sha(__file__),v1_source_sha256=sha(ROOT/'topic1/run.py'),validation_sha256=sha(MBV01_VALIDATION_DATA_PATH))
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Protocol/code changed: use a fresh output directory.'
    else:
        write_json(path,protocol)
        write_json(OUT/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))


def run_task(name,stage):
    task=MatbenchTask(name);info=pd.read_csv(V1/name/'samples.csv').set_index('id')
    assert list(info.index)==list(task.df.index)
    reps=['C']+(['C_structure'] if name=='matbench_jdft2d' else [])
    if stage=='ablation':reps=['A','B']
    for rep in reps:
        matrix=np.load(V1/name/f'features_{rep}.npy')
        families=['ET'] if stage=='ablation' else FAMILIES
        for fold in task.folds:
            train,y=task.get_train_and_val_data(fold);test=task.get_test_data(fold)
            groups=info.loc[train.index,'formula'].to_numpy()
            inner,grouped=inner_folds(np.asarray(y),groups,task.metadata.task_type=='classification',fold)
            manifest=dict(outer_train=list(train.index),outer_test=list(test.index),grouped=grouped,
                          inner=[dict(train=list(train.index[a]),validation=list(train.index[b])) for a,b in inner])
            write_json(OUT/name/'splits'/f'fold_{fold}.json',manifest)
            predictions={};oofs={}
            for family in families:
                _,pred,oof=tune_fit(task,rep,family,fold,matrix,info,inner)
                predictions[family]=pred;oofs[family]=oof
            if stage!='ablation':
                recipe,candidates=select_blend(np.asarray(y),oofs,inner,task.metadata.task_type=='classification')
                pred=np.mean([predictions[f] for f in recipe['members']],axis=0)
                _,target=task.get_test_data(fold,include_target=True)
                save_prediction(task,rep,'Selected',fold,test.index,target,pred,
                                dict(n_train=len(train),n_test=len(test),recipe=recipe,candidate_recipes=candidates,seed=SEED))
        export(task,rep,families+([] if stage=='ablation' else ['Selected']))


def robustness(name):
    task=MatbenchTask(name);cl=task.metadata.task_type=='classification'
    for rep in ['C']+(['C_structure'] if name=='matbench_jdft2d' else []):
        matrix=np.load(V1/name/f'features_{rep}.npy')
        for fold in task.folds:
            train,y=task.get_train_and_val_data(fold);test=task.get_test_data(fold)
            params=json.loads((OUT/name/'runs'/rep/'ET'/f'fold_{fold}.json').read_text())['selected_parameters']
            for seed in [7,2026]:
                family=f'ET_seed{seed}'
                if (OUT/name/'runs'/rep/family/f'fold_{fold}.json').exists():continue
                model=make_model('ET',params,cl,seed=seed)
                start=time.monotonic()
                with threadpool_limits(limits=1):
                    model.fit(matrix[task.df.index.get_indexer(train.index)],y)
                    pred=predict(model,matrix[task.df.index.get_indexer(test.index)],cl)
                _,target=task.get_test_data(fold,include_target=True)
                save_prediction(task,rep,family,fold,test.index,target,pred,dict(n_train=len(train),n_test=len(test),selected_parameters=params,seed=seed,seconds=time.monotonic()-start))
        export(task,rep,['ET_seed7','ET_seed2026'])


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['main','ablation','robustness']);p.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS);args=p.parse_args()
    register()
    for name in args.tasks:
        print('START',args.stage,name,flush=True)
        robustness(name) if args.stage=='robustness' else run_task(name,args.stage)
    print('COMPLETE',args.stage,flush=True)


if __name__=='__main__':main()
