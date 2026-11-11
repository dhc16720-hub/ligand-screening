"""V4: matched physical-representation controls and predeclared inner gate."""
import argparse
import importlib.metadata
import json
import shutil
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from matbench.task import MatbenchTask
from matbench.constants import MBV01_VALIDATION_DATA_PATH
from topic1.run import ROOT, OUT as V1,TASKS,sha,write_json
from topic1 import advanced as base
from topic1.refine_models import predict,select,combine,configs as v3_configs
from topic1.physics_features import OUT,TARGETS,prepare
from topic1 import physics_models as models

V2=ROOT/'outputs/topic1_v2'
V3=ROOT/'outputs/topic1_v3'


def read(path):return pd.read_csv(path,float_precision='round_trip')


def register():
    protocol=dict(tasks=TASKS,new_representation_tasks=TARGETS,seed=42,
                  configurations={kind:models.configs(cl) for kind,cl in [('regression',False),('classification',True)]},
                  physical_features={'alloys':'Miedema inter/amorphous/solid-solution enthalpies + Yang omega/delta; unavailable parameters retained as NaN',
                                     'jdft2d':'CrystalNNFingerprint_ops site mean/std (local geometry), appended to C_structure'},
                  controls='RF/ET on D:128 trees, CatBoost on D:512 iterations; same three-candidate, three-fold searches as matching V2/V3 models on old representation',
                  new_pipeline='D-only three models; V3 select() chooses single/top2 mean/top3 mean/(regression) top3 median using inner OOF',
                  gate='Replace entire V3 pipeline only if paired mean inner loss improvement exceeds paired SE and at least two of three inner folds improve; not a significance test',
                  unchanged_tasks='expt_gap and expt_is_metal carry forward V3 exactly; no new fits',
                  caveat='Exploratory after viewing V1-V3 outer scores, not blind validation. Inner OOF reused for tuning, selection and gating; not unbiased meta-validation.',
                  source_sha256={p:sha(ROOT/p) for p in ['topic1/physics.py','topic1/physics_models.py','topic1/physics_features.py','topic1/advanced.py','topic1/refine_models.py','topic1/run.py']},
                  validation_sha256=sha(MBV01_VALIDATION_DATA_PATH),v3_manifest_sha256=sha(V3/'delivery_manifest.json'))
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Frozen protocol changed'
    else:
        write_json(path,protocol)
        write_json(OUT/'environment.json',{d.metadata['Name']:d.version for d in importlib.metadata.distributions()})


def old_oofs(name,rep,fold,train):
    cl=name=='matbench_glass'
    pool=base.FAMILIES+list(v3_configs(cl))
    data={}
    for family in pool:
        folder=(V2 if family in base.FAMILIES else V3)/name/'runs'/rep/family
        oof=read(folder/f'inner_oof_{fold}.csv')
        assert list(oof.mbid)==list(train.index)
        data[family]=oof.prediction.to_numpy()
    return data


def fit(task,family,fold,matrix,inner):
    folder=OUT/task.dataset_name/'runs/D'/family
    if (folder/f'fold_{fold}.json').exists():
        return read(folder/f'fold_{fold}.csv').prediction.to_numpy(),read(folder/f'inner_oof_{fold}.csv').prediction.to_numpy()
    train,ytrain=task.get_train_and_val_data(fold);test=task.get_test_data(fold)
    cl=task.metadata.task_type=='classification'
    y=np.asarray(ytrain);x=matrix[task.df.index.get_indexer(train.index)]
    traces=[];oofs=[];start=time.monotonic()
    for config in models.configs(cl)[family]:
        oof=np.full(len(y),np.nan)
        for a,b in inner:
            model=models.make_model(family,config,cl)
            with threadpool_limits(limits=1):
                model.fit(x[a],y[a]);oof[b]=predict(model,x[b],cl)
        losses=[base.loss(y[b],oof[b],cl) for _,b in inner]
        traces.append(dict(parameters=config,inner_losses=losses,inner_loss_mean=float(np.mean(losses))))
        oofs.append(oof)
    best=int(np.argmin([r['inner_loss_mean'] for r in traces]));params=traces[best]['parameters']
    model=models.make_model(family,params,cl)
    with threadpool_limits(limits=1):
        model.fit(x,y);pred=predict(model,matrix[task.df.index.get_indexer(test.index)],cl)
    folder.mkdir(parents=True,exist_ok=True)
    assignment=np.empty(len(y),int)
    for j,(_,b) in enumerate(inner):assignment[b]=j
    pd.DataFrame(dict(mbid=train.index,target=np.asarray(y,float),inner_fold=assignment,prediction=oofs[best])).to_csv(folder/f'inner_oof_{fold}.csv',index=False)
    joblib.dump(model,folder/f'model_{fold}.joblib',compress=3)
    _,target=task.get_test_data(fold,include_target=True)
    result=base.save_prediction(task,'D',family,fold,test.index,target,pred,dict(n_train=len(train),n_test=len(test),seed=42,search=traces,
                  selected_parameters=params,selected_inner_loss=traces[best]['inner_loss_mean'],n_inner_fits=9,n_refits=1,seconds=time.monotonic()-start))
    print(task.dataset_name,family,fold,result['metrics'],flush=True)
    return pred,oofs[best]


def run(name):
    task=MatbenchTask(name);base.OUT=OUT
    rep='C_structure' if name=='matbench_jdft2d' else 'C'
    if name not in TARGETS:
        for fold in task.folds:
            test,target=task.get_test_data(fold,include_target=True)
            p=read(V3/name/'runs'/rep/'Selected'/f'fold_{fold}.csv')
            assert list(p.mbid)==list(test.index)
            base.save_prediction(task,'V4','Selected',fold,test.index,target,p.prediction.to_numpy(),dict(provenance='Exact V3 carry-forward, no new training',source_sha256=sha(V3/name/'runs'/rep/'Selected'/f'fold_{fold}.csv')))
        base.export(task,'V4',['Selected']);return
    prepare(name)
    matrix=np.load(OUT/name/'features_D.npy');info=read(V1/name/'samples.csv').set_index('id')
    assert list(info.index)==list(task.df.index)
    for fold in task.folds:
        train,ytrain=task.get_train_and_val_data(fold);test=task.get_test_data(fold)
        cl=task.metadata.task_type=='classification';y=np.asarray(ytrain)
        inner,grouped=base.inner_folds(y,info.loc[train.index,'formula'].to_numpy(),cl,fold)
        manifest=dict(outer_train=list(train.index),outer_test=list(test.index),grouped=grouped,
                      inner=[dict(train=list(train.index[a]),validation=list(train.index[b])) for a,b in inner])
        assert manifest==json.loads((V3/name/'splits'/f'fold_{fold}.json').read_text())
        write_json(OUT/name/'splits'/f'fold_{fold}.json',manifest)
        preds={};oofs={}
        for family in models.configs(cl):preds[family],oofs[family]=fit(task,family,fold,matrix,inner)
        recipe,candidates=select(y,oofs,inner,cl)
        new_pred=combine(preds,recipe);new_oof=combine(oofs,recipe)
        old_pool=old_oofs(name,rep,fold,train)
        old_recipe,_=select(y,old_pool,inner,cl)
        assert old_recipe==json.loads((V3/name/'runs'/rep/'Selected'/f'fold_{fold}.json').read_text())['recipe']
        decision=models.gate(y,combine(old_pool,old_recipe),new_oof,inner,cl)
        old_pred=read(V3/name/'runs'/rep/'Selected'/f'fold_{fold}.csv')
        assert list(old_pred.mbid)==list(test.index)
        final=new_pred if decision['use_new'] else old_pred.prediction.to_numpy()
        _,target=task.get_test_data(fold,include_target=True)
        base.save_prediction(task,'D','Selected',fold,test.index,target,new_pred,dict(recipe=recipe,candidate_recipes=candidates,n_train=len(train),n_test=len(test)))
        base.save_prediction(task,'V4','Selected',fold,test.index,target,final,dict(gate=decision,old_recipe=old_recipe,new_recipe=recipe,n_train=len(train),n_test=len(test)))
    base.export(task,'D',list(models.configs(cl))+['Selected'])
    base.export(task,'V4',['Selected'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS)
    args=parser.parse_args();register()
    for name in args.tasks:
        print('START',name,flush=True);run(name)
    print('COMPLETE',flush=True)
