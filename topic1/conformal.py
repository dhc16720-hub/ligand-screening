"""Separate, fixed-model split-conformal experiment on regression tasks.

No hyperparameter selected using calibration labels; main tuned runs unchanged.
"""
import json
import math
import numpy as np
import pandas as pd
from matbench.task import MatbenchTask
from threadpoolctl import threadpool_limits
from topic1.run import OUT as V1,write_json
from topic1.advanced import OUT,make_model,predict


def conformal_radius(residuals,alpha=.1):
    r=np.sort(np.asarray(residuals,float));k=math.ceil((len(r)+1)*(1-alpha))
    return float(r[k-1]) if k<=len(r) else float('inf')


def main():
    rows=[]
    for name in ['matbench_steels','matbench_jdft2d','matbench_expt_gap']:
        task=MatbenchTask(name)
        rep='C_structure' if name=='matbench_jdft2d' else 'C'
        matrix=np.load(V1/name/f'features_{rep}.npy')
        dest=OUT/name/'conformal';dest.mkdir(parents=True,exist_ok=True)
        for fold in task.folds:
            train,y=task.get_train_and_val_data(fold);test=task.get_test_data(fold)
            order=np.random.default_rng(20261110+fold).permutation(len(train))
            n_cal=math.ceil(.2*len(train));cal=order[:n_cal];proper=order[n_cal:]
            ai=task.df.index.get_indexer(train.index);bi=task.df.index.get_indexer(test.index)
            assert set(train.index[proper]).isdisjoint(train.index[cal]) and set(train.index).isdisjoint(test.index)
            model=make_model('ET',dict(max_features=1.,min_samples_leaf=1),False,seed=42)
            with threadpool_limits(limits=1):
                model.fit(matrix[ai[proper]],np.asarray(y)[proper])
                cp=predict(model,matrix[ai[cal]],False);pred=predict(model,matrix[bi],False)
            residual=np.abs(np.asarray(y)[cal]-cp);q=conformal_radius(residual)
            _,yt=task.get_test_data(fold,include_target=True);yt=np.asarray(yt)
            low,high=pred-q,pred+q
            covered=(yt>=low)&(yt<=high)
            frame=pd.DataFrame(dict(mbid=test.index,fold=fold,target=yt,prediction=pred,lower=low,upper=high,covered=covered))
            frame.to_csv(dest/f'fold_{fold}.csv',index=False)
            pd.DataFrame(dict(mbid=train.index[cal],target=np.asarray(y)[cal],prediction=cp,residual=residual)).to_csv(dest/f'calibration_{fold}.csv',index=False)
            result=dict(task=name,representation=rep,fold=fold,nominal_coverage=.9,coverage=float(covered.mean()),
                        mean_width=float(2*q),mae=float(np.abs(yt-pred).mean()),
                        proper_train_ids=list(train.index[proper]),calibration_ids=list(train.index[cal]),test_ids=list(test.index),
                        parameters=dict(model='ET',n_estimators=128,max_features=1.,min_samples_leaf=1,seed=42),
                        note='Separate fixed model; main tuned results unchanged; exchangeability is an assumption, not guaranteed for materials datasets.')
            write_json(dest/f'fold_{fold}.json',result)
            rows.append({k:result[k] for k in ['task','representation','fold','nominal_coverage','coverage','mean_width','mae']})
    pd.DataFrame(rows).to_csv(OUT/'conformal_metrics.csv',index=False)
    print('Conformal analysis complete: 15 separate fits, calibration disjoint from fitting/test.')


if __name__=='__main__':main()
