"""Paired representation learning curves on official folds, with subset repetitions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import platform
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from matbench.task import MatbenchTask
from threadpoolctl import threadpool_limits

from topic1.run import ROOT, TASKS, pipeline, scores, write_json

OUT = ROOT/'outputs/topic1_small_data'
FRACTIONS = [.2, .5, 1.0]
SEEDS = [7, 42, 2026]
REPRESENTATIONS = ['A', 'B', 'C']
MODELS = ['RF', 'ET']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def configurations():
    for task, fold, fraction, rep, model in itertools.product(TASKS, range(5), FRACTIONS, REPRESENTATIONS, MODELS):
        for seed in ([42] if fraction == 1 else SEEDS):
            yield dict(task=task, fold=fold, fraction=fraction, subset_seed=seed, representation=rep, model=model)


def folder_for(config):
    return OUT/config['task']/f"fold_{config['fold']}"/f"fraction_{config['fraction']:.1f}_seed_{config['subset_seed']}"/config['representation']/config['model']


def subset_positions(n, fraction, seed, fold):
    if fraction == 1: return np.arange(n)
    return np.random.default_rng(seed + 1009*fold).permutation(n)[:max(10, int(n*fraction))]


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), ROOT/'topic1/run.py', ROOT/'topic1/requirements-v4.lock.txt']
    data_paths = [ROOT/'outputs/topic1'/task/name for task in TASKS
                  for name in ['official_folds.json', 'samples.csv', *[f'features_{r}.npy' for r in REPRESENTATIONS]]]
    protocol = dict(tasks=TASKS, official_folds=5, fractions=FRACTIONS, subset_seeds=SEEDS,
                    representations=REPRESENTATIONS, models=MODELS, model_seed=42,
                    trees=100, model_threads=1, expected_fits=len(list(configurations())),
                    sampling='Uniform nested subsets within each official training fold, no target-based sampling. Same IDs across representations and models. Full-data point fitted once.',
                    tuning='None. Fixed RF/ET settings; preprocessing fitted separately on each training subset.',
                    aggregation='Average subset repetitions within each fold first; then mean and SD over five official folds. Repetitions and overlapping folds are not independent samples.',
                    interpretation='Exploratory follow-up on previously inspected benchmarks. Composition A/B/C only; no structural features. Does not replace official full-data results.',
                    hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths+data_paths})
    destination=OUT/'protocol.json'
    if destination.exists():
        if json.loads(destination.read_text()) != protocol:
            raise RuntimeError('Study code/data/protocol changed; use a new output directory')
    else:
        write_json(destination, protocol)
        import importlib.metadata
        write_json(OUT/'environment.json', dict(platform=platform.platform(), python=platform.python_version(),
                   packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))
    return protocol


def validate_checkpoint(folder, config):
    record=json.loads((folder/'result.json').read_text())
    if record['config'] != config: raise RuntimeError(f'Wrong checkpoint config: {folder}')
    for name in ['model.joblib', 'predictions.csv']:
        if not (folder/name).is_file() or sha(folder/name) != record['sha256'][name]:
            raise RuntimeError(f'Damaged checkpoint: {folder/name}')
    return record


def fit_one(config):
    folder=folder_for(config)
    if (folder/'result.json').exists():
        validate_checkpoint(folder, config)
        return config
    task=MatbenchTask(config['task'])
    train,y=task.get_train_and_val_data(config['fold'])
    test=task.get_test_data(config['fold'])
    keep=subset_positions(len(train),config['fraction'],config['subset_seed'],config['fold'])
    ids=train.index[keep]
    matrix=np.load(ROOT/'outputs/topic1'/config['task']/f"features_{config['representation']}.npy")
    a=task.df.index.get_indexer(ids);b=task.df.index.get_indexer(test.index)
    if min(a)<0 or min(b)<0 or set(ids)&set(test.index): raise RuntimeError('Invalid training/test identity')
    cl=task.metadata.task_type=='classification'
    if cl and len(np.unique(np.asarray(y)[keep])) != 2: raise RuntimeError('Training subset has only one class')
    model=pipeline(config['model'],cl)
    model[-1].set_params(n_jobs=1)
    start=time.monotonic()
    with threadpool_limits(limits=1):
        model.fit(matrix[a],np.asarray(y)[keep])
        pred=model.predict_proba(matrix[b])[:,list(model[-1].classes_).index(True)] if cl else model.predict(matrix[b])
    if not np.isfinite(pred).all(): raise RuntimeError('Non-finite predictions')
    _,target=task.get_test_data(config['fold'],include_target=True)
    folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(dict(id=test.index,target=np.asarray(target,float),prediction=pred)).to_csv(folder/'predictions.csv',index=False)
    joblib.dump(model,folder/'model.joblib',compress=3)
    write_json(folder/'result.json',dict(config=config,train_ids=list(ids),test_ids=list(test.index),n_train=len(ids),
               seconds=time.monotonic()-start,metrics=scores(target,pred,cl),
               sha256={name:sha(folder/name) for name in ['model.joblib','predictions.csv']}))
    return config


def run(workers):
    register()
    jobs=list(configurations())
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i,config in enumerate(executor.map(fit_one,jobs),1):
            if i%12==0 or i==len(jobs): print(f'SMALL DATA {i}/{len(jobs)} {config}',flush=True)


def paired_tables(frame):
    # Equal weight per official fold; never count three subsets as three test folds.
    fold=frame.groupby(['task','model','representation','fraction','fold'],as_index=False).agg(
        score=('score','mean'), subset_sd=('score','std'),n_train=('n_train','first'),repeats=('score','size'))
    summary=fold.groupby(['task','model','representation','fraction'],as_index=False).agg(
        mean=('score','mean'),fold_sd=('score','std'),mean_subset_sd=('subset_sd','mean'),n_train_mean=('n_train','mean'))
    pivot=fold.pivot(index=['task','model','fraction','fold'],columns='representation',values='score').reset_index()
    rows=[]
    for base,candidate in [('A','B'),('A','C'),('B','C')]:
        for (task,model,fraction),group in pivot.groupby(['task','model','fraction']):
            direction=1 if task in TASKS[3:] else -1
            gain=direction*(group[candidate]-group[base])
            rows.append(dict(task=task,model=model,fraction=fraction,comparison=f'{candidate}-{base}',
                             mean_improvement=gain.mean(),fold_sd=gain.std(),improved_folds=int((gain>0).sum()),
                             total_folds=len(gain)))
    paired=pd.DataFrame(rows)
    contrast=paired.pivot(index=['task','model','comparison'],columns='fraction',values='mean_improvement')
    contrast['low_data_gain_minus_full']=contrast[.2]-contrast[1.0]
    return fold,summary,paired,contrast.reset_index()


def audit():
    register(); rows=[]; replays=0
    from sklearn.impute import SimpleImputer
    for name in TASKS:
        task=MatbenchTask(name);cl=task.metadata.task_type=='classification'
        matrices={rep:np.load(ROOT/'outputs/topic1'/name/f'features_{rep}.npy') for rep in REPRESENTATIONS}
        for config in (c for c in configurations() if c['task']==name):
            folder=folder_for(config);record=validate_checkpoint(folder,config)
            train,y=task.get_train_and_val_data(config['fold']);test,target=task.get_test_data(config['fold'],include_target=True)
            keep=subset_positions(len(train),config['fraction'],config['subset_seed'],config['fold'])
            ids=list(train.index[keep])
            if record['train_ids']!=ids or record['test_ids']!=list(test.index): raise RuntimeError('Wrong subset/test IDs')
            pred=pd.read_csv(folder/'predictions.csv',float_precision='round_trip');
            if list(pred.id)!=list(test.index): raise RuntimeError('Wrong prediction order')
            np.testing.assert_array_equal(pred.target,np.asarray(target,float))
            matrix=matrices[config['representation']];a=task.df.index.get_indexer(ids);b=task.df.index.get_indexer(test.index)
            bundle=joblib.load(folder/'model.joblib')
            expected=SimpleImputer(strategy='median',keep_empty_features=True).fit(matrix[a])
            np.testing.assert_allclose(bundle[0].statistics_,expected.statistics_,equal_nan=True)
            transformed=expected.transform(matrix[a])
            np.testing.assert_allclose(bundle[1].mean_,transformed.mean(axis=0),rtol=1e-10,atol=1e-10)
            if bundle[-1].n_estimators!=100 or bundle[-1].random_state!=42 or bundle[-1].n_jobs!=1:
                raise RuntimeError('Model budget/seed changed')
            with threadpool_limits(limits=1):
                replay=bundle.predict_proba(matrix[b])[:,list(bundle[-1].classes_).index(True)] if cl else bundle.predict(matrix[b])
            np.testing.assert_allclose(pred.prediction,replay,rtol=1e-11,atol=1e-8)
            metric=scores(target,pred.prediction.to_numpy(),cl)
            for key,value in metric.items(): np.testing.assert_allclose(record['metrics'][key],value,rtol=1e-10,atol=1e-10)
            rows.append(dict(**config,n_train=len(ids),score=metric['roc_auc' if cl else 'mae'],**metric))
            replays+=1
        print(f'SMALL DATA AUDIT {name}: {replays} models replayed',flush=True)
    frame=pd.DataFrame(rows)
    if len(frame)!=1050: raise RuntimeError('Incomplete study')
    frame.to_csv(OUT/'fold_metrics.csv',index=False)
    for label,table in zip(['fold_means','summary','paired_comparisons','low_data_interaction'],paired_tables(frame)):
        table.to_csv(OUT/f'{label}.csv',index=False)
    result=dict(pass_all=True,tasks=TASKS,official_folds=5,fitted_models=1050,replayed_models=replays,
                full_data_repeated=False,paired_subset_identity_verified=True,train_only_preprocessing_verified=True,
                aggregation='Subset means within each fold, then descriptive mean/SD over five folds; no independence-based significance claim.')
    write_json(OUT/'audit.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['run','audit'])
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not __debug__: raise RuntimeError('Assertions must remain enabled')
    if args.stage=='run':
        if not 1<=args.workers<=8: raise RuntimeError('Use 1–8 CPU workers')
        run(args.workers)
    else: audit()


if __name__=='__main__':main()
