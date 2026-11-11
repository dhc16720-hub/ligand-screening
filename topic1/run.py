"""Run fixed, preregistered baselines; never select settings on test labels."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matbench.task import MatbenchTask
from matbench.constants import MBV01_VALIDATION_DATA_PATH
from matminer.featurizers.composition import ElementFraction, ElementProperty
from pymatgen.core import Composition
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, balanced_accuracy_score, accuracy_score, f1_score
from threadpoolctl import threadpool_limits

TASKS = ['matbench_steels', 'matbench_jdft2d', 'matbench_expt_gap', 'matbench_expt_is_metal', 'matbench_glass']
MODELS = ['Linear', 'RF', 'ET', 'Dummy']
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/topic1'
SEED = 42


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(inputs):
    """Stateless descriptors only; all learned preprocessing is fold-local."""
    ef = ElementFraction()
    ep = ElementProperty.from_preset('magpie', impute_nan=False)
    comps = [x.composition if hasattr(x, 'composition') else Composition(x) for x in inputs]
    aa, bb = [], []
    for c in comps:
        frac = np.array(list(c.fractional_composition.get_el_amt_dict().values()))
        aa.append(ef.featurize(c) + [len(frac), float(-np.sum(frac * np.log(frac))), float(np.sum(frac**2))])
        bb.append(ep.featurize(c))
    a, b = np.array(aa, dtype=float), np.array(bb, dtype=float)
    names_a = ef.feature_labels() + ['n_elements', 'composition_entropy', 'fraction_l2_squared']
    names_b = ep.feature_labels()
    matrices = {'A': a, 'B': b, 'C': np.column_stack([a, b])}
    names = {'A': names_a, 'B': names_b, 'C': names_a + names_b}
    if hasattr(inputs.iloc[0], 'lattice'):
        s = np.array([[float(x.density), x.volume / len(x), len(x),
                       max(x.lattice.abc) / min(x.lattice.abc),
                       min(x.lattice.angles), max(x.lattice.angles)] for x in inputs])
        matrices['C_structure'] = np.column_stack([matrices['C'], s])
        names['C_structure'] = names['C'] + ['density_g_cm3', 'volume_per_atom_A3', 'n_sites', 'lattice_length_ratio', 'min_angle_deg', 'max_angle_deg']
    for x in matrices.values():
        x[~np.isfinite(x)] = np.nan
    info = pd.DataFrame({'id': inputs.index, 'formula': [c.reduced_formula for c in comps],
                         'n_elements': [len(c.elements) for c in comps]})
    return matrices, names, info


def estimator(model, classification):
    if model == 'Linear':
        return LogisticRegression(C=1.0, max_iter=3000, random_state=SEED) if classification else Ridge(alpha=10.0)
    if model == 'Dummy':
        return DummyClassifier(strategy='prior') if classification else DummyRegressor(strategy='mean')
    cls = {'RF': (RandomForestRegressor, RandomForestClassifier), 'ET': (ExtraTreesRegressor, ExtraTreesClassifier)}[model][int(classification)]
    return cls(n_estimators=100, max_features=1.0, min_samples_leaf=1, random_state=SEED, n_jobs=4)


def pipeline(model, classification):
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), estimator(model, classification))


def scores(y, p, classification):
    if classification:
        return dict(roc_auc=float(roc_auc_score(y,p)), balanced_accuracy=float(balanced_accuracy_score(y,p > .5)),
                    accuracy=float(accuracy_score(y,p > .5)), f1=float(f1_score(y,p > .5)))
    return dict(mae=float(mean_absolute_error(y,p)), rmse=float(np.sqrt(mean_squared_error(y,p))), r2=float(r2_score(y,p)))


def prepare(task):
    folder = OUT / task.dataset_name
    folder.mkdir(parents=True, exist_ok=True)
    inputs = task.df[task.metadata.input_type]
    matrices, names, info = features(inputs)
    for name, matrix in matrices.items():
        np.save(folder / f'features_{name}.npy', matrix)
    info.to_csv(folder / 'samples.csv', index=False)
    write_json(folder / 'features.json', {k: dict(names=names[k], dimension=x.shape[1], missing_fraction=float(np.isnan(x).mean())) for k,x in matrices.items()})
    # Preserve the exact official ID order. Never regenerate outer folds.
    folds = {}
    for fold in task.folds:
        train, _ = task.get_train_and_val_data(fold)
        test = task.get_test_data(fold)
        assert not set(train.index) & set(test.index)
        assert set(train.index) | set(test.index) == set(inputs.index)
        folds[str(fold)] = dict(train=list(train.index), test=list(test.index))
    assert len(folds) == 5
    assert sorted(i for f in folds.values() for i in f['test']) == sorted(inputs.index)
    write_json(folder / 'official_folds.json', folds)
    write_json(folder / 'metadata.json', dict(task.metadata))
    return matrices, info.set_index('id')


def load_features(task):
    folder = OUT / task.dataset_name
    if not (folder / 'features.json').exists():
        return prepare(task)
    info = pd.read_csv(folder / 'samples.csv').set_index('id')
    assert list(info.index) == list(task.df.index)
    return {k: np.load(folder / f'features_{k}.npy') for k in json.loads((folder / 'features.json').read_text())}, info


def fit_one(task, matrix, info, representation, model, fold, fraction=1.0, learning=False):
    folder = OUT / task.dataset_name / ('learning' if learning else 'runs') / representation / model
    stem = f'fold_{fold}' + (f'_fraction_{fraction:.1f}' if learning else '')
    result_path = folder / f'{stem}.json'
    # A run's configuration and code hashes are checked before any resume.
    if result_path.exists():
        result = json.loads(result_path.read_text())
        return result, pd.read_csv(folder / f'{stem}.csv')['prediction'].to_numpy()
    xtrain, ytrain = task.get_train_and_val_data(fold)
    xtest = task.get_test_data(fold)
    if fraction < 1:
        # Nested subsets for a fixed fold, independent of target values.
        order = np.random.default_rng(SEED + fold).permutation(len(xtrain))
        keep = order[:max(10, int(len(xtrain) * fraction))]
        xtrain, ytrain = xtrain.iloc[keep], ytrain.iloc[keep]
    a, b = task.df.index.get_indexer(xtrain.index), task.df.index.get_indexer(xtest.index)
    assert min(a) >= 0 and min(b) >= 0 and not set(a) & set(b)
    classification = task.metadata.task_type == 'classification'
    pipe = pipeline(model, classification)
    start = time.monotonic()
    with threadpool_limits(limits=1):
        pipe.fit(matrix[a], np.asarray(ytrain))
        pred = pipe.predict_proba(matrix[b])[:, list(pipe[-1].classes_).index(True)] if classification else pipe.predict(matrix[b])
    assert len(pred) == len(b) and np.isfinite(pred).all()
    # Access held-out targets only after predictions have been made.
    _, ytest = task.get_test_data(fold, include_target=True)
    result = dict(task=task.dataset_name, representation=representation, model=model, fold=fold, fraction=fraction,
                  n_train=len(a), n_test=len(b), seconds=time.monotonic()-start,
                  metrics=scores(ytest, pred, classification), parameters=pipe[-1].get_params())
    frame = info.loc[xtest.index].reset_index()
    frame['fold'] = fold
    frame['target'] = np.asarray(ytest, dtype=float)
    frame['prediction'] = pred
    frame['abs_error'] = np.abs(frame.target-pred)
    frame['composition_seen_in_train'] = frame.formula.isin(set(info.loc[xtrain.index,'formula']))
    train_elements = np.any(matrix[a,:118] > 0, axis=0) if representation in ['A','C','C_structure'] else None
    if train_elements is not None:
        frame['unseen_element'] = np.any(matrix[b,:118][:,~train_elements] > 0, axis=1)
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / f'{stem}.csv', index=False)
    write_json(result_path, result)
    print(f'{task.dataset_name} {representation}/{model} fold={fold} fraction={fraction}: {result["metrics"]}', flush=True)
    return result, pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare','run','learning'])
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(seed=SEED, tasks=TASKS, models=MODELS, n_estimators=100, max_features=1.0,
                    linear={'ridge_alpha':10.0, 'logistic_C':1.0}, tuning='none; fixed before evaluation',
                    validation_sha256=sha(MBV01_VALIDATION_DATA_PATH), runner_sha256=sha(__file__))
    protocol_path = OUT / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'Code/config changed: use a new output directory; do not mix results.'
    else:
        write_json(protocol_path, protocol)
        write_json(OUT / 'environment.json', dict(python=platform.python_version(), platform=platform.platform(),
            packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
            matbench_commit=subprocess.check_output(['git','-C',str(ROOT/'third_party/matbench'),'rev-parse','HEAD'],text=True).strip()))
    for name in args.tasks:
        print(f'Loading official task: {name}', flush=True)
        task = MatbenchTask(name)
        if args.stage == 'prepare':
            matrices, info = prepare(task)
            print(f'Prepared {name}: {len(info)} samples; '+str({k:v.shape for k,v in matrices.items()}), flush=True)
            continue
        matrices, info = load_features(task)
        if args.stage == 'learning':
            if name not in ['matbench_steels','matbench_expt_gap']:
                continue
            for fold in task.folds:
                for fraction in [.2,.4,.6,.8]:
                    fit_one(task,matrices['C'],info,'C','RF',fold,fraction,learning=True)
            continue
        for representation, matrix in matrices.items():
            for model in MODELS:
                recorded = MatbenchTask(name, autoload=False)
                recorded.df = task.df
                for fold in task.folds:
                    result, pred = fit_one(task,matrix,info,representation,model,fold)
                    recorded.record(fold,pred,params={'representation':representation,'model':model,'seed':SEED,'parameters':result['parameters']})
                recorded.validate()
                dest = OUT / name / 'official_results' / f'{representation}_{model}.json.gz'
                dest.parent.mkdir(exist_ok=True)
                recorded.to_file(str(dest))
                MatbenchTask.from_file(str(dest)).validate()
    print('STAGE COMPLETE: '+args.stage, flush=True)


if __name__ == '__main__':
    main()
