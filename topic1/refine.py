"""V3 exploratory revision: frozen nested selection, retained negative results."""
import argparse
import importlib.metadata
import json
import platform
import numpy as np
import pandas as pd
from matbench.task import MatbenchTask
from matbench.constants import MBV01_VALIDATION_DATA_PATH
from topic1 import advanced as base
from topic1 import refine_models as models
from topic1.run import ROOT, OUT as V1, TASKS, sha, write_json

OUT = ROOT / 'outputs/topic1_v3'
V2 = ROOT / 'outputs/topic1_v2'


def register():
    protocol = dict(tasks=TASKS, seed=42, iterations=512, inner_folds=3,
                    regression=models.configs(False), classification=models.configs(True),
                    outer='Unchanged official Matbench five folds',
                    primary='C_structure for jdft2d; C for other tasks; jdft2d C is supplementary ablation',
                    reuse='Four V2 baseline inner OOF and outer predictions; identical partitions and features',
                    recipes='Best single, top-two mean, top-three mean; regression also top-three median; choose by mean inner fold loss',
                    objective='MAE in original units for regression; 1-ROC-AUC for classification',
                    caveat='Exploratory after V1/V2 outer results; not a new blind test; 512 boosting iterations versus V2 128 trees, unequal compute',
                    sources={p: sha(ROOT/p) for p in ['topic1/refine.py', 'topic1/refine_models.py', 'topic1/advanced.py', 'topic1/run.py']},
                    validation_sha256=sha(MBV01_VALIDATION_DATA_PATH),
                    v2_manifest_sha256=sha(V2/'delivery_manifest.json'))
    path = OUT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Frozen protocol changed'
    else:
        write_json(path, protocol)
        write_json(OUT/'environment.json', dict(python=platform.python_version(), packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}))


def run(name):
    task = MatbenchTask(name)
    cl = task.metadata.task_type == 'classification'
    base.OUT = OUT
    base.CONFIGS = models.configs(cl)
    base.make_model = models.make_model
    base.predict = models.predict
    info = pd.read_csv(V1/name/'samples.csv').set_index('id')
    assert list(info.index) == list(task.df.index)
    for rep in ['C'] + (['C_structure'] if name == 'matbench_jdft2d' else []):
        matrix = np.load(V1/name/f'features_{rep}.npy')
        for fold in task.folds:
            train, y = task.get_train_and_val_data(fold)
            test = task.get_test_data(fold)
            inner, grouped = base.inner_folds(np.asarray(y), info.loc[train.index, 'formula'].to_numpy(), cl, fold)
            manifest = dict(outer_train=list(train.index), outer_test=list(test.index), grouped=grouped,
                            inner=[dict(train=list(train.index[a]), validation=list(train.index[b])) for a,b in inner])
            assert manifest == json.loads((V2/name/'splits'/f'fold_{fold}.json').read_text())
            write_json(OUT/name/'splits'/f'fold_{fold}.json', manifest)
            preds, oofs = {}, {}
            for family in base.FAMILIES:
                folder = V2/name/'runs'/rep/family
                outer = pd.read_csv(folder/f'fold_{fold}.csv', float_precision='round_trip')
                oof = pd.read_csv(folder/f'inner_oof_{fold}.csv', float_precision='round_trip')
                assert list(outer.mbid) == list(test.index) and list(oof.mbid) == list(train.index)
                assert np.array_equal(oof.target.to_numpy(), np.asarray(y, float))
                preds[family] = outer.prediction.to_numpy()
                oofs[family] = oof.prediction.to_numpy()
            for family in base.CONFIGS:
                _, preds[family], oofs[family] = base.tune_fit(task, rep, family, fold, matrix, info, inner)
            recipe, candidates = models.select(np.asarray(y), oofs, inner, cl)
            pred = models.combine(preds, recipe)
            _, target = task.get_test_data(fold, include_target=True)
            base.save_prediction(task, rep, 'Selected', fold, test.index, target, pred,
                                 dict(recipe=recipe, candidate_recipes=candidates, seed=42, n_train=len(train), n_test=len(test)))
        base.export(task, rep, list(base.CONFIGS)+['Selected'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    args = parser.parse_args()
    register()
    for name in args.tasks:
        print('START', name, flush=True)
        run(name)
    print('COMPLETE', flush=True)
