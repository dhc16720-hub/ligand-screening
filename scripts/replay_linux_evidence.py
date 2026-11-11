"""Replay saved Linux evidence without training or pretending an artifact subset is complete."""
import json
import platform
import sys
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import reproduce_topic1 as runner
from verify_topic1_project import atomic_json, replay_small

sys.path.insert(0, str(runner.ROOT))


def check_available_artifacts(root):
    checked = {}
    missing = []
    manifests = [root/'outputs'/v/'delivery_manifest.json' for v in runner.VERSIONS]
    manifests += [root/'outputs/topic1_final/verification_manifest.json',
                  root/'outputs/topic1_small_data/delivery_manifest.json']
    for manifest in manifests:
        entries = json.loads(manifest.read_text())
        if not isinstance(entries, dict) or not entries:
            raise RuntimeError(f'Empty or invalid evidence manifest: {manifest}')
        for name, expected in entries.items():
            path = root/name
            if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
                raise RuntimeError(f'Unsafe evidence path: {name}')
            if not path.is_file():
                missing.append(name)
                continue
            if runner.digest(path) != expected:
                raise RuntimeError(f'Changed saved evidence: {name}')
            checked[name] = expected
        checked[manifest.relative_to(root).as_posix()] = runner.digest(manifest)
    return checked, sorted(set(missing))


def replay_final_models():
    from topic1 import final_validation as final
    final.register()
    count = 0
    for name in final.TASKS:
        task, matrices, _, _ = final.data(name)
        for stage in ['conditional', 'reselection', 'conformal']:
            if stage == 'reselection' and name not in final.TASKS[:2]:
                continue
            if stage == 'conformal' and name not in final.TASKS[:3]:
                continue
            folders = sorted((final.OUT/name/stage).glob('*'))
            if len(folders) != (5 if stage == 'conformal' else 10):
                raise RuntimeError(f'Incomplete saved model coverage: {name}/{stage}')
            for folder in folders:
                record = json.loads((folder/'result.json').read_text())
                test, target = task.get_test_data(record['fold'], include_target=True)
                frame = pd.read_csv(folder/'predictions.csv', float_precision='round_trip')
                if list(frame.mbid) != list(test.index):
                    raise RuntimeError('Wrong official test identities')
                np.testing.assert_array_equal(frame.target, np.asarray(target, float))
                model = joblib.load(folder/'model.joblib')
                with threadpool_limits(limits=1):
                    actual = model.predict(final.take(matrices, task.df.index.get_indexer(test.index)))
                np.testing.assert_allclose(actual, frame.prediction, rtol=1e-11, atol=1e-8)
                count += 1
        print('SAVED MODEL REPLAY', name, count, flush=True)
    folder = final.OUT/'external'
    frame = pd.read_csv(folder/'predictions.csv', float_precision='round_trip')
    model = joblib.load(folder/'full_training/model.joblib')
    with threadpool_limits(limits=1):
        actual = model.predict({k: np.load(folder/f'features_{k}.npy') for k in ['C', 'D']})
    np.testing.assert_allclose(actual, frame.prediction, rtol=1e-11, atol=1e-8)
    count += 1
    if count != 86:
        raise RuntimeError('Incomplete final model replay')
    return count


def main():
    root = runner.ROOT
    out = root/'outputs/linux_recovery'
    target = out/'replay.json'
    record = dict(status='running', replay_passed=False, training_performed=False,
                  platform=platform.platform(), python=platform.python_version(),
                  unified_check_project_passed=False,
                  scope='Saved model replay and external diagnostics, not complete check-project',
                  source_run=43019918648, source_artifact=13866411278)
    atomic_json(target, record)
    try:
        runner.check_public()
        checked, missing = check_available_artifacts(root)
        record['missing_original_manifest_files'] = missing
        record['complete_original_archive'] = not missing
        for version in runner.VERSIONS + ['topic1_final']:
            audit = json.loads((root/'outputs'/version/'requirement_audit.json').read_text())
            if audit.get('pass_all') is not True:
                raise RuntimeError(f'Original stage audit not passed: {version}')
        record['final_model_replays'] = replay_final_models()
        record['small_data'] = replay_small(root)
        from experiments import external_diagnostics as external
        with patch.object(external, 'OUT', out/'external_diagnostics'):
            external.main()
        after, missing_after = check_available_artifacts(root)
        if after != checked or missing_after != missing:
            raise RuntimeError('Saved evidence changed during replay')
        for path in (out/'external_diagnostics').iterdir():
            if path.is_file():
                checked[path.relative_to(root).as_posix()] = runner.digest(path)
        record.update(status='passed', replay_passed=True, artifact_hashes=checked,
                      original_available_artifacts_unchanged=True,
                      replay_script_sha256=runner.digest(Path(__file__)),
                      limitation='Original Actions artifact omits some historical metadata/models. Original stage audits passed in the source run; this follow-up does not repeat their full audits or claim full archive coverage.')
        atomic_json(target, record)
        print('SAVED LINUX EVIDENCE REPLAY PASSED; not full check-project', flush=True)
    except BaseException as error:
        record.update(status='failed', replay_passed=False, error=str(error))
        atomic_json(target, record)
        raise


if __name__ == '__main__':
    main()
