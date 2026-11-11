#!/usr/bin/env python3
"""Verify all frozen stages and replay final science without changing original outputs."""
import json
import os
from pathlib import Path
import shutil
import tempfile

import reproduce_topic1 as runner

FINAL = 'outputs/topic1_final'
MANIFEST = 'verification_manifest.json'


def required_final_files():
    """Explicit coverage prevents a short or accidentally truncated manifest passing."""
    names = {'protocol.json', 'requirement_audit.json', 'validation_fold_metrics.csv',
             'validation_summary.csv', 'evidence.json', 'datasets/glass_ternary_hipt.json.gz'}

    def add(folder, files):
        names.update(f'{folder}/{name}' for name in files.split())

    for task in runner.TASKS:
        add(f'{task}/equivalence', 'result.json selection.json inner_oof.csv model.joblib')
        for seed in [7, 2026]:
            for fold in range(5):
                add(f'{task}/conditional/seed_{seed}_fold_{fold}',
                    'result.json predictions.csv model.joblib')
                if task in runner.TASKS[:2]:
                    add(f'{task}/reselection/seed_{seed}_fold_{fold}',
                        'result.json predictions.csv model.joblib selection.json inner_oof.csv')
        if task in runner.TASKS[:3]:
            for fold in range(5):
                add(f'{task}/conformal/fold_{fold}',
                    'result.json predictions.csv model.joblib selection.json inner_oof.csv calibration.csv intervals.csv')
        if task in [runner.TASKS[0], runner.TASKS[1], runner.TASKS[4]]:
            for rep in ['C', 'D']:
                for fold in range(5):
                    add(f'{task}/common_pool/{rep}_fold_{fold}', 'result.json predictions.csv')
    add('external', 'result.json predictions.csv features_C.npy features_D.npy features.json')
    add('external/full_training', 'model.joblib selection.json inner_oof.csv')
    return {f'{FINAL}/{name}' for name in names}


def required_files():
    return required_final_files() | {
        f'outputs/{version}/delivery_manifest.json' for version in runner.VERSIONS}


def check_final_audit(root):
    record = json.loads((root/FINAL/'requirement_audit.json').read_text())
    expected = dict(pass_all=True, outer_folds=5, validated_outer_prediction_files=115,
                    replayed_bundles=86, inner_candidate_fits=3789,
                    final_component_refits=200, total_new_fits=3989,
                    external_primary_n=5122, source_checksum_verified=True,
                    conformal_selection_calibration_separated=True, common_pool_reconstructed=True)
    if any((record.get(k) is not v if isinstance(v, bool) else record.get(k) != v)
           for k, v in expected.items()):
        raise RuntimeError('Final audit is incomplete or failed')
    if record.get('official_tasks') != runner.TASKS:
        raise RuntimeError('Final audit does not cover the five official tasks')


def snapshot(root):
    entries = {}
    for name in sorted(required_files()):
        path = root/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f'Missing or unsafe required artifact: {name}')
        entries[name] = runner.digest(path)
    return entries


def replay_science(root, entries):
    """Use the frozen auditor on private copies; only its derived tables are rewritten."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import brier_score_loss
    from topic1 import final_audit, final_validation

    if final_audit.ROOT.resolve() != root.resolve():
        raise RuntimeError('Auditor and verification root differ')
    with tempfile.TemporaryDirectory(prefix='topic1-complete-') as directory:
        scratch = Path(directory)
        for name in sorted(required_final_files()):
            target = scratch/Path(name).relative_to(FINAL)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root/name, target)
            if runner.digest(target) != entries[name]:
                raise RuntimeError(f'Artifact changed during copying: {name}')
        previous_audit, previous_validation = final_audit.OUT, final_validation.OUT
        try:
            final_audit.OUT = final_validation.OUT = scratch
            _, result = final_audit.audit()
        finally:
            final_audit.OUT, final_validation.OUT = previous_audit, previous_validation
        original = json.loads((root/FINAL/'requirement_audit.json').read_text())
        if result != original:
            raise RuntimeError('Replayed scientific audit differs from the saved audit')
        for name in ['validation_fold_metrics.csv', 'validation_summary.csv']:
            pd.testing.assert_frame_equal(pd.read_csv(root/FINAL/name), pd.read_csv(scratch/name),
                                          check_exact=False, rtol=1e-10, atol=1e-10)
        # The historical auditor verifies external predictions and AUC; also check Brier.
        predictions = pd.read_csv(scratch/'external/predictions.csv')
        external = json.loads((scratch/'external/result.json').read_text())
        masks = {'primary': predictions.primary,
                 'nonexact_overlap_removed': ~predictions.exact_training_overlap,
                 'all_rows_diagnostic': np.ones(len(predictions), dtype=bool)}
        for name, mask in masks.items():
            actual = brier_score_loss(predictions.loc[mask, 'target'], predictions.loc[mask, 'prediction'])
            np.testing.assert_allclose(actual, external[name]['brier'], rtol=1e-12, atol=1e-12)


def verify(*, seal=False):
    if not __debug__ or os.environ.get('PYTHONOPTIMIZE'):
        raise RuntimeError('Do not disable scientific audit assertions')
    root = runner.ROOT
    runner.check_outputs(runner.VERSIONS)
    for version in runner.VERSIONS:
        runner.audit_passed(version)
    manifest = root/FINAL/MANIFEST
    creating = not manifest.exists()
    if not creating:
        runner.check_manifest(root, manifest)
        manifest_digest = runner.digest(manifest)
        entries = json.loads(manifest.read_text())
        if set(entries) != required_files():
            raise RuntimeError('Final verification manifest has incomplete or unexpected coverage')
    elif not seal:
        raise RuntimeError('Missing final verification manifest; run seal-final once for existing results')
    else:
        entries = snapshot(root)
    check_final_audit(root)
    replay_science(root, entries)
    # Check again after replay to detect concurrent modification of original artifacts.
    if snapshot(root) != entries:
        raise RuntimeError('Artifacts changed during verification')
    runner.check_outputs(runner.VERSIONS)
    if not creating and (not manifest.is_file() or runner.digest(manifest) != manifest_digest):
        raise RuntimeError('Verification manifest changed during replay')
    if creating:
        # Atomic publication without replacing any existing manifest.
        with tempfile.NamedTemporaryFile(mode='w', dir=manifest.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(entries, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
        try:
            os.link(temporary, manifest)
        finally:
            temporary.unlink()
    print(f'COMPLETE VERIFIED: V1–V4 + final; {len(entries)} final/dependency artifacts; '
          '115 outer predictions, 86 model replays. Original scientific outputs unchanged.', flush=True)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(runner.ROOT))
    verify(seal='--seal' in sys.argv[1:])
