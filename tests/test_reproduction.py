"""Regression coverage for public-clone integrity and historical preservation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('reproduction', Path(__file__).resolve().parents[1]/'scripts/reproduce_topic1.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
with patch.dict(sys.modules, {'reproduce_topic1': r}):
    verification_spec = importlib.util.spec_from_file_location(
        'verification', Path(__file__).resolve().parents[1]/'scripts/verify_topic1_complete.py')
    verification = importlib.util.module_from_spec(verification_spec)
    verification_spec.loader.exec_module(verification)


class ReproductionTests(unittest.TestCase):
    def test_missing_and_empty_manifests_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(r, 'ROOT', root), self.assertRaises(RuntimeError):
                r.check_outputs(['topic1'])
            manifest = root/'manifest.json'; manifest.write_text('{}')
            with self.assertRaises(RuntimeError): r.check_manifest(root, manifest)

    def test_missing_changed_and_escaping_files_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); artifact = root/'data.csv'; artifact.write_text('original')
            manifest = root/'manifest.json'; manifest.write_text(json.dumps({'data.csv': r.digest(artifact)}))
            self.assertEqual(r.check_manifest(root, manifest), 1)
            artifact.write_text('changed')
            with self.assertRaises(RuntimeError): r.check_manifest(root, manifest)
            artifact.unlink()
            with self.assertRaises(RuntimeError): r.check_manifest(root, manifest)
            manifest.write_text(json.dumps({'../escape': '0'*64}))
            with self.assertRaises(RuntimeError): r.check_manifest(root, manifest)

    def test_freeze_requires_complete_scientific_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); folder = root/'outputs/topic1'; folder.mkdir(parents=True)
            (folder/'requirement_audit.json').write_text(json.dumps({'pass_all': True, 'tasks': []}))
            with patch.object(r, 'ROOT', root), self.assertRaises(RuntimeError): r.freeze('topic1')
            self.assertFalse((folder/'delivery_manifest.json').exists())

    def test_existing_manifest_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); folder=root/'outputs/topic1_v3'; folder.mkdir(parents=True)
            (folder/'requirement_audit.json').write_text('{"pass_all": true}')
            item=folder/'data.csv'; item.write_text('original')
            manifest=folder/'delivery_manifest.json'; manifest.write_text(json.dumps({'outputs/topic1_v3/data.csv': r.digest(item)}))
            before=manifest.read_bytes()
            with patch.object(r,'ROOT',root): r.freeze('topic1_v3')
            self.assertEqual(manifest.read_bytes(), before)
            item.write_text('changed')
            with patch.object(r,'ROOT',root), self.assertRaises(RuntimeError): r.freeze('topic1_v3')
            self.assertEqual(manifest.read_bytes(), before)


class CompleteVerificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.root_patch = patch.object(r, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        # Historical audit semantics already have separate tests; retain real hash checks.
        self.audit_patch = patch.object(r, 'audit_passed')
        self.audit_patch.start()
        self.addCleanup(self.audit_patch.stop)
        self.replay_patch = patch.object(verification, 'replay_science')
        self.replay = self.replay_patch.start()
        self.addCleanup(self.replay_patch.stop)
        for name in verification.required_files():
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture')
        for version in r.VERSIONS:
            folder = self.root/'outputs'/version
            item = folder/'evidence.csv'
            item.write_text('original')
            (folder/'delivery_manifest.json').write_text(json.dumps({str(item.relative_to(self.root)): r.digest(item)}))
        self.audit = self.root/verification.FINAL/'requirement_audit.json'
        self.audit.write_text(json.dumps(dict(
            pass_all=True, official_tasks=r.TASKS, outer_folds=5,
            validated_outer_prediction_files=115, replayed_bundles=86,
            inner_candidate_fits=3789, final_component_refits=200, total_new_fits=3989,
            external_primary_n=5122, source_checksum_verified=True,
            conformal_selection_calibration_separated=True, common_pool_reconstructed=True)))
        self.manifest = self.root/verification.FINAL/verification.MANIFEST

    def test_seal_then_verify_preserves_every_original_byte(self):
        before = verification.snapshot(self.root)
        verification.verify(seal=True)
        frozen = self.manifest.read_bytes()
        verification.verify()
        self.assertEqual(before, verification.snapshot(self.root))
        self.assertEqual(frozen, self.manifest.read_bytes())
        self.assertEqual(self.replay.call_count, 2)

    def test_missing_manifest_is_not_silently_created(self):
        with self.assertRaisesRegex(RuntimeError, 'Missing final verification manifest'):
            verification.verify()
        self.replay.assert_not_called()
        self.assertFalse(self.manifest.exists())

    def test_deleted_prediction_or_modified_model_is_rejected_before_replay(self):
        verification.verify(seal=True)
        for suffix in ['predictions.csv', 'model.joblib']:
            path = next(p for p in (self.root/verification.FINAL).rglob(suffix))
            original = path.read_bytes()
            self.replay.reset_mock()
            if suffix == 'predictions.csv': path.unlink()
            else: path.write_text('tampered')
            with self.assertRaises(RuntimeError): verification.verify()
            self.replay.assert_not_called()
            path.write_bytes(original)

    def test_truncated_manifest_does_not_pass_even_with_valid_hashes(self):
        verification.verify(seal=True)
        entries = json.loads(self.manifest.read_text())
        del entries[next(k for k in entries if k.endswith('model.joblib'))]
        self.manifest.write_text(json.dumps(entries))
        self.replay.reset_mock()
        with self.assertRaisesRegex(RuntimeError, 'coverage'): verification.verify()
        self.replay.assert_not_called()

    def test_missing_required_artifact_cannot_be_sealed(self):
        self.audit.unlink()
        with self.assertRaises(RuntimeError): verification.verify(seal=True)
        self.assertFalse(self.manifest.exists())

    def test_pass_flag_alone_cannot_be_sealed(self):
        self.audit.write_text('{"pass_all": true}')
        with self.assertRaisesRegex(RuntimeError, 'incomplete'): verification.verify(seal=True)
        self.replay.assert_not_called()
        self.assertFalse(self.manifest.exists())

    def test_failed_scientific_replay_does_not_publish_manifest(self):
        self.replay.side_effect = RuntimeError('scientific mismatch')
        with self.assertRaisesRegex(RuntimeError, 'scientific mismatch'): verification.verify(seal=True)
        self.assertFalse(self.manifest.exists())

    def test_historical_tampering_is_rejected(self):
        verification.verify(seal=True)
        (self.root/'outputs/topic1/evidence.csv').write_text('tampered')
        self.replay.reset_mock()
        with self.assertRaises(RuntimeError): verification.verify()
        self.replay.assert_not_called()

    def test_modification_during_replay_prevents_sealing(self):
        self.replay.side_effect = lambda *_: self.audit.write_text('changed during audit')
        with self.assertRaisesRegex(RuntimeError, 'changed during verification'):
            verification.verify(seal=True)
        self.assertFalse(self.manifest.exists())

    def test_final_rerun_verifies_before_any_scientific_output_is_written(self):
        verification.verify(seal=True)
        with patch.object(r, 'run', side_effect=RuntimeError('verification failed')), \
                patch.object(r, 'module') as module, patch.object(r, 'download_external') as download:
            with self.assertRaisesRegex(RuntimeError, 'verification failed'): r.stage('final')
        module.assert_not_called()
        download.assert_not_called()


if __name__ == '__main__': unittest.main()
