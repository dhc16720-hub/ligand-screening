"""Whole-project failures must not be hidden by an old success or short manifest."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import verify_topic1_project as project


class ProjectVerificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.patches = [patch.object(project.runner, 'ROOT', self.root),
                        patch.object(project.runner, 'check_public'),
                        patch.object(project.complete, 'verify'),
                        patch.object(project, 'replay_small', return_value={'pass_all': True}),
                        patch.object(project, 'replay_external', return_value={'pass_all': True})]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)
        manifest = self.root/project.runner.SOURCE_MANIFEST
        manifest.parent.mkdir(parents=True); manifest.write_text('{}')

    def read_record(self):
        return json.loads((self.root/project.REPORT).read_text())

    def test_truncated_manifest_rejected_even_when_remaining_hashes_match(self):
        (self.root/'a').write_text('a')
        (self.root/'manifest.json').write_text(json.dumps({'a': project.runner.digest(self.root/'a')}))
        with self.assertRaisesRegex(RuntimeError, 'coverage'):
            project.manifest_entries(self.root, 'manifest.json', {'a', 'missing-model'})

    def test_failure_replaces_previous_success(self):
        project.atomic_json(self.root/project.REPORT, {'pass_all': True})
        with patch.object(project, 'snapshot', side_effect=RuntimeError('missing model')):
            with self.assertRaisesRegex(RuntimeError, 'missing model'): project.verify()
        self.assertFalse(self.read_record()['pass_all'])
        self.assertEqual(self.read_record()['status'], 'failed')

    def test_scientific_replay_failure_cannot_pass(self):
        with patch.object(project, 'snapshot', return_value={'a': 'hash'}), \
             patch.object(project, 'replay_small', side_effect=AssertionError('wrong prediction')):
            with self.assertRaises(AssertionError): project.verify()
        self.assertFalse(self.read_record()['pass_all'])
        self.assertNotIn('external_diagnostics', self.read_record()['stages'])

    def test_external_failure_cannot_pass(self):
        with patch.object(project, 'snapshot', return_value={'a': 'hash'}), \
             patch.object(project, 'replay_external', side_effect=RuntimeError('wrong provenance')):
            with self.assertRaises(RuntimeError): project.verify()
        self.assertFalse(self.read_record()['pass_all'])

    def test_concurrent_artifact_change_rejected(self):
        with patch.object(project, 'snapshot', side_effect=[{'a':'before'}, {'a':'after'}]):
            with self.assertRaisesRegex(RuntimeError, 'changed'): project.verify()
        self.assertFalse(self.read_record()['pass_all'])

    def test_interrupt_invalidates_old_pass(self):
        with patch.object(project, 'snapshot', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): project.verify()
        self.assertEqual(self.read_record()['error']['type'], 'KeyboardInterrupt')
        self.assertFalse(self.read_record()['pass_all'])

    def test_success_records_all_three_scopes_and_artifact_hashes(self):
        with patch.object(project, 'snapshot', return_value={'a':'hash'}):
            project.verify()
        record = self.read_record()
        self.assertTrue(record['pass_all'])
        self.assertEqual(set(record['stages']), {'historical_and_final','small_data','external_diagnostics'})
        self.assertEqual(record['artifact_hashes'], {'a':'hash'})
        self.assertTrue(record['original_artifacts_unchanged'])
        self.assertFalse(record['training_performed'])

    def test_nonfinite_or_changed_numeric_table_rejected(self):
        original = self.root/'original'; replay = self.root/'replay'
        original.mkdir(); replay.mkdir()
        (original/'metrics.csv').write_text('score\n0.75\n')
        for value in ['0.70', 'NaN']:
            (replay/'metrics.csv').write_text('score\n'+value+'\n')
            with self.assertRaises(AssertionError):
                project.compare_tables(original, replay, ['metrics.csv'])
