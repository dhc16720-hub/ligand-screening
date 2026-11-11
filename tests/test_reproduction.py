"""Regression coverage for public-clone integrity and historical preservation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('reproduction', Path(__file__).resolve().parents[1]/'scripts/reproduce_topic1.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


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


if __name__ == '__main__': unittest.main()
