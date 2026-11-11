"""Evidence subsets must reject changed bytes and must expose missing coverage."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import replay_linux_evidence as replay


class SavedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root/'outputs/sample.csv'
        self.data.parent.mkdir()
        self.data.write_bytes(b'value\n1\n')
        entries = {'outputs/sample.csv': hashlib.sha256(self.data.read_bytes()).hexdigest()}
        self.manifests = [self.root/'outputs'/v/'delivery_manifest.json' for v in replay.runner.VERSIONS]
        self.manifests += [self.root/'outputs/topic1_final/verification_manifest.json',
                           self.root/'outputs/topic1_small_data/delivery_manifest.json']
        for path in self.manifests:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(entries))

    def test_missing_file_is_explicit_not_complete(self):
        self.data.unlink()
        checked, missing = replay.check_available_artifacts(self.root)
        self.assertEqual(missing, ['outputs/sample.csv'])
        self.assertNotIn('outputs/sample.csv', checked)

    def test_changed_available_file_is_rejected(self):
        self.data.write_bytes(b'value\n2\n')
        with self.assertRaisesRegex(RuntimeError, 'Changed saved evidence'):
            replay.check_available_artifacts(self.root)

    def test_symlink_is_rejected_even_when_bytes_match(self):
        copy = self.root/'copy.csv'
        copy.write_bytes(self.data.read_bytes())
        self.data.unlink()
        self.data.symlink_to(copy)
        with self.assertRaisesRegex(RuntimeError, 'Unsafe evidence path'):
            replay.check_available_artifacts(self.root)


if __name__ == '__main__':
    unittest.main()
