"""Data-contract regression tests using only temporary, synthetic fixtures.

The .jpg/.png files contain fake bytes, not decodable images. These tests check
IDs, paths, CSV agreement and content hashes; they do not test image decoding.
Run from the project root: python -B docs/skills/tools/test_validate_collection.py
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


VALIDATOR_PATH = Path(__file__).with_name('validate_collection.py')
spec = importlib.util.spec_from_file_location('collection_validator', VALIDATOR_PATH)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class DataContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='gelar_rasa_contract_')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.classes = ['glass', 'paper', 'electronic', 'metal', 'plastic',
                        'cardboard', 'textile', 'foodwaste']
        self.contract = {
            'class_order': self.classes,
            'train_count': 2,
            'test_count': 2,
            'train_id_width': 4,
            'test_id_width': 3,
            'submission_columns': ['image_id', 'waste_category'],
            'sample_submission': 'data/sample.csv',
            'train_labels': 'review/labels.csv',
            'train_label_audit': 'review/audit.csv',
            'train_source_manifest': 'review/manifest.json',
            'train_originals': 'data/train',
            'train_labeled_copies': 'data/train/label',
            'test_images': 'data/test',
        }
        self.originals = self.root / self.contract['train_originals']
        self.copies = self.root / self.contract['train_labeled_copies']
        self.test_images = self.root / self.contract['test_images']
        for label in self.classes:
            (self.copies / label).mkdir(parents=True)
        self.test_images.mkdir(parents=True)
        self.labels = []
        self.audit = []
        self.manifest = []
        for image_id, filename, label in [
            ('0001', '0001.jpg', 'glass'),
            ('0002', '0002.png', 'paper'),
        ]:
            content = ('Integrity fixture, not an image: ' + image_id).encode('ascii')
            (self.originals / filename).write_bytes(content)
            (self.copies / label / filename).write_bytes(content)
            sha256 = hashlib.sha256(content).hexdigest()
            self.labels.append({'image_id': image_id, 'filename': filename,
                                'waste_category': label})
            self.audit.append({'image_id': image_id, 'filename': filename,
                               'waste_category': label, 'source_sha256': sha256})
            self.manifest.append({'image_id': image_id, 'filename': filename,
                                  'sha256': sha256})
        self.sample = [{'image_id': '001', 'waste_category': 'glass'},
                       {'image_id': '002', 'waste_category': 'paper'}]
        for filename in ['001.jpg', '002.png']:
            (self.test_images / filename).write_bytes(b'Fake test image bytes')
        self.write_metadata()

    def write_csv(self, key, fields, rows):
        path = self.root / self.contract[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def write_metadata(self):
        self.write_csv('sample_submission', self.contract['submission_columns'], self.sample)
        self.write_csv('train_labels', ['image_id', 'filename', 'waste_category'], self.labels)
        self.write_csv('train_label_audit',
                       ['image_id', 'filename', 'waste_category', 'source_sha256'], self.audit)
        (self.root / self.contract['train_source_manifest']).write_text(
            json.dumps(self.manifest), encoding='utf-8')

    def verify(self, verify_images=False):
        return validator.verify_data(self.root, self.contract, verify_images)

    def test_valid_contract_in_both_hash_modes(self):
        for verify_images in (False, True):
            with self.subTest(verify_images=verify_images):
                report = self.verify(verify_images)
                self.assertEqual(report['train_label_rows'], 2)
                self.assertEqual(report['sample_test_rows'], 2)
                self.assertEqual(report['test_files'], 2)
                self.assertTrue(report['label_audit_and_copy_paths_agree'])
                self.assertEqual(report['train_image_hashes_verified'],
                                 4 if verify_images else 0)

    def test_filename_must_match_id(self):
        self.labels[0]['filename'] = '0009.jpg'
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'Filename/ID mismatch'):
            self.verify()

    def test_train_id_preserves_leading_zeros(self):
        for rows in (self.labels, self.audit, self.manifest):
            rows[0]['image_id'] = '1'
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'Invalid train ID'):
            self.verify()

    def test_sample_id_preserves_leading_zeros(self):
        self.sample[0]['image_id'] = '1'
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'Invalid sample ID formatting'):
            self.verify()

    def test_duplicate_train_id_is_rejected(self):
        self.labels[1]['image_id'] = self.labels[0]['image_id']
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'Duplicate train ID'):
            self.verify()

    def test_duplicate_sample_id_is_rejected(self):
        self.sample[1]['image_id'] = self.sample[0]['image_id']
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'Duplicate sample ID'):
            self.verify()

    def test_audit_label_disagreement_is_rejected(self):
        self.audit[0]['waste_category'] = 'plastic'
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'Audit/label disagreement'):
            self.verify()

    def test_extra_labeled_copy_is_rejected(self):
        (self.copies / 'plastic' / '0001.jpg').write_bytes(
            (self.originals / '0001.jpg').read_bytes())
        with self.assertRaisesRegex(ValueError, 'Extra/missing labeled copy'):
            self.verify()

    def test_hash_verification_detects_changed_source_or_copy(self):
        paths = [self.originals / '0001.jpg', self.copies / 'glass' / '0001.jpg']
        for path in paths:
            with self.subTest(path=path.relative_to(self.root).as_posix()):
                original = path.read_bytes()
                try:
                    path.write_bytes(b'Tampered integrity fixture')
                    # Structural mode deliberately does not promise content verification.
                    self.assertEqual(self.verify()['train_image_hashes_verified'], 0)
                    with self.assertRaisesRegex(ValueError, 'Image content changed'):
                        self.verify(verify_images=True)
                finally:
                    path.write_bytes(original)

    def test_agreeing_copies_still_must_match_manifest_hash(self):
        for path in [self.originals / '0001.jpg', self.copies / 'glass' / '0001.jpg']:
            path.write_bytes(b'Both changed, so copy equality alone is insufficient')
        with self.assertRaisesRegex(ValueError, 'Image content changed'):
            self.verify(verify_images=True)

    def test_audit_manifest_hash_disagreement_is_rejected(self):
        self.audit[0]['source_sha256'] = '0' * 64
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, 'Audit/manifest hash differs'):
            self.verify()

    def test_contract_path_cannot_escape_fixture_root(self):
        self.contract['sample_submission'] = '../outside.csv'
        with self.assertRaisesRegex(ValueError, 'Path outside expected root'):
            self.verify()

    def test_filename_cannot_traverse_directories(self):
        for filename in ('../0001.jpg', '..\\0001.jpg'):
            with self.subTest(filename=filename):
                self.labels[0]['filename'] = filename
                self.write_metadata()
                with self.assertRaisesRegex(ValueError, 'Invalid filename'):
                    self.verify()

    def test_missing_test_id_is_rejected(self):
        (self.test_images / '002.png').unlink()
        with self.assertRaisesRegex(ValueError, 'Test file IDs differ from sample'):
            self.verify()

    def test_replaced_test_id_is_rejected_even_when_count_matches(self):
        (self.test_images / '002.png').rename(self.test_images / '003.png')
        with self.assertRaisesRegex(ValueError, 'Test file IDs differ from sample'):
            self.verify()

    def test_optimized_cli_fails_without_writing_report(self):
        # Run a temporary copy so even a regression cannot overwrite the real report.
        copied_script = self.root / 'docs/skills/tools/validate_collection.py'
        copied_script.parent.mkdir(parents=True)
        copied_script.write_bytes(VALIDATOR_PATH.read_bytes())
        report_path = copied_script.parent.parent / 'sources/validation.json'
        report_path.parent.mkdir()
        sentinel = b'{"preserve": "temporary report"}\n'
        report_path.write_bytes(sentinel)
        result = subprocess.run(
            [sys.executable, '-B', '-O', str(copied_script), '--no-write'],
            cwd=self.root, capture_output=True, text=True, timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['error_type'], 'ValueError')
        self.assertIn('without Python -O', report['error'])
        self.assertEqual(report_path.read_bytes(), sentinel)


if __name__ == '__main__':
    unittest.main(verbosity=2)
