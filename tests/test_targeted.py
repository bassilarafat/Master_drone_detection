import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from drone_detection.targeted import match_objects, size_bin, safe_crop, require_clean_audit


class TargetedMatchingTests(unittest.TestCase):
    def test_duplicate_and_miss(self):
        targets = [{'class': 0, 'box': [0, 0, 10, 10]},
                   {'class': 0, 'box': [50, 50, 60, 60]}]
        predictions = [{'class': 0, 'box': [0, 0, 10, 10], 'confidence': c}
                       for c in (0.3, 0.9)]
        matches, errors, misses = match_objects(predictions, targets)
        self.assertEqual(matches, [(1, 0, 1.0)])
        self.assertEqual(errors, [(0, 'Duplicate', 1.0)])
        self.assertEqual(misses, [1])

    def test_empty_predictions_preserve_false_negatives(self):
        self.assertEqual(match_objects([], [{'class': 0, 'box': [0, 0, 10, 10]}]),
                         ([], [], [0]))

    def test_wrong_class_cannot_match(self):
        self.assertEqual(len(match_objects([{'class': 1, 'box': [0,0,10,10], 'confidence': .9}],
                                          [{'class': 0, 'box': [0,0,10,10]}])[0]), 0)

    def test_size_boundaries_use_both_dimensions(self):
        self.assertEqual(size_bin([0,0,31,31]), 'Tiny')
        self.assertEqual(size_bin([0,0,32,31]), 'Small')
        self.assertEqual(size_bin([0,0,5,256]), 'Large')

    def test_margin_excludes_nearby_target(self):
        # Crop right edge is 164; GT starts at 175, leaving only 11 pixels.
        targets = [{'class': 0, 'box': [175, 80, 185, 90]}]
        self.assertIsNone(safe_crop([95,95,105,105], targets, (300,300)))
        self.assertEqual(safe_crop([95,95,105,105], [], (300,300)), [36,36,164,164])

    def test_crop_clips_at_image_border(self):
        self.assertEqual(safe_crop([0,0,10,10], [], (100,100)), [0,0,69,69])

    def test_unresolved_audit_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'audit').mkdir()
            (output/'audit/report.json').write_text(json.dumps({'ready_for_experiment':False}))
            with self.assertRaisesRegex(ValueError, 'unresolved'):
                require_clean_audit(SimpleNamespace(output=output, dataset=output/'data'))

    def test_changed_data_invalidates_previously_clean_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'audit').mkdir()
            (output/'audit/report.json').write_text(json.dumps({'ready_for_experiment':True, 'dataset_sha256':'old'}))
            with patch('drone_detection.data_audit.dataset_digest', return_value='changed'):
                with self.assertRaisesRegex(ValueError, 'Data changed'):
                    require_clean_audit(SimpleNamespace(output=output, dataset=output/'data'))
