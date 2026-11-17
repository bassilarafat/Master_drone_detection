"""Controlled-run regression checks without model inference or real training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class ControlledYOLO26Tests(unittest.TestCase):
    def test_training_needs_explicit_execution_before_asset_reads(self):
        from drone_detection.yolo26_controlled import train_controlled
        with patch('drone_detection.yolo26_controlled.load_controls') as load:
            with self.assertRaises(ValueError):
                train_controlled('E0', 17, execute=False)
            load.assert_not_called()

    def test_unsupported_arm_and_seed_fail_before_config_or_dataset_reads(self):
        from drone_detection.yolo26_controlled import dry_run, train_controlled
        for arm, seed in [('E2', 17), ('E3', 17), ('test', 17), ('E0', 0), ('E1', 2026)]:
            with self.subTest(arm=arm, seed=seed):
                with patch('drone_detection.yolo26_controlled.load_controls') as load:
                    with self.assertRaises(ValueError):
                        dry_run(arm, seed)
                    with self.assertRaises(ValueError):
                        train_controlled(arm, seed, execute=True)
                    load.assert_not_called()

    def test_both_architectures_have_identical_explicit_training_and_evaluation_controls(self):
        from drone_detection.yolo26_controls import load_controls
        baseline, protocol, _ = load_controls('E0', require_lock=False)
        p2, _, _ = load_controls('E1', require_lock=False)
        self.assertEqual(baseline['train'], p2['train'])
        self.assertEqual(baseline['evaluation'], p2['evaluation'])
        self.assertEqual(baseline['seeds'], [17, 42, 123])
        self.assertEqual(p2['seeds'], [17, 42, 123])
        train = baseline['train']
        for name, expected in {'optimizer': 'MuSGD', 'epochs': 100, 'imgsz': 960,
                               'batch': 1, 'nbs': 16, 'warmup_epochs': 0,
                               'amp': False, 'nms': True, 'resume': False}.items():
            self.assertEqual(train[name], expected, name)
        self.assertEqual(baseline['checkpoint']['filename'], 'last.pt')
        self.assertEqual(baseline['checkpoint']['epoch'], 100)
        self.assertIs(baseline['evaluation']['nms'], True)
        self.assertEqual(baseline['evaluation']['quantize'], 32)

    def _synthetic_attempt(self, *, integrity_failure=False, training_failure=False):
        """Exercise failure recording without a detector, release, or GPU."""
        from drone_detection.yolo26_controlled import train_controlled
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        config_path = root / 'synthetic_config.yaml'
        config_path.write_text('synthetic fixture configuration; no dataset', encoding='utf-8')
        checkpoint = root / 'synthetic_last.pt'
        checkpoint.write_bytes(b'fixture only: never passed to a model loader')
        config = dict(dataset={'path': str(root / 'not_a_release')},
                      output_root=str(root / 'runs'), experiment_id='E0_Y26',
                      protocol_sha256='fixture-protocol', train={}, evaluation={})
        identity = {'release_id': 'synthetic-fixture-only'}
        with patch('drone_detection.yolo26_controlled.dry_run', return_value={'dataset_release': identity}), \
             patch('drone_detection.yolo26_controlled.load_controls', return_value=(config, {}, config_path)), \
             patch('drone_detection.yolo26_controlled.configure_runtime'), \
             patch('drone_detection.yolo26_controlled.training_view', return_value=(root / 'never_read.yaml', None)), \
             patch('drone_detection.yolo26_controlled._fit', return_value=checkpoint,
                   side_effect=RuntimeError('synthetic training failure') if training_failure else None), \
             patch('drone_detection.yolo26_controlled.pin_release', return_value=identity,
                   side_effect=ValueError('synthetic dataset changed') if integrity_failure else None), \
             patch('drone_detection.yolo26_controlled._evaluate') as evaluate:
            with self.assertRaisesRegex(ValueError if integrity_failure else RuntimeError,
                                        'dataset changed' if integrity_failure else 'training failure'):
                train_controlled('E0', 17, execute=True)
            evaluate.assert_not_called()
        return json.loads((root / 'runs/seed_17/run.json').read_text(encoding='utf-8'))

    def test_post_training_integrity_failure_is_persisted_and_cannot_look_complete(self):
        record = self._synthetic_attempt(integrity_failure=True)
        self.assertEqual(record['status'], 'FAILED_DATASET_INTEGRITY')
        self.assertIs(record['dataset_verified_after'], False)
        self.assertIs(record['test_evaluated'], False)
        self.assertNotIn('summary', record)

    def test_training_failure_preserves_error_even_when_release_verification_passes(self):
        record = self._synthetic_attempt(training_failure=True)
        self.assertEqual(record['status'], 'FAILED')
        self.assertIn('synthetic training failure', record['error'])
        self.assertIs(record['dataset_verified_after'], True)
        self.assertNotIn('summary', record)


if __name__ == '__main__':
    unittest.main()
