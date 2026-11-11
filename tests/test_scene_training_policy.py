"""External no-training policy must stop model loading, not alter frozen data."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from drone_detection.config import ExperimentConfig
from drone_detection.training import run_experiment


class SceneTrainingPolicyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.dataset = self.project / "02_datasets" / "DUT_Anti_UAV_Clean_v1"
        self.dataset.mkdir(parents=True)
        (self.dataset / "FREEZE.json").write_text("{}", encoding="utf-8")
        (self.dataset / "data.yaml").write_text("train: images/train\nval: images/val\n", encoding="utf-8")
        self.weights = self.root / "model.pt"
        self.weights.touch()
        self.config = ExperimentConfig("trial", self.weights, self.dataset / "data.yaml", self.root / "runs")
        self.policy_path = self.project / "configs" / "datasets" / "DUT_Anti_UAV_Clean_v1.training_policy.json"
        self.policy_path.parent.mkdir(parents=True)
        self.policy = dict(schema_version=1, dataset_name=self.dataset.name,
                           dataset_relative_path="02_datasets/DUT_Anti_UAV_Clean_v1", training_allowed=False,
                           reason="Gate 2 confirmed cross-split scene overlap.", documentation="docs/scene_sequence_audit.md")
        self.policy_path.write_text(json.dumps(self.policy), encoding="utf-8")

    def test_clean_v1_is_rejected_before_snapshot_hashing_or_model_loading(self):
        before = {p.name: p.read_bytes() for p in self.dataset.iterdir()}
        with patch("drone_detection.config.PROJECT_ROOT", self.project):
            with patch("drone_detection.dataset_snapshot.verify_snapshot") as verify:
                with patch("drone_detection.models.load_model") as loader:
                    with self.assertRaisesRegex(ValueError, "Training prohibited.*Gate 2"):
                        run_experiment(self.config)
        verify.assert_not_called()
        loader.assert_not_called()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.dataset.iterdir()})

    def test_same_named_dataset_outside_exact_project_root_is_unaffected(self):
        elsewhere = self.root / "independent" / self.dataset.name
        elsewhere.mkdir(parents=True)
        (elsewhere / "FREEZE.json").write_text("{}")
        (elsewhere / "data.yaml").write_text("train: images/train\n")
        config = ExperimentConfig("trial", self.weights, elsewhere / "data.yaml", self.root / "runs")
        with patch("drone_detection.config.PROJECT_ROOT", self.project):
            with patch("drone_detection.dataset_snapshot.verify_snapshot", return_value={"verified": True}) as verify:
                self.assertEqual(config.validate_files(), {"verified": True})
        verify.assert_called_once_with(elsewhere)

    def test_clean_v2_without_policy_still_requires_its_own_freeze_marker(self):
        version2 = self.project / "02_datasets" / "DUT_Anti_UAV_Clean_v2"
        version2.mkdir()
        (version2 / "data.yaml").write_text("train: images/train\n")
        config = ExperimentConfig("trial", self.weights, version2 / "data.yaml", self.root / "runs")
        with patch("drone_detection.config.PROJECT_ROOT", self.project):
            with self.assertRaisesRegex(ValueError, "FREEZE.json"):
                config.validate_files()

    def test_frozen_v2_is_not_blocked_by_v1_policy(self):
        version2 = self.project / "02_datasets" / "DUT_Anti_UAV_Clean_v2"
        version2.mkdir()
        (version2 / "FREEZE.json").write_text("{}")
        (version2 / "data.yaml").write_text("train: images/train\n")
        config = ExperimentConfig("trial", self.weights, version2 / "data.yaml", self.root / "runs")
        with patch("drone_detection.config.PROJECT_ROOT", self.project):
            with patch("drone_detection.dataset_snapshot.verify_snapshot", return_value={"version": 2}):
                self.assertEqual(config.validate_files(), {"version": 2})

    def test_malformed_policy_fails_closed(self):
        for value in ("false", None, 0):
            self.policy["training_allowed"] = value
            self.policy_path.write_text(json.dumps(self.policy), encoding="utf-8")
            with self.subTest(value=value), patch("drone_detection.config.PROJECT_ROOT", self.project):
                with self.assertRaisesRegex(ValueError, "Invalid dataset training policy"):
                    self.config.validate_files()


if __name__ == "__main__":
    unittest.main()
