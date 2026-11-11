"""Frozen data must fail closed before model loading and remain unchanged."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from drone_detection.config import ExperimentConfig
from drone_detection.training import _without_snapshot_cache_writes, run_experiment


class FrozenTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="frozen-training-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "DUT_Anti_UAV_Clean_v1"
        self.dataset.mkdir()
        (self.dataset / "FREEZE.json").write_text("{}", encoding="utf-8")
        self.data = self.dataset / "data.yaml"
        self.data.write_text("train: images/train\nval: images/val\n", encoding="utf-8")
        self.weights = self.root / "weights.pt"
        self.weights.touch()
        self.output = self.root / "runs"
        self.config = ExperimentConfig("trial", self.weights, self.data, self.output)
        self.provenance = {
            "dataset_name": self.dataset.name,
            "manifest_sha256": "a" * 64,
            "freeze_sha256": "b" * 64,
        }

    def test_tampering_blocks_model_loading(self):
        with patch("drone_detection.dataset_snapshot.verify_snapshot", side_effect=ValueError("hash mismatch")):
            with patch("drone_detection.models.load_model") as loader:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    run_experiment(self.config)
                loader.assert_not_called()

    def test_removed_freeze_marker_is_not_treated_as_ordinary_data(self):
        (self.dataset / "FREEZE.json").unlink()
        with patch("drone_detection.models.load_model") as loader:
            with self.assertRaisesRegex(ValueError, "FREEZE.json"):
                run_experiment(self.config)
            loader.assert_not_called()

    def test_alternate_yaml_cannot_substitute_different_splits(self):
        alternate = self.root / "alternate.yaml"
        alternate.write_text(f"path: {self.dataset.as_posix()}\ntrain: elsewhere\n", encoding="utf-8")
        config = ExperimentConfig("trial", self.weights, alternate, self.output)
        with self.assertRaisesRegex(ValueError, "own data.yaml"):
            config.validate_files()

    def test_disk_cache_and_internal_output_are_rejected(self):
        for kwargs in (
            {"train": {"cache": "disk"}},
            {"project": self.dataset / "runs"},
            {"train": {"save_dir": str(self.dataset / "runs")}},
            {"train": {"device": "0,1"}},
        ):
            values = dict(name="trial", model=self.weights, data=self.data, project=self.output)
            values.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ExperimentConfig(**values).validate_files()

    def test_success_records_fingerprint_and_checks_after_training(self):
        self.output.mkdir()
        model = Mock(trainer=SimpleNamespace(save_dir=self.output))
        with patch("drone_detection.dataset_snapshot.verify_snapshot", return_value=self.provenance) as verify:
            with patch("drone_detection.models.load_model", return_value=model):
                with patch("drone_detection.training._without_snapshot_cache_writes", return_value=contextlib.nullcontext()):
                    with contextlib.redirect_stdout(io.StringIO()):
                        run_experiment(self.config)
        self.assertEqual(verify.call_count, 2)
        self.assertIs(model.train.call_args.kwargs["cache"], False)
        record = json.loads((self.output / "experiment.json").read_text(encoding="utf-8"))
        self.assertEqual(record["dataset_snapshot"], self.provenance)
        self.assertTrue(record["dataset_snapshot_verified_after_training"])
        self.assertEqual(record["training_status"], "completed")

    def test_changes_during_training_mark_experiment_failed(self):
        self.output.mkdir()
        model = Mock(trainer=SimpleNamespace(save_dir=self.output))
        model.train.side_effect = lambda **kwargs: (self.dataset / "FREEZE.json").unlink()
        with patch("drone_detection.dataset_snapshot.verify_snapshot", side_effect=[self.provenance, ValueError("extra file")]):
            with patch("drone_detection.models.load_model", return_value=model):
                with patch("drone_detection.training._without_snapshot_cache_writes", return_value=contextlib.nullcontext()):
                    with self.assertRaisesRegex(ValueError, "changed during"):
                        run_experiment(self.config)
        record = json.loads((self.output / "experiment.json").read_text(encoding="utf-8"))
        self.assertFalse(record["dataset_snapshot_verified_after_training"])
        self.assertEqual(record["training_status"], "failed")

    def test_dry_run_checks_snapshot_and_prints_fingerprint(self):
        from drone_detection.cli import main

        output = io.StringIO()
        with patch("drone_detection.cli.load_experiment", return_value=self.config):
            with patch("drone_detection.dataset_snapshot.verify_snapshot", return_value=self.provenance) as verify:
                with patch("drone_detection.models.load_model") as loader:
                    with contextlib.redirect_stdout(output):
                        main(["train", "--config", "unused.yaml", "--dry-run"])
        verify.assert_called_once_with(self.dataset)
        loader.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["dataset_snapshot"], self.provenance)

    def test_self_consistent_replacement_during_training_is_rejected(self):
        self.output.mkdir()
        for field in ("manifest_sha256", "freeze_sha256"):
            with self.subTest(changed_identity=field):
                replaced = {**self.provenance, field: "c" * 64}
                model = Mock(trainer=SimpleNamespace(save_dir=self.output))
                with patch("drone_detection.dataset_snapshot.verify_snapshot", side_effect=[self.provenance, replaced]):
                    with patch("drone_detection.models.load_model", return_value=model):
                        with patch("drone_detection.training._without_snapshot_cache_writes", return_value=contextlib.nullcontext()):
                            with self.assertRaisesRegex(ValueError, f"identity changed: {field}"):
                                run_experiment(self.config)
                record = json.loads((self.output / "experiment.json").read_text(encoding="utf-8"))
                self.assertEqual(record["dataset_snapshot"], self.provenance)
                self.assertFalse(record["dataset_snapshot_verified_after_training"])
                self.assertEqual(record["training_status"], "failed")

    def test_label_cache_suppression_is_limited_to_snapshot_and_restored(self):
        fake_top = ModuleType("ultralytics")
        fake_data = ModuleType("ultralytics.data")
        fake_dataset = ModuleType("ultralytics.data.dataset")
        fake_utils = ModuleType("ultralytics.data.utils")
        original = Mock()
        fake_dataset.save_dataset_cache_file = original
        fake_utils.save_dataset_cache_file = original
        fake_data.dataset, fake_data.utils = fake_dataset, fake_utils
        fake_top.data = fake_data
        modules = {module.__name__: module for module in (fake_top, fake_data, fake_dataset, fake_utils)}
        with patch.dict(sys.modules, modules):
            with _without_snapshot_cache_writes(self.dataset):
                cache = {}
                fake_dataset.save_dataset_cache_file("", self.dataset / "labels" / "train.cache", cache, "1")
                original.assert_not_called()
                self.assertEqual(cache["version"], "1")
                fake_utils.save_dataset_cache_file("", self.root / "other.cache", {}, "1")
                original.assert_called_once()
            self.assertIs(fake_dataset.save_dataset_cache_file, original)
            self.assertIs(fake_utils.save_dataset_cache_file, original)


if __name__ == "__main__":
    unittest.main()
