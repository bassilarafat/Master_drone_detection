"""Release policy and byte-level immutability checks on temporary data only."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from drone_detection.data_audit import digest, normalize_dataset_labels
from drone_detection.dataset_snapshot import create_snapshot, verify_snapshot


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self._unlock_test_files)
        self.source = self.root / "original"
        self.destination = self.root / "DUT_Anti_UAV_Clean_v1"
        for index, split in enumerate(("train", "val", "test")):
            (self.source / "images" / split).mkdir(parents=True)
            (self.source / "labels" / split).mkdir(parents=True)
            self._sample(split, "base", (20 + index * 50, 30, 40))
        (self.source / "data.yaml").write_text("names:\n  0: UAV\n", encoding="utf-8")

    def _unlock_test_files(self):
        for path in self.root.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IREAD | stat.S_IWRITE)

    def _sample(self, split, name, color, label="0 0.5 0.5 0.2 0.2\n"):
        Image.new("RGB", (30, 30), color).save(self.source / "images" / split / (name + ".png"))
        (self.source / "labels" / split / (name + ".txt")).write_text(label, encoding="utf-8")

    def _create(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return create_snapshot(self.source, self.destination)

    def test_clean_policy_and_source_bytes_are_preserved(self):
        # Same pixels, equal numeric labels despite different formatting.
        self._sample("test", "z_copy", (120, 30, 40), "0 .50 .5 .20 .2\n")
        # Identical images with contradictory labels: neither enters training.
        self._sample("train", "conflict_a", (170, 0, 0))
        self._sample("train", "conflict_b", (170, 0, 0), "0 .4 .4 .2 .2\n")
        # A valid row alongside a bad one must not become partial ground truth.
        self._sample("val", "invalid", (0, 170, 0), "0 .4 .4 .2 0\n0 .5 .5 .2 .2\n")
        (self.source / "labels" / "train" / "old_backup.txt").write_text("old", encoding="utf-8")
        before = {p.relative_to(self.source): digest(p) for p in self.source.rglob("*") if p.is_file()}
        result = self._create()
        after = {p.relative_to(self.source): digest(p) for p in self.source.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual({s: r["images"] for s, r in result["counts"].items()},
                         {"train": 1, "val": 1, "test": 1})
        for name in ("conflict_a", "conflict_b"):
            self.assertFalse((self.destination / "images" / "train" / (name + ".png")).exists())
        self.assertFalse((self.destination / "labels" / "val" / "invalid.txt").exists())
        self.assertFalse((self.destination / "images" / "test" / "z_copy.png").exists())
        with (self.destination / "excluded_artifacts.csv").open(newline="", encoding="utf-8") as handle:
            self.assertEqual(next(csv.DictReader(handle))["reason"], "orphan_or_backup_label")
        self.assertEqual(digest(self.source / "labels/train/base.txt"),
                         digest(self.destination / "labels/train/base.txt"))

    def test_changed_label_extra_cache_and_missing_image_are_rejected(self):
        self._create()
        label = self.destination / "labels/train/base.txt"
        original = label.read_bytes()
        label.chmod(stat.S_IREAD | stat.S_IWRITE)
        label.write_text("0 .4 .4 .1 .1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file changed"):
            verify_snapshot(self.destination)
        label.write_bytes(original)
        cache = self.destination / "labels/train.cache"
        cache.write_text("cache", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            verify_snapshot(self.destination)
        cache.unlink()
        image = self.destination / "images/train/base.png"
        image.chmod(stat.S_IREAD | stat.S_IWRITE)
        image.unlink()
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            verify_snapshot(self.destination)

    def test_no_overwrite_and_no_normalization_of_frozen_version(self):
        self._create()
        with self.assertRaises(FileExistsError):
            self._create()
        with self.assertRaisesRegex(ValueError, "cannot be normalized"):
            normalize_dataset_labels(self.destination, self.root / "report", write=True)
        with self.assertRaisesRegex(ValueError, "outside"):
            normalize_dataset_labels(self.destination, self.destination / "report")
        self.assertFalse((self.destination / "report").exists())

    def test_cross_split_exact_group_is_quarantined_in_full(self):
        self._sample("train", "overlap", (240, 40, 50))
        self._sample("test", "overlap", (240, 40, 50))
        self._create()
        for split in ("train", "test"):
            self.assertFalse((self.destination / "images" / split / "overlap.png").exists())

    def test_changed_freeze_metadata_is_rejected(self):
        self._create()
        path = self.destination / "FREEZE.json"
        path.chmod(stat.S_IREAD | stat.S_IWRITE)
        value = json.loads(path.read_text())
        value["counts"]["train"]["objects"] = 123
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity differs"):
            verify_snapshot(self.destination)


if __name__ == "__main__":
    unittest.main()
