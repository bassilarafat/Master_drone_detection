"""Scene allocation and release gates use temporary images, never real data."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import stat
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from drone_detection.data_audit import digest, dump_csv
from drone_detection.dataset_snapshot import create_snapshot, verify_snapshot
from drone_detection.scene_release import (
    FEATURES, REQUIRED_GATE_CHECKS, allocate_scene_groups, finalize_scene_release,
    load_scene_members, prepare_scene_release, write_allocation,
)


class AllocationTests(unittest.TestCase):
    def members(self):
        rows = []
        for index in range(18):
            split = ("train", "val", "test")[index % 3]
            group = f"G{index // 2:02d}"
            rows.append(dict(image=f"{split}/{index:05d}.jpg", old_split=split,
                             old_filename=f"{index:05d}.jpg", scene_group_id=group,
                             allocation_block_id="B_SHARED" if index < 4 else group,
                             images=1, objects=1, tiny=int(index % 4 == 0),
                             small=int(index % 4 == 1), medium=int(index % 4 == 2),
                             large=int(index % 4 == 3), negative_images=0))
        return rows

    def test_scenes_and_conservative_blocks_stay_whole_and_deterministic(self):
        members = self.members()
        first, report = allocate_scene_groups(members, seed=17, restarts=8)
        second, again = allocate_scene_groups(list(reversed(members)), seed=17, restarts=8)
        self.assertEqual(first, second)
        self.assertEqual(report, again)
        groups, blocks = {}, {}
        for row in first:
            groups.setdefault(row["scene_group_id"], set()).add(row["new_split"])
            blocks.setdefault(row["allocation_block_id"], set()).add(row["new_split"])
        self.assertTrue(all(len(value) == 1 for value in groups.values()))
        self.assertTrue(all(len(value) == 1 for value in blocks.values()))
        self.assertEqual({row["new_split"] for row in first}, {"train", "val", "test"})
        self.assertEqual(sum(report["splits"][s]["images"] for s in report["splits"]), len(members))

    def test_independence_has_priority_over_matching_exact_split_sizes(self):
        members = self.members()
        for row in members[:12]:
            row["allocation_block_id"] = "GIANT"
        # Keep each confirmed group in one allocation unit.
        rows, report = allocate_scene_groups(members, target_ratios={"train": .5, "val": .25, "test": .25}, restarts=8)
        giant_splits = {r["new_split"] for r in rows if r["allocation_block_id"] == "GIANT"}
        self.assertEqual(len(giant_splits), 1)
        split = next(iter(giant_splits))
        self.assertGreaterEqual(report["splits"][split]["images"], 12)

    def test_three_nonempty_splits_are_not_fabricated_from_two_units(self):
        members = self.members()[:4]
        with self.assertRaisesRegex(ValueError, "Fewer than three"):
            allocate_scene_groups(members)


class SceneReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self._unlock)
        original = self.root / "original"
        for index, split in enumerate(("train", "val", "test")):
            (original / "images" / split).mkdir(parents=True)
            (original / "labels" / split).mkdir(parents=True)
            for number, name in enumerate(("base", "other")):
                Image.new("RGB", (100, 100), (20 + index * 50, 20 + number * 80, 30)).save(original / "images" / split / f"{name}.jpg")
                (original / "labels" / split / f"{name}.txt").write_text("0 .5 .5 .2 .2\n")
        (original / "data.yaml").write_text("names:\n  0: UAV\n")
        self.source = self.root / "Example_Clean_v1"
        self.destination = self.root / "Example_Clean_v2"
        with contextlib.redirect_stdout(io.StringIO()):
            create_snapshot(original, self.source)
        self.groups = self.root / "scene_groups.csv"
        rows = []
        for index, split in enumerate(("train", "val", "test")):
            for name in ("base", "other"):
                group = "SHARED" if name == "base" and split in ("train", "val") else f"G_{split}_{name}"
                rows.append(dict(image=f"{split}/{name}.jpg", old_split=split,
                                 scene_group_id=group, allocation_block_id=group))
        dump_csv(self.groups, rows, list(rows[0]))
        self.members = load_scene_members(self.source, self.groups)
        self.rows, self.allocation = allocate_scene_groups(self.members, restarts=8)
        self.allocation_dir = self.root / "allocation"
        write_allocation(self.rows, self.allocation, self.allocation_dir)
        self.assignment = self.allocation_dir / "split_assignment.csv"

    def _unlock(self):
        for path in self.root.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IREAD | stat.S_IWRITE)

    def _prepare(self):
        return prepare_scene_release(self.source, self.destination, self.groups, self.assignment)

    def _gate(self, staging):
        return dict(status="PASS", checks={key: 0 for key in REQUIRED_GATE_CHECKS},
                    dataset_manifest_sha256=digest(staging / "dataset_manifest.csv"),
                    split_assignment_sha256=digest(staging / "split_assignment.csv"))

    def test_staging_preserves_source_and_renames_colliding_old_basenames(self):
        before = verify_snapshot(self.source)
        staging = self._prepare()
        self.assertFalse(self.destination.exists())
        self.assertFalse((staging / "FREEZE.json").exists())
        self.assertEqual(verify_snapshot(self.source), before)
        shared = [r for r in self.rows if r["scene_group_id"] == "SHARED"]
        self.assertEqual(len({r["new_split"] for r in shared}), 1)
        self.assertEqual(len({r["new_filename"] for r in shared}), 2)
        for row in self.rows:
            self.assertEqual(digest(staging / row["new_relative_image"]), row["image_sha256"])
            self.assertEqual(digest(staging / row["new_relative_label"]), row["label_sha256"])

    def test_failed_or_unbound_gate_never_freezes_or_publishes(self):
        staging = self._prepare()
        gate = self._gate(staging)
        gate["status"] = "FAIL"
        with self.assertRaisesRegex(ValueError, "must PASS"):
            finalize_scene_release(staging, self.destination, gate)
        gate["status"] = "PASS"
        gate["split_assignment_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "not bound"):
            finalize_scene_release(staging, self.destination, gate)
        self.assertFalse(self.destination.exists())
        self.assertFalse((staging / "FREEZE.json").exists())

    def test_gate_zero_checks_are_explicit_and_changed_labels_are_rejected(self):
        staging = self._prepare()
        gate = self._gate(staging)
        gate["checks"]["unresolved_high_risk_cross_split"] = False
        with self.assertRaisesRegex(ValueError, "integer zero"):
            finalize_scene_release(staging, self.destination, gate)
        gate["checks"]["unresolved_high_risk_cross_split"] = 0
        (staging / self.rows[0]["new_relative_label"]).write_text("0 .5 .5 .3 .3\n")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            finalize_scene_release(staging, self.destination, gate)

    def test_successful_release_is_immutable_and_retains_all_images(self):
        before = verify_snapshot(self.source)
        staging = self._prepare()
        result = finalize_scene_release(staging, self.destination, self._gate(staging),
                                        release_record=self.root / "release.json")
        self.assertEqual(sum(row["images"] for row in result["counts"].values()), 6)
        self.assertEqual(result["gate2_status"], "PASS")
        self.assertFalse(result["test_modified_after_release"])
        self.assertEqual(before, verify_snapshot(self.source))
        verify_snapshot(self.destination)
        self.assertTrue(all(not (p.stat().st_mode & stat.S_IWRITE)
                            for p in self.destination.rglob("*") if p.is_file()))
        with self.assertRaises(FileExistsError):
            self._prepare()

    def test_external_record_cannot_add_a_file_after_freezing_inside_release(self):
        staging = self._prepare()
        with self.assertRaisesRegex(ValueError, "must stay outside"):
            finalize_scene_release(staging, self.destination, self._gate(staging),
                                   release_record=self.destination / "release.json")
        self.assertFalse(self.destination.exists())
        self.assertFalse((staging / "FREEZE.json").exists())


if __name__ == "__main__":
    unittest.main()
