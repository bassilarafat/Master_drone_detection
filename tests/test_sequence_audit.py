"""A provenance audit must preserve uncertainty, including repeated originals."""
import tempfile
import unittest
import contextlib
import csv
import io
import json
import stat
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from drone_detection.sequence_audit import audit_provenance, image_metadata, resolve_original
from drone_detection.dataset_snapshot import create_snapshot, verify_snapshot


class SequenceIdentityTests(unittest.TestCase):
    def source(self, video="video01", frame=100):
        return dict(video_id=video, frame_number=str(frame), original_path=f"{video}/{frame:05d}.jpg")

    def test_missing_exact_match_cannot_invent_video_or_frame_from_filename(self):
        result = resolve_original("different", "different_pixels", {}, {})
        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertEqual(result["verified"], "no")
        self.assertEqual(result["frame_number"], "")
        self.assertEqual(result["video_id"], "")

    def test_reencoded_identical_rgb_can_recover_content_crosswalk(self):
        result = resolve_original("reencoded", "rgb", {}, {"rgb": [self.source()]})
        self.assertEqual(result["verified"], "yes")
        self.assertEqual(result["frame_number"], 100)
        self.assertEqual(result["video_id"], "DUT_Anti_UAV_Tracking_V0/video01")
        self.assertEqual(result["clip_id"], "")
        self.assertEqual(result["evidence_type"], "exact_decoded_rgb_sha256")

    def test_repeated_frame_in_one_video_does_not_choose_arbitrary_index(self):
        result = resolve_original("same", "rgb", {"same": [self.source(frame=100), self.source(frame=101)]}, {})
        self.assertEqual(result["verified"], "yes")
        self.assertEqual(result["status"], "VERIFIED_SEQUENCE_AMBIGUOUS_FRAME")
        self.assertEqual(result["frame_number"], "")

    def test_same_content_in_multiple_videos_remains_ambiguous(self):
        result = resolve_original("same", "rgb", {"same": [self.source(), self.source(video="video02")]}, {})
        self.assertEqual(result["verified"], "no")
        self.assertEqual(result["video_id"], "")
        self.assertEqual(result["status"], "AMBIGUOUS_MULTIPLE_SOURCE_SEQUENCES")

    def test_byte_match_does_not_hide_reencoded_equivalent_in_another_video(self):
        first = self.source()
        second = self.source(video="video02")
        result = resolve_original("same", "rgb", {"same": [first]}, {"rgb": [first, second]})
        self.assertEqual(result["verified"], "no")
        self.assertEqual(result["video_id"], "")
        self.assertEqual(result["status"], "AMBIGUOUS_MULTIPLE_SOURCE_SEQUENCES")

    def test_numeric_export_index_is_separate_from_source_frame_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "00101.jpg"
            Image.new("RGB", (20, 30), "red").save(path)
            result = image_metadata(path, "expected")
            self.assertEqual(result["numeric_export_index"], 101)
            self.assertNotIn("frame_number", result)
            self.assertEqual(result["embedded_identity_claims"], "{}")
            self.assertFalse(result["exif_present"])


class SceneInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self._unlock)
        source = self.root / "source"
        self.dataset = self.root / "Example_Clean_v1"
        for index, split in enumerate(("train", "val", "test")):
            (source / "images" / split).mkdir(parents=True)
            (source / "labels" / split).mkdir(parents=True)
            Image.new("RGB", (30, 30), (index * 70, 20, 30)).save(source / "images" / split / "base.jpg")
            (source / "labels" / split / "base.txt").write_text("0 .5 .5 .2 .2\n")
        (source / "data.yaml").write_text("names:\n  0: UAV\n")
        with contextlib.redirect_stdout(io.StringIO()):
            create_snapshot(source, self.dataset)
        self.original = self.root / "original.csv"
        with self.original.open("w", newline="", encoding="utf-8") as handle:
            fields = ["video_id", "frame_number", "original_path", "sha256", "pixel_sha256"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(dict(video_id="video01", frame_number=1, original_path="video01/00001.jpg",
                                 sha256="0" * 64, pixel_sha256="0" * 64))

    def _unlock(self):
        for path in self.root.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IREAD | stat.S_IWRITE)

    def _audit(self, links):
        path = self.root / "links.json"
        path.write_text(json.dumps(links), encoding="utf-8")
        with patch("drone_detection.sequence_audit.historical_candidates", return_value=[]):
            with contextlib.redirect_stdout(io.StringIO()):
                return audit_provenance(self.dataset, self.original, self.root / "audit", scene_links=path)

    def test_existing_image_referenced_through_wrong_split_is_rejected(self):
        before = verify_snapshot(self.dataset)
        with self.assertRaisesRegex(ValueError, "not an image in this frozen release"):
            self._audit([dict(split_a="train", image_a="../val/base.jpg", split_b="test",
                              image_b="base.jpg", same_scene_confirmed=True, review_scene_id="scene1")])
        self.assertEqual(before, verify_snapshot(self.dataset))

    def test_string_confirmation_does_not_establish_scene_overlap(self):
        result = self._audit([dict(split_a="train", image_a="base.jpg", split_b="test",
                                  image_b="base.jpg", same_scene_confirmed="true", review_scene_id="scene1")])
        self.assertEqual(result["confirmed_reviewed_cross_split_scene_links"], 0)
        self.assertEqual(result["status"], "INDEPENDENCE_UNRESOLVED")
        self.assertFalse(result["independence_pass"])


if __name__ == "__main__":
    unittest.main()
