import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from drone_detection.scene_gate import audit_staging, boundary_candidates
from drone_detection.data_audit import digest


class FreshGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.staging = self.root / "staging"
        self.staging.mkdir()
        for name in ("dataset_manifest.csv", "split_assignment.csv"):
            (self.staging / name).write_text("bound fixture\n")
        self.rows = [dict(image="train/a.jpg", new_split="train", scene_group_id="S1", allocation_block_id="B1", image_sha256="a"),
                     dict(image="test/b.jpg", new_split="test", scene_group_id="S2", allocation_block_id="B2", image_sha256="b")]

    def write_csv(self, name, rows):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def run_gate(self, decisions, expected=None):
        expected = expected or [dict(image_a="train/a.jpg", image_b="test/b.jpg")]
        candidate = self.write_csv("boundary.csv", expected)
        geometry = self.write_csv("geometry.csv", decisions)
        geometry.with_suffix('.csv.metadata.json').write_text(json.dumps(dict(
            candidates_sha256=digest(candidate),complete=True)),encoding='utf-8')
        with patch("drone_detection.scene_gate._validate_staged_bytes", return_value=(self.rows, {})):
            return audit_staging(self.staging, geometry, self.root / "audit", expected_boundary_csv=candidate)

    def test_new_confirmed_crossing_fails_gate_before_publication(self):
        report = self.run_gate([dict(image_a="train/a.jpg", image_b="test/b.jpg", geometry_decision="CONFIRMED_GEOMETRY")])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["checks"]["confirmed_same_scene_train_test"], 1)

    def test_unresolved_high_risk_crossing_cannot_pass_as_no_geometry(self):
        report = self.run_gate([dict(image_a="train/a.jpg", image_b="test/b.jpg", geometry_decision="REVIEW",
                                    cosine_similarity=.99, phash_distance=2)])
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["checks"]["unresolved_high_risk_cross_split"], 1)

    def test_success_needs_feature_identity_as_well_as_zero_crossings(self):
        candidate=self.write_csv('boundary.csv',[dict(image_a='train/a.jpg',image_b='test/b.jpg')])
        geometry=self.write_csv('geometry.csv',[dict(image_a='train/a.jpg',image_b='test/b.jpg',geometry_decision='NO_GEOMETRIC_SUPPORT',
                                                     cosine_similarity=.1,phash_distance=60)])
        feature_index=self.write_csv('features.csv',[dict(image_id=r['image'],sha256=r['image_sha256']) for r in self.rows])
        geometry.with_suffix('.csv.metadata.json').write_text(json.dumps(dict(candidates_sha256=digest(candidate),
                                  feature_index_sha256=digest(feature_index),complete=True)),encoding='utf-8')
        with patch('drone_detection.scene_gate._validate_staged_bytes',return_value=(self.rows,{})):
            result=audit_staging(self.staging,geometry,self.root/'audit',expected_boundary_csv=candidate,feature_index=feature_index)
        self.assertEqual(result['status'],'PASS')
        self.assertTrue(all(v==0 for v in result['checks'].values()))

    def test_missing_new_boundary_evaluation_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            self.run_gate([dict(image_a="train/a.jpg", image_b="test/b.jpg", geometry_decision="NO_GEOMETRIC_SUPPORT")],
                          expected=[dict(image_a="train/a.jpg", image_b="test/b.jpg"), dict(image_a="train/c.jpg", image_b="test/d.jpg")])

    def test_boundary_selection_uses_new_assignment_instead_of_old_split(self):
        assignments = self.write_csv("assign.csv", [
            dict(image="train/a.jpg", new_split="train", new_relative_image="images/train/train_a.jpg"),
            dict(image="train/b.jpg", new_split="test", new_relative_image="images/test/train_b.jpg"),
            dict(image="val/c.jpg", new_split="train", new_relative_image="images/train/val_c.jpg")])
        candidates = self.write_csv("candidates.csv", [dict(image_a="train/a.jpg", image_b="train/b.jpg"),
                                                       dict(image_a="train/a.jpg", image_b="val/c.jpg")])
        output = self.root / "new_boundary.csv"
        result = boundary_candidates(candidates, assignments, output)
        self.assertEqual(result["boundary_pairs"], 1)
        with output.open() as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["image_b"], "train/b.jpg")


if __name__ == "__main__":
    unittest.main()
