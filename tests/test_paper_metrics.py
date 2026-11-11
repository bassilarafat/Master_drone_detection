import csv
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from drone_detection.paper_common import size_bin
from drone_detection.paper_metrics import evaluate_records,export_metrics,match

def target(box=(0,0,20,20),cls=0):return dict(box=list(box),**{'class':cls})
def pred(box=(0,0,20,20),confidence=.9,cls=0):return dict(target(box,cls),confidence=confidence)
def record(name,targets,predictions,verified=False):return dict(image=name,targets=targets,predictions=predictions,negative_verified=verified)

class UnifiedMetricsTests(unittest.TestCase):
    def test_duplicate_predictions_cannot_match_one_gt_twice(self):
        r,*_=evaluate_records([record('a',[target()],[pred(),pred(confidence=.8)])])
        self.assertEqual((r['TP'],r['FP'],r['FN']),(1,1,0));self.assertEqual(r['fp_subtype_counts']['Duplicate'],1)
        self.assertEqual(r['precision'],.5);self.assertEqual(r['mAP50'],1.)

    def test_AP_uses_low_confidence_predictions_below_operating_threshold(self):
        r,*_=evaluate_records([record('a',[target()],[pred(confidence=.1)])])
        self.assertEqual((r['TP'],r['FN']),(0,1));self.assertEqual(r['mAP50_95'],1.)

    def test_high_confidence_background_fp_reduces_AP(self):
        r,*_=evaluate_records([record('a',[target()],[pred((50,50,70,70),.95),pred(confidence=.9)])])
        self.assertAlmostEqual(r['mAP50'],.5);self.assertEqual(r['fp_subtype_counts']['Background'],1)

    def test_missed_gt_limits_recall_and_101_point_AP(self):
        r,*_=evaluate_records([record('a',[target(),target((40,40,60,60))],[pred()])])
        self.assertEqual(r['recall'],.5);self.assertAlmostEqual(r['mAP50'],51/101)

    def test_no_prediction_images_and_verified_negative_denominators(self):
        records=[record('miss',[target()],[]),record('bg',[],[pred()],True),record('clear',[],[],True)]
        r,*_=evaluate_records(records)
        self.assertEqual(r['images'],3);self.assertEqual(r['FN'],1)
        self.assertEqual(r['background_only']['false_positive_rate'],.5);self.assertAlmostEqual(r['FP_per_image'],1/3)

    def test_unverified_or_absent_negative_images_do_not_become_zero_FPR(self):
        r,*_=evaluate_records([record('a',[target()],[pred()]),record('unknown',[],[])])
        self.assertIsNone(r['background_only']['false_positive_rate'])
        r,*_=evaluate_records([record('a',[target()],[pred()])]);self.assertIsNone(r['background_only']['annotation_empty_fp_image_rate'])

    def test_no_gt_AP_and_empty_size_bins_are_not_reported_as_zero(self):
        r,*_=evaluate_records([record('bg',[],[])])
        self.assertIsNone(r['mAP50']);self.assertIsNone(r['recall_by_size']['Large']['recall'])

    def test_fp_subtypes_include_localization_and_class_confusion(self):
        r,*_=evaluate_records([record('a',[target()],[pred((10,0,30,20)),pred(cls=1)])])
        self.assertEqual(r['fp_subtype_counts']['Localization'],1);self.assertEqual(r['fp_subtype_counts']['ClassConfusion'],1)

    def test_original_pixel_area_boundaries(self):
        self.assertEqual([size_bin(Decimal(a)) for a in ['1023.999','1024','9215.999','9216','65535.999','65536']],
            ['Tiny','Small','Small','Medium','Medium','Large'])

    def test_invalid_prediction_and_fake_verified_negative_fail(self):
        with self.assertRaises(ValueError):evaluate_records([record('a',[],[pred(confidence=float('nan'))])])
        with self.assertRaises(ValueError):evaluate_records([record('a',[target()],[],True)])

    def test_clipped_zero_area_predictions_are_counted_not_silently_discarded(self):
        r,*_=evaluate_records([record('a',[target()],[pred((0,0,0,5))])])
        self.assertEqual((r['TP'],r['FP'],r['FN']),(0,1,1));self.assertEqual(r['fp_subtype_counts']['InvalidBox'],1)

    def test_exports_preserve_no_detection_images_and_FN_records(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);export_metrics(out,[record('detected',[target()],[pred()]),record('miss',[target()],[])])
            with (out/'images.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),2)
            with (out/'objects.csv').open() as f:self.assertIn('FN',[r['status'] for r in csv.DictReader(f)])

if __name__=='__main__':unittest.main()
