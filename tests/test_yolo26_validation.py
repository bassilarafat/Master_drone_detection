import json,tempfile,unittest
from pathlib import Path
from drone_detection.dataset_snapshot import _annotations
from drone_detection.yolo26_validation import targets_for,evaluate_validation
from drone_detection.paper_metrics import export_metrics

class ValidationCoordinateTests(unittest.TestCase):
    def test_decimal_parser_targets_match_float_predictions_and_serialize(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);label=root/'a.txt';label.write_text('0 .5 .5 .2 .2\n')
            parsed,issues=_annotations(label);self.assertFalse(issues)
            targets=targets_for(parsed,960,960)
            self.assertEqual(targets[0]['box'],[384.,384.,576.,576.])
            records=[dict(image='a.jpg',targets=targets,predictions=[dict(box=[384.,384.,576.,576.],confidence=.9,**{'class':0})])]
            json.dumps(records)
            summary=export_metrics(root,records)
            self.assertEqual(summary['TP'],1);self.assertEqual(summary['mAP50'],1)
    def test_Test_rejected_before_any_configuration_or_checkpoint_access(self):
        with self.assertRaisesRegex(ValueError,'Test is locked'):
            evaluate_validation(None,None,None,None,split='test')

if __name__=='__main__':unittest.main()
