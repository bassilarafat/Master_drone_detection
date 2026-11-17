import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from drone_detection.yolo26_recovery import latest,atomic_bytes,event

class RecoveryTests(unittest.TestCase):
    def fixture(self,epoch,identity):
        import torch
        return dict(epoch=epoch,optimizer={'state':{}},scaler={'scale':128},controlled_scheduler={'last_epoch':epoch},
                    controlled_live_model_state={'weight':torch.ones(1)},controlled_identity=identity)
    def test_newest_epoch_wins_and_canonical_wins_tie(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            b=Path(d);w=b/'run/weights';w.mkdir(parents=True);ident={'arm':'E0','seed':17}
            torch.save(self.fixture(1,ident),w/'last.pt');torch.save(self.fixture(2,ident),w/'epoch2.pt')
            self.assertEqual(latest(b,ident)[0].name,'epoch2.pt')
            torch.save(self.fixture(2,ident),w/'last.pt');self.assertEqual(latest(b,ident)[0].name,'last.pt')
    def test_corruption_falls_back_but_mismatch_aborts(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            b=Path(d);w=b/'run/weights';w.mkdir(parents=True);ident={'arm':'E0','seed':17}
            torch.save(self.fixture(35,ident),w/'epoch35.pt');(w/'last.pt').write_bytes(b'truncated')
            p,c,errors=latest(b,ident);self.assertEqual(c['epoch'],35);self.assertTrue(errors)
            with self.assertRaisesRegex(ValueError,'mismatch'):latest(b,{'arm':'E0','seed':42})
    def test_test_guard_precedes_configuration_reads(self):
        from drone_detection.yolo26_workflow import check
        with patch('drone_detection.yolo26_workflow.load') as read:
            with self.assertRaisesRegex(ValueError,'Test'):check('E0',17,split='test')
            read.assert_not_called()
    def test_atomic_failure_keeps_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'last.pt';p.write_bytes(b'old')
            with patch('drone_detection.yolo26_recovery.os.replace',side_effect=OSError('power loss')):
                with self.assertRaises(OSError):atomic_bytes(p,b'new')
            self.assertEqual(p.read_bytes(),b'old')
    def test_events_never_overwrite_prior_start(self):
        with tempfile.TemporaryDirectory() as d:
            event(d,'fresh_start');before={p:p.read_bytes() for p in (Path(d)/'events').glob('*')}
            event(d,'resume',previous_last_completed_epoch=36)
            self.assertEqual(len(list((Path(d)/'events').glob('*'))),2)
            for p,b in before.items():self.assertEqual(p.read_bytes(),b)

    def test_finished_run_and_changed_training_settings_refuse_resume(self):
        from drone_detection.yolo26_recovery import validate_resume
        with self.assertRaisesRegex(ValueError,'FINISHED'):validate_resume({'epoch':99},{'batch':4})
        with self.assertRaisesRegex(ValueError,'mismatch'):
            validate_resume({'epoch':35,'train_args':{'batch':1}},{'batch':4})
        self.assertEqual(validate_resume({'epoch':35,'train_args':{'batch':4}},{'batch':4}),36)

if __name__=='__main__':unittest.main()
