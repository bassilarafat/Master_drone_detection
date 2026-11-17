import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from drone_detection.yolo26_laptop import atomic_copy,choose_checkpoint,load

class LaptopRecoveryTests(unittest.TestCase):
    def test_same_shared_settings_for_both_arms(self):
        _,a,_=load('E0',17);_,b,_=load('E1',17)
        self.assertEqual(a['train'],b['train'])
        self.assertEqual(a['train']['batch'],4)
        self.assertTrue(a['train']['amp'])
        self.assertEqual(a['train']['nbs'],16)
        self.assertEqual(a['train']['workers'],2)

    def test_atomic_copy_preserves_previous_destination_if_replace_fails(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b';a.write_bytes(b'new checkpoint');b.write_bytes(b'previous checkpoint')
            with patch('drone_detection.yolo26_laptop.os.replace',side_effect=OSError('interruption')):
                with self.assertRaises(OSError):atomic_copy(a,b)
            self.assertEqual(b.read_bytes(),b'previous checkpoint')

    def test_corrupt_latest_falls_back_to_previous_training_state(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);weights=base/'run/weights';weights.mkdir(parents=True)
            (weights/'last_verified.pt').write_bytes(b'incomplete checkpoint')
            torch.save(dict(epoch=2,optimizer={'state':{}},model=None),weights/'previous_verified.pt')
            selected,state=choose_checkpoint(base)
            self.assertEqual(selected.name,'previous_verified.pt');self.assertEqual(state['epoch'],2)

    def test_inference_only_weights_cannot_resume(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);weights=base/'run/weights';weights.mkdir(parents=True)
            torch.save(dict(epoch=-1,optimizer=None),weights/'last_verified.pt')
            with self.assertRaisesRegex(ValueError,'No intact'):choose_checkpoint(base)

if __name__=='__main__':unittest.main()
