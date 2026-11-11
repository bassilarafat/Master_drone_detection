import tempfile
from pathlib import Path
import unittest
from PIL import Image
from drone_detection.data_audit import digest
from drone_detection.paper_common import bootstrap

class RealLoaderCacheGuardTests(unittest.TestCase):
    def test_label_scanning_and_empty_labels_do_not_write_into_frozen_source(self):
        bootstrap()
        from ultralytics.cfg import get_cfg
        from ultralytics.data.dataset import YOLODataset
        from drone_detection.training import _without_snapshot_cache_writes
        with tempfile.TemporaryDirectory() as d:
            frozen=Path(d)/'frozen';images=frozen/'images/train';labels=frozen/'labels/train'
            images.mkdir(parents=True);labels.mkdir(parents=True)
            for i in range(2):
                Image.new('RGB',(128,96),(20+i*50,50,80)).save(images/f'{i}.png')
                (labels/f'{i}.txt').write_text('0 .5 .5 .2 .2\n' if i==0 else '')
            before={p.relative_to(frozen).as_posix():digest(p) for p in frozen.rglob('*') if p.is_file()}
            with _without_snapshot_cache_writes(frozen):
                ds=YOLODataset(img_path=str(images),imgsz=96,batch_size=1,augment=False,hyp=get_cfg(),rect=False,
                    cache=False,data=dict(nc=1,names={0:'UAV'}))
                self.assertEqual(len(ds),2);self.assertEqual(sum(len(r['cls'])==0 for r in ds.labels),1)
            after={p.relative_to(frozen).as_posix():digest(p) for p in frozen.rglob('*') if p.is_file()}
            self.assertEqual(before,after);self.assertFalse(list(frozen.rglob('*.cache')));self.assertFalse(list(frozen.rglob('*.npy')))

if __name__=='__main__':unittest.main()
