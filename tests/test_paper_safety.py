import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace
from PIL import Image
from drone_detection.data_audit import digest
from drone_detection.paper_audit import inspect_pair
from drone_detection.paper_negatives import verify_row,safe,verify_catalog,verified_rows,propose_hard,FIELDS
from drone_detection.paper_runner import train,evaluate,load_config,training_view
from drone_detection.paper_cli import main
from drone_detection.paths import PROJECT_ROOT

class PaperSafetyTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)
        self.dataset=self.root/'frozen';images=self.dataset/'images/train';labels=self.dataset/'labels/train'
        images.mkdir(parents=True);labels.mkdir(parents=True)
        self.source=images/'positive.png';self.label=labels/'positive.txt'
        Image.new('RGB',(512,512),(80,90,100)).save(self.source);self.label.write_text('0 .25 .25 .1 .1\n')
        self.crop=self.root/'crop.png'
        with Image.open(self.source) as im:im.crop((256,256,512,512)).save(self.crop)
        self.original=dict(new_relative_label='labels/train/positive.txt',image_sha256=digest(self.source),label_sha256=digest(self.label),width='512',height='512')
        self.index={'images/train/positive.png':self.original}
        self.row=dict(sample_id='x',method='hard',source_relative_image='images/train/positive.png',
            source_image_sha256=digest(self.source),source_label_sha256=digest(self.label),crop_path=str(self.crop),crop_sha256=digest(self.crop),
            crop_xyxy='[256,256,512,512]',width='256',height='256',verification='verified_no_uav',reviewer='fixture reviewer',reviewed_at_utc='2026-11-10T12:00:00Z')

    def test_training_requires_explicit_execution_before_loading_config(self):
        with patch('drone_detection.paper_runner.load_config') as load:
            with self.assertRaisesRegex(ValueError,'not started'):train('does-not-exist',execute=False)
            load.assert_not_called()

    def test_Test_rejected_before_model_dataset_or_configuration_reads(self):
        with patch('drone_detection.paper_runner.load_config') as load:
            with self.assertRaisesRegex(ValueError,'Test is locked'):evaluate('none','none','none',split='test')
            load.assert_not_called()

    def test_no_Test_unlock_cli_flag(self):
        with self.assertRaises(SystemExit):main(['evaluate','--config','x','--checkpoint','x','--output','x','--split','test'])

    def test_negative_crop_from_positive_source_is_allowed_only_without_GT(self):
        self.assertEqual(verify_row(self.row,self.dataset,self.index),self.crop)
        with self.assertRaisesRegex(ValueError,'annotated UAV'):
            verify_row(dict(self.row,crop_xyxy='[0,0,256,256]'),self.dataset,self.index)

    def test_negative_sources_cannot_be_val_or_test(self):
        for split in ('val','test'):
            with self.assertRaisesRegex(ValueError,'CURRENT'):
                verify_row(dict(self.row,source_relative_image=f'images/{split}/positive.png'),self.dataset,self.index)

    def test_pending_visual_review_cannot_publish(self):
        with self.assertRaisesRegex(ValueError,'visual verification'):
            verify_row(dict(self.row,verification='pending'),self.dataset,self.index)

    def test_edited_negative_cannot_hide_UAV_even_with_updated_hash(self):
        Image.new('RGB',(256,256),'black').save(self.crop)
        with self.assertRaisesRegex(ValueError,'exact declared source crop'):
            verify_row(dict(self.row,crop_sha256=digest(self.crop)),self.dataset,self.index)

    def test_mismatched_control_size_and_changed_source_label_fail(self):
        with self.assertRaisesRegex(ValueError,'sizes must match'):
            verify_row(dict(self.row,crop_xyxy='[256,256,511,512]'),self.dataset,self.index)
        with self.assertRaisesRegex(ValueError,'Source annotation changed'):
            self.label.write_text('');verify_row(self.row,self.dataset,self.index)

    def test_audit_detects_zero_dimensions_boundaries_and_preserves_bytes(self):
        self.label.write_text('0 .5 .5 .2 0\n0 .95 .5 .2 .2\n')
        before=digest(self.label);r=inspect_pair((self.source,self.label,'train'))
        kinds=[k for k,_ in r['issues']]
        self.assertIn('zero_width_or_height',kinds);self.assertIn('boundary_violation',kinds);self.assertEqual(digest(self.label),before)

    def test_experiment_cannot_override_common_protocol_settings(self):
        import yaml
        c=yaml.safe_load((PROJECT_ROOT/'configs/paper/E0.yaml').read_text())
        c['train']['epochs']=1;path=self.root/'modified.yaml';path.write_text(yaml.safe_dump(c))
        with self.assertRaisesRegex(ValueError,'common controlled parameter'):load_config(path)

    def test_paper_CLI_delegates_without_starting_legacy_training(self):
        from drone_detection.cli import main as entry
        with patch('drone_detection.paper_cli.main') as delegated:
            entry(['paper','dry-run','--config','configs/paper/E0.yaml'])
            delegated.assert_called_once_with(['dry-run','--config','configs/paper/E0.yaml'])

    def test_runtime_training_view_has_no_Test_entry_and_preserves_positive_labels(self):
        import yaml
        from drone_detection.paper_common import write_csv
        rows=[dict(new_split='train',new_relative_image='images/train/positive.png'),
              dict(new_split='val',new_relative_image='images/val/v.png'),
              dict(new_split='test',new_relative_image='images/test/never_open.png')]
        write_csv(self.dataset/'dataset_manifest.csv',rows,list(rows[0]))
        before=digest(self.label)
        cfg=dict(dataset=dict(path=str(self.dataset)),output_root=str(self.root/'runs'),negative_mode='none')
        path,_=training_view(cfg,17);view=yaml.safe_load(path.read_text())
        self.assertNotIn('test',view)
        self.assertNotIn('never_open',Path(view['train']).read_text()+Path(view['val']).read_text())
        self.assertEqual(digest(self.label),before)

    def test_review_cannot_replace_a_proposed_crop_with_another_negative(self):
        from drone_detection.paper_common import write_csv,write_json
        folder=self.root/'proposal';folder.mkdir();catalog=folder/'candidates_manifest.csv';review=folder/'review.csv'
        write_csv(catalog,[dict(self.row,verification='pending')],FIELDS)
        write_json(folder/'proposal.json',dict(candidate_catalog_sha256=digest(catalog)))
        write_csv(review,[self.row],FIELDS);verify_catalog(review,'random')
        write_csv(review,[dict(self.row,crop_xyxy='[0,256,256,512]')],FIELDS)
        with self.assertRaisesRegex(ValueError,'crop identity'):verify_catalog(review,'random')

    def test_insufficient_verified_controls_block_before_any_training(self):
        from drone_detection.paper_common import write_csv
        path=self.root/'review.csv';write_csv(path,[self.row],FIELDS)
        with self.assertRaisesRegex(ValueError,'exactly 2'):verified_rows(path,self.dataset,self.index,'hard',count=2)

    def test_mining_rejects_Val_and_Test_prediction_exports(self):
        folder=self.root/'evaluation';folder.mkdir()
        for split in ('val','test'):
            (folder/'evaluation.json').write_text(json.dumps(dict(split=split,experiment_id='E1',seed=17,status='COMPLETED',dataset_verified_after=True)))
            with patch('drone_detection.paper_negatives.pin_release'):
                with self.assertRaisesRegex(ValueError,'Train-only'):propose_hard(self.dataset,folder,self.root/'never-created')
            self.assertFalse((self.root/'never-created').exists())

    def test_mining_rejects_incomplete_or_failed_integrity_exports(self):
        folder=self.root/'evaluation';folder.mkdir()
        (folder/'evaluation.json').write_text(json.dumps(dict(split='train',experiment_id='E1',seed=17,status='FAILED_DATASET_INTEGRITY',dataset_verified_after=False)))
        with patch('drone_detection.paper_negatives.pin_release'):
            with self.assertRaisesRegex(ValueError,'completed'):propose_hard(self.dataset,folder,self.root/'never-created')

    def _evaluate_fixture(self,fail_after=False):
        from drone_detection.paper_common import bootstrap,write_csv
        bootstrap()
        cfg=self.root/'config.yaml';cfg.write_text('synthetic fixture configuration')
        checkpoint=self.root/'paper_run/run/weights/last.pt';checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'synthetic checkpoint, never loaded')
        c=dict(experiment_id='E1',protocol_sha256='fixture',dataset=dict(path=str(self.dataset)),train=dict(device='cpu'),evaluation=dict(split='val'))
        meta=dict(status='COMPLETED',experiment_id='E1',seed=17,config_sha256=digest(cfg),protocol_sha256='fixture',checkpoint_sha256=digest(checkpoint))
        (self.root/'paper_run/run.json').write_text(json.dumps(meta))
        row=dict(new_split='train',new_relative_image='images/train/positive.png',new_relative_label='labels/train/positive.txt',
            width='512',height='512',image_sha256=digest(self.source),label_sha256=digest(self.label),scene_group_id='fixture')
        write_csv(self.dataset/'dataset_manifest.csv',[row],list(row))
        boxes=SimpleNamespace(xyxy=Mock(),conf=Mock(),cls=Mock())
        boxes.xyxy.cpu.return_value.tolist.return_value=[[102.4,102.4,153.6,153.6]]
        boxes.conf.cpu.return_value.tolist.return_value=[.9];boxes.cls.cpu.return_value.tolist.return_value=[0]
        model=Mock();model.predict.return_value=[SimpleNamespace(boxes=boxes)]
        identity=dict(release_id='synthetic-fixture-only')
        out=self.root/'predictions_out'
        with patch('drone_detection.paper_runner.load_config',return_value=(c,{},cfg)),patch('drone_detection.paper_runner.configure_runtime'),\
             patch('drone_detection.paper_runner.pin_release',side_effect=[identity,ValueError('dataset changed')] if fail_after else [identity,identity]),\
             patch('ultralytics.YOLO',return_value=model):
            if fail_after:
                with self.assertRaisesRegex(ValueError,'dataset changed'):evaluate(cfg,checkpoint,out,split='train')
            else:evaluate(cfg,checkpoint,out,split='train')
        return out

    def test_unified_evaluation_records_actual_Train_scope_without_loading_a_model(self):
        out=self._evaluate_fixture();record=json.loads((out/'evaluation.json').read_text())
        self.assertEqual(record['status'],'COMPLETED');self.assertTrue(record['dataset_verified_after'])
        self.assertEqual(record['settings']['split'],'train');self.assertEqual(json.loads((out/'summary.json').read_text())['split'],'train')

    def test_evaluation_integrity_failure_is_persisted_and_cannot_look_complete(self):
        out=self._evaluate_fixture(fail_after=True);record=json.loads((out/'evaluation.json').read_text())
        self.assertEqual(record['status'],'FAILED_DATASET_INTEGRITY');self.assertFalse(record['dataset_verified_after'])

if __name__=='__main__':unittest.main()
