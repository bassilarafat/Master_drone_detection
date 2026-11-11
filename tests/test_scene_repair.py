import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from drone_detection.data_audit import digest,dump_csv
from drone_detection.scene_repair import repair_allocation,reassign_staging,cached_boundary


class StagingRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.stage=self.root/'.DUT_Anti_UAV_Clean_v2.staging-test'
        for kind in ('images','labels'):
            for split in ('train','val','test'):(self.stage/kind/split).mkdir(parents=True)
        self.members=[];self.old=[]
        for i in range(9):
            split=('train','val','test')[i%3];name=f'{i:05d}.png';newname=split+'_'+name
            ip=self.stage/'images'/split/newname;lp=self.stage/'labels'/split/(Path(newname).stem+'.txt')
            Image.new('RGB',(30,30),(i*20,10,30)).save(ip);lp.write_text('0 .5 .5 .2 .2\n')
            m=dict(image=split+'/'+name,old_split=split,old_filename=name,scene_group_id=f'S{i}',allocation_block_id='MERGED' if i<2 else f'B{i}',
                   source_relative_image='images/'+split+'/'+name,source_relative_label='labels/'+split+'/'+Path(name).stem+'.txt',
                   image_sha256=digest(ip),label_sha256=digest(lp),width=30,height=30,images=1,objects=1,tiny=1,small=0,medium=0,large=0,negative_images=0)
            self.members.append(m)
            prior=dict(m,allocation_block_id=f'B{i}',new_split=split,new_filename=newname,
                       new_relative_image='images/'+split+'/'+newname,new_relative_label='labels/'+split+'/'+Path(newname).stem+'.txt')
            self.old.append(prior)
        self.previous=self.root/'previous.csv';dump_csv(self.previous,self.old,list(self.old[0]))
        dump_csv(self.stage/'dataset_manifest.csv',self.old,list(self.old[0]))
        dump_csv(self.stage/'split_assignment.csv',self.old,list(self.old[0]))
        (self.stage/'BUILD.json').write_text(json.dumps(dict(destination=str(self.root/'DUT_Anti_UAV_Clean_v2'))))

    def test_minimum_change_repair_keeps_the_entire_new_block_together(self):
        rows,report=repair_allocation(self.members,self.previous)
        splits={r['image']:r['new_split'] for r in rows}
        self.assertEqual(splits[self.members[0]['image']],splits[self.members[1]['image']])
        self.assertEqual(len(rows),9)
        self.assertEqual({r['new_split'] for r in rows},{'train','val','test'})
        self.assertLessEqual(report['moved_since_previous'],4)

    def test_file_moves_preserve_bytes_and_refuse_frozen_staging(self):
        rows,report=repair_allocation(self.members,self.previous)
        assign=self.root/'new.csv';dump_csv(assign,rows,list(rows[0]))
        groups=self.root/'groups.csv';dump_csv(groups,[dict(image=m['image'],old_split=m['old_split'],scene_group_id=m['scene_group_id'],allocation_block_id=m['allocation_block_id']) for m in self.members],['image','old_split','scene_group_id','allocation_block_id'])
        stats=self.root/'stats.json';stats.write_text(json.dumps(report))
        result=reassign_staging(self.stage,groups,assign,stats,self.root/'history')
        self.assertGreater(result['moved_files'],0)
        for row in rows:
            self.assertEqual(digest(self.stage/row['new_relative_image']),row['image_sha256'])
            self.assertEqual(digest(self.stage/row['new_relative_label']),row['label_sha256'])
        (self.stage/'FREEZE.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'unpublished'):
            reassign_staging(self.stage,groups,assign,stats,self.root/'another_history')

    def test_cached_metrics_use_new_assignment_and_expose_missing_pairs(self):
        requested=[dict(image_a='train/a.jpg',image_b='test/b.jpg',new_split_a='val',new_split_b='test'),
                   dict(image_a='train/a.jpg',image_b='test/c.jpg',new_split_a='val',new_split_b='train')]
        candidate=self.root/'candidates.csv';dump_csv(candidate,requested,list(requested[0]))
        prior=[dict(image_a='train/a.jpg',image_b='test/b.jpg',new_split_a='train',new_split_b='val',
                    geometry_decision='NO_GEOMETRIC_SUPPORT',direct_geometry_evaluated='')]
        bank=self.root/'bank.csv';dump_csv(bank,prior,list(prior[0]))
        cached=self.root/'cached.csv';missing=self.root/'missing.csv'
        result=cached_boundary(candidate,bank,cached,missing)
        self.assertEqual((result['cached_pairs'],result['new_pairs']),(1,1))
        with cached.open(newline='') as handle:row=next(csv.DictReader(handle))
        self.assertEqual((row['new_split_a'],row['new_split_b']),('val','test'))
        self.assertEqual(row['direct_geometry_evaluated'],'True')
        with missing.open(newline='') as handle:row=next(csv.DictReader(handle))
        self.assertEqual(row['image_b'],'test/c.jpg')


if __name__=='__main__':unittest.main()
