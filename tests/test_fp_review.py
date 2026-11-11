import unittest

from drone_detection.fp_review import summarize, suppress_duplicates, object_transitions


class ReviewTests(unittest.TestCase):
    def test_threshold_rematches_and_counts_lost_true_positives(self):
        records=[dict(image='one.jpg',image_path='one.jpg',
            targets=[{'class':0,'box':[0,0,10,10]}],
            predictions=[{'class':0,'box':[0,0,10,10],'confidence':.4,'pred_id':0},
                         {'class':0,'box':[50,50,60,60],'confidence':.3,'pred_id':1}])]
        baseline,errors=summarize(records)
        corrected,_=summarize(records,.5)
        self.assertEqual((baseline['TP'],baseline['FP'],baseline['FN']),(1,1,0))
        self.assertEqual((corrected['TP'],corrected['FP'],corrected['FN']),(0,0,1))
        self.assertEqual(errors[0]['fp_id'],'one_1')

    def test_nms_does_not_merge_different_classes(self):
        predictions=[{'class':0,'box':[0,0,10,10],'confidence':.9},
                     {'class':0,'box':[0,0,10,10],'confidence':.8},
                     {'class':1,'box':[0,0,10,10],'confidence':.7}]
        self.assertEqual(len(suppress_duplicates(predictions,.5)),2)

    def test_zero_area_annotation_is_retained_as_a_miss(self):
        summary,_=summarize([dict(image='one.jpg',image_path='one.jpg',
            targets=[{'class':0,'box':[0,0,10,0]}],predictions=[])])
        self.assertEqual(summary['FN'],1)
        self.assertEqual(summary['recall_by_size']['Tiny']['FN'],1)

    def test_global_shrink_can_recover_one_target_and_lose_another(self):
        records=[dict(image='over.jpg',image_path='over.jpg',
            targets=[{'class':0,'box':[0,0,10,10]}],
            predictions=[{'class':0,'box':[-2.5,-2.5,12.5,12.5],'confidence':.9,'pred_id':0}]),
            dict(image='tight.jpg',image_path='tight.jpg',
            targets=[{'class':0,'box':[0,0,10,10]}],
            predictions=[{'class':0,'box':[1.4,1.4,8.6,8.6],'confidence':.9,'pred_id':0}])]
        self.assertEqual(summarize([records[0]])[0]['TP'],0)
        self.assertEqual(summarize([records[0]],box_scale=.9)[0]['TP'],1)
        self.assertEqual(summarize([records[1]])[0]['TP'],1)
        self.assertEqual(summarize([records[1]],box_scale=.9)[0]['TP'],0)
        self.assertEqual(records[0]['predictions'][0]['box'],[-2.5,-2.5,12.5,12.5])
        changes=object_transitions(records,box_scale=.9)
        self.assertEqual((changes['recovered_gt'],changes['newly_missed_gt'],changes['net_tp']),(1,1,0))
