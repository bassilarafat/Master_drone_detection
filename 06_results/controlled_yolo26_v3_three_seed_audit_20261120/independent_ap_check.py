from pathlib import Path
import json,hashlib
import numpy as np
root=Path('04_experiments/controlled_yolo26_v3')
out=Path('06_results/controlled_yolo26_v3_three_seed_audit_20261120')
checks=[]
for arm,folder in [('E0','E0_Y26'),('E1','E1_Y26_P2')]:
    for seed in [17,42,123]:
        directory=root/folder/f'seed_{seed}/evaluation/val'
        data=[json.loads(x) for x in (directory/'predictions.jsonl').read_text().splitlines()]
        summary=json.loads((directory/'summary.json').read_text())
        matrices=[]
        for record in data:
            pred=np.array([p['box'] for p in record['predictions']],dtype=float).reshape(-1,4)
            gt=np.array([g['box'] for g in record['targets']],dtype=float).reshape(-1,4)
            assert all(p['class']==0 for p in record['predictions']) and all(g['class']==0 for g in record['targets'])
            lt=np.maximum(pred[:,None,:2],gt[None,:,:2]);rb=np.minimum(pred[:,None,2:],gt[None,:,2:])
            intersection=np.maximum(rb-lt,0).prod(axis=2)
            areaP=(pred[:,2:]-pred[:,:2]).prod(axis=1);areaG=(gt[:,2:]-gt[:,:2]).prod(axis=1)
            union=areaP[:,None]+areaG[None,:]-intersection
            iou=np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
            scores=np.array([p['confidence'] for p in record['predictions']])
            matrices.append((iou,scores))
        apvalues=[]
        for threshold in [.50,.55,.60,.65,.70,.75,.80,.85,.90,.95]:
            allpoints=[]
            for image_index,(matrix,scores) in enumerate(matrices):
                used=np.zeros(matrix.shape[1],dtype=bool)
                hit=np.zeros(len(scores),dtype=int)
                for i in np.argsort(-scores,kind='stable'):
                    candidate=np.where(~used,matrix[i],-1)
                    if len(candidate):
                        j=int(np.argmax(candidate))
                        if candidate[j]>=threshold:used[j]=True;hit[i]=1
                allpoints.extend((float(score),image_index,i,int(hit[i])) for i,score in enumerate(scores))
            allpoints.sort(key=lambda x:(-x[0],x[1],x[2]))
            cumulative=np.cumsum([r[-1] for r in allpoints])
            recall=cumulative/1867
            precision=cumulative/np.arange(1,len(cumulative)+1)
            envelope=np.maximum.accumulate(precision[::-1])[::-1]
            indexes=np.searchsorted(recall,np.arange(101)/100,side='left')
            sampled=np.zeros(101);valid=indexes<len(envelope);sampled[valid]=envelope[indexes[valid]]
            ap=float(np.mean(sampled))
            assert abs(ap-summary['AP_by_iou'][f'{threshold:.2f}'])<1e-12
            apvalues.append(ap)
        assert abs(float(np.mean(apvalues))-summary['mAP50_95'])<1e-12
        checks.append(dict(arm=arm,seed=seed,AP50=apvalues[0],AP50_95=float(np.mean(apvalues)),all_10_thresholds_match=True))
        print(arm,seed,'independent NumPy matching/AP passed',flush=True)
report=dict(status='VERIFIED',method='Independent NumPy IoU, score-ordered greedy matching and reverse precision envelope at 101 recall levels',tolerance=1e-12,runs=checks)
(out/'independent_ap_check.json').write_text(json.dumps(report,indent=2)+'\n')

