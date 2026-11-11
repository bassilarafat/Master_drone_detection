"""Batched exact Euclidean RootSIFT matching; the existing RANSAC policy is unchanged."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .data_audit import digest
from .scene_geometry import METRIC_FIELDS, POLICY, _load_feature, verify_features


def batch_mutual_matches(pairs, *, device="cuda:0", ratio=.75):
    import torch

    if not pairs:
        return []
    na, nb = max(max(len(a), 2) for a,b in pairs), max(max(len(b), 2) for a,b in pairs)
    aa = np.zeros((len(pairs), na, 128), np.float32)
    bb = np.zeros((len(pairs), nb, 128), np.float32)
    lengths_a, lengths_b = [], []
    for i,(a,b) in enumerate(pairs):
        aa[i,:len(a)], bb[i,:len(b)] = a,b
        lengths_a.append(len(a));lengths_b.append(len(b))
    with torch.inference_mode():
        a,b = torch.from_numpy(aa).to(device), torch.from_numpy(bb).to(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        norms_a, norms_b = a.square().sum(2), b.square().sum(2)
        distances = norms_a[:,:,None] + norms_b[:,None,:] - 2*torch.bmm(a,b.transpose(1,2))
        la = torch.tensor(lengths_a, device=device)
        lb = torch.tensor(lengths_b, device=device)
        distances.masked_fill_(torch.arange(na,device=device)[None,:,None]>=la[:,None,None], float('inf'))
        distances.masked_fill_(torch.arange(nb,device=device)[None,None,:]>=lb[:,None,None], float('inf'))
        def refine(query, target, approximate, query_lengths, target_lengths):
            # Matrix products shortlist candidates only. Recompute their
            # distances with coordinate differences; ambiguous shortlist
            # margins use full direct cdist instead of trusting cancellation.
            k=min(8,target.shape[1])
            values, indices=torch.topk(approximate,k,dim=2,largest=False,sorted=True)
            chosen=torch.gather(target[:,None,:,:].expand(-1,query.shape[1],-1,-1),2,
                                indices[:,:,:,None].expand(-1,-1,-1,128))
            exact=(query[:,:,None,:]-chosen).square().sum(3)
            exact.masked_fill_(indices>=target_lengths[:,None,None],float('inf'))
            exact.masked_fill_(torch.arange(query.shape[1],device=device)[None,:,None]>=query_lengths[:,None,None],float('inf'))
            best,positions=torch.topk(exact,2,dim=2,largest=False,sorted=True)
            result_indices=torch.gather(indices,2,positions)
            # Conservative float32 dot/accumulation error bound, scaled by
            # descriptor norms; TF32 is disabled. If the approximate kth
            # candidate cannot certify inclusion of the true nearest two,
            # calculate every direct Euclidean distance for those queries.
            # Eight times eps*D*(norm_a+norm_b) exceeds twice the
            # standard dot/norm accumulation bound for float32 FMA.
            tolerance=8*torch.finfo(torch.float32).eps*query.shape[2]*(
                query.square().sum(2).amax(1)+target.square().sum(2).amax(1))
            uncertain=(values[:,:,-1]-values[:,:,1] <= tolerance[:,None])
            uncertain &= torch.arange(query.shape[1],device=device)[None,:] < query_lengths[:,None]
            uncertain &= target_lengths[:,None] > k
            for batch in range(len(pairs)):
                select=torch.where(uncertain[batch])[0]
                if len(select):
                    direct=torch.cdist(query[batch,select][None],target[batch][None],p=2,
                                       compute_mode='donot_use_mm_for_euclid_dist')[0]
                    direct[:,target_lengths[batch]:]=float('inf')
                    d,di=torch.topk(direct,2,dim=1,largest=False,sorted=True)
                    best[batch,select]=d.square();result_indices[batch,select]=di
            return best.sqrt(), result_indices
        forward,j=refine(a,b,distances,la,lb)
        backward,reverse_indices=refine(b,a,distances.transpose(1,2),lb,la)
        best_j = j[:,:,0]
        good_a = forward[:,:,0] < ratio*forward[:,:,1]
        good_b = backward[:,:,0] < ratio*backward[:,:,1]
        reverse = torch.gather(reverse_indices[:,:,0],1,best_j)
        accepted = good_a & torch.gather(good_b,1,best_j) & (reverse == torch.arange(na,device=device)[None,:])
        batch_id, point_a = torch.where(accepted)
        point_b = best_j[batch_id,point_a]
        matches = torch.stack((batch_id,point_a,point_b),1).cpu().numpy()
    result = [[] for _ in pairs]
    for index,a,b in matches:
        result[int(index)].append((int(a),int(b)))
    return result


def verify_gpu_pairs(candidates, feature_index, output, *, batch_size=16, workers=6, device="cuda:0"):
    candidates, feature_index, output = map(Path,(candidates,feature_index,output))
    settings = dict(candidates_sha256=digest(candidates),feature_index_sha256=digest(feature_index),
                    policy=asdict(POLICY),backend="GPU float32 L2; exact candidate refinement with conservative-margin direct fallback; mutual Lowe ratio; OpenCV RANSAC",
                    batch_size=batch_size,device=device,code_sha256=digest(Path(__file__)))
    metadata = output.with_suffix(output.suffix+'.metadata.json')
    seen = set()
    if output.exists():
        prior=json.loads(metadata.read_text())
        if any(prior.get(k)!=v for k,v in settings.items()):raise ValueError('GPU geometry resume provenance differs')
        with output.open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if not row.get('geometry_decision'):raise ValueError('Incomplete GPU geometry row; preserve and repair before resuming')
                seen.add(tuple(sorted((row['image_a'],row['image_b']))))
    else:
        output.parent.mkdir(parents=True,exist_ok=True)
        metadata.write_text(json.dumps(settings,indent=2),encoding='utf-8')
    with feature_index.open(newline='',encoding='utf-8') as f:
        features={r['image_id']:r for r in csv.DictReader(f)}
    def load(row):
        return tuple(_load_feature(str(feature_index.parent/features[row[key]]['feature_file'])) for key in ('image_a','image_b'))
    def confirm(item):
        row,(a,b),matches=item
        return {**row,**verify_features(a,b,matched_indices=matches), 'direct_geometry_evaluated':True}
    cv2.setNumThreads(1)
    counts=Counter();done=0;start=time.monotonic()
    with candidates.open(newline='',encoding='utf-8') as source:
        reader=csv.DictReader(source)
        fields=list(reader.fieldnames)+[k for k in METRIC_FIELDS+['direct_geometry_evaluated'] if k not in reader.fieldnames]
        existed=output.exists()
        with output.open('a' if existed else 'w',newline='',encoding='utf-8') as dest,ThreadPoolExecutor(max_workers=workers) as pool:
            writer=csv.DictWriter(dest,fieldnames=fields)
            if not existed:writer.writeheader()
            batch=[]
            def flush(rows):
                nonlocal done
                loaded=list(pool.map(load,rows))
                matched=batch_mutual_matches([(a['descriptors'],b['descriptors']) for a,b in loaded],device=device,ratio=POLICY.ratio)
                for result in pool.map(confirm,zip(rows,loaded,matched)):
                    writer.writerow(result);counts[result['geometry_decision']]+=1
                done+=len(rows);dest.flush()
                if done%1024<batch_size:print(f'GPU exact geometry {done+len(seen)} pairs; {dict(counts)}; elapsed={time.monotonic()-start:.1f}s',flush=True)
            for row in reader:
                key=tuple(sorted((row['image_a'],row['image_b'])))
                if key in seen:continue
                batch.append(row)
                if len(batch)==batch_size:flush(batch);batch=[]
            if batch:flush(batch)
    report=dict(settings,processed_new=done,previous_pairs=len(seen),complete=True,counts=dict(counts),elapsed_seconds=time.monotonic()-start)
    metadata.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('candidates','feature-index','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args(argv)
    verify_gpu_pairs(args.candidates,args.feature_index,args.output,batch_size=args.batch_size,workers=args.workers)


if __name__=='__main__':main()
