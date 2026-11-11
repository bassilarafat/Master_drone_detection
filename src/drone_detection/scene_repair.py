"""Repair an unpublished scene split without changing any source image/label."""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data_audit import digest, dump_csv
from .scene_graph import read_csv
from .scene_release import FEATURES, SPLITS, _validate_assignment_rows


def merge_geometry(paths, output):
    """Union content-bound evidence; earlier paths take precedence on duplicates."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    fields=[]
    for path in paths:
        with Path(path).open(newline='',encoding='utf-8') as handle:
            for key in next(csv.reader(handle)):
                if key not in fields:fields.append(key)
    seen=set();counts=Counter();output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8') as dest:
        writer=csv.DictWriter(dest,fieldnames=fields);writer.writeheader()
        for path in paths:
            with Path(path).open(newline='',encoding='utf-8') as handle:
                for row in csv.DictReader(handle):
                    key=tuple(sorted((row['image_a'],row['image_b'])))
                    if key in seen:continue
                    seen.add(key);writer.writerow(row);counts[row['geometry_decision']]+=1
    report=dict(pairs=len(seen),decisions=dict(counts),sources={str(p):digest(p) for p in paths},sha256=digest(output))
    output.with_suffix(output.suffix+'.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def repair_allocation(members, previous_csv, *, seed=17, move_weight=.2):
    """Balance whole new blocks while penalizing unnecessary staging moves."""
    previous={r['image']:r['new_split'] for r in read_csv(previous_csv)}
    if set(previous)!={m['image'] for m in members}:raise ValueError('Previous allocation coverage differs')
    groups=defaultdict(list)
    for m in members:groups[m.get('allocation_block_id') or m['scene_group_id']].append(m)
    units=sorted(groups)
    if len(units)<3:raise ValueError('Three nonempty disjoint splits need at least three allocation blocks')
    vectors=np.array([[sum(int(r.get(f,0)) for r in groups[u]) for f in FEATURES] for u in units],dtype=float)
    totals=vectors.sum(0);old_counts=Counter(m['old_split'] for m in members)
    ratios=np.array([old_counts[s]/len(members) for s in SPLITS]);targets=ratios[:,None]*totals[None,:]
    denominator=np.maximum(targets,1);weights=np.array([3,1,.65,.65,.65,.65,.02]);weights[totals==0]=0
    movement=np.array([[sum(previous[r['image']]!=s for r in groups[u]) for s in SPLITS] for u in units],dtype=float)
    assignments=movement.argmin(1);occupancy=np.bincount(assignments,minlength=3)
    loads=np.array([vectors[assignments==s].sum(0) for s in range(3)])
    def cost(load,s):return float(np.sum(weights*((load-targets[s])/denominator[s])**2))
    rng=np.random.default_rng(seed)
    for _ in range(30):
        changed=False
        for i in rng.permutation(len(units)):
            old=int(assignments[i])
            if occupancy[old]<=1:continue
            new_best=old;delta_best=0.
            for new in range(3):
                if old==new:continue
                delta=cost(loads[old]-vectors[i],old)+cost(loads[new]+vectors[i],new)-cost(loads[old],old)-cost(loads[new],new)
                delta+=move_weight*(movement[i,new]-movement[i,old])/len(members)
                if occupancy[new]==0:delta-=10
                if delta<delta_best-1e-12:new_best,delta_best=new,delta
            if new_best!=old:
                assignments[i]=new_best;loads[old]-=vectors[i];loads[new_best]+=vectors[i]
                occupancy[old]-=1;occupancy[new_best]+=1;changed=True
        if not changed:break
    if np.any(occupancy==0):raise ValueError('Repair did not yield three nonempty splits')
    selected={u:SPLITS[int(assignments[i])] for i,u in enumerate(units)}
    rows=[]
    for m in sorted(members,key=lambda r:r['image']):
        split=selected[m.get('allocation_block_id') or m['scene_group_id']]
        name=f"{m['old_split']}_{m['old_filename']}"
        rows.append(dict(**m,new_split=split,new_filename=name,new_relative_image=f'images/{split}/{name}',new_relative_label=f'labels/{split}/{Path(name).stem}.txt'))
    _validate_assignment_rows(rows)
    summary={s:dict(**{f:sum(int(r.get(f,0)) for r in rows if r['new_split']==s) for f in FEATURES},
                    scene_groups=len({r['scene_group_id'] for r in rows if r['new_split']==s}),
                    allocation_blocks=len({r['allocation_block_id'] for r in rows if r['new_split']==s})) for s in SPLITS}
    report=dict(seed=seed,strategy='whole-block minimum-change repair',movement_penalty=move_weight,
                previous_assignment_sha256=digest(previous_csv),moved_since_previous=sum(previous[r['image']]!=r['new_split'] for r in rows),
                scene_groups=len({r['scene_group_id'] for r in rows}),allocation_blocks=len(units),
                totals=dict(zip(FEATURES,map(int,totals))),splits=summary,target_ratios=dict(zip(SPLITS,map(float,ratios))),
                balance_objective=sum(cost(loads[s],s) for s in range(3)),
                target_relative_deviations={s:dict(zip(FEATURES,map(float,(loads[i]-targets[i])/denominator[i]))) for i,s in enumerate(SPLITS)})
    return rows,report


def reassign_staging(staging, groups_csv, assignment_csv, allocation_report, history):
    """Move files only inside an unpublished staging directory; never overwrite."""
    staging=Path(staging).resolve();history=Path(history).resolve()
    if (staging/'FREEZE.json').exists() or not staging.name.startswith('.DUT_Anti_UAV_Clean_v2.staging-'):
        raise ValueError('Only an unpublished v2 staging directory may be repaired')
    rows=read_csv(assignment_csv);old={r['image']:r for r in read_csv(staging/'dataset_manifest.csv')}
    _validate_assignment_rows(rows)
    if set(old)!={r['image'] for r in rows}:raise ValueError('Repair must preserve every staged image')
    history.mkdir(parents=True,exist_ok=False)
    for name in ('scene_groups.csv','split_assignment.csv','dataset_manifest.csv','dataset_changes.csv','allocation_report.json','BUILD.json'):
        if (staging/name).is_file():shutil.copyfile(staging/name,history/name)
    moved=0
    for row in rows:
        prior=old[row['image']]
        for path_key,hash_key,prefix in [('new_relative_image','image_sha256','images'),('new_relative_label','label_sha256','labels')]:
            if row[hash_key]!=prior[hash_key]:raise ValueError('Repair must retain image/label bytes')
            first=(staging/prior[path_key]).resolve();target=(staging/row[path_key]).resolve()
            allowed=(staging/prefix).resolve()
            if not first.is_relative_to(allowed) or not target.is_relative_to(allowed):raise ValueError('Repair path escaped staging')
            if first==target:continue
            if target.exists() or not first.is_file() or digest(first)!=row[hash_key]:raise ValueError('Unsafe or changed staged move')
            first.rename(target);moved+=1
    shutil.copyfile(groups_csv,staging/'scene_groups.csv');shutil.copyfile(assignment_csv,staging/'split_assignment.csv')
    shutil.copyfile(allocation_report,staging/'allocation_report.json')
    dump_csv(staging/'dataset_manifest.csv',rows,list(rows[0]))
    changes=[dict(image=r['image'],old_split=r['old_split'],new_split=r['new_split'],new_image=r['new_relative_image'],new_label=r['new_relative_label'],
                  image_sha256=r['image_sha256'],label_sha256=r['label_sha256'],change='scene_group_resplit_and_unique_name') for r in rows]
    dump_csv(staging/'dataset_changes.csv',changes,list(changes[0]))
    build=json.loads((staging/'BUILD.json').read_text());build.update(dataset_manifest_sha256=digest(staging/'dataset_manifest.csv'),
        split_assignment_sha256=digest(staging/'split_assignment.csv'),groups_input_sha256=digest(groups_csv),
        last_repair_at_utc=datetime.now(timezone.utc).isoformat(),repair_history=str(history))
    (staging/'BUILD.json').write_text(json.dumps(build,indent=2),encoding='utf-8')
    return dict(moved_files=moved,images=len(rows),staging=str(staging))


def cached_boundary(new_candidates, geometry_bank, cached_output, missing_output):
    """Reuse verified pair metrics by image identity; expose every missing pair."""
    requested={tuple(sorted((r['image_a'],r['image_b']))):r for r in read_csv(new_candidates)}
    with Path(geometry_bank).open(newline='',encoding='utf-8') as handle:
        reader=csv.DictReader(handle);bank_fields=reader.fieldnames
        candidate_fields=list(next(iter(requested.values()))) if requested else []
        fields=list(dict.fromkeys(candidate_fields+bank_fields))
        with Path(cached_output).open('x',newline='',encoding='utf-8') as output:
            writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
            found=set()
            for row in reader:
                key=tuple(sorted((row['image_a'],row['image_b'])))
                if key not in requested:continue
                if row.get('direct_geometry_evaluated') in (None,''):
                    row['direct_geometry_evaluated'] = 'False' if row['geometry_decision']=='SAME_SCENE_TRANSITIVE' else 'True'
                writer.writerow({**row,**requested[key]});found.add(key)
    missing=[row for key,row in requested.items() if key not in found]
    dump_csv(Path(missing_output),missing,candidate_fields)
    return dict(boundary_pairs=len(requested),cached_pairs=len(found),new_pairs=len(missing),bank_sha256=digest(geometry_bank))
