"""Paper commands default to plans/tests; final Test has no unlock option."""
from __future__ import annotations
import argparse
import hashlib
import json
from .paper_common import PROJECT_ROOT, bootstrap, project_path, write_json

def model_probe(output):
    from .paper_runner import load_config,shared_backbone_model
    from .paper_common import outside_dataset
    outside_dataset(project_path(output),project_path('02_datasets/DUT_Anti_UAV_Clean_v2'))
    bootstrap()
    import torch
    threads=torch.get_num_threads();torch.set_num_threads(2)
    reports={};hashes=[]
    try:
        for arm in ('E0','E1'):
            c,_,_=load_config(f'configs/paper/{arm}.yaml')
            model,report=shared_backbone_model(project_path(c['model']['path']),project_path(c['initialization']['path']),17)
            model.eval()
            h=hashlib.sha256()
            for key in sorted(report['loaded_keys']):h.update(key.encode());h.update(model.state_dict()[key].detach().cpu().numpy().tobytes())
            hashes.append(h.hexdigest());strides=list(map(int,model.stride.tolist()))
            expected=[8,16,32] if arm=='E0' else [4,8,16,32]
            if strides!=expected:raise ValueError('Wrong detection scales')
            reports[arm]=dict(report,strides=strides,parameters=sum(p.numel() for p in model.parameters()),backbone_tensor_sha256=h.hexdigest())
            del model
        if hashes[0]!=hashes[1]:raise ValueError('E0 and E1 do not start with identical pretrained backbone tensors')
        result=dict(status='PASS',models=reports,identical_pretrained_backbones=True,
            optimizer_created=False,training_called=False,real_dataset_inference=False,test_evaluated=False,
            note='Only architecture construction/stride dummy tensors and local pretrained tensor transfer; no optimization')
        output=project_path(output);output.parent.mkdir(parents=True,exist_ok=True);write_json(output,result)
        return result
    finally:torch.set_num_threads(threads)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('gate3');a.add_argument('--dataset',type=project_path,default=project_path('02_datasets/DUT_Anti_UAV_Clean_v2'));a.add_argument('--output',type=project_path,required=True)
    a=sub.add_parser('dry-run');a.add_argument('--config',required=True);a.add_argument('--seed',type=int);a.add_argument('--output',type=project_path)
    a=sub.add_parser('model-dry-run');a.add_argument('--output',type=project_path,default=project_path('06_results/paper/dry_runs/model_initialization.json'))
    a=sub.add_parser('yolo26-preflight');a.add_argument('--output',type=project_path,default=project_path('06_results/paper_yolo26/preflight.json'))
    sub.add_parser('yolo26-train-e0')
    a=sub.add_parser('yolo26-dry-run');a.add_argument('--output',type=project_path,required=True)
    a=sub.add_parser('train');a.add_argument('--config',required=True);a.add_argument('--seed',type=int);a.add_argument('--execute-training',action='store_true')
    a=sub.add_parser('evaluate');a.add_argument('--config',required=True);a.add_argument('--checkpoint',type=project_path,required=True);a.add_argument('--output',type=project_path,required=True);a.add_argument('--split',choices=('train','val'),default='val');a.add_argument('--dry-run',action='store_true')
    for command in ('mine-negatives','random-negatives','freeze-negatives'):
        a=sub.add_parser(command);a.add_argument('--dataset',type=project_path,default=project_path('02_datasets/DUT_Anti_UAV_Clean_v2'));a.add_argument('--output',type=project_path,required=True)
        if command=='mine-negatives':a.add_argument('--train-evaluation',type=project_path,required=True)
        else:a.add_argument('--hard-review',type=project_path,required=True)
        if command=='random-negatives':a.add_argument('--exclude-review',type=project_path)
        if command=='freeze-negatives':a.add_argument('--random-review',type=project_path,required=True)
    args=parser.parse_args(argv);bootstrap()
    if args.command=='gate3':
        from .paper_audit import audit
        result=audit(args.dataset,args.output)
    elif args.command=='dry-run':
        from .paper_runner import dry_run
        result=dry_run(args.config,args.seed)
        if args.output:
            from .paper_common import outside_dataset
            outside_dataset(args.output,project_path('02_datasets/DUT_Anti_UAV_Clean_v2'))
            args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,result)
    elif args.command=='model-dry-run':result=model_probe(args.output)
    elif args.command=='yolo26-preflight':
        from .yolo26_preflight import run
        result=run(args.output)
    elif args.command=='yolo26-train-e0':
        from .yolo26_runner import train_e0
        result=train_e0(17)
    elif args.command=='yolo26-dry-run':
        from .yolo26_runner import dry_run_e0
        result=dry_run_e0()
        write_json(args.output, result)
    elif args.command=='train':
        from .paper_runner import train
        result=train(args.config,args.seed,execute=args.execute_training)
    elif args.command=='evaluate':
        from .paper_runner import evaluate
        result=evaluate(args.config,args.checkpoint,args.output,split=args.split,dry=args.dry_run)
    else:
        from .paper_negatives import propose_hard,propose_random,publish_bundle
        if args.command=='mine-negatives':result=propose_hard(args.dataset,args.train_evaluation,args.output)
        elif args.command=='random-negatives':result=propose_random(args.dataset,args.hard_review,args.output,exclude_review=args.exclude_review)
        else:result=publish_bundle(args.dataset,args.hard_review,args.random_review,args.output)
    print(json.dumps(result,indent=2,allow_nan=False))
    if result.get('status')=='FAIL':raise SystemExit(1)

if __name__=='__main__':main()
