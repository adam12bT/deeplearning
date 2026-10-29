"""Supervise training and generate familiar-condition previews on completion."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--tokens-dir',default='checkpoints/tokens_v2')
    parser.add_argument('--out-dir',default='checkpoints/model_compositional_v2')
    parser.add_argument('--run-dir',default='runs/compositional_training')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--n-layer',type=int,default=4)
    parser.add_argument('--n-head',type=int,default=4)
    parser.add_argument('--n-embd',type=int,default=256)
    parser.add_argument('--init-from')
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--patience',type=int,default=6)
    parser.add_argument('--warmup-steps',type=int,default=200)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    run=root/args.run_dir; run.mkdir(parents=True,exist_ok=True)
    model_dir=root/args.out_dir
    command=[sys.executable,'-u','training/train_compositional.py','--epochs',str(args.epochs),
             '--tokens-dir',args.tokens_dir,'--out-dir',args.out_dir,'--run-dir',args.run_dir,
             '--batch-size',str(args.batch_size),'--n-layer',str(args.n_layer),
             '--n-head',str(args.n_head),'--n-embd',str(args.n_embd),
             '--lr',str(args.lr),'--patience',str(args.patience),'--warmup-steps',str(args.warmup_steps)]
    if args.init_from: command+=['--init-from',args.init_from]
    if args.resume: command+=['--resume',str(model_dir/'last.pt')]
    def failure(stage,code):
        (run/'status.json').write_text(json.dumps({'status':'failed','stage':stage,'exit_code':code},indent=2))
        raise SystemExit(code)
    with (run/'train.log').open('a' if args.resume else 'w') as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: failure('training',result.returncode)
    state=json.loads((run/'status.json').read_text())
    if state['status']=='stopped': return
    state['sampling_status']='running'
    (run/'status.json').write_text(json.dumps(state,indent=2))
    # Familiar examples only. Reserved-feature evaluation remains separate.
    scenes=[('day','mountains','grass','river','trees'),
            ('sunset','volcano','rock','lake','none'),
            ('night','plains','grass','lake','settlement')]
    with (run/'samples.log').open('w') as log:
        for values in scenes:
            command=[sys.executable,'-u','generate_compositional.py','--checkpoint',str(model_dir/'best.pt'),
                     '--out-dir',str(run/'samples')]
            for field,value in zip(('sky','landform','ground','water','objects'),values): command+=['--'+field,value]
            result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode: failure('sample_generation',result.returncode)
    state['sampling_status']='completed'
    (run/'status.json').write_text(json.dumps(state,indent=2))
    print('Training and familiar-condition previews completed.',flush=True)


if __name__=='__main__': main()
