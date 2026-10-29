"""Train schema-v2 visual tokens with all five controls and checkpoint provenance."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader,TensorDataset

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from transformer.compositional import CompositionalConfig,CompositionalTransformer
from training.schedules import make_lr_lambda
from mlops.tracking import training_tracking


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(directory):
    meta=json.loads((directory/'condition_names.json').read_text())
    fields=meta['condition_fields']; names=meta['condition_names']
    if fields!=['sky','landform','ground','water','objects']:
        raise ValueError('Unexpected feature ordering')
    identity=json.loads((directory/'tokenizer.json').read_text())
    if digest(identity['checkpoint'])!=identity['checkpoint_sha256']:
        raise ValueError('Tokenizer weights do not match these visual tokens')
    dataset_root=Path(identity.get('dataset_root','dataset'))
    if digest(dataset_root/'manifest.json')!=identity['manifest_sha256']:
        raise ValueError('Manifest does not match these visual tokens')
    manifest={e['filename']:e for e in json.loads((dataset_root/'manifest.json').read_text())}
    schema=json.loads((dataset_root/'schema.json').read_text())
    vocab=json.loads((directory/'vocab_size.json').read_text())['vocab_size']
    datasets={}; file_sets=[]; block_size=None
    for split in ('train','val'):
        data=torch.load(directory/f'{split}_tokens.pt',weights_only=True,map_location='cpu')
        tokens,conditions=data['tokens'],data['conditions']
        if data['condition_fields']!=fields or conditions.shape!=(len(tokens),5):
            raise ValueError('Incomplete or reordered condition labels')
        if tokens.dtype!=torch.long or conditions.dtype!=torch.long:
            raise ValueError('Expected int64 visual and condition tokens')
        if len(tokens)==0 or tokens.ndim!=2 or tokens.min()<0 or tokens.max()>=vocab:
            raise ValueError('Invalid visual tokens')
        if block_size is None: block_size=tokens.shape[1]
        if tokens.shape[1]!=block_size: raise ValueError('Mismatched grids')
        expected=torch.tensor([[names[f].index(manifest[name][f]) for f in fields] for name in data['filenames']])
        if not torch.equal(expected,conditions): raise ValueError('Labels do not match manifest')
        entries=[manifest[name] for name in data['filenames']]
        if any(e['split']!=split for e in entries): raise ValueError('Image assigned to wrong split')
        if any(all(e[k]==v for k,v in rule.items()) for e in entries for rule in schema['heldout_rules']):
            raise ValueError('Reserved feature combination leaked into train/validation')
        files=set(data['filenames'])
        if len(files)!=len(tokens): raise ValueError('Duplicate filenames')
        file_sets.append(files)
        for i,f in enumerate(fields):
            if set(conditions[:,i].tolist())!=set(range(len(names[f]))):
                raise ValueError(f'Incomplete individual feature coverage: {split}/{f}')
        datasets[split]=TensorDataset(tokens,conditions)
    if not file_sets[0].isdisjoint(file_sets[1]): raise ValueError('Train/validation leakage')
    if math.prod(identity['grid'])!=block_size: raise ValueError('Tokenizer grid mismatch')
    return datasets,meta,identity,vocab,block_size


def run_epoch(model,loader,device,optimizer=None,scaler=None,scheduler=None):
    training=optimizer is not None
    model.train(training)
    total=torch.zeros((),device=device); count=0
    with torch.set_grad_enabled(training):
        for index,(tokens,conditions) in enumerate(loader,1):
            tokens=tokens.to(device,non_blocking=True); conditions=conditions.to(device,non_blocking=True)
            if training: optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                _,loss=model(tokens,conditions)
            if training:
                old_scale=scaler.get_scale()
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale()>=old_scale: scheduler.step()
            total+=loss.detach()*len(tokens); count+=len(tokens)
            if index%100==0:
                print(f'  {"train" if training else "val"} {index}/{len(loader)} | loss {(total/count).item():.4f}',flush=True)
    result=(total/count).item()
    if not math.isfinite(result): raise RuntimeError('Non-finite loss')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tokens-dir',default='checkpoints/tokens_v2')
    parser.add_argument('--out-dir',default='checkpoints/model_compositional_v2')
    parser.add_argument('--run-dir',default='runs/compositional_training')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--warmup-steps',type=int,default=200)
    parser.add_argument('--patience',type=int,default=6)
    parser.add_argument('--n-layer',type=int,default=4)
    parser.add_argument('--n-head',type=int,default=4)
    parser.add_argument('--n-embd',type=int,default=256)
    parser.add_argument('--seed',type=int,default=26431)
    parser.add_argument('--resume')
    parser.add_argument('--init-from',help='Warm-start compatible weights on a new dataset with a fresh optimizer')
    parser.add_argument('--mlflow',action='store_true',help='Track this run in the local MLflow dashboard or MLFLOW_TRACKING_URI')
    args=parser.parse_args()
    if min(args.epochs,args.batch_size)<1: parser.error('Epochs and batch size must be positive')
    if args.resume and args.init_from: parser.error('Choose resume or init-from, not both')
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=Path(args.out_dir); run=Path(args.run_dir)
    out.mkdir(parents=True,exist_ok=True); run.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not args.resume: raise SystemExit('Checkpoint exists; use --resume')
    datasets,meta,identity,vocab,block_size=load_data(Path(args.tokens_dir))
    resume=torch.load(args.resume,map_location='cpu',weights_only=True) if args.resume else None
    initial=torch.load(args.init_from,map_location='cpu',weights_only=True) if args.init_from else None
    cfg=CompositionalConfig(vocab,block_size,tuple(len(meta['condition_names'][f]) for f in meta['condition_fields']),
                            args.n_layer,args.n_head,args.n_embd)
    if resume:
        if resume.get('model_type')!='compositional_v2' or resume['conditions_metadata']!=meta or resume['tokenizer']!=identity:
            raise ValueError('Checkpoint model type, features, or tokenizer mismatch')
        if Path(args.resume).resolve().parent!=out.resolve():
            raise ValueError('Resume in the same checkpoint directory to preserve best-model tracking')
        cfg=CompositionalConfig(**resume['config'])
    if initial:
        if initial.get('model_type')!='compositional_v2' or initial['conditions_metadata']!=meta:
            raise ValueError('Warm-start model/condition mismatch')
        for key in ('checkpoint_sha256','grid','tokenizer_type'):
            if initial['tokenizer'].get(key)!=identity.get(key):
                raise ValueError(f'Warm-start tokenizer mismatch: {key}')
        cfg=CompositionalConfig(**initial['config'])
        if cfg.vocab_size!=vocab or cfg.block_size!=block_size:
            raise ValueError('Warm-start visual vocabulary or grid mismatch')
    generator=torch.Generator().manual_seed(args.seed)
    train_loader=DataLoader(datasets['train'],batch_size=args.batch_size,shuffle=True,num_workers=0,
                            pin_memory=device.type=='cuda',generator=generator)
    val_loader=DataLoader(datasets['val'],batch_size=args.batch_size,num_workers=0,pin_memory=device.type=='cuda')
    model=CompositionalTransformer(cfg).to(device)
    if initial: model.load_state_dict(initial['model_state'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01,betas=(.9,.95),fused=device.type=='cuda')
    schedule=make_lr_lambda(args.warmup_steps,len(train_loader)*args.epochs)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start=1; best=float('inf'); patience=0; history=[]
    if resume:
        model.load_state_dict(resume['model_state'])
        optimizer.load_state_dict(resume['optimizer_state']); scaler.load_state_dict(resume['scaler_state'])
        scheduler.load_state_dict(resume['scheduler_state'])
        start=resume['epoch']+1; best=resume['best_val_loss']; patience=resume['epochs_without_improvement']
        if start>args.epochs: raise ValueError('Increase --epochs beyond the completed checkpoint epoch')
        if resume['batch_size']!=args.batch_size: raise ValueError('Keep the same batch size when resuming')
        factor=schedule(scheduler.last_epoch)
        for group,base in zip(optimizer.param_groups,scheduler.base_lrs): group['lr']=base*factor
        scheduler._last_lr=[g['lr'] for g in optimizer.param_groups]
        torch.set_rng_state(resume['torch_rng_state']); generator.set_state(resume['loader_rng_state'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(resume['cuda_rng_state'])
        if (run/'history.json').exists(): history=json.loads((run/'history.json').read_text())
        print(f'Resumed after epoch {start-1}',flush=True)
    def status(stage,**extra):
        (run/'status.json').write_text(json.dumps({'status':stage,'checkpoint_dir':str(out),
            'max_epochs':args.epochs,'condition_fields':meta['condition_fields'],**extra},indent=2))
    parameters=sum(p.numel() for p in model.parameters())
    print(f'Device {device}; parameters={parameters:,}; train={len(datasets["train"])}, val={len(datasets["val"])}',flush=True)
    print(f'Five separate condition prefix tokens; image tokens={block_size}; batches/epoch={len(train_loader)}',flush=True)
    print(f'Max epochs={args.epochs}; patience={args.patience}; tokenizer={identity["checkpoint"]}',flush=True)
    def make_checkpoint(number,training,validation):
        return {'model_type':'compositional_v2','epoch':number,'config':asdict(cfg),
                    'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                    'scheduler_state':scheduler.state_dict(),'scaler_state':scaler.state_dict(),
                    'conditions_metadata':meta,'tokenizer':identity,'tokens_dir':args.tokens_dir,
                    'train_loss':training,'val_loss':validation,'best_val_loss':best,
                    'epochs_without_improvement':patience,'batch_size':args.batch_size,
                    'torch_rng_state':torch.get_rng_state(),'loader_rng_state':generator.get_state(),
                    'cuda_rng_state':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                    'fine_tuned_from':args.init_from or (resume or {}).get('fine_tuned_from')}
    tracking_params={'parameters':parameters,'dataset_manifest_sha256':identity['manifest_sha256'],
                     'tokenizer_sha256':identity['checkpoint_sha256'],
                     **{k:getattr(cfg,k) for k in ('n_layer','n_head','n_embd','block_size','vocab_size')}}
    with training_tracking(args.mlflow,run,tracking_params,{'recording_mode':'live_training'}) as tracking:
        for previous in history:tracking.epoch(previous)
        if initial:
            status('evaluating_baseline',parameters=parameters)
            best=run_epoch(model,val_loader,device)
            tracking.metrics({'baseline_val_loss':best},step=0)
            torch.save(make_checkpoint(0,None,best),out/'best.tmp'); (out/'best.tmp').replace(out/'best.pt')
            (run/'baseline.json').write_text(json.dumps({'checkpoint':args.init_from,
                'checkpoint_sha256':digest(args.init_from),'val_loss_on_candidate':best,
                'tokenizer':identity},indent=2))
            print(f'Warm-start baseline on candidate validation: {best:.6f}',flush=True)
        status('running',epoch=start,parameters=parameters)
        stopped=False
        for number in range(start,args.epochs+1):
            began=time.perf_counter(); status('running',epoch=number,parameters=parameters)
            training=run_epoch(model,train_loader,device,optimizer,scaler,scheduler)
            validation=run_epoch(model,val_loader,device)
            improved=validation<best
            if improved: best=validation; patience=0
            else: patience+=1
            duration=time.perf_counter()-began
            record={'epoch':number,'train_loss':training,'val_loss':validation,'seconds':duration,'lr':scheduler.get_last_lr()[0]}
            history.append(record)
            checkpoint=make_checkpoint(number,training,validation)
            # Save weights before tracking: a tracking failure cannot corrupt checkpoints.
            torch.save(checkpoint,out/'last.tmp'); (out/'last.tmp').replace(out/'last.pt')
            if improved:
                torch.save(checkpoint,out/'best.tmp'); (out/'best.tmp').replace(out/'best.pt')
            (run/'history.json').write_text(json.dumps(history,indent=2))
            tracking.epoch(record)
            tracking.metrics({'best_val_loss':best},step=number)
            print(f'Epoch {number}/{args.epochs} | train {training:.4f} | val {validation:.4f} | '
                  f'{duration:.1f}s | {"new best" if improved else "no improvement"}',flush=True)
            status('running',epoch=number,completed_epoch=number,best_val_loss=best,parameters=parameters)
            if (run/'STOP_AFTER_EPOCH').exists():
                (run/'STOP_AFTER_EPOCH').unlink(); stopped=True
                print('Stopped after saving the epoch, as requested.',flush=True); break
            if args.patience and patience>=args.patience:
                print('Early stopping: validation loss has plateaued.',flush=True); break
        if digest(identity['checkpoint'])!=identity['checkpoint_sha256']:
            raise RuntimeError('Tokenizer changed during training')
        status('stopped' if stopped else 'completed',completed_epoch=number,best_val_loss=best,parameters=parameters)
        for name in ('history.json','status.json','baseline.json'):tracking.artifact(run/name)
        print('Training stopped.' if stopped else 'Training completed.',flush=True)


if __name__=='__main__': main()
