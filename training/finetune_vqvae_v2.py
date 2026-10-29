"""Short, separate VQ-VAE adaptation using only v2 train/validation images."""
import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datasets.compositional_dataset import CompositionalDataset
from tokenizer.checkpoints import load_vqvae
from training.reconstruction_losses import signed_gradient_loss
from tokenizer.compatibility import frozen_tokenizer_digest


def epoch(model,loader,device,optimizer=None,scaler=None,edge_weight=0.,decoder_only=False):
    training=optimizer is not None
    model.train(training)
    if decoder_only:
        # Freezing parameters alone would still update BatchNorm buffers.
        model.encoder.eval(); model.vq.eval()
    keys=('total_loss','recon_loss','codebook_loss','commitment_loss','edge_loss')
    totals={key:torch.zeros((),device=device) for key in keys}
    counts=torch.zeros(model.cfg.num_embeddings,device=device)
    seen=0
    with torch.set_grad_enabled(training):
        for index,(images,_) in enumerate(loader,1):
            images=images.to(device,non_blocking=True)
            if training: optimizer.zero_grad(set_to_none=True)
            # Full-precision validation gives a stable reconstruction comparison.
            if decoder_only:
                # Keep nearest-code assignments at the same precision as the
                # existing extracted tokens, even while the decoder uses AMP.
                with torch.no_grad(),torch.autocast(device.type,enabled=False):
                    vq=model.vq(model.encoder(images.float()))
                with torch.autocast(device.type,dtype=torch.float16,enabled=training and device.type=='cuda'):
                    reconstruction=model.decoder(vq['quantized'])
                recon_loss=torch.nn.functional.mse_loss(reconstruction.float(),images)
                result={'reconstruction':reconstruction,'recon_loss':recon_loss,
                        'total_loss':recon_loss+vq['vq_loss'],'token_ids':vq['token_ids'],
                        'codebook_loss':vq['codebook_loss'],'commitment_loss':vq['commitment_loss']}
            else:
                with torch.autocast(device.type,dtype=torch.float16,enabled=training and device.type=='cuda'):
                    result=model(images)
            result['edge_loss']=signed_gradient_loss(result['reconstruction'],images)
            objective=result['total_loss']+edge_weight*result['edge_loss']
            if training:
                scaler.scale(objective).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer); scaler.update()
            for key in keys: totals[key]+=result[key].detach()*len(images)
            counts+=torch.bincount(result['token_ids'].flatten(),minlength=len(counts))
            seen+=len(images)
            if index%100==0:
                print(f'  {"train" if training else "val"} {index}/{len(loader)} | '
                      f'recon {(totals["recon_loss"]/seen).item():.6f}',flush=True)
    metrics={key:(value/seen).item() for key,value in totals.items()}
    if not all(math.isfinite(v) for v in metrics.values()): raise RuntimeError('Non-finite VQ-VAE loss')
    probabilities=counts/counts.sum()
    metrics['active_codes']=(counts>0).sum().item()
    metrics['code_perplexity']=torch.exp(-(probabilities*probabilities.clamp_min(1e-12).log()).sum()).item()
    metrics['pixel_mse_0_1']=metrics['recon_loss']/4
    metrics['selection_score']=metrics['recon_loss']+edge_weight*metrics['edge_loss']
    return metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--lr',type=float,default=5e-5)
    parser.add_argument('--init-from',default='checkpoints/vqvae_v2/vqvae_best.pt')
    parser.add_argument('--out-dir',default='checkpoints/vqvae_finetune')
    parser.add_argument('--run-dir',default='runs/vqvae_finetune')
    parser.add_argument('--edge-weight',type=float,default=0.)
    parser.add_argument('--seed',type=int,default=62931)
    parser.add_argument('--decoder-only',action='store_true')
    parser.add_argument('--detail-head',action='store_true')
    args=parser.parse_args()
    if args.edge_weight<0 or args.epochs<1 or args.lr<=0: parser.error('Invalid loss weight, epochs, or learning rate')
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    run=Path(args.run_dir); run.mkdir(parents=True,exist_ok=True)
    if (out/'vqvae_best.pt').exists(): raise SystemExit('Output checkpoint already exists; choose a new --out-dir')
    train=CompositionalDataset(split='train'); val=CompositionalDataset(split='val')
    schema=json.loads(Path('dataset/schema.json').read_text())
    for data in (train,val):
        assert not any(all(e[k]==v for k,v in rule.items()) for e in data.entries for rule in schema['heldout_rules'])
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=device.type=='cuda',
                            generator=torch.Generator().manual_seed(args.seed))
    val_loader=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
    model=load_vqvae(Path(args.init_from),device)
    if args.detail_head and not getattr(model.cfg,'decoder_detail_head',False):
        state=model.state_dict()
        candidate=type(model)(replace(model.cfg,decoder_detail_head=True)).to(device)
        result=candidate.load_state_dict(state,strict=False)
        if set(result.missing_keys)!={'decoder.detail_head.0.weight'} or result.unexpected_keys:
            raise RuntimeError('Unexpected state mismatch while adding detail head')
        model=candidate
        del state
    if args.decoder_only:
        model.encoder.requires_grad_(False); model.vq.requires_grad_(False)
    frozen_digest=frozen_tokenizer_digest(model)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,
                                weight_decay=.01,fused=device.type=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    history={'init_from':args.init_from,'epochs_requested':args.epochs,'lr':args.lr,
             'edge_weight':args.edge_weight,'seed':args.seed,
             'decoder_only':args.decoder_only,'detail_head':getattr(model.cfg,'decoder_detail_head',False),
             'initial_encoder_codebook_sha256':frozen_digest,
             'train_images':len(train),'val_images':len(val),'heldout_used':False,
             'manifest_sha256':hashlib.sha256(Path('dataset/manifest.json').read_bytes()).hexdigest(),
             'train':[],'val':[]}
    def status(stage,**extra):
        (run/'status.json').write_text(json.dumps({'stage':stage,**extra},indent=2))
    status('baseline validation')
    print(f'Device {device}; train={len(train)}, validation={len(val)}, batch={args.batch_size}',flush=True)
    baseline=epoch(model,val_loader,device,edge_weight=args.edge_weight,decoder_only=args.decoder_only)
    history['baseline']=baseline
    best=baseline['selection_score']
    best_recon=baseline['recon_loss']
    print('BASELINE',json.dumps(baseline),flush=True)
    def save(path,number,metrics):
        torch.save({'epoch':number,'model_state_dict':model.state_dict(),'config':model.cfg,
                    'optimizer_state_dict':optimizer.state_dict(),'scaler_state':scaler.state_dict(),
                    'val_recon_loss':metrics['recon_loss'],'best_val_recon_loss':best_recon,
                    'edge_weight':args.edge_weight,'selection_score':metrics['selection_score'],
                    'best_selection_score':best,
                    'decoder_only':args.decoder_only,'encoder_codebook_sha256':frozen_tokenizer_digest(model),
                    'init_from':args.init_from,'manifest_sha256':history['manifest_sha256']},path)
    save(out/'vqvae_best.pt',0,baseline)
    for number in range(1,args.epochs+1):
        start=time.perf_counter()
        status('training',epoch=number,epochs=args.epochs)
        training=epoch(model,train_loader,device,optimizer,scaler,edge_weight=args.edge_weight,decoder_only=args.decoder_only)
        validation=epoch(model,val_loader,device,edge_weight=args.edge_weight,decoder_only=args.decoder_only)
        if args.decoder_only and frozen_tokenizer_digest(model)!=frozen_digest:
            raise RuntimeError('Frozen encoder/codebook state changed')
        history['train'].append(training); history['val'].append(validation)
        if validation['selection_score']<best:
            best=validation['selection_score']; best_recon=validation['recon_loss']
            save(out/'vqvae_best.pt',number,validation)
        save(out/'vqvae_last.pt',number,validation)
        (out/'vqvae_history.json').write_text(json.dumps(history,indent=2))
        print(f'Epoch {number}/{args.epochs} | train recon {training["recon_loss"]:.6f} | '
              f'val recon {validation["recon_loss"]:.6f} | edge {validation["edge_loss"]:.6f} | '
              f'active codes {validation["active_codes"]} | '
              f'{time.perf_counter()-start:.1f}s',flush=True)
    status('completed',epochs=args.epochs,best_val_recon_loss=best_recon,
           best_selection_score=best,edge_weight=args.edge_weight,
           baseline_val_recon_loss=baseline['recon_loss'])
    print('Fine-tuning completed; old checkpoint preserved.',flush=True)


if __name__=='__main__':
    main()
