"""Fixed held-out feature requests and matched familiar-condition controls."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image,ImageDraw
from torchvision.utils import save_image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tokenizer.checkpoints import load_vqvae
from transformer.compositional import CompositionalConfig,CompositionalTransformer


def main():
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root=Path('runs/compositional_test_epoch30'); root.mkdir(parents=True,exist_ok=True)
    path=Path('checkpoints/model_compositional_v2/best.pt')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    meta=checkpoint['conditions_metadata']; fields=meta['condition_fields']; names=meta['condition_names']
    identity=checkpoint['tokenizer']
    assert hashlib.sha256(Path(identity['checkpoint']).read_bytes()).hexdigest()==identity['checkpoint_sha256']
    model=CompositionalTransformer(CompositionalConfig(**checkpoint['config'])).to(device).eval()
    model.load_state_dict(checkpoint['model_state'])
    scenes=[('HELD OUT',('night','volcano','sand','river','none')),
            ('SEEN CONTROL',('night','volcano','rock','river','none')),
            ('HELD OUT',('sunset','mountains','grass','river','trees')),
            ('SEEN CONTROL',('day','mountains','grass','river','trees'))]
    directory=Path('checkpoints/tokens_v2')
    data={s:torch.load(directory/f'{s}_tokens.pt',weights_only=True) for s in ('train','val','test')}
    report={'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'epoch':checkpoint['epoch'],'temperature':1.,'top_k':50,'seed':87431,
            'samples_per_request':8,'tokenizer':identity,'results':{}}
    all_conditions=[]
    for kind,values in scenes:
        labels=torch.tensor([names[f].index(v) for f,v in zip(fields,values)])
        counts={s:int((d['conditions']==labels).all(1).sum()) for s,d in data.items()}
        core=[fields.index(f) for f in ('sky','landform','ground')]
        core_counts={s:int((d['conditions'][:,core]==labels[core]).all(1).sum()) for s,d in data.items()}
        assert counts['test']>0
        if kind=='HELD OUT':
            assert counts['train']==0 and counts['val']==0
            assert core_counts['train']==0 and core_counts['val']==0
        else: assert counts['train']>0 and counts['val']>0
        all_conditions.extend([labels]*8)
        report['results']['_'.join(values)]={'kind':kind,'features':dict(zip(fields,values)),
                                           'exact_request_split_counts':counts,
                                           'sky_landform_ground_split_counts':core_counts}
        print(kind,values,counts,flush=True)
    conditions=torch.stack(all_conditions).to(device)
    torch.manual_seed(report['seed'])
    print('Generating 32 images at fixed settings...',flush=True)
    with torch.inference_mode(),torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        generated=model.generate(conditions,temperature=1.,top_k=50).cpu()
    vqvae=load_vqvae(Path(identity['checkpoint']),device)
    images=[]
    with torch.inference_mode():
        h,w=identity['grid']
        for batch in generated.split(8):
            images.extend(((vqvae.decode_from_indices(batch.to(device).reshape(-1,h,w))+1)/2).clamp(0,1).cpu())
    sheet=Image.new('RGB',(8*132,4*160),(25,25,30)); draw=ImageDraw.Draw(sheet)
    nearest_sheet=Image.new('RGB',(4*264,4*160),(25,25,30)); nd=ImageDraw.Draw(nearest_sheet)
    reference=data['train']['tokens'].to(device)
    exact=0
    for row,(kind,values) in enumerate(scenes):
        key='_'.join(values); labels=conditions[row*8].cpu()
        ids=(data['test']['conditions']==labels).all(1).nonzero().flatten()
        total=0.; count=0
        with torch.inference_mode():
            for batch_ids in ids.split(8):
                x=data['test']['tokens'][batch_ids].to(device)
                c=data['test']['conditions'][batch_ids].to(device)
                _,loss=model(x,c)
                total+=loss.item()*len(x); count+=len(x)
        report['results'][key]['test_token_cross_entropy']=total/count
        neighbors=[]
        for col in range(8):
            i=row*8+col
            path=root/f'{key}_{col:02d}.png'; save_image(images[i],path)
            image=Image.open(path).convert('RGB').resize((128,128),Image.Resampling.NEAREST)
            sheet.paste(image,(col*132,row*160+28))
            value,index=(reference==generated[i].to(device)).float().mean(1).max(0)
            score=value.item(); index=index.item(); exact+=score==1.
            neighbors.append({'sample':col,'nearest_training_similarity':score,
                              'nearest_training_filename':data['train']['filenames'][index]})
            if col<4:
                nearest_sheet.paste(image,(col*264,row*160+28))
                with torch.inference_mode():
                    decoded=((vqvae.decode_from_indices(reference[index:index+1].reshape(-1,h,w))+1)/2).clamp(0,1).cpu()[0]
                pixels=(decoded*255).byte().permute(1,2,0).numpy()
                neighbor=Image.fromarray(pixels).resize((128,128),Image.Resampling.NEAREST)
                nearest_sheet.paste(neighbor,(col*264+132,row*160+28))
        report['results'][key]['nearest_training_examples']=neighbors
        draw.text((3,row*160+4),kind+' | '+' / '.join(values))
        nd.text((3,row*160+4),kind+' | '+' / '.join(values)+' | generated / nearest decoded training')
    sheet.save(root/'samples.png'); nearest_sheet.save(root/'nearest_training.png')
    report['exact_training_token_matches']=exact
    report['unique_generated_sequences']=len({hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in generated})
    torch.save({'tokens':generated,'conditions':conditions.cpu(),'condition_fields':fields},root/'generated_tokens.pt')
    (root/'report.json').write_text(json.dumps(report,indent=2))
    print(f'Complete: {root}/samples.png; exact training token matches={exact}/32',flush=True)


if __name__=='__main__': main()
