"""Matched small/large model samples; visual review, not an automated quality score."""
import hashlib
import argparse
import json
import sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizer.checkpoints import load_tokenizer
from transformer.compositional import CompositionalConfig, CompositionalTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=918273)
    parser.add_argument('--out-dir', default='runs/pixel_model_comparison')
    parser.add_argument('--scene-set', choices=('original','extra'), default='original')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out_dir)
    if (out/'report.json').exists():
        parser.error('Choose a new output directory to preserve previous comparisons')
    out.mkdir(parents=True, exist_ok=True)
    scenes = [('night','mountains','rock','lake','none'),
              ('night','plains','grass','lake','settlement'),
              ('day','mountains','grass','river','trees'),
              ('night','volcano','sand','river','none'),
              ('night','volcano','sand','lake','settlement'),
              ('sunset','mountains','grass','ocean','settlement')]
    if args.scene_set == 'extra':
        scenes = [('night','mountains','grass','river','settlement'),
                  ('night','volcano','rock','ocean','settlement'),
                  ('night','plains','sand','lake','trees'),
                  ('night','volcano','sand','ocean','settlement'),
                  ('dawn','mountains','grass','lake','settlement'),
                  ('sunset','mountains','grass','river','trees')]
    manifest = json.loads(Path('dataset/manifest.json').read_text())
    reference = torch.load('checkpoints/tokens_pixel_patches/train_tokens.pt', weights_only=True)['tokens']
    report = {'seed':args.seed, 'temperature':1., 'top_k':50, 'samples_per_scene':4,
              'method':'Same ordered requests, seed, batch size, tokenizer and sampling settings. '
                       'Rows are independent samples, not pixel-aligned targets. '
                       'Token Hamming neighbors are descriptive, not proof about memorization.',
              'models':{}, 'scenes':[]}
    rendered = {}
    for tag, path in [('small','checkpoints/model_pixel_patches/best.pt'),
                      ('large','checkpoints/model_pixel_patches_larger/best.pt')]:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        meta = checkpoint['conditions_metadata']; fields = meta['condition_fields']; names = meta['condition_names']
        identity = checkpoint['tokenizer']
        assert hashlib.sha256(Path(identity['checkpoint']).read_bytes()).hexdigest() == identity['checkpoint_sha256']
        if tag == 'small':
            original_identity = identity
            for values in scenes:
                features = dict(zip(fields,values))
                counts = {s:sum(r['split']==s and all(r[f]==v for f,v in features.items()) for r in manifest)
                          for s in ('train','val','test')}
                report['scenes'].append(dict(features=features,split_counts=counts))
        else:
            assert identity == original_identity
        labels = torch.tensor([[names[f].index(v) for f,v in zip(fields,scene)]
                               for scene in scenes for _ in range(4)], device=device)
        model = CompositionalTransformer(CompositionalConfig(**checkpoint['config'])).to(device).eval()
        model.load_state_dict(checkpoint['model_state'])
        codec = load_tokenizer(Path(identity['checkpoint']),device)
        torch.manual_seed(report['seed'])
        print(f'Generating {tag}: epoch {checkpoint["epoch"]}, 24 samples',flush=True)
        with torch.inference_mode(), torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            tokens = model.generate(labels,temperature=1.,top_k=50)
        with torch.inference_mode():
            images = ((codec.decode_from_indices(tokens.reshape(-1,*identity['grid']))+1)/2).clamp(0,1)
        pixels = (images.permute(0,2,3,1).cpu()*255).round().byte().numpy()
        rendered[tag] = [Image.fromarray(x) for x in pixels]
        tokens = tokens.cpu()
        nearest = []
        for row in tokens:
            distance = (reference != row).sum(1)
            index = int(distance.argmin())
            nearest.append(dict(train_index=index,token_mismatch_fraction=float(distance[index])/len(row)))
        report['models'][tag] = dict(checkpoint=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                epoch=checkpoint['epoch'],best_val_loss=checkpoint['best_val_loss'],
                exact_training_token_matches=sum(x['token_mismatch_fraction']==0 for x in nearest),nearest_tokens=nearest)
        torch.save(tokens,out/f'{tag}_tokens.pt')
        for i,img in enumerate(rendered[tag]): img.save(out/f'{tag}_{i:02d}.png')
        del model,codec,images
        torch.cuda.empty_cache()
    for i,scene in enumerate(scenes):
        sheet = Image.new('RGB',(4*260,2*286+32),'#202028'); draw = ImageDraw.Draw(sheet)
        draw.text((8,4),' / '.join(scene),fill='white')
        for j,tag in enumerate(('small','large')):
            draw.text((8,32+j*286),tag+' model',fill='white')
            for k in range(4):
                sheet.paste(rendered[tag][i*4+k].resize((256,256),Image.Resampling.NEAREST),(k*260,54+j*286))
        sheet.save(out/f'comparison_{i+1:02d}.png')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print('Comparison saved:',out,flush=True)


if __name__=='__main__': main()
