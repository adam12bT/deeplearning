"""Preview opt-in source changes without rebuilding or replacing the dataset."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datasets.generate_compositional import render_scene
from tokenizer.checkpoints import load_tokenizer


def main():
    torch.set_num_threads(4)
    out=Path('runs/clean_source_preview');out.mkdir(exist_ok=True)
    dataset=Path('dataset/manifest.json')
    before=hashlib.sha256(dataset.read_bytes()).hexdigest()
    manifest=json.loads(dataset.read_text())
    codec=load_tokenizer(Path('checkpoints/pixel_patches_candidate/tokenizer.pt'),torch.device('cpu'))
    scenes=[('night','mountains','rock','lake','settlement'),
            ('night','plains','grass','river','settlement'),
            ('night','volcano','sand','lake','settlement'),
            ('dawn','mountains','grass','lake','settlement'),
            ('day','mountains','grass','river','trees'),
            ('sunset','mountains','grass','ocean','settlement')]
    records=[]
    for case,values in enumerate(scenes):
        features=dict(zip(('sky','landform','ground','water','objects'),values))
        rows=sorted([r for r in manifest if all(r[k]==v for k,v in features.items())],
                    key=lambda r:r['filename'])[:4]
        sheet=Image.new('RGB',(1040,898),'#202028');draw=ImageDraw.Draw(sheet)
        draw.text((5,5),' / '.join(values),fill='white')
        for j,title in enumerate(('Existing source','Revised source (preview only)','Revised source through current tokenizer')):
            draw.text((5,28+j*286),title,fill='white')
        for k,row in enumerate(rows):
            old=render_scene(features,row['seed'])
            # Protect default rendering: existing seed must reproduce dataset bytes.
            assert old.tobytes()==Image.open(Path('dataset')/row['path']).convert('RGB').tobytes()
            new=render_scene(features,row['seed'],clean_edges=True)
            pixels=torch.from_numpy(np.array(new).copy()).permute(2,0,1).float()/255
            with torch.inference_mode():
                decoded=((codec.decode_from_indices(codec.encode_to_indices(pixels[None]*2-1))[0]+1)/2).clamp(0,1)
            reconstructed=Image.fromarray((decoded.permute(1,2,0).numpy()*255).round().astype('uint8'))
            for j,img in enumerate((old,new,reconstructed)):
                sheet.paste(img.resize((256,256),Image.Resampling.NEAREST),(k*260,48+j*286))
                img.save(out/f'case{case+1}_{k}_{j}.png')
            records.append(dict(features=features,seed=row['seed'],original=row['path'],
                                visible_objects=new.info['visible_objects'],water_pixels=new.info['water_pixels'],
                                reconstruction_mse=float((pixels-decoded).square().mean())))
        sheet.save(out/f'preview_{case+1:02d}.png')
    assert before==hashlib.sha256(dataset.read_bytes()).hexdigest()
    (out/'report.json').write_text(json.dumps(dict(preview_only=True,training_started=False,
        dataset_unchanged=True,default_rendering_verified_images=len(records),images=records),indent=2))
    print('Saved 24 source comparisons:',out)


if __name__=='__main__':main()
