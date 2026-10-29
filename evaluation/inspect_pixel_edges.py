"""Inspect source/codec errors separately from unpaired generated shapes."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tokenizer.checkpoints import load_tokenizer


def main():
    torch.set_num_threads(4)
    directory=Path('runs/edge_inspection'); directory.mkdir(exist_ok=True)
    samples=Path('runs/pixel_model_comparison_extra')
    comparison=json.loads((samples/'report.json').read_text())
    manifest=json.loads(Path('dataset/manifest.json').read_text())
    checkpoint=torch.load(comparison['models']['small']['checkpoint'],map_location='cpu',weights_only=True)
    identity=checkpoint['tokenizer']
    assert hashlib.sha256(Path(identity['checkpoint']).read_bytes()).hexdigest()==identity['checkpoint_sha256']
    codec=load_tokenizer(Path(identity['checkpoint']),torch.device('cpu'))
    panels=[]; results=[]
    for case in (0,1,4,5):
        features=comparison['scenes'][case]['features']
        rows=sorted([r for r in manifest if r['split'] in ('val','test') and
                     all(r[k]==v for k,v in features.items())],key=lambda r:(r['split']!='val',r['filename']))[:4]
        for j,row in enumerate(rows):
            source=Image.open(Path('dataset')/row['path']).convert('RGB')
            original=torch.from_numpy(np.array(source).copy()).permute(2,0,1).float()/255
            with torch.inference_mode():
                indices=codec.encode_to_indices(original.unsqueeze(0)*2-1)
                decoded=((codec.decode_from_indices(indices)[0]+1)/2).clamp(0,1)
            reconstructed=Image.fromarray((decoded.permute(1,2,0).numpy()*255).round().astype('uint8'))
            small=Image.open(samples/f'small_{case*4+j:02d}.png').convert('RGB')
            large=Image.open(samples/f'large_{case*4+j:02d}.png').convert('RGB')
            a=np.asarray(source).astype('float32')/255; b=np.asarray(reconstructed).astype('float32')/255
            gradients=[]
            for axis in (0,1):
                ga=np.diff(a,axis=axis);gb=np.diff(b,axis=axis)
                edge=np.max(np.abs(ga),axis=2)>.08
                gradients.extend(np.abs(ga-gb)[edge].mean(axis=1).tolist())
            results.append(dict(features=features,source=row['path'],source_split=row['split'],
                generated_index=case*4+j,pixel_mse=float(np.mean((a-b)**2)),
                exact_rgb_pixel_fraction=float(np.mean(np.all(np.asarray(source)==np.asarray(reconstructed),axis=2))),
                source_edge_gradient_mae=float(np.mean(gradients))))
            panels.append((features,[source,reconstructed,small,large]))
    for group in range(4):
        sheet=Image.new('RGB',(4*260,4*294+28),'#202028');draw=ImageDraw.Draw(sheet)
        for k,label in enumerate(('Dataset source','Patch reconstruction','Small generated (unpaired)','Large generated (unpaired)')):
            draw.text((k*260+4,5),label,fill='white')
        for j,(features,images) in enumerate(panels[group*4:group*4+4]):
            y=28+j*294
            draw.text((4,y),' / '.join(features.values()),fill='white')
            for k,img in enumerate(images):sheet.paste(img.resize((256,256),Image.Resampling.NEAREST),(k*260,y+20))
        sheet.save(directory/f'case_{group+1:02d}.png')
    # First fixed example per scene, same crop coordinates for all four columns.
    # Generated images are not reconstructions of the source, so positions differ.
    for label,box in [('sky',(0,0,64,24)),('terrain',(0,20,64,44)),('foreground',(0,40,64,64))]:
        sheet=Image.new('RGB',(4*388,4*176+28),'#202028');draw=ImageDraw.Draw(sheet)
        for k,title in enumerate(('Source','Reconstruction','Small generated','Large generated')):
            draw.text((k*388+4,5),title,fill='white')
        for j,(features,images) in enumerate(panels[::4]):
            y=28+j*176;draw.text((4,y),' / '.join(features.values()),fill='white')
            for k,img in enumerate(images):sheet.paste(img.crop(box).resize((384,144),Image.Resampling.NEAREST),(k*388,y+20))
        sheet.save(directory/f'crop_{label}.png')
    report=dict(source_images=len(results),tokenizer=identity,metrics=results,
                mean_pixel_mse=float(np.mean([r['pixel_mse'] for r in results])),
                mean_source_edge_gradient_mae=float(np.mean([r['source_edge_gradient_mae'] for r in results])),
                note='Fixed first four validation/test filenames per scene; no cherry-picking. '
                     'Only source and reconstruction are paired. Generated outputs share conditions, not layouts. '
                     'No reconstruction MSE is calculated for generated images.')
    (directory/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('source_images','mean_pixel_mse','mean_source_edge_gradient_mae')}))


if __name__=='__main__':main()
