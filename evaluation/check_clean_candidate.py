"""Gate tokenizer fidelity on fixed old/new validation samples before training."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tokenizer.checkpoints import load_tokenizer


def main():
    torch.set_num_threads(4)
    root=Path('dataset_clean_candidate');out=Path('runs/clean_finetune');out.mkdir(exist_ok=True)
    entries=json.loads((root/'manifest.json').read_text())
    codec=load_tokenizer(Path('checkpoints/pixel_patches_candidate/tokenizer.pt'),torch.device('cpu'))
    result={};pictures=[]
    for version in ('original','clean_candidate'):
        rows=sorted([r for r in entries if r['split']=='val' and r['source_version']==version],key=lambda r:r['filename'])
        rows=[rows[i*len(rows)//64] for i in range(64)]
        stats=[]
        for i,row in enumerate(rows):
            image=Image.open(root/row['path']).convert('RGB')
            a=torch.from_numpy(np.array(image).copy()).permute(2,0,1).float()/255
            with torch.inference_mode():b=((codec.decode_from_indices(codec.encode_to_indices(a[None]*2-1))[0]+1)/2).clamp(0,1)
            edges=[]
            for axis in (1,2):
                ga=torch.diff(a,dim=axis);gb=torch.diff(b,dim=axis)
                mask=ga.abs().amax(0)>.08
                edges.extend((ga-gb).abs().mean(0)[mask].tolist())
            stats.append((float((a-b).square().mean()),float(np.mean(edges))))
            if i<8:pictures.append((version,image,Image.fromarray((b.permute(1,2,0).numpy()*255).round().astype('uint8'))))
        mean=np.mean(stats,axis=0)
        result[version]=dict(images=len(rows),pixel_mse=float(mean[0]),edge_gradient_mae=float(mean[1]))
    # Loose fidelity gates catch severe OOD codec failure, not a quality guarantee.
    result['passed']=all(result[v]['pixel_mse']<.0008 and result[v]['edge_gradient_mae']<.04 for v in ('original','clean_candidate'))
    sheet=Image.new('RGB',(8*132,4*156),'#202028');draw=ImageDraw.Draw(sheet)
    for i,(version,a,b) in enumerate(pictures):
        group=i//8;col=i%8
        for j,img in enumerate((a,b)):
            y=(group*2+j)*156;draw.text((col*132,y),('old' if group==0 else 'clean')+(' source' if j==0 else ' codec'),fill='white')
            sheet.paste(img.resize((128,128),Image.Resampling.NEAREST),(col*132,y+20))
    sheet.save(out/'tokenizer_check.png')
    result['selection']='64 evenly spaced filenames per source version, validation only'
    (out/'tokenizer_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    assert result['passed'],'Inspect codec failures before extracting tokens'


if __name__=='__main__':main()
