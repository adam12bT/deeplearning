"""Native 64x64 layered landscapes with five independent condition labels.

Build in a staging directory, inspect previews, then explicitly replace old data.
No AI image service, copied scene crops, or antialiased resizing is involved.
"""
import argparse
import hashlib
import itertools
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

FIELDS = {
    'sky': ['dawn', 'day', 'night', 'sunset'],
    'landform': ['plains', 'mountains', 'volcano'],
    'ground': ['grass', 'sand', 'rock'],
    'water': ['none', 'river', 'lake', 'ocean'],
    'objects': ['none', 'trees', 'settlement'],
}
HOLDOUTS = [dict(sky='night',landform='volcano',ground='sand'),
            dict(sky='sunset',landform='mountains',ground='grass')]
SIZE = 64


def mix(a,b,t):
    return tuple(max(4,min(255,round(x*(1-t)+y*t))) for x,y in zip(a,b))


def light(color,sky):
    if sky=='night': return mix(tuple(round(c*.49) for c in color),(25,32,65),.30)
    if sky=='sunset': return mix(color,(177,101,94),.20)
    if sky=='dawn': return mix(color,(169,148,161),.16)
    return color


def ridge(r,base,amplitude,step=8):
    return [(x,base+r.randint(-amplitude,amplitude)) for x in range(-4,73,step)]


def render_scene(labels,seed,*,clean_edges=False):
    r=random.Random(seed)
    sky,land,ground,water,objects=(labels[k] for k in FIELDS)
    for field,value in labels.items():
        if field not in FIELDS or value not in FIELDS[field]:
            raise ValueError(f'Invalid feature {field}={value}')
    palettes={
        'day':((56,133,190),(191,223,224)),
        'night':((12,18,43),(63,71,101)),
        'dawn':((91,91,139),(243,190,158)),
        'sunset':((100,68,124),(248,164,103)),
    }
    top,bottom=palettes[sky]
    horizon=r.randint(31,38)
    image=Image.new('RGB',(64,64),bottom)
    draw=ImageDraw.Draw(image)
    # Quantized bands, without the former high-contrast checkerboard pattern.
    for y in range(64):
        t=min(1.,y/max(horizon+6,1))
        draw.line((0,y,63,y),fill=mix(top,bottom,round(t*18)/18))
    if sky=='night':
        for _ in range(r.randint(16,30)):
            x,y=r.randrange(64),r.randint(1,27)
            draw.point((x,y),fill=r.choice([(163,181,204),(214,219,215),(125,150,183)]))
    cx,cy=r.randint(8,55),r.randint(6,15)
    radius=r.choice([3,4,5])
    if clean_edges:
        # Keep celestial silhouettes clear of mountains drawn from y=12.
        cy=max(radius+1,min(cy,10-radius))
    if sky=='night':
        moon=Image.new('L',(64,64)); md=ImageDraw.Draw(moon)
        md.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),fill=255)
        if r.random()<.45:
            md.ellipse((cx-radius+3,cy-radius-1,cx+radius+3,cy+radius-1),fill=0)
        image.paste((232,229,195),(0,0),moon)
    else:
        draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),fill=(255,229,155))
        draw.line((cx-radius+1,cy+radius,cx+radius-1,cy+radius),fill=(251,190,124))
    for _ in range(r.randint(1,4)):
        x,y=r.randint(-5,56),r.randint(4,22)
        width=r.randint(6,16)
        cloud=mix(bottom,(237,224,208),.48) if sky!='night' else (52,61,87)
        if clean_edges and (x <= cx+radius+2 and x+width >= cx-radius-2
                            and y-1 <= cy+radius+2 and y+1 >= cy-radius-2):
            continue
        draw.rectangle((x,y,min(63,x+width),y+1),fill=cloud)
        draw.rectangle((x+2,y-1,min(63,x+width-2),y+1),fill=cloud)
    # Every layer draws over a complete backdrop; no untouched black canvas.
    ground_colors={'grass':(91,136,76),'sand':(197,158,96),'rock':(122,127,126)}
    base=light(ground_colors[ground],sky)
    far=mix(light((110,128,129),sky),bottom,.30)
    draw.polygon(ridge(r,horizon,3)+[(68,64),(-4,64)],fill=far)
    draw.polygon(ridge(r,horizon+5,3)+[(68,64),(-4,64)],fill=base)
    draw.polygon(ridge(r,horizon+17,3,12)+[(68,64),(-4,64)],fill=mix(base,(40,55,48),.13))
    if land=='mountains':
        for layer in range(2):
            color=light((87+layer*20,103+layer*13,112+layer*6),sky)
            for x in range(-10,76, r.randint(14,22)):
                peakx=x+r.randint(7,16); peaky=r.randint(12,25)+layer*6
                width=r.randint(16,26); foot=horizon+8+layer*3
                draw.polygon([(peakx-width,foot),(peakx,peaky),(peakx+width,foot)],fill=color)
                draw.polygon([(peakx,peaky),(peakx+width,foot),(peakx+3,foot)],fill=mix(color,(28,39,58),.25))
                if ground=='rock' or r.random()<.45:
                    cap=light((210,216,210),sky)
                    if clean_edges:
                        draw.polygon([(peakx-4,peaky+5),(peakx,peaky),(peakx+4,peaky+5),
                                      (peakx+1,peaky+4),(peakx,peaky+5),(peakx-1,peaky+4)],fill=cap)
                    else:
                        draw.polygon([(peakx-4,peaky+6),(peakx,peaky),(peakx+5,peaky+7),(peakx+1,peaky+5)],fill=cap)
    elif land=='volcano':
        peakx=r.randint(16,48); peaky=r.randint(18,27)
        foot=horizon+r.randint(5,10); half=r.randint(14,23)
        if clean_edges:
            peakx=max(half+1,min(peakx,62-half))
        color=light((81,78,79),sky)
        draw.polygon([(peakx-half,foot),(peakx-4,peaky+2),(peakx+3,peaky),(peakx+half,foot)],fill=color)
        draw.polygon([(peakx,peaky+2),(peakx+half,foot),(peakx+4,foot)],fill=mix(color,(35,34,49),.30))
        draw.line((peakx-3,peaky+2,peakx+3,peaky+1),fill=(255,181,67),width=2)
        points=[(peakx,peaky+2),(peakx+r.randint(-3,3),peaky+7),
                (peakx+r.randint(-6,6),foot-1)]
        draw.line(points,fill=(231,78,44),width=2)
        draw.line(points,fill=(255,182,67),width=1)
        drift=r.choice([-1,1])
        for j in range(r.randint(3,6)):
            x=peakx+drift*j; y=peaky-3-j*2
            shade=mix(light((133,128,129),sky),bottom,min(.7,j*.13))
            draw.rectangle((x-2,y,x+2+j//2,y+2),fill=shade)
    # Sparse, low-contrast ground marks rather than dense noise.
    for _ in range(95):
        x,y=r.randrange(64),r.randint(horizon+9,63)
        shade=mix(base,r.choice([(205,196,158),(48,73,59),(151,139,108)]),.23)
        draw.line((x,y,min(63,x+r.randint(0,2)),y),fill=shade)
    water_mask=Image.new('L',(64,64)); wd=ImageDraw.Draw(water_mask)
    wc=light((51,126,159),sky)
    edge=light((190,179,125) if ground=='sand' else (99,126,95),sky)
    if water=='river':
        start=r.randint(18,46); bend=r.randint(-15,15)
        bend_amplitude=r.choice([3,4]); foreground_width=r.randint(9,13)
        centers=[]; left=[]; right=[]
        begin=horizon+7
        for y in range(begin,64):
            t=(y-begin)/max(1,63-begin)
            x=start+bend*t+math.sin(t*math.pi*1.3)*bend_amplitude
            width=1+t*foreground_width
            centers.append((x,y)); left.append((round(x-width),y)); right.append((round(x+width),y))
        polygon=left+right[::-1]
        draw.polygon(polygon,fill=wc); wd.polygon(polygon,fill=255)
        draw.line(left,fill=edge,width=1); draw.line(right,fill=edge,width=1)
    elif water=='lake':
        x,y=r.randint(17,45),r.randint(horizon+11,54)
        w,h=r.randint(12,22),r.randint(4,7)
        box=(max(0,x-w),y-h,min(63,x+w),min(63,y+h))
        draw.ellipse(box,fill=edge)
        inner=(box[0]+1,box[1]+1,box[2]-1,box[3]-1)
        draw.ellipse(inner,fill=wc); wd.ellipse(inner,fill=255)
    elif water=='ocean':
        coast=ridge(r,max(horizon+15,52) if clean_edges else max(horizon+9,46),2,8)
        polygon=coast+[(68,64),(-4,64)]
        draw.polygon(polygon,fill=wc); wd.polygon(polygon,fill=255)
        draw.line(coast,fill=edge,width=1)
    if water!='none':
        for _ in range(65):
            x,y=r.randrange(64),r.randint(horizon+7,63)
            if water_mask.getpixel((x,y)):
                color=mix(wc,light((192,213,205),sky),r.uniform(.15,.38))
                for xx in range(x,min(64,x+r.randint(2,5))):
                    if water_mask.getpixel((xx,y)): draw.point((xx,y),fill=color)
    # Place props on ground only, with foreground depth controlled by y.
    placed=[]
    if objects!='none':
        for _ in range(100):
            if len(placed)>=r.randint(4,7): break
            x,y=r.randint(5,58),r.randint(max(horizon+4,39) if clean_edges else max(horizon+7,41),61)
            if clean_edges:
                # Reserve the whole prop footprint, including the roof/canopy.
                if any(water_mask.getpixel((xx,yy)) for xx in range(x-5,x+6)
                       for yy in range(y-12,y+1)):
                    continue
                if any(abs(x-px)<11 and abs(y-py)<13 for px,py in placed):
                    continue
            if any(water_mask.getpixel((xx,yy)) for xx,yy in [(x,y),(x-3,y),(x+3,y),(x,y-2)]): continue
            if any(abs(x-px)<7 and abs(y-py)<5 for px,py in placed): continue
            placed.append((x,y))
        if not placed:
            raise ValueError(f'No dry ground for requested objects: {labels}, seed={seed}')
        for x,y in sorted(placed,key=lambda p:p[1]):
            if objects=='trees':
                height=r.randint(7,12)
                trunk=light((95,72,51),sky)
                draw.line((x,y,x,y-height),fill=trunk,width=1)
                leaves=light((45,103,64),sky)
                if ground=='sand':
                    for dx,dy in [(-6,-1),(-4,-3),(5,-2),(6,1)]:
                        draw.line((x,y-height,x+dx,y-height+dy),fill=leaves,width=1)
                else:
                    draw.polygon([(x,y-height-2),(x-4,y-4),(x+4,y-4)],fill=leaves)
                    draw.polygon([(x,y-height+1),(x-5,y-2),(x+4,y-2)],fill=mix(leaves,light((103,153,81),sky),.3))
            else:
                w,h=r.randint(4,6),r.randint(4,7)
                wall=light((195,178,139),sky); roof=light((141,77,61),sky)
                draw.rectangle((x-w//2,y-h,x+w//2,y),fill=wall)
                draw.polygon([(x-w//2-1,y-h),(x,y-h-4),(x+w//2+1,y-h)],fill=roof)
                draw.rectangle((x,y-2,x+1,y),fill=light((79,69,57),sky))
                draw.point((x-1,y-h+2),fill=(246,196,98) if sky=='night' else (77,122,139))
    image.info['visible_objects']=len(placed)
    image.info['water_pixels']=sum(bool(p) for p in water_mask.tobytes())
    return image


def reserved(labels):
    return any(all(labels[key]==value for key,value in rule.items()) for rule in HOLDOUTS)


def make_seed(labels,index,master):
    text=json.dumps([master,list(labels.values()),index],separators=(',',':'))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',default='dataset_v2_build')
    parser.add_argument('--per-combination',type=int,default=32)
    parser.add_argument('--preview-only',action='store_true')
    parser.add_argument('--clean-edges',action='store_true')
    parser.add_argument('--seed',type=int,default=92731)
    args=parser.parse_args()
    root=Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f'Refusing to overwrite nonempty directory: {root}')
    root.mkdir(parents=True,exist_ok=True)
    preview=root/'previews'; preview.mkdir()
    choices=[dict(zip(FIELDS,values)) for values in itertools.product(*FIELDS.values())]
    # Show deliberate scenes, including moon + sand + volcano and river.
    examples=[
        ('day','mountains','grass','river','trees'),
        ('night','volcano','sand','river','none'),
        ('sunset','plains','sand','ocean','trees'),
        ('dawn','mountains','rock','lake','settlement'),
        ('night','plains','grass','lake','settlement'),
        ('day','volcano','rock','none','none'),
        ('sunset','mountains','grass','river','trees'),
        ('dawn','plains','sand','none','settlement'),
    ]
    sheet=Image.new('RGB',(6*132,len(examples)*154),(25,25,30)); draw=ImageDraw.Draw(sheet)
    for row,values in enumerate(examples):
        labels=dict(zip(FIELDS,values))
        draw.text((4,row*154+4),' / '.join(values))
        for col in range(6):
            image=render_scene(labels,make_seed(labels,col,args.seed),clean_edges=args.clean_edges)
            image.save(preview/f'example_{row}_{col}.png')
            sheet.paste(image.resize((128,128),Image.Resampling.NEAREST),(col*132,row*154+22))
    sheet.save(preview/'overview.png')
    schema={'schema_version':2,'size':[64,64],'condition_fields':list(FIELDS),
            'condition_names':FIELDS,'heldout_rules':HOLDOUTS,'seed':args.seed,
            'clean_edges':args.clean_edges,
            'per_combination':args.per_combination,'generator':'datasets/generate_compositional.py'}
    (root/'schema.json').write_text(json.dumps(schema,indent=2))
    if args.preview_only:
        print(f'Preview saved: {preview / "overview.png"}'); return
    if args.per_combination<8 or args.per_combination%8:
        raise SystemExit('--per-combination must be a positive multiple of 8, at least 8')
    manifest=[]
    train_count=args.per_combination*3//4
    val_count=args.per_combination//8
    for labels in choices:
        name='v2_'+'_'.join(labels.values())
        for index in range(args.per_combination):
            split=('test' if reserved(labels) else
                   'train' if index<train_count else
                   'val' if index<train_count+val_count else 'test')
            seed=make_seed(labels,index,args.seed)
            filename=f'{name}_{seed:016x}.png'
            path=Path(split)/name/filename
            (root/path.parent).mkdir(parents=True,exist_ok=True)
            image=render_scene(labels,seed,clean_edges=args.clean_edges)
            image.save(root/path,format='PNG')
            manifest.append({'schema_version':2,'filename':filename,'path':path.as_posix(),
                             'split':split,'mood':name,**labels,'terrain':labels['landform'],
                             'seed':seed,'size':[64,64],'channels':3,
                             'visible_objects':image.info['visible_objects'],
                             'water_pixels':image.info['water_pixels'],
                             'split_reason':'heldout_feature_rule' if reserved(labels) else 'seen_combination'})
        if len(manifest)%1024==0: print(f'Generated {len(manifest)}/{len(choices)*args.per_combination}',flush=True)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Completed {len(manifest)} images in {root}',flush=True)


if __name__=='__main__':
    main()
