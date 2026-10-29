"""Generate with a schema-v2 five-control Transformer and its exact tokenizer."""
import argparse
import hashlib
import json
import io
from pathlib import Path

import torch
from torchvision.utils import save_image,make_grid

from tokenizer.checkpoints import load_vqvae, load_tokenizer
from tokenizer.vqvae import VQVAE
from transformer.compositional import CompositionalConfig,CompositionalTransformer
from tokenizer.compatibility import assert_same_tokenizer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='checkpoints/model_compositional_v2/best.pt')
    for field in ('sky','landform','ground','water','objects'): parser.add_argument('--'+field,required=True)
    parser.add_argument('--num-images',type=int,default=4)
    parser.add_argument('--temperature',type=float,default=1.)
    parser.add_argument('--top-k',type=int,default=50)
    parser.add_argument('--seed',type=int,default=75321)
    parser.add_argument('--out-dir',default='generated_v2')
    parser.add_argument('--decoder-checkpoint',help='Optional decoder-only replacement; encoder/codebook must be identical')
    args=parser.parse_args()
    if args.num_images<1: parser.error('--num-images must be positive')
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    checkpoint_bytes=Path(args.checkpoint).read_bytes()
    checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint=torch.load(io.BytesIO(checkpoint_bytes),map_location='cpu',weights_only=True)
    del checkpoint_bytes
    if checkpoint.get('model_type')!='compositional_v2': parser.error('Expected a five-control v2 checkpoint')
    meta=checkpoint['conditions_metadata']; fields=meta['condition_fields']; names=meta['condition_names']
    values=[getattr(args,f) for f in fields]
    for f,value in zip(fields,values):
        if value not in names[f]: parser.error(f'Unknown {f}={value}; choices: {names[f]}')
    identity=checkpoint['tokenizer']; vqvae_path=Path(identity['checkpoint'])
    if hashlib.sha256(vqvae_path.read_bytes()).hexdigest()!=identity['checkpoint_sha256']:
        parser.error('Tokenizer checkpoint changed; refusing to mix incompatible visual tokens')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=CompositionalTransformer(CompositionalConfig(**checkpoint['config'])).to(device).eval()
    model.load_state_dict(checkpoint['model_state'])
    vqvae=load_tokenizer(vqvae_path,device)
    decoder_override=None
    if args.decoder_checkpoint:
        if not isinstance(vqvae, VQVAE):
            parser.error('Decoder overrides apply only to VQ-VAE tokenizers, not pixel-patch codebooks')
        replacement=load_vqvae(Path(args.decoder_checkpoint),device)
        try:
            assert_same_tokenizer(vqvae,replacement)
        except ValueError as error:
            parser.error(str(error))
        vqvae=replacement
        decoder_override={'checkpoint':args.decoder_checkpoint,
                          'sha256':hashlib.sha256(Path(args.decoder_checkpoint).read_bytes()).hexdigest(),
                          'encoder_codebook_match_verified':True}
    conditions=torch.tensor([[names[f].index(v) for f,v in zip(fields,values)]]*args.num_images,device=device)
    # Model construction can consume the CPU RNG; isolate sampling from
    # whether a compatible decoder override was instantiated.
    torch.manual_seed(args.seed)
    with torch.inference_mode(),torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        tokens=model.generate(conditions,temperature=args.temperature,top_k=args.top_k)
    with torch.inference_mode():
        h,w=identity['grid']
        images=((vqvae.decode_from_indices(tokens.reshape(-1,h,w))+1)/2).clamp(0,1)
    destination=Path(args.out_dir)/'_'.join(values); destination.mkdir(parents=True,exist_ok=True)
    for i,image in enumerate(images): save_image(image,destination/f'seed_{args.seed}_{i:03d}.png')
    save_image(make_grid(images,nrow=min(4,len(images))),destination/f'seed_{args.seed}_grid.png')
    torch.save({'tokens':tokens.cpu(),'conditions':conditions.cpu(),'condition_fields':fields},
               destination/f'seed_{args.seed}_tokens.pt')
    (destination/f'seed_{args.seed}_metadata.json').write_text(json.dumps({
        'checkpoint':args.checkpoint,'epoch':checkpoint['epoch'],'features':dict(zip(fields,values)),
        'seed':args.seed,'temperature':args.temperature,'top_k':args.top_k,
        'tokenizer':identity,'decoder_override':decoder_override,'checkpoint_sha256':checkpoint_sha256},indent=2))
    print(f'Saved {len(images)} images in {destination}',flush=True)


if __name__=='__main__': main()
