"""Export only inference weights and required source into an uploadable Space."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import torch

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='checkpoints/model_pixel_patches_larger/best.pt')
args = parser.parse_args()
target = root / 'deployment/hf_space'
checkpoint = torch.load(root / args.checkpoint, map_location='cpu', weights_only=True)
identity = checkpoint['tokenizer']
source = root / identity['checkpoint']
assert hashlib.sha256(source.read_bytes()).hexdigest() == identity['checkpoint_sha256']
export = {key: checkpoint[key] for key in ('config','model_state','conditions_metadata','tokenizer','epoch')}
export['source_checkpoint_sha256'] = hashlib.sha256((root / args.checkpoint).read_bytes()).hexdigest()
torch.save(export, target / 'model.pt')
shutil.copy2(source, target / 'tokenizer.pt')
(target / 'transformer').mkdir(exist_ok=True)
(target / 'transformer/__init__.py').write_text('')
for name in ('compositional.py', 'blocks.py'):
    shutil.copy2(root / 'transformer' / name, target / 'transformer' / name)
shutil.copy2(root / 'tokenizer/pixel_patches.py', target / 'pixel_patches.py')
print(f'Exported epoch {checkpoint["epoch"]} into {target}')
