"""Schema-v2 loader: images and all five independent condition IDs."""
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class CompositionalDataset(Dataset):
    def __init__(self,manifest_path='dataset/manifest.json',search_root='dataset',split='train',return_mood=True):
        path=Path(manifest_path)
        schema=json.loads((path.parent/'schema.json').read_text())
        if schema['schema_version']!=2: raise ValueError('Expected schema-v2 data')
        if split not in ('train','val','test'): raise ValueError(f'Invalid split {split}')
        self.condition_fields=schema['condition_fields']
        self.condition_names=schema['condition_names']
        self.entries=[e for e in json.loads(path.read_text()) if e['split']==split]
        if not self.entries: raise ValueError(f'Empty {split} split')
        self.root=Path(search_root).resolve()
        self.return_mood=return_mood

    def __len__(self): return len(self.entries)

    def __getitem__(self,index):
        entry=self.entries[index]
        path=(self.root/entry['path']).resolve()
        path.relative_to(self.root)
        with Image.open(path) as image:
            tensor=pil_to_tensor(image.convert('RGB')).float()/127.5-1.
        if not self.return_mood: return tensor
        labels=torch.tensor([self.condition_names[field].index(entry[field])
                             for field in self.condition_fields],dtype=torch.long)
        return tensor,labels

    @staticmethod
    def denormalize(tensor): return (tensor*.5+.5).clamp(0,1)
