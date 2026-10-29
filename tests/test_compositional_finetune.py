"""Exercise dataset provenance and retention of a non-improved warm-start model."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from training.train_compositional import load_data, main
from transformer.compositional import CompositionalConfig, CompositionalTransformer


class FineTuneTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.data=self.root/'data';self.data.mkdir()
        self.tokens=self.root/'tokens';self.tokens.mkdir()
        fields=['sky','landform','ground','water','objects']
        self.meta=dict(condition_fields=fields,condition_names={f:['a','b'] for f in fields})
        rows=[]
        for split in ('train','val'):
            filenames=[f'{split}{i}.png' for i in range(2)]
            for i,name in enumerate(filenames):rows.append(dict(filename=name,split=split,**{f:['a','b'][i] for f in fields}))
            torch.save(dict(tokens=torch.arange(16).reshape(2,8),conditions=torch.tensor([[0]*5,[1]*5]),
                            condition_fields=fields,filenames=filenames),self.tokens/f'{split}_tokens.pt')
        (self.data/'manifest.json').write_text(json.dumps(rows))
        (self.data/'schema.json').write_text(json.dumps(dict(heldout_rules=[])))
        codec=self.root/'codec.pt';codec.write_bytes(b'identity fixture, never deserialized')
        self.identity=dict(checkpoint=str(codec),checkpoint_sha256=hashlib.sha256(codec.read_bytes()).hexdigest(),
                           manifest_sha256=hashlib.sha256((self.data/'manifest.json').read_bytes()).hexdigest(),
                           dataset_root=str(self.data),grid=[2,4],tokenizer_type='pixel_patch_v1')
        for name,value in [('condition_names',self.meta),('tokenizer',self.identity),('vocab_size',dict(vocab_size=16))]:
            (self.tokens/f'{name}.json').write_text(json.dumps(value))
        cfg=CompositionalConfig(16,8,(2,2,2,2,2),1,2,16,0.)
        self.parent=self.root/'parent.pt'
        self.weights=CompositionalTransformer(cfg).state_dict()
        torch.save(dict(model_type='compositional_v2',config=asdict(cfg),model_state=self.weights,
                        conditions_metadata=self.meta,tokenizer=dict(self.identity,manifest_sha256='old-manifest')),self.parent)

    def test_candidate_dataset_root_and_manifest_hash(self):
        data,*_=load_data(self.tokens)
        self.assertEqual(len(data['train']),2)
        with (self.data/'manifest.json').open('a') as stream:stream.write(' ')
        with self.assertRaisesRegex(ValueError,'Manifest'):load_data(self.tokens)

    def arguments(self):
        return ['train','--tokens-dir',str(self.tokens),'--out-dir',str(self.root/'out'),
                '--run-dir',str(self.root/'run'),'--init-from',str(self.parent),'--epochs','1',
                '--batch-size','2','--lr','0','--warmup-steps','0']

    def test_no_improvement_keeps_baseline_weights(self):
        digest=hashlib.sha256(self.parent.read_bytes()).hexdigest()
        with patch('sys.argv',self.arguments()),patch('torch.cuda.is_available',return_value=False):main()
        best=torch.load(self.root/'out/best.pt',weights_only=True)
        last=torch.load(self.root/'out/last.pt',weights_only=True)
        self.assertEqual(best['epoch'],0)
        self.assertEqual(last['epoch'],1)
        self.assertEqual(best['tokenizer'],self.identity)
        for key,value in self.weights.items():self.assertTrue(torch.equal(best['model_state'][key],value))
        self.assertEqual(digest,hashlib.sha256(self.parent.read_bytes()).hexdigest())

    def test_incompatible_tokenizer_cannot_warm_start(self):
        parent=torch.load(self.parent,weights_only=True)
        parent['tokenizer']['checkpoint_sha256']='incompatible'
        torch.save(parent,self.parent)
        with patch('sys.argv',self.arguments()),patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(ValueError,'tokenizer mismatch'):main()


if __name__=='__main__':unittest.main()
