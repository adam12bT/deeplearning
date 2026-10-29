from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader,TensorDataset

from tokenizer.vqvae import VQVAE,VQVAEConfig
from tokenizer.checkpoints import load_vqvae
from training.finetune_vqvae_v2 import epoch,frozen_tokenizer_digest
from tokenizer.compatibility import assert_same_tokenizer


class DecoderDetailTests(unittest.TestCase):
    def models(self):
        torch.set_num_threads(2); torch.manual_seed(8)
        cfg=VQVAEConfig(base_channels=8,mid_channels=8,hidden_channels=8,embedding_dim=8,num_embeddings=16)
        old=VQVAE(cfg).eval(); new=VQVAE(replace(cfg,decoder_detail_head=True)).eval()
        result=new.load_state_dict(old.state_dict(),strict=False)
        self.assertEqual(result.missing_keys,['decoder.detail_head.0.weight'])
        self.assertFalse(result.unexpected_keys)
        return old,new

    def test_zero_head_preserves_outputs_and_32_grid(self):
        old,new=self.models(); images=torch.rand(2,3,64,64)*2-1
        with torch.inference_mode():
            old_codes=old.encode_to_indices(images); new_codes=new.encode_to_indices(images)
            self.assertEqual(tuple(new_codes.shape),(2,32,32))
            self.assertTrue(torch.equal(old_codes,new_codes))
            torch.testing.assert_close(old.decode_from_indices(old_codes),new.decode_from_indices(new_codes),rtol=0,atol=0)

    def test_head_has_independent_subpixel_phases(self):
        _,model=self.models()
        with torch.no_grad():
            model.decoder.detail_head[0].weight[:4,0,0,0]=torch.tensor([.1,.2,.3,.4])
            output=model.decoder.detail_head(torch.ones(1,8,2,2))
        torch.testing.assert_close(output[0,0,:2,:2],torch.tensor([[.1,.2],[.3,.4]]))

    def test_decoder_step_does_not_change_encoder_or_codebook(self):
        _,model=self.models()
        model.encoder.requires_grad_(False); model.vq.requires_grad_(False)
        before=frozen_tokenizer_digest(model)
        images=torch.rand(2,3,8,8)*2-1
        loader=DataLoader(TensorDataset(images,torch.zeros(2,5,dtype=torch.long)),batch_size=2)
        optimizer=torch.optim.Adam(model.decoder.parameters(),lr=1e-3)
        epoch(model,loader,torch.device('cpu'),optimizer,torch.amp.GradScaler('cuda',enabled=False),
              edge_weight=.1,decoder_only=True)
        self.assertEqual(before,frozen_tokenizer_digest(model))
        gradient=model.decoder.detail_head[0].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(),0.)

    def test_new_checkpoint_loads_with_existing_loader(self):
        _,model=self.models()
        with torch.no_grad(): model.decoder.detail_head[0].weight.fill_(.01)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'candidate.pt'
            torch.save({'config':model.cfg,'model_state_dict':model.state_dict()},path)
            restored=load_vqvae(path,torch.device('cpu'))
        grid=torch.zeros(1,32,32,dtype=torch.long)
        with torch.inference_mode():
            torch.testing.assert_close(model.decode_from_indices(grid),restored.decode_from_indices(grid),rtol=0,atol=0)

    def test_compatible_override_allows_decoder_changes(self):
        old,new=self.models()
        with torch.no_grad(): new.decoder.detail_head[0].weight.fill_(.01)
        assert_same_tokenizer(old,new)

    def test_override_rejects_codebook_or_normalization_changes(self):
        for change in ('codebook','normalization'):
            with self.subTest(change=change):
                old,new=self.models()
                with torch.no_grad():
                    if change=='codebook': new.vq.codebook.weight[0,0]+=.01
                    else: new.encoder.net[1].running_mean[0]+=.01
                with self.assertRaises(ValueError): assert_same_tokenizer(old,new)


if __name__=='__main__': unittest.main()
