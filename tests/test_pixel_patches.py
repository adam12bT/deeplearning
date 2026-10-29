import unittest
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from tokenizer.pixel_patches import PixelPatchTokenizer
from tokenizer.checkpoints import load_tokenizer


class PixelPatchTests(unittest.TestCase):
    def test_exact_patch_codebook_preserves_pixels_and_edge_phases(self):
        rgb = torch.zeros(1, 3, 8, 8, dtype=torch.uint8)
        rgb[:, 0, :, 3:] = 255
        rgb[:, 1, 5:, :] = 127
        images = rgb.float() / 127.5 - 1
        patches = PixelPatchTokenizer.patches(images).reshape(-1, 12)
        model = PixelPatchTokenizer(torch.unique(patches, dim=0))
        indices = model.encode_to_indices(images)
        restored = ((model.decode_from_indices(indices) + 1) * 127.5).round().byte()
        self.assertTrue(torch.equal(rgb, restored))
        self.assertEqual(tuple(indices.shape), (1, 4, 4))

    def test_independent_subpixels_do_not_get_smoothed(self):
        patch = torch.arange(12).float().reshape(1, 12) / 12
        model = PixelPatchTokenizer(patch)
        output = (model.decode_from_indices(torch.zeros(1, 1, 1, dtype=torch.long)) + 1) / 2
        expected = F.pixel_shuffle(patch.reshape(1, 12, 1, 1), 2)
        torch.testing.assert_close(output, expected)

    def test_grid_size_stays_32_by_32(self):
        model = PixelPatchTokenizer(torch.tensor([[0.] * 12, [1.] * 12]))
        codes = model.encode_to_indices(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(codes.shape), (2, 32, 32))
        self.assertEqual(tuple(model.decode_from_indices(codes).shape), (2, 3, 64, 64))

    def test_reject_invalid_codebooks(self):
        with self.assertRaises(ValueError):
            PixelPatchTokenizer(torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            PixelPatchTokenizer(torch.full((2, 12), float('nan')))

    def test_checkpoint_loader_preserves_codebook(self):
        codebook = torch.arange(24).reshape(2, 12).float() / 24
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'patches.pt'
            torch.save({'model_type': 'pixel_patch_v1', 'codebook': codebook,
                        'config': {'num_embeddings': 2, 'patch_size': 2, 'in_channels': 3}}, path)
            restored = load_tokenizer(path, torch.device('cpu'))
        torch.testing.assert_close(restored.codebook, codebook, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
