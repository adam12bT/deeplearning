import unittest

import torch
import torch.nn.functional as F

from training.reconstruction_losses import signed_gradient_loss


class ReconstructionLossTests(unittest.TestCase):
    def step(self):
        image=torch.zeros(1,3,8,8)
        image[:,:,:,4:]=1
        return image

    def test_sharp_target_beats_blurred_reconstruction(self):
        target=self.step()
        blurred=F.avg_pool2d(target,3,stride=1,padding=1)
        self.assertEqual(signed_gradient_loss(target,target).item(),0.)
        self.assertGreater(signed_gradient_loss(blurred,target).item(),0.)

    def test_reversed_contrast_is_not_treated_as_the_same_edge(self):
        target=self.step()
        self.assertGreater(signed_gradient_loss(1-target,target).item(),0.)

    def test_finite_backpropagation_for_low_precision_input(self):
        target=self.step()
        prediction=F.avg_pool2d(target,3,stride=1,padding=1).half().requires_grad_()
        loss=signed_gradient_loss(prediction,target.half())
        self.assertEqual(loss.dtype,torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(),0.)

    def test_brightness_offsets_still_require_pixel_loss(self):
        target=self.step()
        self.assertAlmostEqual(signed_gradient_loss(target+.25,target).item(),0.)
        self.assertGreater(F.mse_loss(target+.25,target).item(),0.)


if __name__=='__main__':
    unittest.main()
