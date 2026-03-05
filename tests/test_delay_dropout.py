import unittest

import torch
from DCLS.construct.modules import Dcls1d

from delay_dropout import forward_with_delay_dropout


class DelayDropoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.layer = Dcls1d(
            in_channels=4,
            out_channels=6,
            kernel_count=2,
            dilated_kernel_size=9,
            bias=False,
            version="gauss",
        )
        self.x = torch.randn(3, 4, 32)

    def test_sigma_zero_matches_original_forward(self):
        out_base = self.layer(self.x)
        out_sigma_zero = forward_with_delay_dropout(self.layer, self.x, True, 0.0)
        self.assertTrue(torch.allclose(out_base, out_sigma_zero, atol=1e-6, rtol=1e-6))

    def test_inference_path_matches_original_forward(self):
        out_base = self.layer(self.x)
        out_eval = forward_with_delay_dropout(self.layer, self.x, False, 1.0)
        self.assertTrue(torch.allclose(out_base, out_eval, atol=1e-6, rtol=1e-6))

    def test_training_noise_is_transient_and_fresh(self):
        p_before = self.layer.P.detach().clone()

        torch.manual_seed(0)
        out1 = forward_with_delay_dropout(self.layer, self.x, True, 1.0)
        self.assertTrue(torch.allclose(self.layer.P.detach(), p_before, atol=0, rtol=0))

        torch.manual_seed(1)
        out2 = forward_with_delay_dropout(self.layer, self.x, True, 1.0)
        self.assertTrue(torch.allclose(self.layer.P.detach(), p_before, atol=0, rtol=0))

        self.assertFalse(torch.allclose(out1, out2, atol=1e-7, rtol=1e-7))

    def test_delay_parameter_still_receives_gradients(self):
        optimizer = torch.optim.SGD([self.layer.P], lr=1e-2)
        p_before = self.layer.P.detach().clone()

        optimizer.zero_grad()
        out = forward_with_delay_dropout(self.layer, self.x, True, 1.0)
        loss = out.pow(2).mean()
        loss.backward()

        self.assertIsNotNone(self.layer.P.grad)
        self.assertGreater(float(self.layer.P.grad.norm().item()), 0.0)

        optimizer.step()
        self.assertFalse(torch.allclose(self.layer.P.detach(), p_before, atol=1e-9, rtol=1e-9))


if __name__ == "__main__":
    unittest.main()
