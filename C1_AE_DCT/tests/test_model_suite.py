"""Automated Unit Test Suite for TASK 12 — 1D-CNN Autoencoder."""

import sys
import pytest
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_layer_shapes_and_parameters():
    """Verify layer-by-layer intermediate shapes, parameter counts, and NaN/Inf checks."""
    import torch
    from src.model import C1Autoencoder

    batch_size = 2
    in_channels = 4
    seq_len = 512

    torch.manual_seed(42)
    x = torch.randn(batch_size, in_channels, seq_len)

    expected_params = {
        16: 36724,
        8:  31596,
        4:  29032,
        2:  27750,
    }

    for db, exp_param_count in expected_params.items():
        model = C1Autoencoder(d_b=db)
        model.eval()

        # 1. Verify total trainable parameters count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[TEST] d_b={db:<2} -> Total parameters: {total_params} (expected: {exp_param_count})")
        assert total_params == exp_param_count, f"Parameter count mismatch for d_b={db}: {total_params} != {exp_param_count}"

        # 2. Layer-by-layer output shape checks using forward hooks
        shapes = {}

        def get_hook(name):
            def hook(module, input_tensor, output_tensor):
                shapes[name] = list(output_tensor.shape)
            return hook

        # Register hooks on encoder layers
        model.encoder[0].register_forward_hook(get_hook("enc1"))
        model.encoder[2].register_forward_hook(get_hook("enc2"))
        model.encoder[4].register_forward_hook(get_hook("enc3"))
        model.encoder[6].register_forward_hook(get_hook("latent"))

        with torch.no_grad():
            output = model(x)

        # Assert shape sequence:
        # enc1: [2, 16, 256]
        # enc2: [2, 32, 128]
        # enc3: [2, 64, 64]
        # latent: [2, d_b, 32]
        # output: [2, 4, 512]
        assert shapes["enc1"] == [2, 16, 256], f"enc1 shape mismatch: {shapes['enc1']}"
        assert shapes["enc2"] == [2, 32, 128], f"enc2 shape mismatch: {shapes['enc2']}"
        assert shapes["enc3"] == [2, 64, 64], f"enc3 shape mismatch: {shapes['enc3']}"
        assert shapes["latent"] == [2, db, 32], f"latent shape mismatch: {shapes['latent']}"
        assert list(output.shape) == [2, 4, 512], f"output shape mismatch: {list(output.shape)}"

        # 3. Assert NO NaN or Inf values in output
        assert not torch.isnan(output).any(), f"Output contains NaN for d_b={db}"
        assert not torch.isinf(output).any(), f"Output contains Inf for d_b={db}"

        print(f"       -> enc1: {shapes['enc1']}, enc2: {shapes['enc2']}, enc3: {shapes['enc3']}")
        print(f"       -> latent: {shapes['latent']}, output: {list(output.shape)}")

    print("[SUCCESS] All layer shapes, parameter counts, and NaN/Inf checks passed!")


if __name__ == "__main__":
    print("======================================================")
    print("   TASK 12 — AUTOMATED UNIT TEST SUITE FOR 1D-CNN AE ")
    print("======================================================")
    test_layer_shapes_and_parameters()
    print("======================================================")
    print(">>> ALL AUTOENCODER MODEL TESTS PASSED CLEANLY! <<<")
    print("======================================================")
