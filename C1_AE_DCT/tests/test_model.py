"""Unit test for TASK 11 — 1D-CNN Autoencoder Architecture."""

import sys
import pytest
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_autoencoder_architecture_and_shapes():
    """Verify input shape [B, 4, 512], latent shape [B, d_b, 32], and output shape [B, 4, 512]."""
    import torch
    from src.model import C1Autoencoder

    batch_size = 8
    seq_len = 512
    in_channels = 4

    x = torch.randn(batch_size, in_channels, seq_len)

    db_expected_m = [
        (16, 512),
        (8, 256),
        (4, 128),
        (2, 64),
    ]

    for db, expected_m in db_expected_m:
        model = C1Autoencoder(d_b=db)
        model.eval()

        # Test forward pass with return_latent=True
        with torch.no_grad():
            rec_x, latent = model(x, return_latent=True)

        assert rec_x.shape == (batch_size, 4, 512), f"Reconstruction shape mismatch: {rec_x.shape}"
        assert latent.shape == (batch_size, db, 32), f"Latent shape mismatch for db={db}: {latent.shape}"
        assert latent.numel() // batch_size == expected_m, f"Latent total elements per sample mismatch for db={db}"

        # Test encode and decode separately
        with torch.no_grad():
            latent_encoded = model.encode(x)
            rec_decoded = model.decode(latent_encoded)

        assert latent_encoded.shape == (batch_size, db, 32)
        assert rec_decoded.shape == (batch_size, 4, 512)

        # Test decode with flattened latent [B, M]
        flattened_latent = latent_encoded.view(batch_size, -1)
        assert flattened_latent.shape[1] == expected_m
        with torch.no_grad():
            rec_from_flat = model.decode(flattened_latent)
        assert rec_from_flat.shape == (batch_size, 4, 512)

        print(f"[TEST] d_b={db:<2} -> Latent shape {list(latent.shape)}, M={expected_m}, Output shape {list(rec_x.shape)} passed!")


def test_no_batchnorm_or_dropout():
    """Verify model contains NO BatchNorm or Dropout layers as specified."""
    import torch
    import torch.nn as nn
    from src.model import C1Autoencoder

    model = C1Autoencoder(d_b=16)

    for module in model.modules():
        assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout, nn.Dropout1d)), (
            f"Forbidden layer found in model: {type(module)}"
        )

    print("[TEST] Verified NO BatchNorm or Dropout layers in model!")


if __name__ == "__main__":
    print("=== TASK 11 - 1D-CNN AUTOENCODER VERIFICATION ===")
    test_autoencoder_architecture_and_shapes()
    test_no_batchnorm_or_dropout()
    print("=== ALL AUTOENCODER ARCHITECTURE TESTS PASSED CLEANLY ===")
