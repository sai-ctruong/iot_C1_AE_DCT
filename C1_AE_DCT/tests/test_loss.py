"""Unit test for TASK 13 — Z-Score Reconstruction Loss Function."""

import sys
import pytest
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_reconstruction_mse_loss():
    """Verify reconstruction_mse output, zero error, gradient flow, and shape checks."""
    import torch
    from src.loss import reconstruction_mse, ReconstructionMSELoss
    from src.model import C1Autoencoder

    batch_size = 4
    channels = 4
    seq_len = 512

    torch.manual_seed(42)
    # Z-score normalized tensors (mean ~ 0, std ~ 1)
    target = torch.randn(batch_size, channels, seq_len)
    pred = target + 0.1 * torch.randn(batch_size, channels, seq_len)

    # 1. Scalar output test
    loss = reconstruction_mse(pred, target)
    assert loss.ndim == 0, f"Loss must be scalar (0D tensor), got shape {loss.shape}"
    assert loss.item() > 0, f"Loss for non-identical tensors must be positive, got {loss.item()}"

    # 2. Exact match zero loss test
    zero_loss = reconstruction_mse(target, target)
    assert abs(zero_loss.item()) < 1e-7, f"Loss for identical tensors must be 0.0, got {zero_loss.item()}"

    # 3. Model integration & gradient backward pass test
    model = C1Autoencoder(d_b=16)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    rec_x = model(target)
    criterion = ReconstructionMSELoss()
    loss_model = criterion(rec_x, target)

    optimizer.zero_grad()
    loss_model.backward()
    optimizer.step()

    assert loss_model.item() > 0
    print(f"[TEST] Model training step with reconstruction_mse passed! Loss: {loss_model.item():.6f}")

    # 4. Shape mismatch exception check
    invalid_pred = torch.randn(batch_size, 3, seq_len)
    with pytest.raises(ValueError):
        reconstruction_mse(invalid_pred, target)

    print("[SUCCESS] All reconstruction MSE loss tests passed cleanly!")


if __name__ == "__main__":
    print("=== TASK 13 - RECONSTRUCTION LOSS FUNCTION VERIFICATION ===")
    test_reconstruction_mse_loss()
    print("=== ALL LOSS FUNCTION TESTS PASSED CLEANLY ===")
