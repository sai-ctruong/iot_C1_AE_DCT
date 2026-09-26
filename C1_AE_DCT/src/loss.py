"""Loss functions for Autoencoder training in C1_AE_DCT project."""

from typing import Union

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


def reconstruction_mse(
    pred: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Compute Mean Squared Error (MSE) loss on Z-score normalized signal domain across all 4 channels equally.

    Formulas:
    Loss_MSE = mean((pred - target) ** 2)

    DESIGN RATIONALE & EXCLUSIONS:
    - Optimization Domain: MSE is computed on Z-score normalized signals [B, 4, 512] during training.
    - Equal Channel Weighting: Each channel (PPG, ACCx, ACCy, ACCz) has equal weight (1.0).
    - Exclusions: NO HR labels, NO Activity labels, NO PRD as loss, NO test-tuned channel weights.
    - Evaluation Note: Clinical distortion metrics (PRD, PRDN, RMSE) are computed AFTER denormalizing
      back to original physical units during evaluation.

    Parameters:
    -----------
    pred : torch.Tensor
        Model output tensor of shape [B, 4, 512] in normalized domain.
    target : torch.Tensor
        Ground truth target tensor of shape [B, 4, 512] in normalized domain.

    Returns:
    --------
    loss : torch.Tensor
        Scalar MSE loss tensor.
    """
    if torch is None:
        raise ImportError("PyTorch is required for reconstruction_mse. Please install requirements.txt.")

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch between pred {pred.shape} and target {target.shape}")

    if pred.ndim != 3 or pred.shape[1] != 4:
        raise ValueError(f"Expected input shape [B, 4, 512], got {pred.shape}")

    return torch.mean((pred - target) ** 2)


class ReconstructionMSELoss(nn.Module if nn is not None else object):
    """PyTorch Module wrapper for equal-weighted 4-channel Z-score Reconstruction MSE loss."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return reconstruction_mse(pred, target)
