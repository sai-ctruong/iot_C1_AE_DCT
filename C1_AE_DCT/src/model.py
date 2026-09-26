"""1D-CNN Autoencoder Architecture for C1_AE_DCT project."""

from typing import Tuple, Union, Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class C1Autoencoder(nn.Module if nn is not None else object):
    """
    1D-CNN Autoencoder for multi-channel PPG/ACC compression (4 channels, 512 length).

    Input shape:  [B, 4, 512]
    Latent shape: [B, d_b, 32] -> M = 32 * d_b total latent values per window
    Output shape: [B, 4, 512]

    Supported Budget Factors d_b in {16, 8, 4, 2}:
    - d_b = 16 => M = 512 latent values
    - d_b = 8  => M = 256 latent values
    - d_b = 4  => M = 128 latent values
    - d_b = 2  => M = 64 latent values

    Strict Exclusions:
    - NO BatchNorm
    - NO Dropout
    - NO Attention
    - NO Skip connection across bottleneck
    """

    def __init__(self, d_b: int = 16):
        if nn is None or torch is None:
            raise ImportError("PyTorch is required for C1Autoencoder. Please install requirements.txt.")

        super().__init__()

        if d_b not in [16, 8, 4, 2]:
            raise ValueError(f"d_b must be one of [16, 8, 4, 2], got {d_b}")

        self.d_b = d_b
        self.latent_dim = 32 * d_b

        # Encoder: 3 Conv1d + ReLU + 1 Bottleneck Conv1d (linear)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5, stride=2, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=d_b, kernel_size=5, stride=2, padding=2, bias=True),
            # Bottleneck output: Linear (no activation)
        )

        # Decoder: 3 ConvTranspose1d + ReLU + 1 Output ConvTranspose1d (linear)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=d_b, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels=16, out_channels=4, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
            # Final output: Linear (no activation)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input [B, 4, 512] to latent representation [B, d_b, 32]."""
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent tensor [B, d_b, 32] or [B, M] back to reconstructed [B, 4, 512]."""
        if latent.ndim == 2:
            # Reshape flattened latent [B, 32 * d_b] to [B, d_b, 32]
            latent = latent.view(-1, self.d_b, 32)
        elif latent.ndim == 3 and latent.shape[1:] != (self.d_b, 32):
            raise ValueError(f"Expected latent shape (B, {self.d_b}, 32), got {latent.shape}")

        return self.decoder(latent)

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [B, 4, 512].
        return_latent : bool
            If True, returns tuple of (reconstruction, latent_tensor).

        Returns:
        --------
        reconstruction : torch.Tensor
            Reconstructed signal of shape [B, 4, 512].
        latent : torch.Tensor (optional)
            Latent tensor of shape [B, d_b, 32].
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)

        if return_latent:
            return reconstruction, latent
        return reconstruction
