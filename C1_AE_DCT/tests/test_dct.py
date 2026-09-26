"""Unit test for TASK 6 — DCT-II Baseline Top-K Encoding and Decoding."""
import sys
import pytest
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_full_k_exact_reconstruction():
    """Verify that when K=2048 (all coefficients retained), reconstruction is exact (error ~ 0)."""
    import numpy as np
    from src.baseline_dct import dct_encode_topk, dct_decode, dct_reconstruct

    np.random.seed(42)
    # 4 channels x 512 samples
    window = np.random.randn(4, 512).astype(np.float32)

    rec_signal, topk_vals, topk_idx, ch_counts = dct_reconstruct(window, k=2048)

    assert rec_signal.shape == (4, 512)
    assert len(topk_vals) == 2048
    assert len(topk_idx) == 2048
    assert sum(ch_counts.values()) == 2048

    # Check exact reconstruction with full energy
    np.testing.assert_allclose(window, rec_signal, rtol=1e-5, atol=1e-5, err_msg="Full K reconstruction mismatch!")
    print("[TEST] Full K=2048 exact reconstruction passed!")


def test_topk_selection_and_channel_counts():
    """Verify Top-K selection across 2048 coefficients without forcing K/4 per channel."""
    import numpy as np
    from src.baseline_dct import dct_encode_topk

    np.random.seed(100)
    # Channel 0 (PPG) has large high-frequency amplitude, others are small
    window = np.zeros((4, 512), dtype=np.float32)
    t = np.linspace(0, 1, 512)
    window[0] = 100.0 * np.sin(2 * np.pi * 10 * t)  # High energy on PPG
    window[1] = 0.01 * np.sin(2 * np.pi * 2 * t)   # Very low energy on ACCx
    window[2] = 0.01 * np.cos(2 * np.pi * 2 * t)   # Very low energy on ACCy
    window[3] = 0.01 * np.sin(2 * np.pi * 1 * t)   # Very low energy on ACCz

    k = 64
    topk_vals, topk_idx, ch_counts, sparse_dct = dct_encode_topk(window, k=k)

    assert len(topk_vals) == 64
    assert sum(ch_counts.values()) == 64
    
    # PPG should have significantly more than 16 (K/4) coefficients retained because of its high energy
    assert ch_counts["PPG"] > 16, f"PPG should dominate Top-K, got {ch_counts}"
    print(f"[TEST] Top-K dynamic channel allocation passed! Channel counts: {ch_counts}")


def test_tie_break_rule():
    """Verify tie-break rule: equal absolute values favor smaller flat index j."""
    import numpy as np
    from src.baseline_dct import dct_encode_topk

    # Create dummy matrix where DCT output has equal magnitude coefficients
    # In time domain, impulse signals
    window = np.zeros((4, 512), dtype=np.float32)
    window[0, 0] = 5.0
    window[1, 0] = 5.0
    window[2, 0] = 5.0
    window[3, 0] = 5.0

    k = 2
    topk_vals, topk_idx, ch_counts, sparse_dct = dct_encode_topk(window, k=k)

    assert len(topk_idx) == 2
    # Smaller indices (e.g. channel 0 before channel 1) must come first due to stable sort tie-break
    assert topk_idx[0] < topk_idx[1], f"Tie break failed: {topk_idx}"
    print(f"[TEST] Tie-break rule (smaller index j first) passed! Top indices: {topk_idx}")


def test_batch_reconstruction():
    """Verify batch processing (B, 4, 512)."""
    import numpy as np
    from src.baseline_dct import dct_encode_topk, dct_decode

    np.random.seed(7)
    batch_window = np.random.randn(8, 4, 512).astype(np.float32)

    topk_vals, topk_idx, ch_counts_list, sparse_dct = dct_encode_topk(batch_window, k=128)
    rec_batch = dct_decode(sparse_dct)

    assert rec_batch.shape == (8, 4, 512)
    assert topk_vals.shape == (8, 128)
    assert topk_idx.shape == (8, 128)
    assert len(ch_counts_list) == 8

    print("[TEST] Batch processing (8, 4, 512) passed!")


if __name__ == "__main__":
    print("=== TASK 6 - DCT-II BASELINE VERIFICATION ===")
    test_full_k_exact_reconstruction()
    test_topk_selection_and_channel_counts()
    test_tie_break_rule()
    test_batch_reconstruction()
    print("=== ALL DCT BASELINE TESTS PASSED CLEANLY ===")
