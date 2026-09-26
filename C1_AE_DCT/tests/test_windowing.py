"""Unit test for TASK 5 — 8s Windowing & Metadata Verification."""
import sys
import pytest
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_windowing_train_val_overlap():
    """Verify Train/Val windowing has step=256 (4s) and output shape [4, 512]."""
    import numpy as np
    from src.windowing import create_windows

    fs = 64.0
    # 20 seconds of signal = 20 * 64 = 1280 samples
    signal = np.random.randn(1280, 4).astype(np.float32)

    windows, meta_list, stats = create_windows(
        signal, fs=fs, window_sec=8.0, split="train", subject="S1", fold=1
    )

    # Window 0: 0..512 (0s - 8s)
    # Window 1: 256..768 (4s - 12s)
    # Window 2: 512..1024 (8s - 16s)
    # Window 3: 768..1280 (12s - 20s)
    # Total windows = 4, leftover = 0

    assert windows.shape == (4, 4, 512), f"Expected shape (4, 4, 512), got {windows.shape}"
    assert len(meta_list) == 4
    assert stats["windows_generated"] == 4
    assert stats["leftover_samples_dropped"] == 0

    # Check step size in metadata
    assert meta_list[0]["start_index"] == 0
    assert meta_list[1]["start_index"] == 256
    assert meta_list[2]["start_index"] == 512
    assert meta_list[3]["start_index"] == 768

    print("[TEST] Train/Val 50% overlap windowing (step=256, shape=[4,512]) passed!")


def test_windowing_test_no_overlap():
    """Verify Test windowing has step=512 (8s, no overlap) and output shape [4, 512]."""
    import numpy as np
    from src.windowing import create_windows

    fs = 64.0
    # 20 seconds of signal = 1280 samples
    signal = np.random.randn(1280, 4).astype(np.float32)

    windows, meta_list, stats = create_windows(
        signal, fs=fs, window_sec=8.0, split="test", subject="S2", fold=1
    )

    # Window 0: 0..512 (0s - 8s)
    # Window 1: 512..1024 (8s - 16s)
    # Remaining samples = 1280 - 1024 = 256 samples (< 512), so 256 samples dropped!
    # Total windows = 2, leftover = 256 samples (4 seconds)

    assert windows.shape == (2, 4, 512), f"Expected shape (2, 4, 512), got {windows.shape}"
    assert len(meta_list) == 2
    assert stats["windows_generated"] == 2
    assert stats["leftover_samples_dropped"] == 256
    assert stats["leftover_seconds_dropped"] == 4.0

    assert meta_list[0]["start_index"] == 0
    assert meta_list[1]["start_index"] == 512

    print("[TEST] Test 0% overlap windowing (step=512, leftover dropped) passed!")


def test_metadata_fields():
    """Verify all required metadata fields exist and hold correct subject/fold/split info."""
    import numpy as np
    from src.windowing import create_windows

    signal = np.random.randn(600, 4).astype(np.float32)
    windows, meta_list, stats = create_windows(
        signal, fs=64.0, window_sec=8.0, subject="S3", fold=2, split="val"
    )

    assert len(meta_list) == 1
    m = meta_list[0]

    required_keys = ["subject", "start_time", "start_index", "fold", "split", "window_id"]
    for key in required_keys:
        assert key in m, f"Missing metadata key: {key}"

    assert m["subject"] == "S3"
    assert m["fold"] == 2
    assert m["split"] == "val"
    assert m["start_index"] == 0
    assert m["end_index"] == 512
    assert m["window_id"] == "S3_val_f2_w0000"

    print("[TEST] Metadata fields verification passed!")


def test_no_cross_subject_leftover_concatenation():
    """Verify leftovers from subject A are NOT concatenated with subject B."""
    import numpy as np
    from src.windowing import create_windows

    fs = 64.0
    # Subject A: 600 samples (512 win + 88 leftover)
    # Subject B: 600 samples (512 win + 88 leftover)
    subj_a_signal = np.random.randn(600, 4).astype(np.float32)
    subj_b_signal = np.random.randn(600, 4).astype(np.float32)

    win_a, meta_a, stats_a = create_windows(subj_a_signal, fs=fs, subject="S4", split="test")
    win_b, meta_b, stats_b = create_windows(subj_b_signal, fs=fs, subject="S5", split="test")

    assert len(win_a) == 1
    assert len(win_b) == 1
    assert stats_a["leftover_samples_dropped"] == 88
    assert stats_b["leftover_samples_dropped"] == 88
    assert meta_a[0]["subject"] == "S4"
    assert meta_b[0]["subject"] == "S5"

    print("[TEST] No cross-subject leftover concatenation passed!")


if __name__ == "__main__":
    print("=== TASK 5 - WINDOWING VERIFICATION ===")
    test_windowing_train_val_overlap()
    test_windowing_test_no_overlap()
    test_metadata_fields()
    test_no_cross_subject_leftover_concatenation()
    print("=== ALL WINDOWING TESTS PASSED CLEANLY ===")
