"""Unit test for TASK 3 — Resample ACC 32 Hz -> 64 Hz & PPG-ACC Alignment."""
import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_acc_resampling_and_alignment():
    """Test 8 seconds ACC (32 Hz -> 64 Hz, 256 -> 512 samples) and PPG alignment."""
    import numpy as np
    from src.resample import resample_acc, align_ppg_acc

    duration_sec = 8.0
    fs_acc_in = 32.0
    fs_target = 64.0

    num_acc_samples = int(duration_sec * fs_acc_in)  # 256 samples
    num_ppg_samples = int(duration_sec * fs_target)  # 512 samples

    # Generate synthetic 8-second ACC signal (3 channels: x, y, z)
    t_acc = np.linspace(0, duration_sec, num_acc_samples, endpoint=False)
    acc_x = np.sin(2 * np.pi * 1.5 * t_acc)
    acc_y = np.cos(2 * np.pi * 1.5 * t_acc)
    acc_z = 0.5 * np.sin(2 * np.pi * 0.5 * t_acc)
    acc_data = np.column_stack([acc_x, acc_y, acc_z])  # Shape: (256, 3)

    # Generate synthetic 8-second PPG signal (64 Hz)
    t_ppg = np.linspace(0, duration_sec, num_ppg_samples, endpoint=False)
    ppg_data = np.sin(2 * np.pi * 1.0 * t_ppg)  # Shape: (512,)

    # Step 1: Check duration before resampling
    dur_acc_before = len(acc_data) / fs_acc_in
    assert dur_acc_before == duration_sec, f"Expected {duration_sec}s, got {dur_acc_before}s"
    print(f"[TEST] ACC input duration: {dur_acc_before:.2f}s ({len(acc_data)} samples at {fs_acc_in} Hz)")

    # Step 2: Resample ACC (32 Hz -> 64 Hz)
    acc_resampled, _ = resample_acc(acc_data, fs_in=32.0, fs_out=64.0)

    # Step 3: Check duration and length after resampling
    dur_acc_after = len(acc_resampled) / fs_target
    print(f"[TEST] ACC resampled duration: {dur_acc_after:.2f}s ({len(acc_resampled)} samples at {fs_target} Hz)")
    assert abs(dur_acc_before - dur_acc_after) < 1e-4, "Duration before and after resampling must match!"
    assert len(acc_resampled) == num_ppg_samples, f"Expected {num_ppg_samples} samples, got {len(acc_resampled)}"

    # Step 4: Align PPG and Resampled ACC into 4 channels [PPG, ACCx, ACCy, ACCz]
    output_4ch = align_ppg_acc(ppg_data, acc_resampled, fs=64.0)

    print(f"[TEST] Output 4-channel matrix shape: {output_4ch.shape}")
    assert output_4ch.shape == (512, 4), f"Expected shape (512, 4), got {output_4ch.shape}"

    # Verify channel ordering: [PPG, ACCx, ACCy, ACCz]
    np.testing.assert_allclose(output_4ch[:, 0], ppg_data, err_msg="Channel 0 must be PPG")
    np.testing.assert_allclose(output_4ch[:, 1:], acc_resampled, err_msg="Channels 1-3 must be [ACCx, ACCy, ACCz]")

    print("[SUCCESS] Resampling & Alignment test passed cleanly!")

if __name__ == "__main__":
    print("=== TASK 3 — RESAMPLE ACC 32 Hz -> 64 Hz VERIFICATION ===")
    test_acc_resampling_and_alignment()
