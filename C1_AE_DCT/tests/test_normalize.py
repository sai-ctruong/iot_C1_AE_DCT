"""Unit test for TASK 4 — Z-Score Train-Only Normalization Pipeline."""
import sys
import json
import pytest
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_zscore_normalize_denormalize_reconstruction():
    """Verify denormalize(normalize(x)) ~= x."""
    import numpy as np
    from src.normalize import compute_norm_stats, normalize, denormalize

    np.random.seed(42)
    # Generate 1000 samples of 4 channels [PPG, ACCx, ACCy, ACCz]
    train_data = np.random.randn(1000, 4).astype(np.float32) * np.array([2.5, 0.8, 1.2, 0.5]) + np.array([50.0, 0.1, -0.2, 9.8])

    # Compute stats on Train
    stats = compute_norm_stats(train_data, ddof=0)

    assert len(stats["mean"]) == 4
    assert len(stats["std"]) == 4

    # Normalize Train
    train_norm = normalize(train_data, stats)
    
    # Check normalized mean ≈ 0 and std ≈ 1
    np.testing.assert_allclose(np.mean(train_norm, axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.std(train_norm, axis=0, ddof=0), 1.0, atol=1e-5)

    # Denormalize
    train_rec = denormalize(train_norm, stats)

    # Verify reconstruction exactness denormalize(normalize(x)) ~= x
    np.testing.assert_allclose(train_data, train_rec, rtol=1e-5, atol=1e-5, err_msg="Reconstruction mismatch!")
    print("[TEST] denormalize(normalize(x)) ~= x passed!")


def test_val_test_uses_train_stats():
    """Verify Val/Test set normalization uses exact Train statistics without recomputing stats."""
    import numpy as np
    from src.normalize import compute_norm_stats, normalize, denormalize

    np.random.seed(123)
    train_data = np.random.randn(500, 4).astype(np.float32) + 10.0
    val_data = np.random.randn(100, 4).astype(np.float32) + 12.0  # Shifted distribution

    stats_train = compute_norm_stats(train_data, ddof=0)
    
    # Normalize Val using Train stats
    val_norm = normalize(val_data, stats_train)
    val_rec = denormalize(val_norm, stats_train)

    np.testing.assert_allclose(val_data, val_rec, rtol=1e-5, atol=1e-5)
    print("[TEST] Val data normalized with Train stats reconstructs cleanly!")


def test_zero_std_error():
    """Verify ValueError is raised if any channel has zero or non-positive std."""
    import numpy as np
    from src.normalize import compute_norm_stats

    # Channel 2 is constant (std = 0)
    invalid_data = np.array([
        [1.0, 2.0, 5.0, 3.0],
        [2.0, 4.0, 5.0, 6.0],
        [3.0, 6.0, 5.0, 9.0],
    ], dtype=np.float32)

    with pytest.raises(ValueError) as excinfo:
        compute_norm_stats(invalid_data, ddof=0)

    assert "Invalid standard deviation" in str(excinfo.value) or "constant" in str(excinfo.value)
    print("[TEST] Zero std error exception raised correctly!")


def test_save_load_norm_stats_folds(tmp_path):
    """Test saving and loading norm_stats_fold1.json through norm_stats_fold5.json in isolated test temp directory."""
    import numpy as np
    from src.normalize import compute_norm_stats, save_norm_stats, load_norm_stats

    test_configs_dir = tmp_path / "test_configs"
    test_configs_dir.mkdir(exist_ok=True)

    for fold_idx in range(1, 6):
        dummy_train = np.random.randn(200, 4).astype(np.float32) * fold_idx + (fold_idx * 10)
        stats = compute_norm_stats(dummy_train, ddof=0)

        filepath = test_configs_dir / f"norm_stats_fold{fold_idx}.json"
        save_norm_stats(stats, filepath)
        assert filepath.exists(), f"File {filepath} was not created"

        loaded_stats = load_norm_stats(filepath)
        assert loaded_stats["channels"] == ["PPG", "ACCx", "ACCy", "ACCz"]
        assert len(loaded_stats["mean"]) == 4
        assert len(loaded_stats["std"]) == 4

    print("[SUCCESS] Fold 1-5 norm_stats JSON serialization and loading verified in isolated temp dir!")



if __name__ == "__main__":
    import tempfile
    print("=== TASK 4 - Z-SCORE TRAIN-ONLY NORMALIZATION VERIFICATION ===")
    test_zscore_normalize_denormalize_reconstruction()
    test_val_test_uses_train_stats()
    test_zero_std_error()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_save_load_norm_stats_folds(Path(tmpdir))
    print("=== ALL NORMALIZATION TESTS PASSED CLEANLY ===")

