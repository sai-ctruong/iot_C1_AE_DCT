"""Unit test for TASK 2 — Subject-Wise 5-Fold Split Validation."""
import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_folds, validate_folds, print_folds_table

def test_load_and_validate_folds():
    """Verify that load_folds and validate_folds work and satisfy all prompt constraints."""
    folds = load_folds()
    
    # Run strict assertion suite
    is_valid = validate_folds(folds)
    assert is_valid is True, "validate_folds did not return True"
    
    print("[SUCCESS] All 5-fold split validation checks passed!")

if __name__ == "__main__":
    print("=== TASK 2 - SUBJECT-WISE 5-FOLD SPLIT VERIFICATION ===")
    folds = load_folds()
    validate_folds(folds)
    print_folds_table(folds)
    print("\n[SUCCESS] validate_folds asserts passed cleanly!")
