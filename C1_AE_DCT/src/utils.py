"""Utility functions for C1_AE_DCT project."""
import os
import sys
import platform
import json
from pathlib import Path
from typing import Dict, Any, List

def load_config(config_path: str = "configs/config.json") -> Dict[str, Any]:
    """Load configuration file (JSON or YAML fallback)."""
    path = Path(config_path)
    if not path.exists():
        base_dir = Path(__file__).resolve().parent.parent
        path = base_dir / config_path
    
    if path.suffix in [".yaml", ".yml"]:
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            json_path = path.with_suffix(".json")
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            raise ImportError("PyYAML is not installed. Run 'pip install -r requirements.txt'.")
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def load_folds(folds_path: str = "configs/folds.json") -> Dict[str, Dict[str, List[str]]]:
    """Load deterministic 5-fold subject split configuration from JSON."""
    path = Path(folds_path)
    if not path.exists():
        base_dir = Path(__file__).resolve().parent.parent
        path = base_dir / folds_path
    
    if not path.exists():
        raise FileNotFoundError(f"Folds configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        folds = json.load(f)
    
    return folds

def validate_folds(folds: Dict[str, Dict[str, List[str]]]) -> bool:
    """Validate subject-wise 5-fold partition correctness.
    
    Validation Criteria:
    1. Train, Val, Test do not intersect in the same fold.
    2. Each fold has exactly 10 Train, 2 Val, 3 Test subjects.
    3. Every subject S1–S15 appears in Test set exactly once across the 5 folds.
    4. Deterministic static partition (5 folds present).
    """
    assert len(folds) == 5, f"Expected 5 folds, found {len(folds)}"
    
    all_test_subjects = []
    expected_all_subjects = {f"S{i}" for i in range(1, 16)}

    for fold_name, split in folds.items():
        train_set = set(split["train"])
        val_set = set(split["val"])
        test_set = set(split["test"])

        # 1. Assert no overlap between Train, Val, Test in the same fold
        assert train_set.isdisjoint(val_set), f"{fold_name}: Train and Val overlap ({train_set & val_set})"
        assert train_set.isdisjoint(test_set), f"{fold_name}: Train and Test overlap ({train_set & test_set})"
        assert val_set.isdisjoint(test_set), f"{fold_name}: Val and Test overlap ({val_set & test_set})"

        # 2. Assert exact counts: 10 Train, 2 Val, 3 Test
        assert len(split["train"]) == 10, f"{fold_name}: Train count is {len(split['train'])}, expected 10"
        assert len(split["val"]) == 2, f"{fold_name}: Val count is {len(split['val'])}, expected 2"
        assert len(split["test"]) == 3, f"{fold_name}: Test count is {len(split['test'])}, expected 3"

        # 3. Collect test subjects to verify single coverage across 5 folds
        all_test_subjects.extend(split["test"])

    # Verify S1–S15 test coverage
    test_counts = {subj: all_test_subjects.count(subj) for subj in expected_all_subjects}
    for subj, count in test_counts.items():
        assert count == 1, f"Subject {subj} appears in Test set {count} times across 5 folds (expected exactly 1)"

    assert set(all_test_subjects) == expected_all_subjects, "Test set union across 5 folds does not equal S1-S15"

    return True

def print_folds_table(folds: Dict[str, Dict[str, List[str]]]) -> None:
    """Print formatted ASCII table of the 5-fold split for visual verification."""
    header = f"| {'Fold':<8} | {'Train (10 Subjects)':<45} | {'Val (2)':<12} | {'Test (3)':<15} |"
    separator = "+" + "-" * 10 + "+" + "-" * 47 + "+" + "-" * 14 + "+" + "-" * 17 + "+"
    
    print("\n=== SUBJECT-WISE 5-FOLD SPLIT TABLE ===")
    print(separator)
    print(header)
    print(separator)

    for fold_name, split in folds.items():
        train_str = ", ".join(split["train"])
        val_str = ", ".join(split["val"])
        test_str = ", ".join(split["test"])
        print(f"| {fold_name:<8} | {train_str:<45} | {val_str:<12} | {test_str:<15} |")

    print(separator)

def get_system_info() -> Dict[str, Any]:
    """Inspect and log current Python environment and system specs."""
    info = {
        "python_version": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor() or platform.machine(),
        "packages": {}
    }

    for pkg_name in ["torch", "numpy", "scipy", "yaml"]:
        try:
            mod = __import__(pkg_name)
            info["packages"][pkg_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            info["packages"][pkg_name] = "Not installed"

    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        else:
            info["gpu_name"] = "N/A"
            info["vram_gb"] = "N/A"
    except ImportError:
        info["cuda_available"] = False
        info["gpu_name"] = "Torch not installed"
        info["vram_gb"] = "N/A"

    return info

if __name__ == "__main__":
    folds = load_folds()
    validate_folds(folds)
    print_folds_table(folds)
