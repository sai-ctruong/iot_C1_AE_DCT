"""Verification script for module imports and environment configuration."""
import os
import sys
from pathlib import Path

# Add project root directory to sys.path to allow src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_project_modules_import():
    """Test importing all project modular components without dependencies error."""
    try:
        import src.dataset as dataset
        import src.baseline_dct as baseline_dct
        import src.codec as codec
        import src.model as model
        import src.train as train
        import src.evaluate as evaluate
        import src.make_report as make_report
        import src.utils as utils
        import src.resample as resample
        import src.normalize as norm_mod
        import src.windowing as windowing
        import src.metrics as metrics
        import src.loss as loss
        import src.results_schema as results_schema
        import src.aggregation as aggregation

        assert hasattr(dataset, "C1Dataset")
        assert hasattr(baseline_dct, "dct_encode_topk")
        assert hasattr(baseline_dct, "dct_decode")
        assert hasattr(baseline_dct, "dct_reconstruct")
        assert hasattr(baseline_dct, "BaselineDCT")
        assert hasattr(codec, "encode_ae_bytes")
        assert hasattr(codec, "decode_ae_bytes")
        assert hasattr(codec, "encode_dct_bytes")
        assert hasattr(codec, "decode_dct_bytes")
        assert hasattr(codec, "C1Codec")
        assert hasattr(model, "C1Autoencoder")
        assert hasattr(train, "train_model")
        assert hasattr(train, "set_seed")
        assert hasattr(train, "evaluate_fair_val_loss")
        assert hasattr(evaluate, "compute_channel_metrics")
        assert hasattr(evaluate, "evaluate_window_metrics")
        assert hasattr(evaluate, "evaluate_dataset_metrics")
        assert hasattr(make_report, "generate_report")
        assert hasattr(utils, "get_system_info")
        assert hasattr(resample, "resample_acc")
        assert hasattr(resample, "align_ppg_acc")
        assert hasattr(norm_mod, "compute_norm_stats")
        assert hasattr(norm_mod, "normalize")
        assert hasattr(norm_mod, "denormalize")
        assert hasattr(windowing, "create_windows")
        assert hasattr(metrics, "compute_cr_dim")
        assert hasattr(metrics, "get_budget_mapping_table")
        assert hasattr(loss, "reconstruction_mse")
        assert hasattr(loss, "ReconstructionMSELoss")
        assert hasattr(results_schema, "append_result")
        assert hasattr(results_schema, "validate_results_schema")
        assert hasattr(results_schema, "join_ae_dct_results")
        assert hasattr(aggregation, "export_aggregation_reports")

        import src.reporting as reporting
        assert hasattr(reporting, "generate_table8_byte_cost")
        assert hasattr(reporting, "generate_table9_ae_parameter_count")
        assert hasattr(reporting, "generate_all_plots_and_tables")

        import src.reconstruction_visualization as recon_vis
        assert hasattr(recon_vis, "find_failure_cases")
        assert hasattr(recon_vis, "export_failure_cases_csv")
        assert hasattr(recon_vis, "plot_single_window_comparison")
        assert hasattr(recon_vis, "generate_reconstruction_task23_artifacts")
        print("[SUCCESS] All project stubs (src/*) imported successfully!")
    except Exception as e:
        print(f"[FAIL] Module import error: {e}")
        raise e

def test_config_loading():
    """Test reading the configuration file."""
    try:
        from src.utils import load_config
        config = load_config("configs/config.json")
        assert config["project"]["name"] == "C1_AE_DCT"
        print("[SUCCESS] Configuration file (configs/config.json) loaded correctly!")
    except Exception as e:
        print(f"[FAIL] Configuration loading error: {e}")
        raise e

def test_folds_validation():
    """Test loading and validating the 5-fold split configuration."""
    try:
        from src.utils import load_folds, validate_folds
        folds = load_folds()
        assert validate_folds(folds) is True
        print("[SUCCESS] Subject-Wise 5-Fold split validated correctly!")
    except Exception as e:
        print(f"[FAIL] Folds validation error: {e}")
        raise e

def check_external_packages():
    """Check status of required external packages (torch, numpy, scipy, yaml)."""
    print("\n--- External Packages Status ---")
    packages = ["torch", "numpy", "scipy", "yaml", "psutil", "matplotlib", "pytest"]
    installed = []
    missing = []
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            print(f"  [OK] {pkg}: {version}")
            installed.append(pkg)
        except ImportError:
            print(f"  [MISSING] {pkg}: Not installed (Run: pip install -r requirements.txt)")
            missing.append(pkg)
    return installed, missing

if __name__ == "__main__":
    print("=== RUNNING IMPORT & CONFIG VERIFICATION TEST ===")
    test_project_modules_import()
    test_config_loading()
    test_folds_validation()
    check_external_packages()
    print("\n=== ALL CORE SETUP TESTS PASSED ===")
