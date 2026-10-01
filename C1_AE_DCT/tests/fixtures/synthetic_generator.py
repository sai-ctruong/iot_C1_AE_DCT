"""Synthetic results generator fixture for test suite ONLY.

NEVER TO BE CALLED BY PRODUCTION CODE IN src/.
"""

from pathlib import Path
from typing import Union, List, Dict, Any
import numpy as np

from src.results_schema import create_result_row
from src.aggregation import save_records_to_csv

CHANNEL_NAMES = ["PPG", "ACCx", "ACCy", "ACCz"]


def generate_synthetic_full_results(output_path: Union[str, Path]) -> Path:
    """Generate a valid synthetic results.csv covering 5 folds, 15 subjects, 4 d_b budgets for unit tests ONLY."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    records = []

    fold_test_map = {
        1: ["S1", "S2", "S3"],
        2: ["S4", "S5", "S6"],
        3: ["S7", "S8", "S9"],
        4: ["S10", "S11", "S12"],
        5: ["S13", "S14", "S15"],
    }

    for fold, subjs in fold_test_map.items():
        for subj in subjs:
            for db in [16, 8, 4, 2]:
                m_dim = 32 * db
                nbytes = 16 + 4 * m_dim
                cr_dim_ae = 2048.0 / float(m_dim)
                k_dct_dim = m_dim
                k_dct_byte = (4 * m_dim) // 6
                cr_dim_dct_dim = 2048.0 / float(k_dct_dim)
                cr_dim_dct_byte = 2048.0 / float(k_dct_byte)
                cr_b64 = 8192.0 / float(nbytes)
                cr_bnative = 5120.0 / float(nbytes)

                for w in range(4):  # 4 windows per subject
                    w_id = f"win_{fold:02d}_{subj}_{w:04d}"

                    for ch in CHANNEL_NAMES:
                        ae_prd = float(2.0 + (16 - db) * 0.4 + np.random.randn() * 0.2)
                        dct_dim_prd = float(3.5 + (16 - db) * 0.6 + np.random.randn() * 0.3)
                        dct_byte_prd = float(4.0 + (16 - db) * 0.7 + np.random.randn() * 0.3)

                        # AE equal_dim
                        r_ae_dim = create_result_row(
                            fold=fold, subject=subj, seed=42, dataset="PPG-DaLiA", method="AE",
                            comparison_type="equal_dim", db=db, M=m_dim, K=0, representation_count=m_dim,
                            channel=ch, window_id=w_id, start_index=w*512, nbytes=nbytes,
                            CR_dim=cr_dim_ae, CR_byte_64=cr_b64, CR_byte_native=cr_bnative,
                            PRD=ae_prd, PRDN=ae_prd*2.1, RMSE=ae_prd*0.02, valid_prd=True, valid_prdn=True,
                            metric_valid=True, checkpoint=f"ae_f{fold}_db{db}.pt", config_id=f"ae_f{fold}_db{db}"
                        )
                        # AE equal_byte
                        r_ae_byte = create_result_row(
                            fold=fold, subject=subj, seed=42, dataset="PPG-DaLiA", method="AE",
                            comparison_type="equal_byte", db=db, M=m_dim, K=0, representation_count=m_dim,
                            channel=ch, window_id=w_id, start_index=w*512, nbytes=nbytes,
                            CR_dim=cr_dim_ae, CR_byte_64=cr_b64, CR_byte_native=cr_bnative,
                            PRD=ae_prd, PRDN=ae_prd*2.1, RMSE=ae_prd*0.02, valid_prd=True, valid_prdn=True,
                            metric_valid=True, checkpoint=f"ae_f{fold}_db{db}.pt", config_id=f"ae_f{fold}_db{db}"
                        )

                        # DCT equal_dim
                        nbytes_dct_dim = 16 + 6 * m_dim
                        r_dct_dim = create_result_row(
                            fold=fold, subject=subj, seed=42, dataset="PPG-DaLiA", method="DCT",
                            comparison_type="equal_dim", db=db, M=m_dim, K=k_dct_dim, representation_count=k_dct_dim,
                            channel=ch, window_id=w_id, start_index=w*512, nbytes=nbytes_dct_dim,
                            CR_dim=cr_dim_dct_dim, CR_byte_64=8192.0/nbytes_dct_dim, CR_byte_native=5120.0/nbytes_dct_dim,
                            PRD=dct_dim_prd, PRDN=dct_dim_prd*2.2, RMSE=dct_dim_prd*0.025, valid_prd=True, valid_prdn=True,
                            metric_valid=True, checkpoint="N/A", config_id=f"dct_eqdim_db{db}"
                        )
                        # DCT equal_byte
                        r_dct_byte = create_result_row(
                            fold=fold, subject=subj, seed=42, dataset="PPG-DaLiA", method="DCT",
                            comparison_type="equal_byte", db=db, M=m_dim, K=k_dct_byte, representation_count=k_dct_byte,
                            channel=ch, window_id=w_id, start_index=w*512, nbytes=nbytes,
                            CR_dim=cr_dim_dct_byte, CR_byte_64=cr_b64, CR_byte_native=cr_bnative,
                            PRD=dct_byte_prd, PRDN=dct_byte_prd*2.2, RMSE=dct_byte_prd*0.025, valid_prd=True, valid_prdn=True,
                            metric_valid=True, checkpoint="N/A", config_id=f"dct_eqbyte_db{db}"
                        )

                        records.extend([r_ae_dim, r_ae_byte, r_dct_dim, r_dct_byte])

    save_records_to_csv(records, out_file)
    return out_file
