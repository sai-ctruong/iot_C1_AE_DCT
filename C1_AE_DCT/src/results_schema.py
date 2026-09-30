"""Results Schema module for C1_AE_DCT (TASK 20).

Defines the standard result schema, appending helpers, schema validation, and 
joining utilities for automated table/figure generation without manual Word editing.
"""

from typing import Dict, List, Any, Union, Optional
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

# Mandatory columns defined in TASK 20, TASK 2 & TASK 3 specification
RESULT_SCHEMA_COLUMNS = [
    "fold",
    "subject",
    "seed",
    "dataset",
    "method",
    "comparison_type",
    "db",
    "M",
    "K",
    "representation_count",
    "channel",
    "window_id",
    "start_index",
    "nbytes",
    "CR_dim",
    "CR_byte_64",
    "CR_byte_native",
    "PRD",
    "PRDN",
    "RMSE",
    "valid_prd",
    "valid_prdn",
    "metric_valid",
    "checkpoint",
    "config_id",
]

# Optional/Alias columns for explicit joining
ALL_SCHEMA_COLUMNS = RESULT_SCHEMA_COLUMNS + ["budget"]

JOIN_KEYS = ["fold", "subject", "window_id", "channel", "budget", "comparison_type"]


def create_empty_results_df() -> Any:
    """Create an empty DataFrame or list with standard TASK 20 schema columns."""
    if pd is not None:
        return pd.DataFrame(columns=ALL_SCHEMA_COLUMNS)
    return []


def create_result_row(
    fold: int,
    subject: str,
    seed: int,
    method: str,
    db: int,
    K: int,
    channel: str,
    window_id: Union[str, int],
    start_index: int,
    nbytes: int,
    CR_dim: float,
    CR_byte_64: float,
    CR_byte_native: float,
    PRD: float,
    PRDN: float,
    RMSE: float,
    metric_valid: bool,
    checkpoint: str,
    config_id: str,
    dataset: str = "PPG-DaLiA",
    comparison_type: str = "equal_byte",
    valid_prd: Optional[bool] = None,
    valid_prdn: Optional[bool] = None,
    M: Optional[int] = None,
    representation_count: Optional[int] = None,
    budget: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Construct a single validated result dictionary following TASK 20, TASK 2 & TASK 3 schema.
    """
    if budget is None:
        budget = db
    if M is None:
        M = 32 * db
    if representation_count is None:
        representation_count = M if method == "AE" else K
    if valid_prd is None:
        valid_prd = bool(metric_valid) and not np.isnan(PRD) if PRD is not None else False
    if valid_prdn is None:
        valid_prdn = bool(metric_valid) and not np.isnan(PRDN) if PRDN is not None else False

    row = {
        "fold": int(fold),
        "subject": str(subject),
        "seed": int(seed),
        "dataset": str(dataset),
        "method": str(method),
        "comparison_type": str(comparison_type),
        "db": int(db),
        "budget": int(budget),
        "M": int(M),
        "K": int(K),
        "representation_count": int(representation_count),
        "channel": str(channel),
        "window_id": str(window_id),
        "start_index": int(start_index),
        "nbytes": int(nbytes),
        "CR_dim": float(CR_dim),
        "CR_byte_64": float(CR_byte_64),
        "CR_byte_native": float(CR_byte_native),
        "PRD": float(PRD) if PRD is not None else float("nan"),
        "PRDN": float(PRDN) if PRDN is not None else float("nan"),
        "RMSE": float(RMSE) if RMSE is not None else float("nan"),
        "valid_prd": bool(valid_prd),
        "valid_prdn": bool(valid_prdn),
        "metric_valid": bool(metric_valid),
        "checkpoint": str(checkpoint),
        "config_id": str(config_id),
    }

    return row


def append_result(
    results_container: Union[List[Dict[str, Any]], Any],
    row_or_dict: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Union[List[Dict[str, Any]], Any]:
    """
    Append a result row to a results list or pandas DataFrame.

    Can pass either a dict via row_or_dict or keyword arguments.
    """
    if row_or_dict is None and kwargs:
        row_dict = create_result_row(**kwargs)
    elif isinstance(row_or_dict, dict):
        row_dict = row_or_dict.copy()
        if "budget" not in row_dict:
            row_dict["budget"] = row_dict.get("db", 0)
    else:
        raise ValueError("Must provide a dict or keyword arguments to append_result.")

    if pd is not None and isinstance(results_container, pd.DataFrame):
        new_df = pd.DataFrame([row_dict])
        return pd.concat([results_container, new_df], ignore_index=True)
    elif isinstance(results_container, list):
        results_container.append(row_dict)
        return results_container
    else:
        return [row_dict]


def validate_results_schema(
    results: Union[Any, List[Dict[str, Any]]]
) -> bool:
    """
    Validate that a results DataFrame or list of dicts complies strictly with TASK 20 schema.

    Validation Rules:
    1. All mandatory columns present.
    2. Non-empty.
    3. Channel in ['PPG', 'ACCx', 'ACCy', 'ACCz'].
    4. fold >= 1.
    5. Correct data types and no illegal NaNs in metadata columns.
    """
    if pd is not None and isinstance(results, pd.DataFrame):
        if results.empty:
            raise ValueError("Results DataFrame is empty.")

        missing_cols = set(RESULT_SCHEMA_COLUMNS) - set(results.columns)
        if missing_cols:
            raise ValueError(f"Missing mandatory columns in results: {missing_cols}")

        invalid_channels = set(results["channel"].unique()) - {"PPG", "ACCx", "ACCy", "ACCz"}
        if invalid_channels:
            raise ValueError(f"Invalid channel names found: {invalid_channels}")

        if (results["fold"] < 1).any():
            raise ValueError("Found fold index < 1 in results.")

        return True

    elif isinstance(results, list):
        if not results:
            raise ValueError("Results list is empty.")

        for idx, row in enumerate(results):
            missing_cols = set(RESULT_SCHEMA_COLUMNS) - set(row.keys())
            if missing_cols:
                raise ValueError(f"Row {idx} missing mandatory columns: {missing_cols}")

            if row["channel"] not in {"PPG", "ACCx", "ACCy", "ACCz"}:
                raise ValueError(f"Row {idx} has invalid channel: {row['channel']}")

            if int(row["fold"]) < 1:
                raise ValueError(f"Row {idx} has invalid fold: {row['fold']}")

        return True

    else:
        raise ValueError("Input must be a Pandas DataFrame or a List of Dicts.")


def join_ae_dct_results(
    results_ae: Union[Any, List[Dict[str, Any]]],
    results_dct: Union[Any, List[Dict[str, Any]]],
    join_on: Optional[List[str]] = None
) -> Union[Any, List[Dict[str, Any]]]:
    """
    Join AE and DCT detailed results on fold + subject + window_id + channel + budget (or db).
    """
    validate_results_schema(results_ae)
    validate_results_schema(results_dct)

    if join_on is None:
        # Check if 'comparison_type' and 'budget' exist in both, else fallback to db
        sample_ae = results_ae[0] if isinstance(results_ae, list) else results_ae.iloc[0].to_dict()
        sample_dct = results_dct[0] if isinstance(results_dct, list) else results_dct.iloc[0].to_dict()
        join_on = ["fold", "subject", "window_id", "channel"]
        if "budget" in sample_ae and "budget" in sample_dct:
            join_on.append("budget")
        else:
            join_on.append("db")
        if "comparison_type" in sample_ae and "comparison_type" in sample_dct:
            join_on.append("comparison_type")

    if pd is not None and isinstance(results_ae, pd.DataFrame) and isinstance(results_dct, pd.DataFrame):
        merged = pd.merge(
            results_ae,
            results_dct,
            on=join_on,
            suffixes=("_ae", "_dct")
        )
        return merged
    else:
        # Fallback list of dicts joining
        ae_list = results_ae if isinstance(results_ae, list) else results_ae.to_dict("records")
        dct_list = results_dct if isinstance(results_dct, list) else results_dct.to_dict("records")

        dct_map = {
            tuple(r[k] for k in join_on): r for r in dct_list
        }

        merged_list = []
        for r_ae in ae_list:
            key = tuple(r_ae[k] for k in join_on)
            if key in dct_map:
                r_dct = dct_map[key]
                merged_row = {f"{k}_ae": v for k, v in r_ae.items()}
                for k, v in r_dct.items():
                    merged_row[f"{k}_dct"] = v
                for k in join_on:
                    merged_row[k] = r_ae[k]
                merged_list.append(merged_row)

        return merged_list
