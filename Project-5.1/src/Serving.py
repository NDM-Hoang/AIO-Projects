"""
SERVING UTILITIES
=================

Helpers to load artifacts, prepare single-record inputs, run the full
processing pipeline, and generate predictions using the persisted best model.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
MODELS_DIR = PROJECT_ROOT / "models"


def load_artifacts() -> Tuple[Any, List[str], Dict[str, Any], Dict[str, Any]]:
    """Load persisted model, feature names, configs, and default values.

    Returns:
        model: trained estimator loaded from models/best_model.pkl
        feature_names: ordered list of feature columns used for training
        transform_config: dict of transformation params
        defaults: Series of default values per raw column (from train_data)
    """
    model_path = MODELS_DIR / "best_model.pkl"
    features_path = MODELS_DIR / "best_model_features.json"
    transform_cfg_path = DATA_INTERIM / "transformation_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features list: {features_path}")

    model = joblib.load(model_path)
    with open(features_path, "r") as f:
        feature_names = json.load(f)

    transform_config = {}
    if transform_cfg_path.exists():
        with open(transform_cfg_path, "r") as f:
            transform_config = json.load(f)

    # Defaults from training raw split
    train_raw_path = DATA_PROCESSED / "train_data.csv"
    if not train_raw_path.exists():
        raise FileNotFoundError("Missing processed train_data.csv. Run pipeline first.")
    train_df = pd.read_csv(train_raw_path)
    defaults_series = train_df.drop(columns=["SalePrice"]).median(numeric_only=True)
    defaults: Dict[str, Any] = {}
    if hasattr(defaults_series, "to_dict"):
        defaults = defaults_series.to_dict()
    else:
        # Fallback if defaults_series is not a Series
        for col in train_df.select_dtypes(include=[np.number]).columns:
            if col != "SalePrice":
                defaults[col] = train_df[col].median()
    # For categorical defaults, use mode
    for col in train_df.select_dtypes(include=["object"]).columns:
        if col == "SalePrice":
            continue
        defaults[col] = (
            train_df[col].mode().iloc[0] if not train_df[col].mode().empty else "None"
        )

    return model, feature_names, transform_config, defaults


def prepare_single_record(
    raw_input: Dict[str, Any], defaults: Dict[str, Any]
) -> pd.DataFrame:
    """Build a one-row raw dataframe (close to original columns) with defaults.

    Any missing keys are filled from defaults. Extra keys are ignored.
    """
    # Use columns from train_data for consistency
    train_raw_path = DATA_PROCESSED / "train_data.csv"
    train_df = pd.read_csv(train_raw_path)
    raw_cols = [c for c in train_df.columns if c != "SalePrice"]

    filled = {}
    for col in raw_cols:
        if col in raw_input and raw_input[col] is not None:
            filled[col] = raw_input[col]
        else:
            # Use appropriate default based on column type
            if col in train_df.select_dtypes(include=["object"]).columns:
                filled[col] = defaults.get(col, "None")
            else:
                filled[col] = defaults.get(col, 0)

    df_one = pd.DataFrame([filled], columns=raw_cols)
    return df_one


def run_full_processing(df_raw_one_row: pd.DataFrame) -> pd.DataFrame:
    """Apply FE → Transformation → Encoding to match training schema.

    Note: Preprocessing was applied before split in the project pipeline;
    at serving time, we assume inputs are already clean or resemble processed
    train_data columns. If needed, minimal guards can be added here.
    """
    from src.FeatureEngineering import FeatureEngineer
    from src.Transformation import SkewnessTransformer
    from src.Encoding import SklearnEncodingPipeline

    # Start from a copy
    df = df_raw_one_row.copy()

    # Ensure all categorical columns have proper string values
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if col in df.columns:
            # Replace any non-string values with "None"
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(["nan", "None", "null"], "None")

    # We need a dummy target to satisfy downstream code; use median SalePrice
    # It will be dropped by scaler split inside Encoding
    train_raw_path = DATA_PROCESSED / "train_data.csv"
    train_df = pd.read_csv(train_raw_path)
    df["SalePrice"] = float(train_df["SalePrice"].median())

    # 1) Feature Engineering
    fe = FeatureEngineer(df)
    df_fe = (
        fe.engineer_garage_features()
        .engineer_area_features()
        .engineer_basement_features()
        .engineer_age_features()
        .engineer_quality_features()
        .get_dataframe()
    )

    # Save temporary single-row CSV for downstream modules expecting file IO
    tmp_test_fe = DATA_PROCESSED / "_tmp_test_fe.csv"
    df_fe.to_csv(tmp_test_fe, index=False)

    # 2) Transformation
    transformer = SkewnessTransformer(
        processed_dir=str(DATA_PROCESSED), interim_dir=str(DATA_INTERIM)
    )
    # Fit on full training FE data, apply to single-row test FE
    train_fe_path = DATA_PROCESSED / "train_fe.csv"
    if not train_fe_path.exists():
        raise FileNotFoundError(
            "Missing train_fe.csv. Run pipeline first (preprocess→fe→transform→encode)."
        )
    transformer.run_pipeline(train_path=str(train_fe_path), test_path=str(tmp_test_fe))

    tmp_test_trans = DATA_PROCESSED / "test_transformed.csv"

    # 3) Encoding (updated to use SklearnEncodingPipeline)
    train_transformed_path = DATA_PROCESSED / "train_transformed.csv"
    if not train_transformed_path.exists():
        raise FileNotFoundError(
            "Missing train_transformed.csv. Run pipeline first (transform step)."
        )

    enc = SklearnEncodingPipeline()
    train_encoded_df, test_encoded_df = enc.fit_transform(
        train_path=str(train_transformed_path),
        test_path=str(tmp_test_trans),
    )

    # Persist outputs to expected locations so downstream code can read them
    enc.save(
        train_encoded=train_encoded_df,
        test_encoded=test_encoded_df,
        processed_dir=str(DATA_PROCESSED),
        interim_dir=str(DATA_INTERIM),
        train_output="train_encoded.csv",
        test_output="test_encoded.csv",
        config_output="encoding_config.json",
    )

    # Load encoded single-row from test output
    encoded = pd.read_csv(DATA_PROCESSED / "test_encoded.csv")
    return encoded


def predict_single(input_dict: Dict) -> Dict:
    """Run end-to-end single-record prediction.

    Returns:
        dict with fields: prediction_log, prediction, currency
    """
    model, feature_names, transform_config, defaults = load_artifacts()
    df_raw = prepare_single_record(input_dict, defaults)
    df_encoded = run_full_processing(df_raw)

    # Align to training feature order and drop target
    X = df_encoded.drop(columns=["SalePrice"], errors="ignore")
    # Add any missing columns with zeros
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    # Ensure correct column order
    X = X[feature_names]

    y_log = model.predict(X)[0]

    # Inverse-transform if target was log1p
    saleprice = float(np.expm1(y_log))

    return {
        "prediction_log": float(y_log),
        "prediction": saleprice,
        "currency": "USD",
    }
