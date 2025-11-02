"""
Helper script to save the best model from existing model_comparison.csv
This is useful if you've already trained models but need to persist the best one.
"""

import pandas as pd
import json
import joblib
import ast
from pathlib import Path


def save_best_model_from_csv():
    """Load best model config from CSV and re-fit to save it"""

    models_dir = Path("models")
    csv_path = models_dir / "model_comparison.csv"

    if not csv_path.exists():
        print(f"❌ {csv_path} not found. Please run model training first.")
        return False

    # Load comparison data
    print("📊 Loading model comparison data...")
    results_df = pd.read_csv(csv_path)
    results_df = results_df.sort_values("RMSE")

    best_row = results_df.iloc[0]
    best_name = best_row["Model"]
    best_params_raw = best_row["Best_Params"]

    print(f"🎯 Best model: {best_name}")
    print(f"   RMSE: {best_row['RMSE']:.4f}")
    print(f"   R²: {best_row['R²']:.4f}")

    # Parse parameters
    if isinstance(best_params_raw, str):
        try:
            best_params = ast.literal_eval(best_params_raw)
        except:
            print(f"⚠️ Could not parse parameters: {best_params_raw}")
            return False
    else:
        best_params = best_params_raw

    # Load training data
    processed_dir = Path("data/processed")
    train_path = processed_dir / "train_encoded.csv"

    if not train_path.exists():
        print(f"❌ Training data not found: {train_path}")
        print("   Please run the encoding step first.")
        return False

    print(f"\n📊 Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop("SalePrice", axis=1)
    y_train = train_df["SalePrice"]

    print(f"   Features: {X_train.shape[1]}, Samples: {X_train.shape[0]}")

    # Build model mapping
    model_mapping = {}

    # Linear models
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor

    model_mapping["Ridge"] = Ridge
    model_mapping["Lasso"] = Lasso
    model_mapping["ElasticNet"] = ElasticNet
    model_mapping["Huber"] = HuberRegressor

    # Tree-based models
    try:
        import lightgbm as lgb

        model_mapping["LightGBM"] = lgb.LGBMRegressor
    except ImportError:
        pass

    try:
        import xgboost as xgb

        model_mapping["XGBoost"] = xgb.XGBRegressor
    except ImportError:
        pass

    if best_name not in model_mapping:
        print(f"\n❌ Model '{best_name}' is not supported.")
        print(f"   Supported models: {', '.join(model_mapping.keys())}")
        return False

    # Create and train model
    print(f"\n🔧 Creating {best_name} model with best parameters...")
    BestModelClass = model_mapping[best_name]

    # Set default params
    if best_name in ["LightGBM", "XGBoost"]:
        default_params = {"random_state": 42, "verbosity": 0, "verbose": -1}
        if best_name == "LightGBM":
            default_params["verbose"] = -1
        elif best_name == "XGBoost":
            default_params["verbosity"] = 0
        best_params = {**default_params, **best_params}
    elif best_name in ["Ridge", "Lasso", "ElasticNet"]:
        if "random_state" not in best_params:
            best_params["random_state"] = 42

    print(f"   Parameters: {best_params}")

    model = BestModelClass(**best_params)
    print("   Training model...")
    model.fit(X_train, y_train)

    # Save model
    model_path = models_dir / "best_model.pkl"
    joblib.dump(model, model_path)
    print(f"   ✓ Model saved to: {model_path}")

    # Save features
    features_path = models_dir / "best_model_features.json"
    with open(features_path, "w") as f:
        json.dump(list(X_train.columns), f, indent=2)
    print(f"   ✓ Features saved to: {features_path}")

    print(f"\n✅ Successfully saved best model ({best_name})!")
    return True


if __name__ == "__main__":
    save_best_model_from_csv()
