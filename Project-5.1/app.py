"""
HOUSE PRICE PREDICTION PIPELINE
================================

Main orchestration script for the entire ML workflow.

Workflow:
1. Preprocessing (fix logic, fill nulls, split 85/15)
2. Feature Engineering (create derived features)
3. Transformation (reduce skewness)
4. Encoding (categorical encoding + scaling)
5. Modeling (train & evaluate)

Data Flow:
  data/raw/
    └─ train-house-prices-advanced-regression-techniques.csv
  ↓
  [Preprocessing: BƯỚC 0-2]
    ├─ BƯỚC 0: Fix MasVnrType/Area logic (delete 2, fill 5)
    ├─ BƯỚC 1: Fill 6940 null values
    ├─ BƯỚC 2: Fix Garage logic (81 rows)
    └─ Result: 1458 × 81 clean data
  ↓
  [Split 85/15]
  ↓
  data/processed/
    ├─ train_data.csv
    ├─ test_data.csv
    ├─ train_fe.csv
    ├─ test_fe.csv
    ├─ train_transformed.csv
    ├─ test_transformed.csv
    ├─ train_encoded.csv     (ready for model)
    └─ test_encoded.csv
  ↓
  models/
    └─ best_model.pkl

Usage:
    python app.py --step all           # Run full pipeline
    python app.py --step preprocess    # Only preprocessing + split
    python app.py --step fe            # Up to Feature Engineering
    python app.py --step transform     # Up to Transformation
    python app.py --step encode        # Up to Encoding (ready for model)
    python app.py --step model         # Placeholder
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class Pipeline:
    """House Price Prediction Pipeline"""

    def __init__(
        self,
        raw_data_path="data/raw/train-house-prices-advanced-regression-techniques.csv",
    ):
        self.raw_data_path = Path(raw_data_path)
        self.processed_dir = Path("data/processed")
        self.interim_dir = Path("data/interim")
        self.models_dir = Path("models")

        # Create directories
        for d in [self.processed_dir, self.interim_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def run_preprocessing(self):
        """STEP 1: Preprocessing (includes split)"""
        print("\n" + "=" * 100)
        print("STEP 1: PREPROCESSING (Fix Logic + Fill Nulls + Split 85/15)")
        print("=" * 100)

        try:
            from src.Preprocessing import Preprocessor
            from sklearn.model_selection import train_test_split

            # Load raw data
            if not self.raw_data_path.exists():
                print(f"❌ Raw data not found: {self.raw_data_path}")
                return False

            df_raw = pd.read_csv(self.raw_data_path)
            print(f"\nLoaded raw data: {df_raw.shape}")

            # Run preprocessing pipeline
            preprocessor = Preprocessor(df_raw)
            df_clean = (
                preprocessor.step0_fix_masonry_veneer_logic()
                .step1_fill_missing_values()
                .step2_fix_garage_logic()
                .get_summary()
                .get_dataframe()
            )

            # Save cleaned data
            preprocessed_path = self.processed_dir / "train_preprocessed.csv"
            df_clean.to_csv(preprocessed_path, index=False)
            print(f"\n✓ Saved preprocessed data: {preprocessed_path}")
            print(f"  Shape: {df_clean.shape}")
            print(f"  Null values: {df_clean.isnull().sum().sum()}")

            # Split into train/test (85/15)
            print(f"\nSplitting into train/test (85/15)...")
            target = df_clean["SalePrice"]
            X = df_clean.drop("SalePrice", axis=1)

            X_train, X_test, y_train, y_test = train_test_split(
                X, target, test_size=0.15, random_state=42
            )

            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

            train_path = self.processed_dir / "train_data.csv"
            test_path = self.processed_dir / "test_data.csv"

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            print(f"  Train: {train_df.shape}")
            print(f"  Test: {test_df.shape}")
            print(f"\n✓ Preprocessing complete (with split)")

            return True

        except Exception as e:
            print(f"\n❌ Error in preprocessing: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_feature_engineering(self):
        """STEP 2: Feature Engineering"""
        print("\n" + "=" * 100)
        print("STEP 2: FEATURE ENGINEERING")
        print("=" * 100)

        try:
            from src.FeatureEngineering import FeatureEngineer

            train_path = self.processed_dir / "train_data.csv"
            test_path = self.processed_dir / "test_data.csv"

            if not train_path.exists() or not test_path.exists():
                print(
                    "❌ Preprocessed train/test data not found. Run preprocessing first."
                )
                return False

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            engineer_train = FeatureEngineer(train_df)
            train_engineered = (
                engineer_train.engineer_garage_features()
                .engineer_area_features()
                .engineer_basement_features()
                .engineer_age_features()
                .engineer_quality_features()
                .get_dataframe()
            )

            engineer_test = FeatureEngineer(test_df)
            test_engineered = (
                engineer_test.engineer_garage_features()
                .engineer_area_features()
                .engineer_basement_features()
                .engineer_age_features()
                .engineer_quality_features()
                .get_dataframe()
            )

            train_engineered.to_csv(self.processed_dir / "train_fe.csv", index=False)
            test_engineered.to_csv(self.processed_dir / "test_fe.csv", index=False)

            print(f"\n✓ Feature Engineering complete")
            print(f"  Train: {train_engineered.shape}")
            print(f"  Test: {test_engineered.shape}")
            return True

        except Exception as e:
            print(f"\n❌ Error in feature engineering: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_transformation(self):
        """STEP 3: Transformation"""
        print("\n" + "=" * 100)
        print("STEP 3: TRANSFORMATION (Skewness Reduction)")
        print("=" * 100)

        try:
            from src.Transformation import SkewnessTransformer

            train_path = self.processed_dir / "train_fe.csv"
            test_path = self.processed_dir / "test_fe.csv"

            if not train_path.exists() or not test_path.exists():
                print(
                    "❌ Feature engineered data not found. Run feature engineering first."
                )
                return False

            transformer = SkewnessTransformer(
                processed_dir=str(self.processed_dir), interim_dir=str(self.interim_dir)
            )
            transformer.run_pipeline(
                train_path=str(train_path), test_path=str(test_path)
            )

            print("\n✓ Transformation complete")
            return True

        except Exception as e:
            print(f"\n❌ Error in transformation: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_encoding(self):
        """STEP 4: Encoding"""
        print("\n" + "=" * 100)
        print("STEP 4: ENCODING (Categorical + Scaling)")
        print("=" * 100)

        try:
            from src.Encoding import SklearnEncodingPipeline

            train_path = self.processed_dir / "train_transformed.csv"
            test_path = self.processed_dir / "test_transformed.csv"

            if not train_path.exists() or not test_path.exists():
                print("❌ Transformed data not found. Run transformation first.")
                return False

            encoder = SklearnEncodingPipeline()
            train_encoded, test_encoded = encoder.fit_transform(
                train_path=str(train_path), test_path=str(test_path)
            )
            encoder.save(
                train_encoded,
                test_encoded,
                processed_dir=str(self.processed_dir),
                interim_dir=str(self.interim_dir),
            )

            print("\n✓ Encoding complete")
            print(f"  Train: {train_encoded.shape}")
            print(f"  Test: {test_encoded.shape}")
            return True

        except Exception as e:
            print(f"\n❌ Error in encoding: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_modeling(self):
        """STEP 5: Modeling - Train and evaluate models"""
        print("\n" + "=" * 100)
        print("STEP 5: MODELING - Train & Evaluate Models")
        print("=" * 100)

        try:
            from src.Modeling import ModelTrainer
            import joblib
            from sklearn.linear_model import Ridge, Lasso, ElasticNet

            # Load encoded data
            train_path = self.processed_dir / "train_encoded.csv"
            test_path = self.processed_dir / "test_encoded.csv"

            if not train_path.exists() or not test_path.exists():
                print("❌ Encoded data not found. Run encoding first.")
                return False

            print(f"\n📊 Loading encoded data...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print(f"  Train: {train_df.shape}")
            print(f"  Test: {test_df.shape}")

            # Separate features and target
            X_train = train_df.drop("SalePrice", axis=1)
            y_train = train_df["SalePrice"]
            X_test = test_df.drop("SalePrice", axis=1)
            y_test = test_df["SalePrice"]

            print(f"  Features: {X_train.shape[1]}")
            print(f"  Target: SalePrice (log-transformed)")

            # Initialize ModelTrainer
            print(f"\n🤖 Initializing ModelTrainer...")
            trainer = ModelTrainer(models_dir=str(self.models_dir))

            # Train all models
            print(f"\n🚀 Training all models...")
            results_df = trainer.train_all_models(X_train, y_train, X_test, y_test)

            # Save results
            print(f"\n💾 Saving results...")
            trainer.save_results(results_df)

            # Generate report
            print(f"\n📋 Generating report...")
            report = trainer.generate_report(results_df)

            # Save report
            report_path = Path("reports") / "ModelReport.md"
            report_path.parent.mkdir(exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

            print(f"\n✅ Modeling complete!")
            print(f"  Best model: {trainer.best_model_name}")
            print(f"  RMSE: {trainer.best_model_info['RMSE']:.4f}")
            print(f"  R²: {trainer.best_model_info['R²']:.4f}")
            print(f"  Results saved to: {self.models_dir}")
            print(f"  Report saved to: {report_path}")

            # Persist best model artifact for serving
            try:
                best_name = trainer.best_model_name
                best_params_raw = trainer.best_model_info.get("Best_Params", {})

                # Handle Best_Params which might be a string (from CSV) or dict
                import json
                import ast

                if isinstance(best_params_raw, str):
                    try:
                        # Safely parse string representation of dict
                        best_params = (
                            ast.literal_eval(best_params_raw)
                            if best_params_raw.startswith("{")
                            else {}
                        )
                    except:
                        best_params = {}
                else:
                    best_params = best_params_raw

                # Build comprehensive model mapping
                model_mapping = {}

                # Linear models
                model_mapping["Ridge"] = Ridge
                model_mapping["Lasso"] = Lasso
                model_mapping["ElasticNet"] = ElasticNet

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

                # Huber
                try:
                    from sklearn.linear_model import HuberRegressor

                    model_mapping["Huber"] = HuberRegressor
                except ImportError:
                    pass

                if best_name in model_mapping:
                    print(
                        f"\n💾 Re-fitting best model ({best_name}) with best params for persistence..."
                    )
                    BestModelClass = model_mapping[best_name]

                    # Set default params for tree-based models
                    if best_name in ["LightGBM", "XGBoost"]:
                        default_params = {
                            "random_state": 42,
                            "verbosity": 0,
                            "verbose": -1,
                        }
                        if best_name == "LightGBM":
                            default_params["verbose"] = -1
                        elif best_name == "XGBoost":
                            default_params["verbosity"] = 0
                        best_params = {**default_params, **best_params}
                    elif best_name in ["Ridge", "Lasso", "ElasticNet"]:
                        if "random_state" not in best_params:
                            best_params["random_state"] = 42

                    best_model = BestModelClass(**best_params)
                    best_model.fit(X_train, y_train)

                    model_path = self.models_dir / "best_model.pkl"
                    joblib.dump(best_model, model_path)

                    features_path = self.models_dir / "best_model_features.json"
                    with open(features_path, "w") as f:
                        json.dump(list(X_train.columns), f, indent=2)

                    print(f"  ✓ Saved best model ({best_name}) to: {model_path}")
                    print(f"  ✓ Saved feature names to: {features_path}")
                else:
                    print(
                        f"⚠️ Best model '{best_name}' is not in supported models list."
                    )
                    print(f"   Supported models: {', '.join(model_mapping.keys())}")
            except Exception as persist_err:
                import traceback

                print(f"\n⚠️ Failed to persist best model: {persist_err}")
                traceback.print_exc()

            return True

        except Exception as e:
            print(f"\n❌ Error in modeling: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_pipeline(self, steps=["preprocess", "fe", "transform", "encode", "model"]):
        """Run specified pipeline steps"""

        print("\n" + "=" * 100)
        print("HOUSE PRICE PREDICTION PIPELINE")
        print("=" * 100)
        print(f"\nConfiguration:")
        print(f"  Raw data: {self.raw_data_path}")
        print(f"  Processed dir: {self.processed_dir}")
        print(f"  Steps: {', '.join(steps)}")

        results = {}

        step_map = {
            "preprocess": self.run_preprocessing,
            "fe": self.run_feature_engineering,
            "transform": self.run_transformation,
            "encode": self.run_encoding,
            "model": self.run_modeling,
        }

        for step in steps:
            if step in step_map:
                results[step] = step_map[step]()
                if not results[step]:
                    print(f"\n⚠️  Pipeline stopped at step '{step}'")
                    break
            else:
                print(f"❌ Unknown step: {step}")

        # Summary
        print("\n" + "=" * 100)
        print("PIPELINE SUMMARY")
        print("=" * 100)

        for step, success in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {step.upper()}")

        all_success = all(results.values())
        if all_success:
            print(f"\n✅ Pipeline completed successfully!")
            print(f"\nOutput files in {self.processed_dir}:")
            for f in sorted(self.processed_dir.glob("*.csv")):
                print(f"  - {f.name}")

        return all_success


def main():
    # --- Argument parser: Lấy tham số command line ---
    parser = argparse.ArgumentParser(description="House Price Prediction Pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "preprocess", "fe", "transform", "encode", "model"],
        default="all",
        help="Pipeline step to run (chọn bước nào để chạy: all, preprocess, fe, transform, encode, model)",
    )
    parser.add_argument(
        "--raw-data",
        default="data/raw/train-house-prices-advanced-regression-techniques.csv",
        help="Path to raw data (đường dẫn đến file dữ liệu gốc)",
    )

    args = parser.parse_args()

    # --- Mapping bước command line sang các bước thực tế của pipeline ---
    # (VD: 'all' = chạy toàn bộ, 'fe' = preprocess + feature engineering, ...)
    step_map = {
        "all": ["preprocess", "fe", "transform", "encode", "model"],
        "preprocess": ["preprocess"],
        "fe": ["preprocess", "fe"],
        "transform": ["preprocess", "fe", "transform"],
        "encode": ["preprocess", "fe", "transform", "encode"],
        "model": ["preprocess", "fe", "transform", "encode", "model"],
    }

    # --- Khởi tạo và chạy pipeline ---
    pipeline = Pipeline(raw_data_path=args.raw_data)
    success = pipeline.run_pipeline(steps=step_map[args.step])

    # --- Exit code: 0 (thành công), 1 (thất bại) ---
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
