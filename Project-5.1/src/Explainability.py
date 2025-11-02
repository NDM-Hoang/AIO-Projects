"""
EXPLAINABILITY MODULE
=====================

Provide Explainable AI (XAI) capabilities using SHAP values to understand
how models make predictions for house price estimation.

Features:
- Global feature importance (overall model behavior)
- Local explanations (individual prediction explanations)
- Feature impact visualization
- Support for linear models (Lasso, Ridge, ElasticNet) and tree models (XGBoost, LightGBM)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

warnings.filterwarnings("ignore")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Install with: pip install shap")


class ModelExplainer:
    """
    Explainability wrapper for house price prediction models using SHAP.

    Supports both linear models (Lasso, Ridge, ElasticNet) and tree-based models
    (XGBoost, LightGBM) with appropriate SHAP explainers.
    """

    def __init__(
        self,
        model,
        model_name: str,
        X_train: pd.DataFrame,
        feature_names: List[str] = None,
    ):
        """
        Initialize ModelExplainer.

        Args:
            model: Trained model object
            model_name: Name of the model (e.g., 'Lasso', 'XGBoost')
            X_train: Training features (used as background data)
            feature_names: Optional list of feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")

        self.model = model
        self.model_name = model_name
        self.X_train = X_train
        self.feature_names = feature_names if feature_names else list(X_train.columns)

        # Determine model type and initialize appropriate explainer
        self._initialize_explainer()

        print(f"✅ ModelExplainer initialized for {model_name}")
        print(f"   Explainer type: {type(self.explainer).__name__}")
        print(f"   Background samples: {len(X_train)}")

    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        model_type = type(self.model).__name__

        # Check if it's a tree-based model
        if model_type in ["LGBMRegressor", "XGBRegressor"]:
            # Use TreeExplainer for tree models (fast and exact)
            self.explainer = shap.TreeExplainer(self.model)
            self.explainer_type = "tree"

        # Check if it's a linear model
        elif model_type in ["Lasso", "Ridge", "ElasticNet"]:
            # Use LinearExplainer for linear models (exact and fast)
            # For linear models, we can use a sample of training data
            background_data = (
                self.X_train.sample(min(100, len(self.X_train)), random_state=42)
                if len(self.X_train) > 100
                else self.X_train
            )
            self.explainer = shap.LinearExplainer(
                self.model,
                background_data,
                feature_perturbation="correlation_dependent",
            )
            self.explainer_type = "linear"
        else:
            # Fallback to KernelExplainer for unknown model types (slower but works)
            print(
                f"⚠️ Unknown model type {model_type}, using KernelExplainer (may be slow)"
            )
            background_data = (
                self.X_train.sample(min(50, len(self.X_train)), random_state=42)
                if len(self.X_train) > 50
                else self.X_train
            )
            self.explainer = shap.KernelExplainer(self.model.predict, background_data)
            self.explainer_type = "kernel"

    def get_global_feature_importance(self, max_features: int = 20) -> pd.DataFrame:
        """
        Get global feature importance using SHAP values.

        Args:
            max_features: Maximum number of top features to return

        Returns:
            DataFrame with feature names and their mean absolute SHAP values
        """
        print(f"\n📊 Computing global feature importance...")

        # Calculate SHAP values for a sample of training data
        sample_size = min(100, len(self.X_train))
        X_sample = self.X_train.sample(sample_size, random_state=42)

        shap_values = self.explainer.shap_values(X_sample)

        # Handle multi-dimensional output (shouldn't happen for regression, but just in case)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create DataFrame
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names[: len(mean_abs_shap)],
                "importance": mean_abs_shap,
                "importance_abs": np.abs(mean_abs_shap),
            }
        )

        # Sort by absolute importance
        importance_df = importance_df.sort_values("importance_abs", ascending=False)

        # Return top features
        return importance_df.head(max_features)

    def explain_prediction(self, X_instance: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP values.

        Args:
            X_instance: Single row DataFrame with features aligned to training data

        Returns:
            Dictionary containing:
            - shap_values: SHAP values for each feature
            - base_value: Base value (model output when all features are at baseline)
            - prediction: Model prediction
            - feature_contributions: Features sorted by contribution magnitude
        """
        print(f"\n🔍 Explaining prediction...")

        # Calculate SHAP values for this instance
        shap_values = self.explainer.shap_values(X_instance)

        # Handle different output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]  # Get first row if multiple

        # Get base value
        if hasattr(self.explainer, "expected_value"):
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
        else:
            base_value = float(
                self.model.predict(self.X_train.mean().values.reshape(1, -1))[0]
            )

        # Get prediction
        prediction = float(self.model.predict(X_instance)[0])

        # Create feature contributions DataFrame
        contributions = pd.DataFrame(
            {
                "feature": self.feature_names[: len(shap_values)],
                "shap_value": shap_values,
                "abs_contribution": np.abs(shap_values),
            }
        )

        # Sort by absolute contribution
        contributions = contributions.sort_values("abs_contribution", ascending=False)

        return {
            "shap_values": (
                shap_values.tolist()
                if isinstance(shap_values, np.ndarray)
                else shap_values
            ),
            "base_value": float(base_value),
            "prediction": prediction,
            "feature_contributions": contributions.to_dict("records"),
            "feature_names": self.feature_names[: len(shap_values)],
        }

    def get_summary_plot_data(
        self, max_display: int = 20, sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Get data for summary plot visualization.

        Args:
            max_display: Maximum number of features to display
            sample_size: Number of samples to use for SHAP value calculation

        Returns:
            Dictionary with data for visualization
        """
        print(f"\n📊 Computing summary plot data...")

        # Sample training data
        X_sample = self.X_train.sample(
            min(sample_size, len(self.X_train)), random_state=42
        )

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Get feature importance for top features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-max_display:][::-1]

        # Select top features
        top_shap = shap_values[:, top_indices]
        top_features = [self.feature_names[i] for i in top_indices]
        top_X = X_sample.iloc[:, top_indices]

        return {
            "shap_values": top_shap.tolist(),
            "feature_values": top_X.values.tolist(),
            "feature_names": top_features,
        }

    def get_waterfall_plot_data(
        self, X_instance: pd.DataFrame, max_features: int = 15
    ) -> Dict[str, Any]:
        """
        Get data for waterfall plot (explains one prediction step-by-step).

        Args:
            X_instance: Single row DataFrame
            max_features: Maximum number of features to show

        Returns:
            Dictionary with waterfall plot data
        """
        explanation = self.explain_prediction(X_instance)

        # Get top contributing features
        contributions = pd.DataFrame(explanation["feature_contributions"])
        top_contributions = contributions.head(max_features)

        # Sort by SHAP value (for waterfall display)
        top_contributions = top_contributions.sort_values("shap_value")

        return {
            "base_value": explanation["base_value"],
            "prediction": explanation["prediction"],
            "features": top_contributions["feature"].tolist(),
            "shap_values": top_contributions["shap_value"].tolist(),
            "feature_values": (
                X_instance[top_contributions["feature"]].values[0].tolist()
                if len(X_instance) > 0
                else []
            ),
        }


def load_explainer_from_artifacts(
    models_dir: str = "models", processed_dir: str = "data/processed"
) -> ModelExplainer:
    """
    Load model and training data to create a ModelExplainer.

    Args:
        models_dir: Directory containing model artifacts
        processed_dir: Directory containing processed training data

    Returns:
        ModelExplainer instance
    """
    models_path = Path(models_dir)
    processed_path = Path(processed_dir)

    # Load model
    model_path = models_path / "best_model.pkl"
    features_path = models_path / "best_model_features.json"
    config_path = models_path / "best_model_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    # Load feature names
    with open(features_path, "r") as f:
        feature_names = json.load(f)

    # Load model config to get model name
    with open(config_path, "r") as f:
        config = json.load(f)
    model_name = config.get("best_model", "Unknown")

    # Load training data
    train_encoded_path = processed_path / "train_encoded.csv"
    if not train_encoded_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_encoded_path}")

    train_df = pd.read_csv(train_encoded_path)
    X_train = train_df[feature_names]

    return ModelExplainer(model, model_name, X_train, feature_names)


def main():
    """Example usage of ModelExplainer."""
    if not SHAP_AVAILABLE:
        print("⚠️ SHAP not available. Install with: pip install shap")
        return

    try:
        explainer = load_explainer_from_artifacts()

        # Get global feature importance
        importance = explainer.get_global_feature_importance()
        print("\n📊 Top 10 Most Important Features:")
        print(importance.head(10))

        print("\n✅ ModelExplainer module loaded successfully")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
