"""
Script to create additional visualization plots for model evaluation:
- Residuals plot
- Actual vs Predicted plot
- Feature importance plot (for tree-based models)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Visualization
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ Visualization libraries not available")

# LightGBM for feature importance
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def load_model_and_data():
    """Load best model and test data."""
    models_dir = Path("models")
    processed_dir = Path("data/processed")

    # Load model
    model_path = models_dir / "best_model.pkl"
    features_path = models_dir / "best_model_features.json"
    config_path = models_dir / "best_model_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    with open(features_path, "r") as f:
        feature_names = json.load(f)

    with open(config_path, "r") as f:
        config = json.load(f)

    model_name = config.get("best_model", "Unknown")

    # Load test data
    test_path = processed_dir / "test_encoded.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    test_df = pd.read_csv(test_path)

    # Ensure we have the correct features (model might expect specific order)
    if "SalePrice" in test_df.columns:
        X_test = test_df[feature_names]
        y_test = test_df["SalePrice"]
    else:
        raise ValueError("SalePrice column not found in test data")

    return model, model_name, X_test, y_test, feature_names


def create_residuals_plot(model, X_test, y_test, model_name, save_path):
    """Create residuals plot and actual vs predicted plot."""
    if not VISUALIZATION_AVAILABLE:
        return

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate residuals
    residuals = y_test - y_pred

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Set style
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except:
        try:
            plt.style.use("seaborn-darkgrid")
        except:
            plt.style.use("default")

    # 1. Residuals Plot
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.6, s=30, color="#3498db")
    ax1.axhline(
        y=0, color="red", linestyle="--", linewidth=2, label="Perfect Prediction"
    )
    ax1.set_xlabel("Predicted Values (log scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Residuals (Actual - Predicted)", fontsize=11, fontweight="bold")
    ax1.set_title(
        f"{model_name} Residuals Plot\n(Homogeneity Check)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add annotations
    rmse = np.sqrt(np.mean(residuals**2))
    ax1.text(
        0.05,
        0.95,
        f"RMSE: {rmse:.4f}",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # 2. Actual vs Predicted Plot
    ax2 = axes[1]
    ax2.scatter(y_test, y_pred, alpha=0.6, s=30, color="#2ecc71", zorder=2)

    # Perfect prediction line: y = x (diagonal line)
    # Calculate the range that includes both actual and predicted values
    all_values = np.concatenate(
        [
            y_test.values if hasattr(y_test, "values") else y_test,
            y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred),
        ]
    )
    min_val = np.min(all_values)
    max_val = np.max(all_values)

    # Add a small margin for better visualization
    margin = (max_val - min_val) * 0.05
    line_min = min_val - margin
    line_max = max_val + margin

    # Draw the perfect prediction line (y = x) - this should be a 45-degree diagonal
    ax2.plot(
        [line_min, line_max],
        [line_min, line_max],
        "r--",
        linewidth=2,
        label="Perfect Prediction (y=x)",
        zorder=1,
    )

    # Set equal aspect ratio and same limits for both axes to ensure 45-degree line
    ax2.set_aspect("equal", adjustable="box")

    ax2.set_xlabel("Actual Values (log scale)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Predicted Values (log scale)", fontsize=11, fontweight="bold")
    ax2.set_title(
        f"{model_name} Actual vs Predicted\n(Prediction Accuracy)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Calculate R²
    from sklearn.metrics import r2_score

    r2 = r2_score(y_test, y_pred)
    ax2.text(
        0.05,
        0.95,
        f"R²: {r2:.4f}",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Residuals plot saved: {save_path}")


def create_feature_importance_plot(model, model_name, feature_names, save_path):
    """Create feature importance plot for tree-based models."""
    if not VISUALIZATION_AVAILABLE:
        return

    # Check if model supports feature_importances_
    if not hasattr(model, "feature_importances_"):
        print(f"⚠️ Model {model_name} does not support feature_importances_")
        return

    # Get feature importance
    importances = model.feature_importances_

    # Create DataFrame
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except:
        try:
            plt.style.use("seaborn-darkgrid")
        except:
            plt.style.use("default")

    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    bars = ax.barh(range(len(importance_df)), importance_df["importance"], color=colors)

    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df["feature"], fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{model_name} - Top 20 Feature Importance\n(Which features matter most?)",
        fontsize=12,
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importance_df["importance"])):
        ax.text(
            val + val * 0.01,
            i,
            f"{val:.2f}",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Feature importance plot saved: {save_path}")


def main():
    """Create all additional plots."""
    print("\n" + "=" * 60)
    print("CREATING ADDITIONAL VISUALIZATION PLOTS")
    print("=" * 60)

    if not VISUALIZATION_AVAILABLE:
        print("❌ Visualization libraries not available")
        return

    try:
        # Load model and data
        print("\n📊 Loading model and test data...")
        model, model_name, X_test, y_test, feature_names = load_model_and_data()

        print(f"   Model: {model_name}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {len(feature_names)}")

        models_dir = Path("models")

        # 1. Residuals and Actual vs Predicted plot
        print("\n📈 Creating residuals plot...")
        residuals_path = models_dir / "model_residuals.png"
        create_residuals_plot(model, X_test, y_test, model_name, residuals_path)

        # 2. Feature importance plot (if supported)
        print("\n📊 Creating feature importance plot...")
        importance_path = models_dir / "model_feature_importance.png"
        create_feature_importance_plot(
            model, model_name, feature_names, importance_path
        )

        print("\n✅ All additional plots created successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
