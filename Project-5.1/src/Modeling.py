"""
MODELING MODULE
===============

Production-ready model training, hyperparameter optimization, and evaluation
for house price prediction using both linear and tree-based models.

Features:
- Ridge, Lasso, ElasticNet with regularization
- LightGBM, XGBoost with ensemble methods
- 5-Fold Cross-Validation
- Comprehensive evaluation metrics
- Model persistence and configuration saving
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Machine Learning
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# Visualization
try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Tree-based models (optional)
try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class ModelTrainer:
    """
    Comprehensive model training and evaluation class for house price prediction.

    Supports both linear regularization models and tree-based ensemble methods
    with hyperparameter optimization and cross-validation.
    """

    def __init__(self, models_dir: str = "models", random_state: int = 42):
        """
        Initialize ModelTrainer.

        Args:
            models_dir: Directory to save models and configurations
            random_state: Random state for reproducibility
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Initialize results storage
        self.results = []
        self.best_model = None
        self.best_model_name = None

        print(f"✅ ModelTrainer initialized")
        print(f"   Models directory: {self.models_dir}")
        print(f"   LightGBM available: {LIGHTGBM_AVAILABLE}")
        print(f"   XGBoost available: {XGBOOST_AVAILABLE}")

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"
    ) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "Model": model_name,
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2,
            "RMSLE": rmse,  # Since target is log-transformed
        }

    def train_linear_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> List[Dict[str, Any]]:
        """
        Train linear models with regularization (Ridge, Lasso, ElasticNet).

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            List of model results
        """
        print("\n" + "=" * 60)
        print("TRAINING LINEAR MODELS WITH REGULARIZATION")
        print("=" * 60)

        results = []

        # Define hyperparameter grids
        ridge_params = {"alpha": [0.001, 0.01, 0.1, 1, 10, 50, 100]}
        lasso_params = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]}
        elasticnet_params = {
            "alpha": [0.001, 0.01, 0.1, 1],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
        huber_params = {
            "epsilon": [1.2, 1.35, 1.5, 1.75, 2.0],
            "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
            "fit_intercept": [True],
        }

        # Initialize models
        models = {
            "Ridge": Ridge(random_state=self.random_state),
            "Lasso": Lasso(random_state=self.random_state, max_iter=2000),
            "ElasticNet": ElasticNet(random_state=self.random_state, max_iter=2000),
            "Huber": HuberRegressor(max_iter=1000),
        }

        param_grids = {
            "Ridge": ridge_params,
            "Lasso": lasso_params,
            "ElasticNet": elasticnet_params,
            "Huber": huber_params,
        }

        # Train each model
        for model_name, model in models.items():
            print(f"\n🔧 Training {model_name}...")

            # Grid search with cross-validation
            search = GridSearchCV(
                model,
                param_grids[model_name],
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=0,
            )

            search.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = search.predict(X_test)
            metrics = self.calculate_metrics(y_test, y_pred, model_name)
            metrics.update(
                {"Best_Params": search.best_params_, "CV_Score": -search.best_score_}
            )

            results.append(metrics)

            print(f"   Best params: {search.best_params_}")
            print(f"   CV Score: {-search.best_score_:.6f}")
            print(f"   Test RMSE: {metrics['RMSE']:.4f}")
            print(f"   Test R²: {metrics['R²']:.4f}")

        return results

    def train_tree_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> List[Dict[str, Any]]:
        """
        Train tree-based models (LightGBM, XGBoost).

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            List of model results
        """
        print("\n" + "=" * 60)
        print("TRAINING TREE-BASED MODELS")
        print("=" * 60)

        results = []

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            print("\n🌳 Training LightGBM...")

            lgb_params = {
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "num_leaves": [31, 50, 100, 200],
                "max_depth": [3, 5, 7, 10],
                "min_child_samples": [20, 50, 100],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            }

            lgb_model = lgb.LGBMRegressor(random_state=self.random_state, verbose=-1)

            search = RandomizedSearchCV(
                lgb_model,
                lgb_params,
                n_iter=30,
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0,
            )

            search.fit(X_train, y_train)

            y_pred = search.predict(X_test)
            metrics = self.calculate_metrics(y_test, y_pred, "LightGBM")
            metrics.update(
                {"Best_Params": search.best_params_, "CV_Score": -search.best_score_}
            )

            results.append(metrics)

            print(f"   Best params: {search.best_params_}")
            print(f"   CV Score: {-search.best_score_:.6f}")
            print(f"   Test RMSE: {metrics['RMSE']:.4f}")
            print(f"   Test R²: {metrics['R²']:.4f}")
        else:
            print("⚠️ LightGBM not available")

        # XGBoost
        if XGBOOST_AVAILABLE:
            print("\n🌳 Training XGBoost...")

            xgb_params = {
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 10],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "gamma": [0, 0.1, 0.2],
            }

            xgb_model = xgb.XGBRegressor(random_state=self.random_state, verbosity=0)

            search = RandomizedSearchCV(
                xgb_model,
                xgb_params,
                n_iter=30,
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0,
            )

            search.fit(X_train, y_train)

            y_pred = search.predict(X_test)
            metrics = self.calculate_metrics(y_test, y_pred, "XGBoost")
            metrics.update(
                {"Best_Params": search.best_params_, "CV_Score": -search.best_score_}
            )

            results.append(metrics)

            print(f"   Best params: {search.best_params_}")
            print(f"   CV Score: {-search.best_score_:.6f}")
            print(f"   Test RMSE: {metrics['RMSE']:.4f}")
            print(f"   Test R²: {metrics['R²']:.4f}")
        else:
            print("⚠️ XGBoost not available")

        return results

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """
        Train all models and return comprehensive results.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            DataFrame with all model results
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE MODEL TRAINING & EVALUATION")
        print("=" * 80)

        # Train linear models
        linear_results = self.train_linear_models(X_train, y_train, X_test, y_test)

        # Train tree-based models
        tree_results = self.train_tree_models(X_train, y_train, X_test, y_test)

        # Combine all results
        all_results = linear_results + tree_results
        self.results = all_results

        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values("RMSE").reset_index(drop=True)

        # Identify best model
        self.best_model_name = results_df.iloc[0]["Model"]
        self.best_model_info = results_df.iloc[0]

        print("\n" + "=" * 60)
        print("FINAL MODEL COMPARISON")
        print("=" * 60)
        print(results_df[["Model", "RMSE", "MAE", "R²", "CV_Score"]].round(4))

        print(f"\n🥇 BEST MODEL: {self.best_model_name}")
        print(f"   RMSE: {self.best_model_info['RMSE']:.4f}")
        print(f"   R²: {self.best_model_info['R²']:.4f}")

        return results_df

    def save_model(self, model, model_name: str, feature_names: List[str]) -> None:
        """
        Save trained model and configuration.

        Args:
            model: Trained model object
            model_name: Name of the model
            feature_names: List of feature names
        """
        # Save model
        model_path = self.models_dir / f"{model_name.lower()}_model.pkl"
        joblib.dump(model, model_path)

        # Save feature names
        features_path = self.models_dir / f"{model_name.lower()}_features.json"
        with open(features_path, "w") as f:
            json.dump(feature_names, f, indent=2)

        print(f"💾 Model saved: {model_path}")
        print(f"💾 Features saved: {features_path}")

    def save_results(self, results_df: pd.DataFrame) -> None:
        """
        Save model comparison results and best model configuration.

        Args:
            results_df: DataFrame with model results
        """
        # Save model comparison
        results_path = self.models_dir / "model_comparison.csv"
        results_df.to_csv(results_path, index=False)

        # Save best model configuration
        best_config = {
            "best_model": self.best_model_name,
            "best_params": self.best_model_info["Best_Params"],
            "performance": {
                "RMSE": float(self.best_model_info["RMSE"]),
                "MAE": float(self.best_model_info["MAE"]),
                "R²": float(self.best_model_info["R²"]),
                "CV_Score": float(self.best_model_info["CV_Score"]),
            },
            "feature_count": len(results_df.columns) - 5,  # Exclude metric columns
            "training_samples": len(
                results_df
            ),  # This will be updated with actual data
            "test_samples": len(results_df),  # This will be updated with actual data
        }

        config_path = self.models_dir / "best_model_config.json"
        with open(config_path, "w") as f:
            json.dump(best_config, f, indent=2)

        print(f"💾 Results saved: {results_path}")
        print(f"💾 Config saved: {config_path}")

        # Generate visualization plots
        self.create_comparison_plots(results_df)

    def create_comparison_plots(self, results_df: pd.DataFrame) -> None:
        """
        Create comprehensive visualization plots for model comparison.

        Args:
            results_df: DataFrame with model results
        """
        if not VISUALIZATION_AVAILABLE:
            print("⚠️ Visualization libraries not available. Skipping plots.")
            return

        # Sort by RMSE for better visualization
        df_sorted = results_df.sort_values("RMSE").copy()

        # Set style (try modern seaborn style, fallback to default)
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except:
            try:
                plt.style.use("seaborn-darkgrid")
            except:
                plt.style.use("default")
        sns.set_palette("husl")

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # 1. RMSE Comparison (Bar Chart)
        ax1 = plt.subplot(2, 3, 1)
        colors = [
            "#2ecc71" if model == self.best_model_name else "#3498db"
            for model in df_sorted["Model"]
        ]
        bars1 = ax1.barh(df_sorted["Model"], df_sorted["RMSE"], color=colors)
        ax1.set_xlabel("RMSE (Lower is Better)", fontsize=10, fontweight="bold")
        ax1.set_title("Model RMSE Comparison", fontsize=12, fontweight="bold")
        ax1.invert_yaxis()
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars1, df_sorted["RMSE"])):
            ax1.text(
                val + 0.001, i, f"{val:.4f}", va="center", fontsize=9, fontweight="bold"
            )
        ax1.grid(axis="x", alpha=0.3)

        # 2. R² Comparison (Bar Chart)
        ax2 = plt.subplot(2, 3, 2)
        bars2 = ax2.barh(df_sorted["Model"], df_sorted["R²"], color=colors)
        ax2.set_xlabel("R² Score (Higher is Better)", fontsize=10, fontweight="bold")
        ax2.set_title("Model R² Comparison", fontsize=12, fontweight="bold")
        ax2.invert_yaxis()
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, df_sorted["R²"])):
            ax2.text(
                val + 0.005, i, f"{val:.4f}", va="center", fontsize=9, fontweight="bold"
            )
        ax2.grid(axis="x", alpha=0.3)

        # 3. MAE Comparison (Bar Chart)
        ax3 = plt.subplot(2, 3, 3)
        bars3 = ax3.barh(df_sorted["Model"], df_sorted["MAE"], color=colors)
        ax3.set_xlabel("MAE (Lower is Better)", fontsize=10, fontweight="bold")
        ax3.set_title("Model MAE Comparison", fontsize=12, fontweight="bold")
        ax3.invert_yaxis()
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars3, df_sorted["MAE"])):
            ax3.text(
                val + 0.001, i, f"{val:.4f}", va="center", fontsize=9, fontweight="bold"
            )
        ax3.grid(axis="x", alpha=0.3)

        # 4. CV Score Comparison (Bar Chart)
        ax4 = plt.subplot(2, 3, 4)
        bars4 = ax4.barh(df_sorted["Model"], df_sorted["CV_Score"], color=colors)
        ax4.set_xlabel("CV Score (Lower is Better)", fontsize=10, fontweight="bold")
        ax4.set_title(
            "Cross-Validation Score Comparison", fontsize=12, fontweight="bold"
        )
        ax4.invert_yaxis()
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars4, df_sorted["CV_Score"])):
            ax4.text(
                val + 0.0005,
                i,
                f"{val:.6f}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
        ax4.grid(axis="x", alpha=0.3)

        # 5. Comprehensive Metrics Comparison (Grouped Bar Chart)
        ax5 = plt.subplot(2, 3, 5)
        metrics = ["RMSE", "MAE", "R²", "CV_Score"]
        x = np.arange(len(df_sorted["Model"]))
        width = 0.2

        # Normalize metrics for comparison (R² is already 0-1, normalize others)
        rmse_norm = df_sorted["RMSE"] / df_sorted["RMSE"].max()
        mae_norm = df_sorted["MAE"] / df_sorted["MAE"].max()
        r2_norm = df_sorted["R²"]  # Already normalized
        cv_norm = df_sorted["CV_Score"] / df_sorted["CV_Score"].max()

        ax5.bar(x - 1.5 * width, rmse_norm, width, label="RMSE (norm)", alpha=0.8)
        ax5.bar(x - 0.5 * width, mae_norm, width, label="MAE (norm)", alpha=0.8)
        ax5.bar(x + 0.5 * width, r2_norm, width, label="R²", alpha=0.8)
        ax5.bar(x + 1.5 * width, cv_norm, width, label="CV Score (norm)", alpha=0.8)

        ax5.set_xlabel("Models", fontsize=10, fontweight="bold")
        ax5.set_ylabel("Normalized Score", fontsize=10, fontweight="bold")
        ax5.set_title(
            "All Metrics Comparison (Normalized)", fontsize=12, fontweight="bold"
        )
        ax5.set_xticks(x)
        ax5.set_xticklabels(df_sorted["Model"], rotation=45, ha="right")
        ax5.legend(loc="upper left", fontsize=8)
        ax5.grid(axis="y", alpha=0.3)

        # 6. Ranking Visualization (Heatmap-style)
        ax6 = plt.subplot(2, 3, 6)
        # Calculate ranks (1 is best)
        rank_df = pd.DataFrame(
            {
                "Model": df_sorted["Model"],
                "RMSE_Rank": df_sorted["RMSE"].rank(),
                "MAE_Rank": df_sorted["MAE"].rank(),
                "R²_Rank": df_sorted["R²"].rank(ascending=False),  # Higher is better
                "CV_Rank": df_sorted["CV_Score"].rank(),
            }
        )
        rank_df["Avg_Rank"] = rank_df[
            ["RMSE_Rank", "MAE_Rank", "R²_Rank", "CV_Rank"]
        ].mean(axis=1)
        rank_df = rank_df.sort_values("Avg_Rank")

        rank_matrix = rank_df[["RMSE_Rank", "MAE_Rank", "R²_Rank", "CV_Rank"]].values
        im = ax6.imshow(
            rank_matrix, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=len(df_sorted)
        )
        ax6.set_xticks(range(4))
        ax6.set_xticklabels(["RMSE", "MAE", "R²", "CV"], fontsize=9)
        ax6.set_yticks(range(len(rank_df)))
        ax6.set_yticklabels(rank_df["Model"], fontsize=9)
        ax6.set_title(
            "Model Ranking Heatmap\n(Lower rank = Better)",
            fontsize=12,
            fontweight="bold",
        )

        # Add text annotations
        for i in range(len(rank_df)):
            for j in range(4):
                text = ax6.text(
                    j,
                    i,
                    f"{rank_matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                )

        plt.colorbar(im, ax=ax6, label="Rank (1=Best)")

        # Add overall title
        fig.suptitle(
            "Model Performance Comparison Dashboard",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])

        # Save plot
        plot_path = self.models_dir / "model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Comparison plots saved: {plot_path}")

        # Also create a simple summary plot
        fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: RMSE and R² side by side
        ax1_sum = axes[0]
        x_pos = np.arange(len(df_sorted))
        width = 0.35

        bars1_sum = ax1_sum.bar(
            x_pos - width / 2,
            df_sorted["RMSE"],
            width,
            label="RMSE",
            color="#e74c3c",
            alpha=0.8,
        )
        ax1_sum_twin = ax1_sum.twinx()
        bars2_sum = ax1_sum_twin.bar(
            x_pos + width / 2,
            df_sorted["R²"],
            width,
            label="R²",
            color="#27ae60",
            alpha=0.8,
        )

        ax1_sum.set_xlabel("Models", fontsize=11, fontweight="bold")
        ax1_sum.set_ylabel("RMSE", fontsize=11, fontweight="bold", color="#e74c3c")
        ax1_sum_twin.set_ylabel(
            "R² Score", fontsize=11, fontweight="bold", color="#27ae60"
        )
        ax1_sum.set_xticks(x_pos)
        ax1_sum.set_xticklabels(df_sorted["Model"], rotation=45, ha="right")
        ax1_sum.set_title("RMSE vs R² Score Comparison", fontsize=12, fontweight="bold")
        ax1_sum.grid(axis="y", alpha=0.3)

        # Right: Best model highlight
        ax2_sum = axes[1]
        best_idx = df_sorted[df_sorted["Model"] == self.best_model_name].index[0]
        colors_sum = [
            "#2ecc71" if i == best_idx else "#95a5a6" for i in range(len(df_sorted))
        ]

        bars_sum = ax2_sum.barh(df_sorted["Model"], df_sorted["R²"], color=colors_sum)
        ax2_sum.set_xlabel("R² Score", fontsize=11, fontweight="bold")
        ax2_sum.set_title(
            f'Best Model: {self.best_model_name}\n(R² = {df_sorted.iloc[best_idx]["R²"]:.4f})',
            fontsize=12,
            fontweight="bold",
        )
        ax2_sum.invert_yaxis()
        ax2_sum.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars_sum, df_sorted["R²"])):
            ax2_sum.text(
                val + 0.01, i, f"{val:.4f}", va="center", fontsize=9, fontweight="bold"
            )

        plt.tight_layout()

        summary_plot_path = self.models_dir / "model_summary.png"
        plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Summary plot saved: {summary_plot_path}")

    def generate_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate comprehensive model performance report.

        Args:
            results_df: DataFrame with model results

        Returns:
            Report text
        """
        report = f"""
# MODEL PERFORMANCE REPORT
==========================

## Executive Summary
- **Best Model**: {self.best_model_name}
- **RMSE**: {self.best_model_info['RMSE']:.4f}
- **R²**: {self.best_model_info['R²']:.4f}
- **MAE**: {self.best_model_info['MAE']:.4f}

## Visualizations
Comprehensive comparison plots have been generated:
- **model_comparison.png**: Detailed dashboard with 6 comparison charts
  - RMSE, R², MAE, CV Score comparisons
  - Normalized metrics comparison
  - Ranking heatmap
- **model_summary.png**: Quick summary with RMSE vs R² and best model highlight

## Model Comparison
"""

        for _, row in results_df.iterrows():
            report += f"""
### {row['Model']}
- **RMSE**: {row['RMSE']:.4f}
- **R²**: {row['R²']:.4f}
- **MAE**: {row['MAE']:.4f}
- **CV Score**: {row['CV_Score']:.6f}
- **Best Parameters**: {row['Best_Params']}
"""

        report += f"""
## Recommendations
1. **Primary Model**: {self.best_model_name} shows best performance
2. **Regularization**: Linear models provide good baseline with interpretability
3. **Ensemble**: Tree-based models offer higher accuracy for complex patterns
4. **Cross-Validation**: All models validated with 5-fold CV

## Next Steps
1. Deploy {self.best_model_name} for production
2. Monitor model performance over time
3. Consider ensemble methods for further improvement
4. Regular retraining with new data
"""

        return report


def main():
    """Example usage of ModelTrainer."""
    # This would be called from the main pipeline
    print("ModelTrainer module loaded successfully")
    print("Use ModelTrainer class for comprehensive model training and evaluation")


if __name__ == "__main__":
    main()
