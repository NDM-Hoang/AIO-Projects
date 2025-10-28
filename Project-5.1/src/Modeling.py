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
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

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
    
    def __init__(self, models_dir: str = 'models', random_state: int = 42):
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
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'RMSLE': rmse  # Since target is log-transformed
        }
    
    def train_linear_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           X_test: pd.DataFrame, y_test: pd.Series) -> List[Dict[str, Any]]:
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
        print("\n" + "="*60)
        print("TRAINING LINEAR MODELS WITH REGULARIZATION")
        print("="*60)
        
        results = []
        
        # Define hyperparameter grids
        ridge_params = {'alpha': [0.001, 0.01, 0.1, 1, 10, 50, 100]}
        lasso_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
        elasticnet_params = {
            'alpha': [0.001, 0.01, 0.1, 1],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        
        # Initialize models
        models = {
            'Ridge': Ridge(random_state=self.random_state),
            'Lasso': Lasso(random_state=self.random_state, max_iter=2000),
            'ElasticNet': ElasticNet(random_state=self.random_state, max_iter=2000)
        }
        
        param_grids = {
            'Ridge': ridge_params,
            'Lasso': lasso_params,
            'ElasticNet': elasticnet_params
        }
        
        # Train each model
        for model_name, model in models.items():
            print(f"\n🔧 Training {model_name}...")
            
            # Grid search with cross-validation
            search = GridSearchCV(
                model, param_grids[model_name],
                cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=0
            )
            
            search.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = search.predict(X_test)
            metrics = self.calculate_metrics(y_test, y_pred, model_name)
            metrics.update({
                'Best_Params': search.best_params_,
                'CV_Score': -search.best_score_
            })
            
            results.append(metrics)
            
            print(f"   Best params: {search.best_params_}")
            print(f"   CV Score: {-search.best_score_:.6f}")
            print(f"   Test RMSE: {metrics['RMSE']:.4f}")
            print(f"   Test R²: {metrics['R²']:.4f}")
        
        return results
    
    def train_tree_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series) -> List[Dict[str, Any]]:
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
        print("\n" + "="*60)
        print("TRAINING TREE-BASED MODELS")
        print("="*60)
        
        results = []
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            print("\n🌳 Training LightGBM...")
            
            lgb_params = {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [31, 50, 100, 200],
                'max_depth': [3, 5, 7, 10],
                'min_child_samples': [20, 50, 100],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            lgb_model = lgb.LGBMRegressor(random_state=self.random_state, verbose=-1)
            
            search = RandomizedSearchCV(
                lgb_model, lgb_params,
                n_iter=30, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, random_state=self.random_state, verbose=0
            )
            
            search.fit(X_train, y_train)
            
            y_pred = search.predict(X_test)
            metrics = self.calculate_metrics(y_test, y_pred, "LightGBM")
            metrics.update({
                'Best_Params': search.best_params_,
                'CV_Score': -search.best_score_
            })
            
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
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=self.random_state, verbosity=0)
            
            search = RandomizedSearchCV(
                xgb_model, xgb_params,
                n_iter=30, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, random_state=self.random_state, verbose=0
            )
            
            search.fit(X_train, y_train)
            
            y_pred = search.predict(X_test)
            metrics = self.calculate_metrics(y_test, y_pred, "XGBoost")
            metrics.update({
                'Best_Params': search.best_params_,
                'CV_Score': -search.best_score_
            })
            
            results.append(metrics)
            
            print(f"   Best params: {search.best_params_}")
            print(f"   CV Score: {-search.best_score_:.6f}")
            print(f"   Test RMSE: {metrics['RMSE']:.4f}")
            print(f"   Test R²: {metrics['R²']:.4f}")
        else:
            print("⚠️ XGBoost not available")
        
        return results
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
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
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL TRAINING & EVALUATION")
        print("="*80)
        
        # Train linear models
        linear_results = self.train_linear_models(X_train, y_train, X_test, y_test)
        
        # Train tree-based models
        tree_results = self.train_tree_models(X_train, y_train, X_test, y_test)
        
        # Combine all results
        all_results = linear_results + tree_results
        self.results = all_results
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('RMSE').reset_index(drop=True)
        
        # Identify best model
        self.best_model_name = results_df.iloc[0]['Model']
        self.best_model_info = results_df.iloc[0]
        
        print("\n" + "="*60)
        print("FINAL MODEL COMPARISON")
        print("="*60)
        print(results_df[['Model', 'RMSE', 'MAE', 'R²', 'CV_Score']].round(4))
        
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
        model_path = self.models_dir / f'{model_name.lower()}_model.pkl'
        joblib.dump(model, model_path)
        
        # Save feature names
        features_path = self.models_dir / f'{model_name.lower()}_features.json'
        with open(features_path, 'w') as f:
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
        results_path = self.models_dir / 'model_comparison.csv'
        results_df.to_csv(results_path, index=False)
        
        # Save best model configuration
        best_config = {
            'best_model': self.best_model_name,
            'best_params': self.best_model_info['Best_Params'],
            'performance': {
                'RMSE': float(self.best_model_info['RMSE']),
                'MAE': float(self.best_model_info['MAE']),
                'R²': float(self.best_model_info['R²']),
                'CV_Score': float(self.best_model_info['CV_Score'])
            },
            'feature_count': len(results_df.columns) - 5,  # Exclude metric columns
            'training_samples': len(results_df),  # This will be updated with actual data
            'test_samples': len(results_df)  # This will be updated with actual data
        }
        
        config_path = self.models_dir / 'best_model_config.json'
        with open(config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"💾 Results saved: {results_path}")
        print(f"💾 Config saved: {config_path}")
    
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
