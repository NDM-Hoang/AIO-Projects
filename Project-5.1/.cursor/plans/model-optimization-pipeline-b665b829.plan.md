<!-- b665b829-2b80-4fda-8416-d8daa07e540b d2abab20-f730-40c4-8a24-6f89f1050011 -->
# Model Optimization & Selection Pipeline

## Overview

Implement model training, hyperparameter optimization, and selection for house price prediction using both linear regularization models and tree-based ensemble models, with proper cross-validation and metrics evaluation.

## Phase 1: Notebook Exploration & Experimentation

### 1.1 Create Model Comparison Notebook

**File**: `notebooks/model_optimization.ipynb`

- Load processed data from `data/processed/train_encoded.csv` and `test_encoded.csv`
- Separate features (X) and target (SalePrice)
- Implement evaluation framework with metrics: RMSE, MAE, R², RMSLE

### 1.2 Implement Linear Models with Regularization

Train and tune:

- **Ridge Regression** (L2): `alpha` range [0.001, 0.01, 0.1, 1, 10, 50, 100]
- **Lasso Regression** (L1): `alpha` range [0.0001, 0.001, 0.01, 0.1, 1, 10]
- **ElasticNet** (L1+L2): `alpha` [0.001, 0.01, 0.1, 1], `l1_ratio` [0.1, 0.3, 0.5, 0.7, 0.9]

Use **RandomizedSearchCV** (50 iterations) → then **GridSearchCV** for fine-tuning top performers.

**CV Strategy**: 5-Fold Cross-Validation (appropriate for 1239 samples, balances bias-variance tradeoff)

### 1.3 Implement Tree-Based Models

Train and tune:

- **LightGBM**: learning_rate, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree
- **XGBoost**: learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma
- **Optional CatBoost**: learning_rate, depth, l2_leaf_reg

Use **RandomizedSearchCV** (30-50 iterations per model) with 5-Fold CV.

### 1.4 Model Comparison & Analysis

- Compare all models using CV scores and holdout test performance
- Create visualization: bar charts for RMSE/R², learning curves, residual plots
- Feature importance analysis for top models
- Identify best performing model(s)

## Phase 2: Production Implementation

### 2.1 Create Modeling Module

**File**: `src/Modeling.py`

Implement class `ModelTrainer` with methods:

- `train_linear_models()`: Ridge, Lasso, ElasticNet with hyperparameter tuning
- `train_tree_models()`: LightGBM, XGBoost with hyperparameter tuning
- `cross_validate()`: 5-Fold CV with multiple metrics
- `evaluate_model()`: Comprehensive evaluation on test set
- `save_model()`: Save best model to `models/` directory
- `generate_report()`: Model performance summary

### 2.2 Update Main Pipeline

**File**: `app.py`

Replace placeholder `run_modeling()` method:

- Load encoded train/test data
- Initialize ModelTrainer
- Train all models with hyperparameter optimization
- Compare and select best model
- Save best model + configuration to `models/`
- Generate performance report

### 2.3 Configuration & Outputs

Create outputs:

- `models/best_model.pkl`: Serialized best model
- `models/model_comparison.json`: All model scores
- `models/hyperparameters.json`: Optimal hyperparameters
- `reports/ModelReport.md`: Comprehensive analysis in Vietnamese

## Key Technical Details

**Loss Metrics**:

- Primary: RMSE (aligns with Kaggle competition)
- Secondary: R², MAE, RMSLE
- Use neg_mean_squared_error for sklearn CV scoring

**Data Handling**:

- Train: 1239 samples (85%) for CV and training
- Test: 219 samples (15%) for final evaluation
- Target already log-transformed, may need inverse transform for interpretable metrics

**Hyperparameter Search**:

- RandomizedSearchCV first (broad exploration, faster)
- GridSearchCV for refinement (narrow ranges around best)
- 5-Fold CV (optimal for ~1200 samples)

**Model Persistence**:

- Use joblib/pickle for model serialization
- Save scaler configs from encoding step
- Document feature names and order

### To-dos

- [ ] Create model_optimization.ipynb notebook with data loading, train/test split verification, and metric evaluation framework
- [ ] Implement Ridge, Lasso, ElasticNet with RandomizedSearchCV then GridSearchCV hyperparameter tuning in notebook
- [ ] Implement LightGBM and XGBoost with RandomizedSearchCV hyperparameter tuning in notebook
- [ ] Compare all models with visualizations (bar charts, learning curves, residual plots) and identify best performer
- [ ] Create src/Modeling.py with ModelTrainer class implementing train, evaluate, and save methods
- [ ] Update app.py run_modeling() method to integrate ModelTrainer and generate outputs
- [ ] Run full pipeline, verify model outputs, and generate comprehensive ModelReport.md