
# MODEL PERFORMANCE REPORT
==========================

## Executive Summary
- **Best Model**: Lasso
- **RMSE**: 0.1264
- **R²**: 0.9035
- **MAE**: 0.0863

## Model Comparison

### Lasso
- **RMSE**: 0.1264
- **R²**: 0.9035
- **MAE**: 0.0863
- **CV Score**: 0.020086
- **Best Parameters**: {'alpha': 0.01}

### ElasticNet
- **RMSE**: 0.1279
- **R²**: 0.9013
- **MAE**: 0.0881
- **CV Score**: 0.019886
- **Best Parameters**: {'alpha': 0.1, 'l1_ratio': 0.1}

### XGBoost
- **RMSE**: 0.1322
- **R²**: 0.8946
- **MAE**: 0.0868
- **CV Score**: 0.018098
- **Best Parameters**: {'subsample': 1.0, 'min_child_weight': 3, 'max_depth': 3, 'learning_rate': 0.2, 'gamma': 0, 'colsample_bytree': 1.0}

### LightGBM
- **RMSE**: 0.1355
- **R²**: 0.8892
- **MAE**: 0.0899
- **CV Score**: 0.017368
- **Best Parameters**: {'subsample': 1.0, 'num_leaves': 50, 'min_child_samples': 50, 'max_depth': 10, 'learning_rate': 0.1, 'colsample_bytree': 0.8}

### Ridge
- **RMSE**: 0.1544
- **R²**: 0.8562
- **MAE**: 0.0916
- **CV Score**: 0.021131
- **Best Parameters**: {'alpha': 0.01}

## Recommendations
1. **Primary Model**: Lasso shows best performance
2. **Regularization**: Linear models provide good baseline with interpretability
3. **Ensemble**: Tree-based models offer higher accuracy for complex patterns
4. **Cross-Validation**: All models validated with 5-fold CV

## Next Steps
1. Deploy Lasso for production
2. Monitor model performance over time
3. Consider ensemble methods for further improvement
4. Regular retraining with new data
