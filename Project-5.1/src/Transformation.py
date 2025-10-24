"""
TRANSFORMATION MODULE
======================

Purpose:
- Apply transformations to reduce skewness
- Log1p for positive features
- Yeo-Johnson for zero-inflated/negative features
- Bin count features (KitchenAbvGr)
- Cross-fit strategy: fit on train, apply on val/test
- Save transformation parameters for reproducibility

Strategy:
- Binary flags: NO transform
- Residuals: NO transform (already orthogonalized)
- Target (SalePrice): log1p(x)
- Main scale (GrLivArea): log1p(x)
- Area features: log1p(x)
- Zero-inflated: Yeo-Johnson or log1p (if all ≥ 0)
- Rare/ultra-skew: Only flags (Phase 2)
- Count: Bin or flag
"""

import pandas as pd
import numpy as np
import json
from scipy.stats import skew, yeojohnson
from sklearn.preprocessing import PowerTransformer
import warnings
warnings.filterwarnings("ignore")


class SkewnessTransformer:
    """Apply transformations to reduce skewness with cross-fit strategy"""
    
    def __init__(self, processed_dir='data/processed', interim_dir='data/interim'):
        self.train_data = None
        self.test_data = None
        self.transformation_params = {}
        self.feature_strategy = {}
        self.processed_dir = processed_dir
        self.interim_dir = interim_dir
        
    def load_splits(self, train_path, test_path):
        """Load train/test splits"""
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        
        print(f"✓ Train: {self.train_data.shape}")
        print(f"✓ Test:  {self.test_data.shape}")
        return self
    
    def _identify_features_strategy(self):
        """Identify transformation strategy for each feature"""
        print("\n" + "=" * 90)
        print("STEP 1: Identify Feature Transformation Strategy")
        print("=" * 90)
        
        numeric_df = self.train_data.select_dtypes(include=[np.number])
        
        # Define feature groups
        binary_flags = [
            'HasGarage', 'HasBasement', 'HasFireplace', 'HasMasonryVeneer',
            'Has2ndFlr', 'GarageSameAsHouse'
        ]
        
        residual_features = [
            'BasementResid', 'MasVnrAreaResid', 'SecondFlrShare_resid'
        ]
        
        ultra_rare = [
            'PoolArea', 'MiscVal', '3SsnPorch', 'LowQualFinSF'
        ]
        
        # Strategy assignment
        strategies = {
            'SalePrice': 'target_log',
            'GrLivArea': 'log',
            'LotArea': 'log',
            'AvgRoomSize': 'log',
            'LotFrontage': 'log',
            'KitchenAbvGr': 'bin_count',
            'ExtraFireplaces': 'log',
        }
        
        # Assign strategies
        for col in numeric_df.columns:
            if col == 'Id':
                self.feature_strategy[col] = 'skip'
            elif col in binary_flags:
                self.feature_strategy[col] = 'skip_binary'
            elif col in residual_features:
                self.feature_strategy[col] = 'skip_residual'
            elif col in ultra_rare:
                self.feature_strategy[col] = 'skip_ultra_rare'
            elif col in strategies:
                self.feature_strategy[col] = strategies[col]
            else:
                # Auto-detect: check if all positive
                col_data = numeric_df[col].dropna()
                if (col_data > 0).all():
                    self.feature_strategy[col] = 'log'
                elif (col_data >= 0).all():
                    self.feature_strategy[col] = 'log_zero_inflated'
                else:
                    self.feature_strategy[col] = 'yeo_johnson'
        
        # Print summary
        print("\nFeature Transformation Strategy:")
        for strategy_type in set(self.feature_strategy.values()):
            features_in_group = [f for f, s in self.feature_strategy.items() if s == strategy_type]
            print(f"\n  {strategy_type:<20} ({len(features_in_group)} features):")
            for feat in sorted(features_in_group)[:10]:
                print(f"    ├─ {feat}")
            if len(features_in_group) > 10:
                print(f"    └─ ... +{len(features_in_group) - 10} more")
        
        return self
    
    def _transform_target(self):
        """Transform SalePrice using log1p"""
        print("\n" + "=" * 90)
        print("STEP 2: Transform Target (SalePrice)")
        print("=" * 90)
        
        if 'SalePrice' in self.train_data.columns:
            y_train = self.train_data['SalePrice'].values
            y_test = self.test_data['SalePrice'].values
            
            original_skew_train = skew(y_train)
            
            y_train_transformed = np.log1p(y_train)
            y_test_transformed = np.log1p(y_test)
            
            new_skew_train = skew(y_train_transformed)
            
            print(f"\nSalePrice Transformation:")
            print(f"  Train original skew: {original_skew_train:+.3f}")
            print(f"  Train transformed skew: {new_skew_train:+.3f}")
            print(f"  Reduction: {((original_skew_train - new_skew_train) / original_skew_train * 100):.1f}%")
            
            self.train_data['SalePrice'] = y_train_transformed
            self.test_data['SalePrice'] = y_test_transformed
            
            self.transformation_params['SalePrice'] = {
                'method': 'log1p',
                'original_skew': float(original_skew_train),
                'transformed_skew': float(new_skew_train),
                'inverse': 'np.expm1(x)'
            }
        
        return self
    
    def _bin_kitchen_abvgr(self):
        """Bin KitchenAbvGr into categories"""
        print("\n" + "=" * 90)
        print("STEP 3: Bin KitchenAbvGr (Count Feature)")
        print("=" * 90)
        
        if 'KitchenAbvGr' in self.train_data.columns:
            print("\nOriginal KitchenAbvGr distribution (train):")
            print(self.train_data['KitchenAbvGr'].value_counts().sort_index())
            
            # Create bin: 0/1, 2, 3+
            def bin_kitchen(x):
                if x <= 1:
                    return 0
                elif x == 2:
                    return 1
                else:
                    return 2
            
            self.train_data['KitchenAbvGr_Binned'] = self.train_data['KitchenAbvGr'].apply(bin_kitchen)
            self.test_data['KitchenAbvGr_Binned'] = self.test_data['KitchenAbvGr'].apply(bin_kitchen)
            
            print("\nBinned distribution (train):")
            print(self.train_data['KitchenAbvGr_Binned'].value_counts().sort_index())
            
            # Create HasMultiKitchen flag
            self.train_data['HasMultiKitchen'] = (self.train_data['KitchenAbvGr'] >= 2).astype(int)
            self.test_data['HasMultiKitchen'] = (self.test_data['KitchenAbvGr'] >= 2).astype(int)
            
            print("\nHasMultiKitchen distribution (train):")
            print(self.train_data['HasMultiKitchen'].value_counts())
            
            # Remove original
            self.train_data = self.train_data.drop('KitchenAbvGr', axis=1)
            self.test_data = self.test_data.drop('KitchenAbvGr', axis=1)
            
            self.transformation_params['KitchenAbvGr'] = {
                'method': 'bin',
                'bins': {0: '<=1', 1: '2', 2: '>=3'},
                'flags': ['KitchenAbvGr_Binned', 'HasMultiKitchen']
            }
            
            print("\n✓ Created: KitchenAbvGr_Binned, HasMultiKitchen")
        
        return self
    
    def _transform_features_log(self):
        """Apply log1p to positive features"""
        print("\n" + "=" * 90)
        print("STEP 4: Log1p Transform (Positive Features)")
        print("=" * 90)
        
        log_features = [f for f, s in self.feature_strategy.items() if s == 'log']
        
        print(f"\nApplying log1p to {len(log_features)} features:")
        
        for feat in log_features:
            if feat in self.train_data.columns:
                original_skew = skew(self.train_data[feat].dropna())
                
                self.train_data[f'{feat}_log'] = np.log1p(self.train_data[feat])
                self.test_data[f'{feat}_log'] = np.log1p(self.test_data[feat])
                
                transformed_skew = skew(self.train_data[f'{feat}_log'].dropna())
                
                print(f"  {feat:<30} {original_skew:+.3f} → {transformed_skew:+.3f}")
                
                self.transformation_params[feat] = {
                    'method': 'log1p',
                    'original_skew': float(original_skew),
                    'transformed_skew': float(transformed_skew)
                }
                
                # Keep original for now (will drop after)
                self.train_data = self.train_data.drop(feat, axis=1)
                self.test_data = self.test_data.drop(feat, axis=1)
        
        return self
    
    def _transform_features_yeo_johnson(self):
        """Apply Yeo-Johnson to zero-inflated/negative features"""
        print("\n" + "=" * 90)
        print("STEP 5: Yeo-Johnson Transform (Zero-Inflated/Negative)")
        print("=" * 90)
        
        yj_features = [f for f, s in self.feature_strategy.items() 
                      if s in ['yeo_johnson', 'log_zero_inflated']]
        
        # Fit on train, transform both
        numeric_df_train = self.train_data[[f for f in yj_features if f in self.train_data.columns]].copy()
        numeric_df_test = self.test_data[[f for f in yj_features if f in self.test_data.columns]].copy()
        
        print(f"\nApplying Yeo-Johnson to {len(yj_features)} features:")
        
        # Create transformer
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        
        # Fit on train
        pt.fit(numeric_df_train)
        
        # Transform
        train_transformed = pt.transform(numeric_df_train)
        test_transformed = pt.transform(numeric_df_test)
        
        for idx, feat in enumerate([f for f in yj_features if f in self.train_data.columns]):
            original_skew = skew(numeric_df_train[feat].dropna())
            transformed_skew = skew(train_transformed[:, idx])
            
            print(f"  {feat:<30} {original_skew:+.3f} → {transformed_skew:+.3f} (λ={pt.lambdas_[idx]:.3f})")
            
            self.train_data[f'{feat}_yj'] = train_transformed[:, idx]
            self.test_data[f'{feat}_yj'] = test_transformed[:, idx]
            
            self.transformation_params[feat] = {
                'method': 'yeo_johnson',
                'lambda': float(pt.lambdas_[idx]),
                'original_skew': float(original_skew),
                'transformed_skew': float(transformed_skew)
            }
            
            # Drop original
            if feat in self.train_data.columns:
                self.train_data = self.train_data.drop(feat, axis=1)
            if feat in self.test_data.columns:
                self.test_data = self.test_data.drop(feat, axis=1)
        
        return self
    
    def _check_transformation_quality(self):
        """Check if transformations reduced skewness effectively"""
        print("\n" + "=" * 90)
        print("STEP 6: Transformation Quality Check")
        print("=" * 100)
        
        print("\nSkewness Reduction Summary:")
        print(f"{'Feature':<30} {'Original':<12} {'Transformed':<12} {'Reduction %':<12}")
        print("-" * 100)
        
        skew_reduced_count = 0
        for feat, params in self.transformation_params.items():
            if 'original_skew' in params and 'transformed_skew' in params:
                orig = params['original_skew']
                trans = params['transformed_skew']
                if orig != 0:
                    reduction = ((abs(orig) - abs(trans)) / abs(orig) * 100)
                else:
                    reduction = 0
                
                status = "✓" if abs(trans) < abs(orig) else "⚠"
                print(f"{feat:<30} {orig:+9.3f}    {trans:+9.3f}    {reduction:+9.1f}%  {status}")
                
                if abs(trans) < abs(orig):
                    skew_reduced_count += 1
        
        print(f"\n✓ {skew_reduced_count} features had skewness reduced")
        
        return self
    
    def _save_transformation_config(self, output_path='transformation_config.json'):
        """Save transformation parameters for reproducibility"""
        print("\n" + "=" * 90)
        print("STEP 7: Save Transformation Configuration")
        print("=" * 90)
        
        full_path = self.interim_dir + '/' + output_path
        with open(full_path, 'w') as f:
            json.dump(self.transformation_params, f, indent=2)
        
        print(f"✓ Saved: {full_path}")
        return self
    
    def save_transformed_data(self, train_output='train_transformed.csv', 
                             test_output='test_transformed.csv'):
        """Save transformed datasets"""
        print("\n" + "=" * 90)
        print("STEP 8: Save Transformed Data")
        print("=" * 90)
        
        train_path = self.processed_dir + '/' + train_output
        test_path = self.processed_dir + '/' + test_output
        
        self.train_data.to_csv(train_path, index=False)
        self.test_data.to_csv(test_path, index=False)
        
        print(f"✓ Train: {train_path} ({self.train_data.shape})")
        print(f"✓ Test:  {test_path} ({self.test_data.shape})")
        
        # Summary
        print(f"\nFeature Changes:")
        print(f"  Train columns: {self.train_data.shape[1]}")
        print(f"  Test columns:  {self.test_data.shape[1]}")
        
        return self
    
    def run_pipeline(self, train_path='train_data.csv', test_path='test_data.csv'):
        """Run complete transformation pipeline"""
        return (self
                .load_splits(train_path, test_path)
                ._identify_features_strategy()
                ._transform_target()
                ._bin_kitchen_abvgr()
                ._transform_features_log()
                ._transform_features_yeo_johnson()
                ._check_transformation_quality()
                ._save_transformation_config()
                .save_transformed_data())


# ============================================================================
# ADDITIONAL ANALYSIS
# ============================================================================

def analyze_remaining_skewness(train_path='train_transformed.csv'):
    """Analyze skewness of transformed features"""
    print("\n" + "=" * 90)
    print("ADDITIONAL: Analyze Remaining Skewness")
    print("=" * 90)
    
    df = pd.read_csv(train_path)
    numeric_df = df.select_dtypes(include=[np.number])
    
    skew_values = {col: skew(numeric_df[col].dropna()) for col in numeric_df.columns}
    skew_sorted = sorted(skew_values.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\nTop 15 Features by Remaining Skewness:")
    print(f"{'Feature':<40} {'Skewness':<12} {'Status':<20}")
    print("-" * 100)
    
    for feat, skew_val in skew_sorted[:15]:
        if abs(skew_val) > 1:
            status = "⚠ Still high (>1)"
        elif abs(skew_val) > 0.5:
            status = "~ Moderate (0.5-1)"
        else:
            status = "✓ Good (<0.5)"
        
        print(f"{feat:<40} {skew_val:+8.3f}      {status:<20}")
    
    # Count by category
    high_skew = sum(1 for _, s in skew_sorted if abs(s) > 1)
    moderate_skew = sum(1 for _, s in skew_sorted if 0.5 < abs(s) <= 1)
    low_skew = sum(1 for _, s in skew_sorted if abs(s) <= 0.5)
    
    print(f"\nSummary:")
    print(f"  High (|skew| > 1):     {high_skew} features")
    print(f"  Moderate (0.5-1):      {moderate_skew} features")
    print(f"  Good (< 0.5):          {low_skew} features")


# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("=" * 90)
    print("TRANSFORMATION PIPELINE")
    print("=" * 90)
    
    # Run pipeline
    transformer = SkewnessTransformer()
    transformer.run_pipeline(
        train_path='train_data.csv',
        test_path='test_data.csv'
    )
    
    # Analyze remaining skewness
    analyze_remaining_skewness(train_path='train_transformed.csv')
    
    print("\n" + "=" * 90)
    print("✅ TRANSFORMATION COMPLETE")
    print("=" * 90)
    print("\nNext steps:")
    print("  1. Categorical encoding (Encoding.py)")
    print("  2. Feature scaling/normalization")
    print("  3. Model training")
