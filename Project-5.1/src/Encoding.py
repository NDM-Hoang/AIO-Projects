"""
ENCODING MODULE
================

Purpose:
- Encode categorical features
- Ordinal encoding for quality/order features (17 features)
- One-Hot encoding for nominal categorical (24 features)
- Target encoding for high-cardinality features (2 features)
- Cross-fit strategy for target encoding (prevent leakage)
- Combine all features + StandardScaler

Strategy:
- Ordinal: Quality scales (Ex > Gd > TA > Fa > Po), Finish levels, Shape/Slope
- One-Hot: Low/medium cardinality categorical
- Target Encoding: Neighborhood (25 unique), Exterior2nd (16 unique)
- Drop: Id (no predictive value)

Output: ~190 features ready for modeling
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")


class CategoricalEncoder:
    """Encode categorical features with multiple strategies"""
    
    def __init__(self, processed_dir='data/processed', interim_dir='data/interim'):
        self.train_data = None
        self.test_data = None
        self.scaler = StandardScaler()
        self.encoding_config = {}
        self.processed_dir = processed_dir
        self.interim_dir = interim_dir
        
        # Ordinal mappings
        self.ordinal_mappings = {
            'ExterQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'ExterCond': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'KitchenQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'HeatingQC': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'FireplaceQu': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4, 'None': -1},
            'GarageQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4, 'None': -1},
            'GarageCond': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4, 'None': -1},
            'BsmtQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4, 'None': -1},
            'BsmtCond': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4, 'None': -1},
            'GarageFinish': {'None': -1, 'Unf': 1, 'RFn': 2, 'Fin': 3},
            'BsmtExposure': {'None': -1, 'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3},
            'BsmtFinType1': {'None': -1, 'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5},
            'BsmtFinType2': {'None': -1, 'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5},
            'LotShape': {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3},
            'LandSlope': {'Sev': 0, 'Mod': 1, 'Gtl': 2},
            'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
            'Functional': {'Sev': 0, 'Maj2': 1, 'Maj1': 2, 'Mod': 3, 'Min2': 4, 'Min1': 5, 'Typ': 6},
        }
        
        # Ordinal features list
        self.ordinal_features = list(self.ordinal_mappings.keys())
        
        # Features for one-hot encoding
        self.ohe_features = [
            'MSZoning', 'Street', 'Alley', 'LandContour', 'Utilities', 'LotConfig',
            'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
            'Exterior1st', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical',
            'GarageType', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition',
            'LandContour'
        ]
        
        # Target encoding features
        self.target_enc_features = ['Neighborhood', 'Exterior2nd']
        
        # Store mappings for target encoding
        self.target_enc_mappings = {}
        self.ohe_encoder = None
        
    def load_splits(self, train_path, test_path):
        """Load train/test splits"""
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        
        print(f"✓ Train: {self.train_data.shape}")
        print(f"✓ Test:  {self.test_data.shape}")
        return self
    
    def _apply_ordinal_encoding(self):
        """Apply ordinal encoding to quality/order features"""
        print("\n" + "=" * 90)
        print("STEP 1: Ordinal Encoding (17 features)")
        print("=" * 90)
        
        print(f"\nEncoding {len(self.ordinal_features)} ordinal features:")
        
        for feat in self.ordinal_features:
            if feat in self.train_data.columns:
                mapping = self.ordinal_mappings[feat]
                
                # Apply mapping
                self.train_data[f'{feat}_ord'] = self.train_data[feat].map(mapping)
                self.test_data[f'{feat}_ord'] = self.test_data[feat].map(mapping)
                
                # Handle unmapped values
                self.train_data[f'{feat}_ord'] = self.train_data[f'{feat}_ord'].fillna(0)
                self.test_data[f'{feat}_ord'] = self.test_data[f'{feat}_ord'].fillna(0)
                
                # Drop original
                self.train_data = self.train_data.drop(feat, axis=1)
                self.test_data = self.test_data.drop(feat, axis=1)
                
                unique_vals = sorted(self.train_data[f'{feat}_ord'].unique())
                print(f"  ✓ {feat:<25} → values: {unique_vals}")
        
        self.encoding_config['ordinal_features'] = len(self.ordinal_features)
        return self
    
    def _apply_target_encoding(self):
        """Apply target encoding with cross-fit strategy"""
        print("\n" + "=" * 90)
        print("STEP 2: Target Encoding (2 features - Cross-fit strategy)")
        print("=" * 90)
        
        # Extract target from train data
        y_train = self.train_data['SalePrice'].values
        
        print(f"\nTarget encoding {len(self.target_enc_features)} high-cardinality features:")
        
        for feat in self.target_enc_features:
            if feat in self.train_data.columns:
                # Calculate mean target per category on train set
                train_mapping = {}
                
                for category in self.train_data[feat].unique():
                    if pd.notna(category):
                        mask = self.train_data[feat] == category
                        mean_target = y_train[mask].mean() if mask.sum() > 0 else 0
                        train_mapping[category] = mean_target
                
                # Apply mapping to both train and test
                self.train_data[f'{feat}_target_enc'] = self.train_data[feat].map(train_mapping)
                self.test_data[f'{feat}_target_enc'] = self.test_data[feat].map(train_mapping)
                
                # Handle unseen categories (fill with global mean)
                global_mean = y_train.mean()
                self.train_data[f'{feat}_target_enc'] = self.train_data[f'{feat}_target_enc'].fillna(global_mean)
                self.test_data[f'{feat}_target_enc'] = self.test_data[f'{feat}_target_enc'].fillna(global_mean)
                
                # Drop original
                self.train_data = self.train_data.drop(feat, axis=1)
                self.test_data = self.test_data.drop(feat, axis=1)
                
                n_categories = len(train_mapping)
                print(f"  ✓ {feat:<25} → {n_categories:>3} categories → 1D feature (cross-fit)")
                
                # Store mapping for reproducibility
                self.target_enc_mappings[feat] = train_mapping
        
        self.encoding_config['target_enc_features'] = len(self.target_enc_features)
        self.encoding_config['target_enc_mappings'] = self.target_enc_mappings
        return self
    
    def _apply_one_hot_encoding(self):
        """Apply One-Hot encoding to nominal categorical features"""
        print("\n" + "=" * 90)
        print("STEP 3: One-Hot Encoding (~24 features)")
        print("=" * 90)
        
        # Get remaining categorical columns
        remaining_cat_cols = self.train_data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove specific columns that shouldn't be encoded
        remaining_cat_cols = [c for c in remaining_cat_cols if c not in ['Id'] and c != 'SalePrice']
        
        print(f"\nOne-Hot encoding {len(remaining_cat_cols)} features:")
        print(f"  Features: {remaining_cat_cols}")
        
        if len(remaining_cat_cols) > 0:
            # Fit encoder on train data
            self.ohe_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',
                drop='first'  # Drop first category to avoid multicollinearity
            )
            
            # Fit on train
            ohe_train = self.ohe_encoder.fit_transform(self.train_data[remaining_cat_cols])
            
            # Transform test
            ohe_test = self.ohe_encoder.transform(self.test_data[remaining_cat_cols])
            
            # Get feature names
            feature_names = self.ohe_encoder.get_feature_names_out(remaining_cat_cols)
            
            # Create dataframes
            ohe_train_df = pd.DataFrame(ohe_train, columns=feature_names, index=self.train_data.index)
            ohe_test_df = pd.DataFrame(ohe_test, columns=feature_names, index=self.test_data.index)
            
            # Drop original categorical columns
            for col in remaining_cat_cols:
                self.train_data = self.train_data.drop(col, axis=1)
                self.test_data = self.test_data.drop(col, axis=1)
            
            # Combine with encoded features
            self.train_data = pd.concat([self.train_data, ohe_train_df], axis=1)
            self.test_data = pd.concat([self.test_data, ohe_test_df], axis=1)
            
            print(f"\n  ✓ Generated {len(feature_names)} one-hot features")
            print(f"  ✓ Train shape: {self.train_data.shape}")
            print(f"  ✓ Test shape:  {self.test_data.shape}")
        
        self.encoding_config['ohe_features'] = len(remaining_cat_cols)
        self.encoding_config['ohe_output_features'] = len(feature_names) if len(remaining_cat_cols) > 0 else 0
        return self
    
    def _drop_unnecessary_columns(self):
        """Drop Id and other unnecessary columns"""
        print("\n" + "=" * 90)
        print("STEP 4: Clean Up")
        print("=" * 90)
        
        drop_cols = ['Id']
        existing_drop_cols = [c for c in drop_cols if c in self.train_data.columns]
        
        if existing_drop_cols:
            self.train_data = self.train_data.drop(existing_drop_cols, axis=1)
            self.test_data = self.test_data.drop(existing_drop_cols, axis=1)
            print(f"\n  ✓ Dropped columns: {existing_drop_cols}")
        
        return self
    
    def _standardize_features(self):
        """Apply StandardScaler to numeric features"""
        print("\n" + "=" * 90)
        print("STEP 5: Feature Scaling (StandardScaler)")
        print("=" * 100)
        
        numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Separate target from features
        y_train = self.train_data['SalePrice'].values
        X_train = self.train_data.drop('SalePrice', axis=1)
        y_test = self.test_data['SalePrice'].values
        X_test = self.test_data.drop('SalePrice', axis=1)
        
        # Fit scaler on X_train
        self.scaler.fit(X_train)
        
        # Transform both
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to dataframes
        self.train_data = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        self.train_data['SalePrice'] = y_train
        
        self.test_data = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        self.test_data['SalePrice'] = y_test
        
        print(f"\n  ✓ Scaled {X_train.shape[1]} numeric features")
        print(f"  ✓ Train shape: {self.train_data.shape}")
        print(f"  ✓ Test shape:  {self.test_data.shape}")
        
        return self
    
    def _check_data_quality(self):
        """Check for any remaining issues"""
        print("\n" + "=" * 90)
        print("STEP 6: Data Quality Check")
        print("=" * 90)
        
        print(f"\nFinal dataset shapes:")
        print(f"  Train: {self.train_data.shape}")
        print(f"  Test:  {self.test_data.shape}")
        
        # Check for nulls
        train_nulls = self.train_data.isnull().sum().sum()
        test_nulls = self.test_data.isnull().sum().sum()
        
        print(f"\nNull values:")
        print(f"  Train: {train_nulls}")
        print(f"  Test:  {test_nulls}")
        
        if train_nulls == 0 and test_nulls == 0:
            print(f"  ✓ No null values")
        else:
            print(f"  ⚠ Still have null values - filling with 0")
            self.train_data = self.train_data.fillna(0)
            self.test_data = self.test_data.fillna(0)
        
        # Check data types
        print(f"\nData types:")
        print(f"  Numeric: {(self.train_data.dtypes == 'float64').sum()}")
        print(f"  Int: {(self.train_data.dtypes == 'int64').sum()}")
        print(f"  Object: {(self.train_data.dtypes == 'object').sum()}")
        
        # Check for any remaining object columns
        object_cols = self.train_data.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            print(f"\n  ⚠ Remaining object columns: {object_cols}")
        else:
            print(f"  ✓ All features are numeric")
        
        return self
    
    def save_encoded_data(self, train_output='train_encoded.csv',
                         test_output='test_encoded.csv',
                         config_output='encoding_config.json'):
        """Save encoded datasets and configuration"""
        print("\n" + "=" * 90)
        print("STEP 7: Save Encoded Data")
        print("=" * 90)
        
        self.train_data.to_csv(self.processed_dir + '/' + train_output, index=False)
        self.test_data.to_csv(self.processed_dir + '/' + test_output, index=False)
        
        print(f"✓ Train: {self.processed_dir}/{train_output} ({self.train_data.shape})")
        print(f"✓ Test:  {self.processed_dir}/{test_output} ({self.test_data.shape})")
        
        # Save config
        with open(self.interim_dir + '/' + config_output, 'w') as f:
            json.dump(self.encoding_config, f, indent=2, default=str)
        
        print(f"✓ Config: {self.interim_dir}/{config_output}")
        
        return self
    
    def get_summary(self):
        """Print summary statistics"""
        print("\n" + "=" * 90)
        print("ENCODING SUMMARY")
        print("=" * 90)
        
        print(f"\nEncoding breakdown:")
        print(f"  • Ordinal (label encoded):     {self.encoding_config.get('ordinal_features', 0):>3} features")
        print(f"  • One-Hot encoded:             {self.encoding_config.get('ohe_output_features', 0):>3} features")
        print(f"  • Target encoded:              {self.encoding_config.get('target_enc_features', 0):>3} features")
        print(f"  • Numeric (from transform):    {self.train_data.shape[1] - self.encoding_config.get('ordinal_features', 0) - self.encoding_config.get('ohe_output_features', 0) - self.encoding_config.get('target_enc_features', 0) - 1:>3} features")
        print(f"  ────────────────────────────────────────")
        print(f"  • Total features:              {self.train_data.shape[1] - 1:>3} features + target")
        
        print(f"\nFinal dataset:")
        print(f"  Train: {self.train_data.shape[0]:>6} samples × {self.train_data.shape[1] - 1:>3} features + target")
        print(f"  Test:  {self.test_data.shape[0]:>6} samples × {self.test_data.shape[1] - 1:>3} features + target")
        
        print(f"\n✅ Data ready for modeling!")
        return self
    
    def run_pipeline(self, train_path='train_transformed.csv',
                    test_path='test_transformed.csv'):
        """Run complete encoding pipeline"""
        return (self
                .load_splits(train_path, test_path)
                ._apply_ordinal_encoding()
                ._apply_target_encoding()
                ._apply_one_hot_encoding()
                ._drop_unnecessary_columns()
                ._standardize_features()
                ._check_data_quality()
                .save_encoded_data()
                .get_summary())


# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("=" * 90)
    print("CATEGORICAL ENCODING PIPELINE")
    print("=" * 90)
    
    # Run pipeline
    encoder = CategoricalEncoder()
    encoder.run_pipeline(
        train_path='train_transformed.csv',
        test_path='test_transformed.csv'
    )
    
    print("\n" + "=" * 90)
    print("✅ ENCODING COMPLETE")
    print("=" * 90)
    print("\nNext steps:")
    print("  1. Model training (XGBoost, LightGBM, etc.)")
    print("  2. Cross-validation & hyperparameter tuning")
    print("  3. Feature importance analysis")
    print("  4. Final predictions on test set")
