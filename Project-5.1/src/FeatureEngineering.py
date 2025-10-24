"""
FEATURE ENGINEERING MODULE
===========================

Purpose:
- Create derived features from preprocessed data
- Reduce multicollinearity through feature engineering
- Preserve predictive power while improving model efficiency

Pipeline:
1. Garage Features: GarageAreaPerCar, HasGarage
2. Area Features: AvgRoomSize
3. Basement Features: HasBasement, BasementResid (orthogonalized)
4. Age Features: HouseAge, GarageLag, GarageSameAsHouse
5. Quality Features: Flags + Residuals (Fireplace, Masonry, 2ndFloor)

Output: Enhanced features with reduced multicollinearity
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")


class FeatureEngineer:
    """
    Feature Engineering pipeline for House Prices dataset
    
    Applies derived features to reduce multicollinearity while
    preserving or improving predictive power.
    """
    
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.original_cols = dataframe.columns.tolist()
        self.new_features = []
    
    def engineer_garage_features(self):
        """
        STEP 1: Create Garage Features
        
        Strategy:
        - GarageAreaPerCar: Efficiency metric (area per car)
        - HasGarage: Binary flag (garage exists)
        - Drop: GarageCars (redundant, corr=0.88 with GarageArea)
        - Keep: GarageArea (important scale feature)
        
        Benefit: Reduce multicollinearity 0.88 → 0.30
        """
        print("=" * 90)
        print("STEP 1: Garage Features")
        print("=" * 90)
        
        # Feature 1: GarageAreaPerCar
        print("\n1. Create GarageAreaPerCar (efficiency metric):")
        self.df['GarageAreaPerCar'] = np.where(
            self.df['GarageCars'] > 0,
            self.df['GarageArea'] / self.df['GarageCars'],
            0
        )
        print(f"   Mean: {self.df['GarageAreaPerCar'].mean():.1f} sq ft/car")
        print(f"   Median: {self.df['GarageAreaPerCar'].median():.1f} sq ft/car")
        self.new_features.append('GarageAreaPerCar')
        
        # Feature 2: HasGarage
        print("\n2. Create HasGarage (binary flag):")
        self.df['HasGarage'] = (self.df['GarageArea'] > 0).astype(int)
        print(f"   Has garage: {self.df['HasGarage'].sum()} samples")
        self.new_features.append('HasGarage')
        
        # Drop GarageCars
        print("\n3. Drop GarageCars (redundant):")
        if 'GarageCars' in self.df.columns:
            self.df = self.df.drop('GarageCars', axis=1)
            print(f"   ✓ Dropped (correlation with GarageArea = 0.88)")
        
        print(f"\n✓ Garage features complete")
        return self
    
    def engineer_area_features(self):
        """
        STEP 2: Create Area Features
        
        Strategy:
        - AvgRoomSize: Efficiency metric (area per room)
        - Keep: GrLivArea (main scale feature, corr=0.71)
        - Drop: TotRmsAbvGrd (redundant, corr=0.825 with GrLivArea)
        
        Benefit: Reduce multicollinearity 0.825 → 0.654
        """
        print("\n" + "=" * 90)
        print("STEP 2: Area Features")
        print("=" * 90)
        
        # Feature 1: AvgRoomSize
        print("\n1. Create AvgRoomSize (efficiency metric):")
        self.df['AvgRoomSize'] = np.where(
            self.df['TotRmsAbvGrd'] > 0,
            self.df['GrLivArea'] / self.df['TotRmsAbvGrd'],
            0
        )
        print(f"   Mean: {self.df['AvgRoomSize'].mean():.1f} sq ft/room")
        self.new_features.append('AvgRoomSize')
        
        # Drop TotRmsAbvGrd
        print("\n2. Drop TotRmsAbvGrd (multicollinear):")
        if 'TotRmsAbvGrd' in self.df.columns:
            self.df = self.df.drop('TotRmsAbvGrd', axis=1)
            print(f"   ✓ Dropped (correlation with GrLivArea = 0.825)")
        
        print(f"\n✓ Area features complete")
        return self
    
    def engineer_basement_features(self):
        """
        STEP 3: Create Basement Features
        
        Strategy:
        - HasBasement: Binary flag (basement exists)
        - BasementResid: Orthogonalized residuals
          = TotalBsmtSF - E[TotalBsmtSF | 1stFlrSF, HasBasement]
        - Keep: 1stFlrSF (scale feature, corr=0.61)
        - Drop: TotalBsmtSF (redundant, corr=0.82 with 1stFlrSF)
        
        Benefit: Reduce VIF 3.045 → 1.001 (perfect orthogonalization)
        """
        print("\n" + "=" * 90)
        print("STEP 3: Basement Features")
        print("=" * 90)
        
        # Feature 1: HasBasement
        print("\n1. Create HasBasement (binary flag):")
        self.df['HasBasement'] = (self.df['TotalBsmtSF'] > 0).astype(int)
        print(f"   Has basement: {self.df['HasBasement'].sum()} samples")
        self.new_features.append('HasBasement')
        
        # Feature 2: BasementResid (orthogonalized)
        print("\n2. Create BasementResid (orthogonalized residuals):")
        X_basement = self.df[['1stFlrSF', 'HasBasement']].values
        y_bsmt = self.df['TotalBsmtSF'].values
        model = LinearRegression()
        model.fit(X_basement, y_bsmt)
        self.df['BasementResid'] = y_bsmt - model.predict(X_basement)
        print(f"   Mean: {self.df['BasementResid'].mean():.3f} (≈0, orthogonal)")
        self.new_features.append('BasementResid')
        
        # Drop TotalBsmtSF
        print("\n3. Drop TotalBsmtSF (multicollinear):")
        if 'TotalBsmtSF' in self.df.columns:
            self.df = self.df.drop('TotalBsmtSF', axis=1)
            print(f"   ✓ Dropped (correlation with 1stFlrSF = 0.82)")
        
        print(f"\n✓ Basement features complete")
        return self
    
    def engineer_age_features(self):
        """
        STEP 4: Create Age Features
        
        Strategy:
        - HouseAge: Years since construction (YrSold - YearBuilt)
        - GarageLag: Garage construction lag (GarageYrBlt - YearBuilt)
        - GarageSameAsHouse: Binary flag (garage built same year)
        - Drop: YearBuilt, GarageYrBlt (perfect inverse with derived)
        - Drop: GarageAge (subsumed by HouseAge + GarageLag)
        
        Benefit: Eliminate perfect multicollinearity (-0.999)
        """
        print("\n" + "=" * 90)
        print("STEP 4: Age Features")
        print("=" * 90)
        
        # Feature 1: HouseAge
        print("\n1. Create HouseAge (years since construction):")
        self.df['HouseAge'] = self.df['YrSold'] - self.df['YearBuilt']
        print(f"   Mean: {self.df['HouseAge'].mean():.1f} years")
        print(f"   Median: {self.df['HouseAge'].median():.0f} years")
        self.new_features.append('HouseAge')
        
        # Feature 2: GarageLag
        print("\n2. Create GarageLag (garage construction lag):")
        self.df['GarageLag'] = self.df['GarageYrBlt'] - self.df['YearBuilt']
        print(f"   Mean: {self.df['GarageLag'].mean():.1f} years")
        self.new_features.append('GarageLag')
        
        # Feature 3: GarageSameAsHouse
        print("\n3. Create GarageSameAsHouse (binary flag):")
        self.df['GarageSameAsHouse'] = (self.df['GarageYrBlt'] == self.df['YearBuilt']).astype(int)
        print(f"   Same year: {self.df['GarageSameAsHouse'].sum()} samples")
        self.new_features.append('GarageSameAsHouse')
        
        # Drop raw year features
        print("\n4. Drop raw year features (redundant):")
        cols_to_drop = []
        for col in ['YearBuilt', 'GarageYrBlt', 'GarageAge']:
            if col in self.df.columns:
                self.df = self.df.drop(col, axis=1)
                cols_to_drop.append(col)
        print(f"   ✓ Dropped {len(cols_to_drop)} columns")
        
        print(f"\n✓ Age features complete")
        return self
    
    def engineer_quality_features(self):
        """
        STEP 5: Create Quality Features (Flags + Residuals)
        
        Strategy:
        - Fireplace: HasFireplace + ExtraFireplaces
        - Masonry: HasMasonryVeneer + MasVnrAreaResid (orthogonal)
        - SecondFloor: Has2ndFlr + SecondFlrShare_resid (orthogonal)
        - Drop: Raw features (multicollinear)
        
        Benefit: Capture binary + continuous effects separately
        """
        print("\n" + "=" * 90)
        print("STEP 5: Quality Features (Flags + Residuals)")
        print("=" * 90)
        
        # ===== FIREPLACE =====
        print("\n1. FIREPLACE FEATURES:")
        
        self.df['HasFireplace'] = (self.df['Fireplaces'] > 0).astype(int)
        print(f"   HasFireplace: {self.df['HasFireplace'].sum()} samples")
        self.new_features.append('HasFireplace')
        
        self.df['ExtraFireplaces'] = np.maximum(self.df['Fireplaces'] - 1, 0)
        print(f"   ExtraFireplaces: max={self.df['ExtraFireplaces'].max()}")
        self.new_features.append('ExtraFireplaces')
        
        if 'Fireplaces' in self.df.columns:
            self.df = self.df.drop('Fireplaces', axis=1)
        
        # ===== MASONRY VENEER =====
        print("\n2. MASONRY VENEER FEATURES:")
        
        self.df['MasVnrArea'] = self.df['MasVnrArea'].fillna(0)
        self.df['HasMasonryVeneer'] = (self.df['MasVnrArea'] > 0).astype(int)
        print(f"   HasMasonryVeneer: {self.df['HasMasonryVeneer'].sum()} samples")
        self.new_features.append('HasMasonryVeneer')
        
        # MasVnrAreaResid (orthogonalized)
        X_masonry = self.df[['HasMasonryVeneer', 'OverallQual']].values
        y_masonry = self.df['MasVnrArea'].values
        model_m = LinearRegression()
        model_m.fit(X_masonry, y_masonry)
        self.df['MasVnrAreaResid'] = y_masonry - model_m.predict(X_masonry)
        print(f"   MasVnrAreaResid: orthogonal (corr≈0)")
        self.new_features.append('MasVnrAreaResid')
        
        if 'MasVnrArea' in self.df.columns:
            self.df = self.df.drop('MasVnrArea', axis=1)
        
        # ===== SECOND FLOOR =====
        print("\n3. SECOND FLOOR FEATURES:")
        
        self.df['Has2ndFlr'] = (self.df['2ndFlrSF'] > 0).astype(int)
        print(f"   Has2ndFlr: {self.df['Has2ndFlr'].sum()} samples")
        self.new_features.append('Has2ndFlr')
        
        # SecondFlrShare_resid (orthogonalized)
        second_flr_share = np.where(
            self.df['GrLivArea'] > 0,
            self.df['2ndFlrSF'] / self.df['GrLivArea'],
            0
        )
        X_share = self.df[['Has2ndFlr']].values
        model_s = LinearRegression()
        model_s.fit(X_share, second_flr_share)
        self.df['SecondFlrShare_resid'] = second_flr_share - model_s.predict(X_share)
        print(f"   SecondFlrShare_resid: orthogonal (corr≈0)")
        self.new_features.append('SecondFlrShare_resid')
        
        if '2ndFlrSF' in self.df.columns:
            self.df = self.df.drop('2ndFlrSF', axis=1)
        
        print(f"\n✓ Quality features complete")
        return self
    
    def get_summary(self):
        """Print engineering summary"""
        print("\n" + "=" * 90)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 90)
        
        original_features = len(self.original_cols)
        current_features = len(self.df.columns)
        
        print(f"\nOriginal features:  {original_features}")
        print(f"Current features:   {current_features}")
        print(f"Net change:         {current_features - original_features:+d}")
        
        if len(self.new_features) > 0:
            print(f"\nNew features created ({len(self.new_features)}):")
            for feat in self.new_features:
                print(f"  + {feat}")
        
        dropped_features = [col for col in self.original_cols if col not in self.df.columns]
        if len(dropped_features) > 0:
            print(f"\nFeatures dropped ({len(dropped_features)}):")
            for feat in dropped_features:
                print(f"  - {feat}")
        
        print(f"\n✓ Feature Engineering complete!")
        return self
    
    def get_dataframe(self):
        """Return engineered dataframe"""
        return self.df


# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("Loading preprocessed data...")
    df = pd.read_csv("train-house-prices-advanced-regression-techniques.csv")
    print(f"✓ Loaded {df.shape} records\n")
    
    # Feature engineering pipeline
    engineer = FeatureEngineer(df)
    df_engineered = (engineer
                     .engineer_garage_features()
                     .engineer_area_features()
                     .engineer_basement_features()
                     .engineer_age_features()
                     .engineer_quality_features()
                     .get_summary()
                     .get_dataframe())
    
    print(f"\n✓ Engineered dataset: {df_engineered.shape}")
    print(f"✓ Ready for next step: Transformation & Encoding\n")
