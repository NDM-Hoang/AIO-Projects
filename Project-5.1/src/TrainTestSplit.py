"""
TRAIN/TEST SPLIT MODULE
========================

Purpose:
- Split data into train (85%) and test (15%) sets
- Apply preprocessing to both sets consistently
- Apply feature engineering to both sets
- Save splits for modeling

Strategy:
- Train set: 85% → used for training & validation
- Test set: 15% → held out for final evaluation
- Preprocessing & FE applied to both

Output:
- train_data.csv / test_data.csv
- Feature list for reference
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
import warnings

warnings.filterwarnings("ignore")


class DataSplitter:
    """Handle train/test split with preprocessing and feature engineering"""

    def __init__(self, data_path, test_size=0.15, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Load raw data"""
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded data: {self.df.shape}")
        return self

    def apply_preprocessing(self):
        """Apply preprocessing to entire dataset"""
        print("\n" + "=" * 90)
        print("STEP 1: Preprocessing")
        print("=" * 90)

        # Fill categorical nulls
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna("None")
        print(f"✓ Filled {len(categorical_cols)} categorical columns with 'None'")

        # Fill numeric nulls
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        null_cols = []
        for col in numeric_cols:
            if self.df[col].isna().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                null_cols.append(col)
        print(f"✓ Filled {len(null_cols)} numeric columns with median")

        print(f"  Final shape after preprocessing: {self.df.shape}")
        return self

    def apply_feature_engineering(self):
        """Apply feature engineering to entire dataset"""
        print("\n" + "=" * 90)
        print("STEP 2: Feature Engineering")
        print("=" * 90)

        # 1. Garage features
        self.df["GarageAreaPerCar"] = np.where(
            self.df["GarageCars"] > 0, self.df["GarageArea"] / self.df["GarageCars"], 0
        )
        self.df["HasGarage"] = (self.df["GarageArea"] > 0).astype(int)
        self.df = self.df.drop("GarageCars", axis=1)
        print("✓ Garage features: GarageAreaPerCar, HasGarage")

        # 2. Area features
        self.df["AvgRoomSize"] = np.where(
            self.df["TotRmsAbvGrd"] > 0,
            self.df["GrLivArea"] / self.df["TotRmsAbvGrd"],
            0,
        )
        self.df = self.df.drop("TotRmsAbvGrd", axis=1)
        print("✓ Area features: AvgRoomSize")

        # 3. Basement features (orthogonalized)
        self.df["HasBasement"] = (self.df["TotalBsmtSF"] > 0).astype(int)
        X_basement = self.df[["1stFlrSF", "HasBasement"]].values
        y_bsmt = self.df["TotalBsmtSF"].values
        model = LinearRegression()
        model.fit(X_basement, y_bsmt)
        self.df["BasementResid"] = y_bsmt - model.predict(X_basement)
        self.df = self.df.drop("TotalBsmtSF", axis=1)
        print("✓ Basement features: HasBasement, BasementResid (orthogonalized)")

        # 4. Age features
        self.df["HouseAge"] = self.df["YrSold"] - self.df["YearBuilt"]
        self.df["GarageLag"] = self.df["GarageYrBlt"] - self.df["YearBuilt"]
        self.df["GarageSameAsHouse"] = (
            self.df["GarageYrBlt"] == self.df["YearBuilt"]
        ).astype(int)
        self.df = self.df.drop(["YearBuilt", "GarageYrBlt"], axis=1)
        if "GarageAge" in self.df.columns:
            self.df = self.df.drop("GarageAge", axis=1)
        print("✓ Age features: HouseAge, GarageLag, GarageSameAsHouse")

        # 5. Fireplace features
        self.df["HasFireplace"] = (self.df["Fireplaces"] > 0).astype(int)
        self.df["ExtraFireplaces"] = np.maximum(self.df["Fireplaces"] - 1, 0)
        self.df = self.df.drop("Fireplaces", axis=1)
        print("✓ Fireplace features: HasFireplace, ExtraFireplaces")

        # 6. Masonry veneer features (orthogonalized)
        self.df["MasVnrArea"] = self.df["MasVnrArea"].fillna(0)
        self.df["HasMasonryVeneer"] = (self.df["MasVnrArea"] > 0).astype(int)
        X_masonry = self.df[["HasMasonryVeneer", "OverallQual"]].values
        y_masonry = self.df["MasVnrArea"].values
        model_m = LinearRegression()
        model_m.fit(X_masonry, y_masonry)
        self.df["MasVnrAreaResid"] = y_masonry - model_m.predict(X_masonry)
        self.df = self.df.drop("MasVnrArea", axis=1)
        print(
            "✓ Masonry veneer features: HasMasonryVeneer, MasVnrAreaResid (orthogonalized)"
        )

        # 7. Second floor features (orthogonalized)
        self.df["Has2ndFlr"] = (self.df["2ndFlrSF"] > 0).astype(int)
        self.df["SecondFlrShare"] = np.where(
            self.df["GrLivArea"] > 0, self.df["2ndFlrSF"] / self.df["GrLivArea"], 0
        )
        X_share = self.df[["Has2ndFlr"]].values
        y_share = self.df["SecondFlrShare"].values
        model_s = LinearRegression()
        model_s.fit(X_share, y_share)
        self.df["SecondFlrShare_resid"] = y_share - model_s.predict(X_share)
        self.df = self.df.drop(["SecondFlrShare", "2ndFlrSF"], axis=1)
        print(
            "✓ Second floor features: Has2ndFlr, SecondFlrShare_resid (orthogonalized)"
        )

        print(f"\n  Final shape after FE: {self.df.shape}")
        return self

    def split_data(self):
        """Split data into train/test"""
        print("\n" + "=" * 90)
        print("STEP 3: Train/Test Split")
        print("=" * 90)

        # Extract target
        y = self.df["SalePrice"].copy()
        X = self.df.drop("SalePrice", axis=1)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Store
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        print(f"\nTrain set: {self.X_train.shape}")
        print(f"  X_train shape: {self.X_train.shape}")
        print(f"  y_train shape: {self.y_train.shape}")
        print(
            f"  Percentage: {len(self.X_train) / (len(self.X_train) + len(self.X_test)) * 100:.1f}%"
        )

        print(f"\nTest set: {self.X_test.shape}")
        print(f"  X_test shape: {self.X_test.shape}")
        print(f"  y_test shape: {self.y_test.shape}")
        print(
            f"  Percentage: {len(self.X_test) / (len(self.X_train) + len(self.X_test)) * 100:.1f}%"
        )

        return self

    def save_splits(self, output_dir="./"):
        """Save train/test splits to CSV"""
        print("\n" + "=" * 90)
        print("STEP 4: Save Splits")
        print("=" * 90)

        # Combine X and y for saving
        train_data = self.X_train.copy()
        train_data["SalePrice"] = self.y_train.values

        test_data = self.X_test.copy()
        test_data["SalePrice"] = self.y_test.values

        # Save
        train_path = f"{output_dir}/train_data.csv"
        test_path = f"{output_dir}/test_data.csv"

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        print(
            f"✓ Saved train_data.csv ({train_data.shape[0]} rows, {train_data.shape[1]} cols)"
        )
        print(
            f"✓ Saved test_data.csv ({test_data.shape[0]} rows, {test_data.shape[1]} cols)"
        )

        # Save feature list
        features_path = f"{output_dir}/feature_list.txt"
        with open(features_path, "w") as f:
            f.write("FEATURE LIST FOR MODEL\n")
            f.write(f"Total features: {len(self.X_train.columns)}\n")
            f.write(f"Target: SalePrice\n\n")
            f.write("Features:\n")
            for i, feat in enumerate(self.X_train.columns, 1):
                f.write(f"{i:3d}. {feat}\n")

        print(f"✓ Saved feature_list.txt ({len(self.X_train.columns)} features)")

        return self

    def get_summary(self):
        """Print summary statistics"""
        print("\n" + "=" * 90)
        print("SUMMARY")
        print("=" * 90)

        print(f"\nDataset Summary:")
        print(f"  Raw data: {self.df.shape[0]} rows")
        print(f"  After preprocessing & FE: {self.df.shape}")
        print(f"  Features (excluding target): {self.X_train.shape[1]}")

        print(f"\nTrain/Test Split (random_state={self.random_state}):")
        print(f"  Train: {self.X_train.shape[0]} samples (85%)")
        print(f"  Test:  {self.X_test.shape[0]} samples (15%)")

        print(f"\nTarget Variable (SalePrice):")
        print(
            f"  Train - Mean: ${self.y_train.mean():,.0f}, Std: ${self.y_train.std():,.0f}"
        )
        print(
            f"  Test  - Mean: ${self.y_test.mean():,.0f}, Std: ${self.y_test.std():,.0f}"
        )
        print(
            f"  Train - Min: ${self.y_train.min():,.0f}, Max: ${self.y_train.max():,.0f}"
        )
        print(
            f"  Test  - Min: ${self.y_test.min():,.0f}, Max: ${self.y_test.max():,.0f}"
        )

        print(f"\nData Types:")
        numeric_count = self.X_train.select_dtypes(include=[np.number]).shape[1]
        categorical_count = self.X_train.select_dtypes(include=["object"]).shape[1]
        print(f"  Numeric features: {numeric_count}")
        print(f"  Categorical features: {categorical_count}")

        print(f"\n✅ Ready for modeling!")
        return self

    def run_pipeline(self, output_dir="./"):
        """Run complete pipeline"""
        return (
            self.load_data()
            .apply_preprocessing()
            .apply_feature_engineering()
            .split_data()
            .save_splits(output_dir)
            .get_summary()
        )


# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("=" * 90)
    print("TRAIN/TEST SPLIT PIPELINE")
    print("=" * 90)

    # Run pipeline
    splitter = DataSplitter(
        data_path="train-house-prices-advanced-regression-techniques.csv",
        test_size=0.15,
        random_state=42,
    )

    splitter.run_pipeline(output_dir="./")

    print("\n" + "=" * 90)
    print("✅ PIPELINE COMPLETE")
    print("=" * 90)
    print("\nNext steps:")
    print("  1. Load train_data.csv for model training")
    print("  2. Load test_data.csv for final evaluation")
    print("  3. Apply transformations (Log target, Yeo-Johnson features)")
    print("  4. Encode categorical variables")
    print("  5. Train models")
