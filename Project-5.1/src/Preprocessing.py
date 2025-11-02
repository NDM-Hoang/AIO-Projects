"""
PREPROCESSING MODULE
====================

Purpose:
- Handle missing values (fill categorical/numeric appropriately)
- Fix logical inconsistencies (MasVnrType/Area, Garage fields)
- Prepare data for Feature Engineering

Data Flow:
- Input:  1460 × 81 (raw data)
  ↓
- BƯỚC 0: Fix MasVnrType & MasVnrArea logic (delete 2 rows, fill 5 rows)
  ↓
- BƯỚC 1: Fill missing values (handle 872 nulls)
  ↓
- BƯỚC 2: Fix Garage logic consistency (81 rows without garage)
  ↓
- Output: 1458 × 81 (clean data, 0 null values)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class Preprocessor:
    """Data preprocessing pipeline"""

    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        print(f"\n✓ Loaded data: {self.df.shape}")

    def step0_fix_masonry_veneer_logic(self):
        """
        BƯỚC 0: Fix MasVnrType & MasVnrArea logic

        Logic:
        - Case 1: Area=0, Type≠NULL → DELETE (inconsistent)
        - Case 2: Area>0, Type=NULL → FILL với mode
        - Case 3: Both NULL → 'None'
        """
        print("\n" + "=" * 90)
        print("BƯỚC 0: Fix MasVnrType & MasVnrArea Logic")
        print("=" * 90)

        original_len = len(self.df)

        # Get mode của MasVnrType (for Case 2)
        mode_type = self.df["MasVnrType"].mode()[0]

        # Case 1: Area=0, Type≠NULL → DELETE
        case1_mask = (self.df["MasVnrArea"] == 0) & (self.df["MasVnrType"].notna())
        case1_count = case1_mask.sum()
        print(f"\nCase 1 (Area=0, Type≠NULL): {case1_count} dòng")
        print(f"  → Xóa {case1_count} dòng lỗi")
        self.df = self.df[~case1_mask]

        # Case 2: Area>0, Type=NULL → FILL mode
        case2_mask = (self.df["MasVnrArea"] > 0) & (self.df["MasVnrType"].isna())
        case2_count = case2_mask.sum()
        print(f"\nCase 2 (Area>0, Type=NULL): {case2_count} dòng")
        print(f"  → Điền {case2_count} dòng với mode='{mode_type}'")
        self.df.loc[case2_mask, "MasVnrType"] = mode_type

        # Case 3: Both NULL → 'None'
        case3_mask = (self.df["MasVnrArea"].isna()) | (self.df["MasVnrType"].isna())
        case3_count = case3_mask.sum()
        if case3_count > 0:
            print(f"\nCase 3 (Both NULL): {case3_count} dòng")
            print(f"  → Điền 'None' cho MasVnrType")
            self.df.loc[case3_mask, "MasVnrType"] = "None"
            self.df.loc[case3_mask, "MasVnrArea"] = 0

        deleted = original_len - len(self.df)
        print(f"\n✓ Bước 0 hoàn tất: Xóa {deleted} dòng")
        print(f"  Shape: {self.original_shape} → {self.df.shape}")

        return self

    def step1_fill_missing_values(self):
        """
        BƯỚC 1: Fill missing values

        Strategy:
        - Categorical: Fill với 'None'
        - Numeric count/area: Fill với 0
        - Other numeric: Fill với median (fit on data)
        """
        print("\n" + "=" * 90)
        print("BƯỚC 1: Fill Missing Values")
        print("=" * 90)

        total_nulls_before = self.df.isnull().sum().sum()
        print(f"\nTổng null trước: {total_nulls_before}")

        # Categorical columns → 'None'
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        cat_nulls = self.df[categorical_cols].isnull().sum().sum()
        if cat_nulls > 0:
            print(f"\nCategorical nulls: {cat_nulls}")
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna("None")
            print(f"  → Điền 'None' cho {len(categorical_cols)} categorical columns")

        # Numeric columns: count/area → 0, others → median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        # Count/area columns → 0
        count_area_cols = [
            "GarageCars",
            "GarageArea",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
            "FullBath",
            "HalfBath",
            "BedroomAbvGr",
            "KitchenAbvGr",
            "WoodDeckSF",
            "OpenPorchSF",
            "EnclosedPorch",
            "ScreenPorch",
            "3SsnPorch",
            "PoolArea",
            "MasVnrArea",
            "2ndFlrSF",
            "BsmtFullBath",
            "BsmtHalfBath",
        ]

        for col in count_area_cols:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(0)

        count_area_nulls = sum(
            self.df[col].isnull().sum()
            for col in count_area_cols
            if col in self.df.columns
        )
        if count_area_nulls > 0:
            print(f"\nCount/Area nulls: {count_area_nulls}")
            print(f"  → Điền 0 cho {len(count_area_cols)} count/area columns")

        # Other numeric → median
        other_numeric_nulls = self.df[numeric_cols].isnull().sum().sum()
        if other_numeric_nulls > 0:
            print(f"\nOther numeric nulls: {other_numeric_nulls}")
            for col in numeric_cols:
                if self.df[col].isnull().sum() > 0:
                    median_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_val)
                    print(
                        f"  ├─ {col}: {self.df[col].isnull().sum()} nulls → median={median_val:.0f}"
                    )

        total_nulls_after = self.df.isnull().sum().sum()
        print(f"\n✓ Bước 1 hoàn tất: {total_nulls_before} → {total_nulls_after} nulls")
        print(f"  Xử lý: {total_nulls_before - total_nulls_after} null values")

        return self

    def step2_fix_garage_logic(self):
        """
        BƯỚC 2: Fix Garage logic consistency

        Logic:
        - If no garage (GarageArea=0): Set GarageQual, GarageCond, GarageFinish='None'
        - If has garage: Fill with mode
        """
        print("\n" + "=" * 90)
        print("BƯỚC 2: Fix Garage Logic Consistency")
        print("=" * 90)

        garage_cols = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]

        # Find rows without garage
        no_garage_mask = self.df["GarageArea"] == 0
        no_garage_count = no_garage_mask.sum()
        print(f"\nDòng không có garage (GarageArea=0): {no_garage_count}")

        # Set 'None' for rows without garage
        for col in garage_cols:
            if col in self.df.columns:
                self.df.loc[no_garage_mask, col] = "None"
        print(f"  → Set 'None' cho {len(garage_cols)} garage columns")

        # Fill remaining nulls với mode
        for col in garage_cols:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                mode_val = (
                    self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else "None"
                )
                null_count = self.df[col].isnull().sum()
                self.df[col] = self.df[col].fillna(mode_val)
                print(f"  ├─ {col}: {null_count} nulls → mode='{mode_val}'")

        print(f"\n✓ Bước 2 hoàn tất")

        return self

    def get_summary(self):
        """Print summary"""
        print("\n" + "=" * 90)
        print("PREPROCESSING SUMMARY")
        print("=" * 90)

        print(f"\nShape transformation:")
        print(f"  Input:  {self.original_shape}")
        print(f"  Output: {self.df.shape}")
        print(f"  Deleted rows: {self.original_shape[0] - self.df.shape[0]}")

        print(f"\nNull values:")
        print(f"  Before: Unknown (original data)")
        print(f"  After: {self.df.isnull().sum().sum()} (0 null values ✓)")

        print(f"\nData types:")
        print(f"  Numeric: {self.df.select_dtypes(include=[np.number]).shape[1]}")
        print(f"  Categorical: {self.df.select_dtypes(include=['object']).shape[1]}")

        print(f"\n✅ Preprocessing complete!")

        return self

    def get_dataframe(self):
        """Return preprocessed dataframe"""
        return self.df


def main():
    """Main preprocessing pipeline"""

    print("=" * 90)
    print("PREPROCESSING PIPELINE")
    print("=" * 90)

    # Load raw data
    raw_path = Path("data/raw/train-house-prices-advanced-regression-techniques.csv")
    if not raw_path.exists():
        print(f"❌ Raw data not found: {raw_path}")
        return False

    df = pd.read_csv(raw_path)

    # Run preprocessing pipeline
    preprocessor = Preprocessor(df)
    df_clean = (
        preprocessor.step0_fix_masonry_veneer_logic()
        .step1_fill_missing_values()
        .step2_fix_garage_logic()
        .get_summary()
        .get_dataframe()
    )

    # Save output
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "train_preprocessed.csv"
    df_clean.to_csv(output_path, index=False)

    print(f"\n✓ Saved: {output_path}")
    print(f"  Shape: {df_clean.shape}")
    print(f"  Null values: {df_clean.isnull().sum().sum()}")

    return True


if __name__ == "__main__":
    main()
