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

    def __init__(self, processed_dir="data/processed", interim_dir="data/interim"):
        self.train_data = None  # DataFrame chứa dữ liệu train
        self.test_data = None  # DataFrame chứa dữ liệu test
        self.transformation_params = {}  # Lưu tham số biến đổi cho tái lập
        self.feature_strategy = {}  # Lưu chiến lược biến đổi mỗi đặc trưng
        self.processed_dir = processed_dir  # Đường dẫn lưu data đã xử lý
        self.interim_dir = interim_dir  # Đường dẫn lưu config tạm thời

    def load_splits(self, train_path, test_path):
        """Load train/test splits"""
        # Đọc tập train và test vào DataFrame
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

        numeric_df = self.train_data.select_dtypes(
            include=[np.number]
        )  # Lọc ra các đặc trưng số

        # Định nghĩa nhóm cờ nhị phân (không cần biến đổi)
        binary_flags = [
            "HasGarage",
            "HasBasement",
            "HasFireplace",
            "HasMasonryVeneer",
            "Has2ndFlr",
            "GarageSameAsHouse",
        ]

        # Đặc trưng phần dư (đã được trực giao hóa, không biến đổi)
        residual_features = ["BasementResid", "MasVnrAreaResid", "SecondFlrShare_resid"]

        # Các đặc trưng cực kỳ hiếm (không biến đổi)
        ultra_rare = ["PoolArea", "MiscVal", "3SsnPorch", "LowQualFinSF"]

        # Gán chiến lược trước cho một số cột quan trọng
        strategies = {
            "SalePrice": "target_log",  # Đích cần log1p
            "GrLivArea": "log",  # Diện tích chính log1p
            "LotArea": "log",  # Diện tích đất log1p
            "AvgRoomSize": "log",  # Trung bình kích thước phòng log1p
            "LotFrontage": "log",
            "KitchenAbvGr": "bin_count",  # Đặc trưng đếm - phân bin
            "ExtraFireplaces": "log",
        }

        # Gán chiến lược biến đổi cho các đặc trưng số còn lại
        for col in numeric_df.columns:
            if col == "Id":  # cột Id bỏ qua
                self.feature_strategy[col] = "skip"
            elif col in binary_flags:  # cờ nhị phân bỏ qua
                self.feature_strategy[col] = "skip_binary"
            elif col in residual_features:  # phần dư bỏ qua
                self.feature_strategy[col] = "skip_residual"
            elif col in ultra_rare:  # đặc trưng cực hiếm bỏ qua
                self.feature_strategy[col] = "skip_ultra_rare"
            elif col in strategies:  # cột đã khai báo sẵn chiến lược
                self.feature_strategy[col] = strategies[col]
            else:
                # Tự động phát hiện kiểu biến đổi
                col_data = numeric_df[col].dropna()
                if (col_data > 0).all():  # Nếu toàn số dương
                    self.feature_strategy[col] = "log"
                elif (col_data >= 0).all():  # >=0, khả năng zero-inflated
                    self.feature_strategy[col] = "log_zero_inflated"
                else:  # Có số âm, dùng Yeo-Johnson
                    self.feature_strategy[col] = "yeo_johnson"

        # In ra tóm tắt chiến lược cho từng nhóm
        print("\nFeature Transformation Strategy:")
        for strategy_type in set(self.feature_strategy.values()):
            features_in_group = [
                f for f, s in self.feature_strategy.items() if s == strategy_type
            ]
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

        if "SalePrice" in self.train_data.columns:
            # Lưu giá trị ban đầu
            y_train = self.train_data["SalePrice"].values
            y_test = self.test_data["SalePrice"].values

            original_skew_train = skew(y_train)  # Độ lệch skew trước khi transform

            # Dùng log1p để giảm skew
            y_train_transformed = np.log1p(y_train)
            y_test_transformed = np.log1p(y_test)

            new_skew_train = skew(y_train_transformed)  # Skew sau khi biến đổi

            print(f"\nSalePrice Transformation:")
            print(f"  Train original skew: {original_skew_train:+.3f}")
            print(f"  Train transformed skew: {new_skew_train:+.3f}")
            print(
                f"  Reduction: {((original_skew_train - new_skew_train) / original_skew_train * 100):.1f}%"
            )

            # Ghi đè dữ liệu
            self.train_data["SalePrice"] = y_train_transformed
            self.test_data["SalePrice"] = y_test_transformed

            # Lưu lại thông tin biến đổi cho tái lập/đảo ngược
            self.transformation_params["SalePrice"] = {
                "method": "log1p",
                "original_skew": float(original_skew_train),
                "transformed_skew": float(new_skew_train),
                "inverse": "np.expm1(x)",  # công thức inverse
            }

        return self

    def _bin_kitchen_abvgr(self):
        """Bin KitchenAbvGr into categories"""
        print("\n" + "=" * 90)
        print("STEP 3: Bin KitchenAbvGr (Count Feature)")
        print("=" * 90)

        if "KitchenAbvGr" in self.train_data.columns:
            print("\nOriginal KitchenAbvGr distribution (train):")
            print(self.train_data["KitchenAbvGr"].value_counts().sort_index())

            # Hàm phân nhóm KitchenAbvGr thành 3 bin:
            # - 0 và 1: bin 0
            # - 2:      bin 1
            # - >=3:    bin 2
            def bin_kitchen(x):
                if x <= 1:
                    return 0
                elif x == 2:
                    return 1
                else:
                    return 2

            # Tạo cột binned mới
            self.train_data["KitchenAbvGr_Binned"] = self.train_data[
                "KitchenAbvGr"
            ].apply(bin_kitchen)
            self.test_data["KitchenAbvGr_Binned"] = self.test_data[
                "KitchenAbvGr"
            ].apply(bin_kitchen)

            print("\nBinned distribution (train):")
            print(self.train_data["KitchenAbvGr_Binned"].value_counts().sort_index())

            # Tạo cờ "HasMultiKitchen": 1 nếu có ≥2 kitchen
            self.train_data["HasMultiKitchen"] = (
                self.train_data["KitchenAbvGr"] >= 2
            ).astype(int)
            self.test_data["HasMultiKitchen"] = (
                self.test_data["KitchenAbvGr"] >= 2
            ).astype(int)

            print("\nHasMultiKitchen distribution (train):")
            print(self.train_data["HasMultiKitchen"].value_counts())

            # Xóa cột cũ KitchenAbvGr
            self.train_data = self.train_data.drop("KitchenAbvGr", axis=1)
            self.test_data = self.test_data.drop("KitchenAbvGr", axis=1)

            # Lưu thông tin về binning và flag
            self.transformation_params["KitchenAbvGr"] = {
                "method": "bin",
                "bins": {0: "<=1", 1: "2", 2: ">=3"},
                "flags": ["KitchenAbvGr_Binned", "HasMultiKitchen"],
            }

            print("\n✓ Created: KitchenAbvGr_Binned, HasMultiKitchen")

        return self

    def _transform_features_log(self):
        """Apply log1p to positive features"""
        print("\n" + "=" * 90)
        print("STEP 4: Log1p Transform (Positive Features)")
        print("=" * 90)

        # Chọn ra những đặc trưng áp dụng log1p
        log_features = [f for f, s in self.feature_strategy.items() if s == "log"]

        print(f"\nApplying log1p to {len(log_features)} features:")

        for feat in log_features:
            if feat in self.train_data.columns:
                original_skew = skew(
                    self.train_data[feat].dropna()
                )  # Skew trước biến đổi

                # Biến đổi bằng log1p, ghi sang cột mới
                self.train_data[f"{feat}_log"] = np.log1p(self.train_data[feat])
                self.test_data[f"{feat}_log"] = np.log1p(self.test_data[feat])

                transformed_skew = skew(
                    self.train_data[f"{feat}_log"].dropna()
                )  # Skew sau biến đổi

                print(f"  {feat:<30} {original_skew:+.3f} → {transformed_skew:+.3f}")

                # Lưu thông tin phép biến đổi
                self.transformation_params[feat] = {
                    "method": "log1p",
                    "original_skew": float(original_skew),
                    "transformed_skew": float(transformed_skew),
                }

                # Loại bỏ đặc trưng gốc (chỉ giữ đặc trưng mới)
                self.train_data = self.train_data.drop(feat, axis=1)
                self.test_data = self.test_data.drop(feat, axis=1)

        return self

    def _transform_features_yeo_johnson(self):
        """Apply Yeo-Johnson to zero-inflated/negative features"""
        print("\n" + "=" * 90)
        print("STEP 5: Yeo-Johnson Transform (Zero-Inflated/Negative)")
        print("=" * 90)

        # Chọn ra đặc trưng cần biến đổi Yeo-Johnson hoặc log1p cho zero-inflated
        yj_features = [
            f
            for f, s in self.feature_strategy.items()
            if s in ["yeo_johnson", "log_zero_inflated"]
        ]

        # Chỉ lấy những cột hiện diện thực tế trong data
        numeric_df_train = self.train_data[
            [f for f in yj_features if f in self.train_data.columns]
        ].copy()
        numeric_df_test = self.test_data[
            [f for f in yj_features if f in self.test_data.columns]
        ].copy()

        print(f"\nApplying Yeo-Johnson to {len(yj_features)} features:")

        # Tạo transformer (không chuẩn hóa mean/std)
        pt = PowerTransformer(method="yeo-johnson", standardize=False)

        # Fit transformer trên train data
        pt.fit(numeric_df_train)

        # Áp dụng biến đổi trên cả hai tập train/test
        train_transformed = pt.transform(numeric_df_train)
        test_transformed = pt.transform(numeric_df_test)

        # Thêm cột mới và lưu lại thông tin biến đổi
        for idx, feat in enumerate(
            [f for f in yj_features if f in self.train_data.columns]
        ):
            original_skew = skew(numeric_df_train[feat].dropna())
            transformed_skew = skew(train_transformed[:, idx])

            print(
                f"  {feat:<30} {original_skew:+.3f} → {transformed_skew:+.3f} (λ={pt.lambdas_[idx]:.3f})"
            )

            self.train_data[f"{feat}_yj"] = train_transformed[:, idx]
            self.test_data[f"{feat}_yj"] = test_transformed[:, idx]

            self.transformation_params[feat] = {
                "method": "yeo_johnson",
                "lambda": float(pt.lambdas_[idx]),
                "original_skew": float(original_skew),
                "transformed_skew": float(transformed_skew),
            }

            # Xóa đặc trưng gốc
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
        print(
            f"{'Feature':<30} {'Original':<12} {'Transformed':<12} {'Reduction %':<12}"
        )
        print("-" * 100)

        skew_reduced_count = 0  # Đếm số đặc trưng giảm skew sau khi transform
        for feat, params in self.transformation_params.items():
            if "original_skew" in params and "transformed_skew" in params:
                orig = params["original_skew"]
                trans = params["transformed_skew"]
                if orig != 0:
                    reduction = (abs(orig) - abs(trans)) / abs(orig) * 100
                else:
                    reduction = 0

                status = "✓" if abs(trans) < abs(orig) else "⚠"
                print(
                    f"{feat:<30} {orig:+9.3f}    {trans:+9.3f}    {reduction:+9.1f}%  {status}"
                )

                if abs(trans) < abs(orig):
                    skew_reduced_count += 1

        print(f"\n✓ {skew_reduced_count} features had skewness reduced")

        return self

    def _save_transformation_config(self, output_path="transformation_config.json"):
        """Save transformation parameters for reproducibility"""
        print("\n" + "=" * 90)
        print("STEP 7: Save Transformation Configuration")
        print("=" * 90)

        # Lưu file thông tin biến đổi để sau có thể tái lập hoặc áp dụng cho test
        full_path = self.interim_dir + "/" + output_path
        with open(full_path, "w") as f:
            json.dump(self.transformation_params, f, indent=2)

        print(f"✓ Saved: {full_path}")
        return self

    def save_transformed_data(
        self, train_output="train_transformed.csv", test_output="test_transformed.csv"
    ):
        """Save transformed datasets"""
        print("\n" + "=" * 90)
        print("STEP 8: Save Transformed Data")
        print("=" * 90)

        # Đường dẫn lưu file
        train_path = self.processed_dir + "/" + train_output
        test_path = self.processed_dir + "/" + test_output

        # Lưu data đã biến đổi ra file CSV
        self.train_data.to_csv(train_path, index=False)
        self.test_data.to_csv(test_path, index=False)

        print(f"✓ Train: {train_path} ({self.train_data.shape})")
        print(f"✓ Test:  {test_path} ({self.test_data.shape})")

        # Tóm tắt số lượng cột (biến) sau biến đổi
        print(f"\nFeature Changes:")
        print(f"  Train columns: {self.train_data.shape[1]}")
        print(f"  Test columns:  {self.test_data.shape[1]}")

        return self

    def run_pipeline(self, train_path="train_data.csv", test_path="test_data.csv"):
        """Run complete transformation pipeline"""
        # Chạy toàn bộ pipeline: đọc, xác định chiến lược, biến đổi, lưu lại,...
        return (
            self.load_splits(train_path, test_path)
            ._identify_features_strategy()
            ._transform_target()
            ._bin_kitchen_abvgr()
            ._transform_features_log()
            ._transform_features_yeo_johnson()
            ._check_transformation_quality()
            ._save_transformation_config()
            .save_transformed_data()
        )


# ============================================================================
# ADDITIONAL ANALYSIS
# ============================================================================


def analyze_remaining_skewness(train_path="train_transformed.csv"):
    """Analyze skewness of transformed features"""
    print("\n" + "=" * 90)
    print("ADDITIONAL: Analyze Remaining Skewness")
    print("=" * 90)

    # Đọc tập đã transform để kiểm tra lại độ skew còn tồn tại
    df = pd.read_csv(train_path)
    numeric_df = df.select_dtypes(include=[np.number])

    # Tính toán skew từng đặc trưng số
    skew_values = {col: skew(numeric_df[col].dropna()) for col in numeric_df.columns}
    skew_sorted = sorted(skew_values.items(), key=lambda x: abs(x[1]), reverse=True)

    # In ra 15 đặc trưng có skew lớn nhất sau transform
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

    # Đếm số lượng đặc trưng theo từng ngưỡng skew
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
    transformer.run_pipeline(train_path="train_data.csv", test_path="test_data.csv")

    # Analyze remaining skewness sau khi transform
    analyze_remaining_skewness(train_path="train_transformed.csv")

    print("\n" + "=" * 90)
    print("✅ TRANSFORMATION COMPLETE")
    print("=" * 90)
    print("\nNext steps:")
    print("  1. Categorical encoding (Encoding.py)")
    print("  2. Feature scaling/normalization")
    print("  3. Model training")
