"""
SKLEARN ENCODING PIPELINE
=========================

Mục tiêu:
- Ordinal encode các cột chất lượng/thứ bậc (Ex > Gd > TA > Fa > Po, v.v.)
- One-hot encode các cột phân loại nominal (ít/mid cardinality)
- Target encode các cột high-cardinality (Neighborhood, Exterior2nd)
    -> dùng TargetEncoder với cross-fit K-fold cv=5 để tránh leakage
- Chuẩn hoá tất cả feature numeric cuối cùng bằng StandardScaler
- Xuất ra DataFrame X_train, X_test đã encode + scale, kèm SalePrice

Yêu cầu:
- scikit-learn >= phiên bản có sklearn.preprocessing.TargetEncoder
- pandas, numpy

Lưu ý:
- Ta drop 'Id' ra khỏi X trước khi encode vì không mang ý nghĩa dự báo.
- Các cột ordinal sẽ được encode theo đúng thứ tự mình định nghĩa.
- Các cột nominal sẽ one-hot (drop='first' để giảm đa cộng tuyến).
- Các cột high-cardinality sẽ target-encode (cv=5 nội bộ).
- Numeric còn lại (ví dụ diện tích, age, ratios, flags 0/1...) sẽ đi qua imputer + giữ nguyên (sau đó tất cả được scale).
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    TargetEncoder,      # cần sklearn có TargetEncoder
    StandardScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class SklearnEncodingPipeline:
    def __init__(self):
        # ===== 1. Định nghĩa các nhóm cột =====
        # a) Ordinal mappings (thứ bậc chất lượng / mức độ)
        #    Key: cột; Value: dict {category: rank}, rank nhỏ = "tệ hơn"
        self.ordinal_mappings = {
            'ExterQual':     {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'ExterCond':     {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'KitchenQual':   {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'HeatingQC':     {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'FireplaceQu':   {'None': -1, 'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'GarageQual':    {'None': -1, 'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'GarageCond':    {'None': -1, 'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'BsmtQual':      {'None': -1, 'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'BsmtCond':      {'None': -1, 'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'GarageFinish':  {'None': -1, 'Unf': 1, 'RFn': 2, 'Fin': 3},
            'BsmtExposure':  {'None': -1, 'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3},
            'BsmtFinType1':  {'None': -1, 'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5},
            'BsmtFinType2':  {'None': -1, 'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5},
            'LotShape':      {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3},
            'LandSlope':     {'Sev': 0, 'Mod': 1, 'Gtl': 2},
            'PavedDrive':    {'N': 0, 'P': 1, 'Y': 2},
            'Functional':    {'Sev': 0, 'Maj2': 1, 'Maj1': 2, 'Mod': 3,
                               'Min2': 4, 'Min1': 5, 'Typ': 6},
        }
        self.ordinal_cols = list(self.ordinal_mappings.keys())

        # b) Nominal (one-hot)
        self.ohe_cols = [
            'MSZoning', 'Street', 'Alley', 'LandContour', 'Utilities', 'LotConfig',
            'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
            'Exterior1st', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical',
            'GarageType', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition',
            'LandContour'  # nếu lặp lại thì lát nữa sẽ deduplicate
        ]

        # c) High-cardinality → TargetEncoder
        self.target_enc_cols = ['Neighborhood', 'Exterior2nd']

        # d) Tên cột bị drop chắc chắn
        self.drop_cols = ['Id']

        # sau khi fit, mình sẽ lưu:
        self.pipeline_ = None
        self.feature_names_ = None

    def _build_pipeline(self, X):
        """
        Tạo sklearn Pipeline với:
        - OrdinalEncoder cho self.ordinal_cols
        - SimpleImputer + OneHotEncoder cho self.ohe_cols
        - SimpleImputer + TargetEncoder cho self.target_enc_cols
        - SimpleImputer cho numeric
        - Cuối cùng StandardScaler cho toàn bộ vector đã encode
        """

        # Lấy danh sách cột thực sự tồn tại trong X
        ordinal_cols = [c for c in self.ordinal_cols if c in X.columns]
        ohe_cols     = [c for c in self.ohe_cols if c in X.columns]
        target_cols  = [c for c in self.target_enc_cols if c in X.columns]

        # numeric_cols = tất cả numeric còn lại không nằm trong ba nhóm trên + không phải drop_cols
        candidate_numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [
            c for c in candidate_numeric
            if c not in self.drop_cols
            and c not in ordinal_cols
            and c not in ohe_cols
            and c not in target_cols
        ]

        # ----- OrdinalEncoder -----
        # Ta cần đưa thứ tự category cho từng cột ordinal
        # sklearn.OrdinalEncoder muốn categories=[list_for_col1, list_for_col2, ...]
        ordinal_categories = []
        for col in ordinal_cols:
            mapping = self.ordinal_mappings[col]   # ex: {'None': -1, 'Po':0, ...}
            # sắp xếp key theo giá trị rank tăng dần
            ordered_levels = sorted(mapping.keys(), key=lambda k: mapping[k])
            ordinal_categories.append(ordered_levels)

        ordinal_encoder = OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=-1,
        )

        # ----- OneHotEncoder -----
        # Nhiều cột nominal có NA => Imputer(most_frequent) trước OHE
        ohe_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(
                handle_unknown='ignore',
                drop='first',            # tránh multicollinearity quá mạnh
                sparse_output=False,     # output dense để scaler xử lý được
            )),
        ])

        # ----- TargetEncoder -----
        # Với TargetEncoder, ta cũng impute most_frequent trước
        # TargetEncoder trong sklearn có tham số cv=5 mặc định để cross-fit
        # và fit_transform() sẽ tự dùng cross-fitting để tránh leakage.
        # Khi Pipeline.fit_transform chạy cho train, nó sẽ gọi đúng logic này.
        tgt_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('tgt', TargetEncoder(
                cv=5,              # cross-fit K-fold =5
                smooth='auto',     # smoothing shrink về global mean
                random_state=0,
            )),
        ])

        # ----- Numeric -----
        # Impute median cho numeric còn lại (LotFrontage,...)
        num_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
        ])

        # ----- ColumnTransformer -----
        pre = ColumnTransformer(
            transformers=[
                ('ord', ordinal_encoder, ordinal_cols),
                ('ohe', ohe_pipe, ohe_cols),
                ('tgt', tgt_pipe, target_cols),
                ('num', num_pipe, numeric_cols),
            ],
            remainder='drop',
        )

        # ----- Full pipeline -----
        full = Pipeline(steps=[
            ('pre', pre),
            ('scaler', StandardScaler()),
        ])

        return full

    def fit_transform(self, train_path, test_path):
        """
        - Đọc train/test csv
        - Tách SalePrice ra khỏi X
        - Build pipeline và fit trên train
        - Transform train & test
        - Lưu lại DataFrame sau encode/scale
        """
        df_train = pd.read_csv(train_path)
        df_test  = pd.read_csv(test_path)

        # Bỏ cột Id nếu có
        for c in self.drop_cols:
            if c in df_train.columns:
                df_train = df_train.drop(columns=[c])
            if c in df_test.columns:
                df_test = df_test.drop(columns=[c])

        # Tách target
        y_train = df_train['SalePrice'].values
        y_test  = df_test['SalePrice'].values if 'SalePrice' in df_test.columns else None

        X_train = df_train.drop(columns=['SalePrice'])
        X_test  = df_test.drop(columns=['SalePrice'])

        # Build & fit pipeline
        self.pipeline_ = self._build_pipeline(X_train)

        # fit_transform trên train (TargetEncoder sẽ dùng cross-fit nội bộ)
        X_train_arr = self.pipeline_.fit_transform(X_train, y_train)

        # transform test (TargetEncoder.transform sẽ encode test bằng stats fit trên full train)
        X_test_arr  = self.pipeline_.transform(X_test)

        # Lấy tên cột sau transform
        pre = self.pipeline_.named_steps['pre']
        feat_names = pre.get_feature_names_out()
        self.feature_names_ = feat_names

        # Convert về DataFrame
        train_encoded = pd.DataFrame(X_train_arr, columns=feat_names, index=X_train.index)
        train_encoded['SalePrice'] = y_train

        test_encoded = pd.DataFrame(X_test_arr, columns=feat_names, index=X_test.index)
        if y_test is not None:
            test_encoded['SalePrice'] = y_test

        return train_encoded, test_encoded

    def save(self, train_encoded, test_encoded,
             processed_dir='data/processed',
             interim_dir='data/interim',
             train_output='train_encoded.csv',
             test_output='test_encoded.csv',
             config_output='encoding_config.json'):
        """Lưu kết quả encode ra csv + metadata đơn giản."""
        import json
        import os

        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(interim_dir, exist_ok=True)

        train_encoded.to_csv(f"{processed_dir}/{train_output}", index=False)
        test_encoded.to_csv(f"{processed_dir}/{test_output}", index=False)

        config = {
            "n_features_after_encoding": len(self.feature_names_),
            "feature_names": list(self.feature_names_),
            "ordinal_cols": self.ordinal_cols,
            "ohe_cols": self.ohe_cols,
            "target_enc_cols": self.target_enc_cols,
            "dropped_cols": self.drop_cols,
        }

        with open(f"{interim_dir}/{config_output}", "w") as f:
            json.dump(config, f, indent=2)

        print("✓ Saved:")
        print(f"  - {processed_dir}/{train_output}  ({train_encoded.shape})")
        print(f"  - {processed_dir}/{test_output}   ({test_encoded.shape})")
        print(f"  - {interim_dir}/{config_output}")


# ===========================
# Ví dụ chạy (tương đương run_pipeline cũ):
# ===========================
if __name__ == "__main__":
    enc = SklearnEncodingPipeline()
    train_df_enc, test_df_enc = enc.fit_transform(
        train_path="train_transformed.csv",
        test_path="test_transformed.csv",
    )
    enc.save(train_df_enc, test_df_enc)
    print("Done. Dataset đã encode + scale, sẵn sàng train model.")
