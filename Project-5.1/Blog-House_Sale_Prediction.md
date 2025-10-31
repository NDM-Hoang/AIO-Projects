# I. Giới thiệu:

Bạn đã bao giờ tự hỏi: **"Tại sao căn nhà này lại đắt gấp 4 lần căn kia?"**
Trong project này, bọn mình không chỉ build một model dự báo giá nhà. Bọn mình xây dựng một **production-ready pipeline** từ A-Z, với focus vào:

- 🎯 **Data Quality:** Không bỏ qua một null value, một logical error nào
- 🔒 **Zero Leakage:** Early split + cross-fit encoding đúng chuẩn
- 📊 **Feature Engineering:** Mỗi feature mới đều có ý nghĩa real-estate rõ ràng
- 🎨 **Transformation:** Giảm skewness từ 2.009 → 0.205 (89.8%!)
- 🚀 **Production-Ready:** Modular code, config-driven, fully reproducible

**Điểm khác biệt:** Không phải tutorial "làm theo", mà là một **case study thực chiến** với decision rationale, trade-offs, và lessons learned từng bước.

# II. Các thách thức:

#### 1️⃣ **Missing Values Chaos**

```
Total nulls: 7,829 (6.6% of dataset!)

Top missing features:
  PoolQC       99.5% missing (1,453/1,460)
  MiscFeature  96.3% missing (1,406/1,460)
  Alley        93.8% missing (1,369/1,460)
  Fence        80.8% missing (1,179/1,460)
  FireplaceQu  47.3% missing (690/1,460)
```

💡 **Insight:** Missing ≠ Error! `PoolQC` missing nghĩa là "no pool", không phải lỗi nhập liệu.

#### 2️⃣ **Logical Inconsistencies**

```python
# Example: MasVnrArea vs MasVnrType
Case 1: MasVnrArea = 0, MasVnrType = 'BrkFace' ❌
  → Không có veneer nhưng lại có type? DELETE!

Case 2: MasVnrArea = 288, MasVnrType = NULL ⚠️
  → Có veneer nhưng thiếu type? FILL with mode!
```

#### 3️⃣ **Extreme Skewness**

```
SalePrice skewness: 2.009 (highly right-skewed)
  → Violates normality assumption cho linear regression
  → Cần transform!
```

![Missing Values and Logical Errors Illustration](images/Pasted%20image%2020251028104423.png)

#### 4️⃣ **High Cardinality Categoricals**

```
Neighborhood: 25 unique values
Exterior2nd:  16 unique values
  → One-hot encoding = 41 sparse columns!
  → Giải pháp: Target encoding
```

#### 5️⃣ High Correlation Features

Các cặp biến có tương quan > 80%:

| ID   | **Feature 1** | **Feature 2** | **Correlation** |
| ---- | ------------- | ------------- | --------------- |
| 988  | GarageCars    | GarageArea    | 0.882475        |
| 1025 | GarageArea    | GarageCars    | 0.882475        |
| 246  | YearBuilt     | GarageYrBlt   | 0.825667        |
| 931  | GarageYrBlt   | YearBuilt     | 0.825667        |
| 614  | GrLivArea     | TotRmsAbvGrd  | 0.825489        |
| 867  | TotRmsAbvGrd  | GrLivArea     | 0.825489        |
| 493  | 1stFlrSF      | TotalBsmtSF   | 0.819530        |
| 456  | TotalBsmtSF   | 1stFlrSF      | 0.819530        |

# III. Cách xử lý và các bước trong Data Preparationeparation

## 1. Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  RAW DATA (1,460 × 81)                                       │
│  ├─ 7,829 nulls                                              │
│  ├─ Logical errors                                           │
│  └─ Mixed data types                                         │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 1: PREPROCESSING                                       │
│  ├─ Fix MasVnr logic (delete 2 rows)                        │
│  ├─ Fill 6,940 nulls                                        │
│  ├─ Fix Garage consistency                                  │
│  └─ Train/Test Split (85/15)                                │
└──────────────────────────────────────────────────────────────┘
                  │                          │
                  ▼                          ▼
         ┌─────────────┐            ┌─────────────┐
         │ TRAIN       │            │ TEST        │
         │ 1,239 × 81  │            │ 219 × 81    │
         └─────────────┘            └─────────────┘
                  │                          │
                  ▼                          ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 2: FEATURE ENGINEERING                                 │
│  ├─ Create derived features (+6)                            │
│  ├─ Orthogonalize residuals                                 │
│  └─ Binary flags (Has* features)                            │
└──────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 3: TRANSFORMATION                                      │
│  ├─ log1p(SalePrice): skew 2.009 → 0.205                   │
│  ├─ log1p for 15 features                                   │
│  ├─ Yeo-Johnson for 9 features                              │
│  └─ Binning: KitchenAbvGr                                   │
└──────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 4: ENCODING                                            │
│  ├─ Ordinal: 17 features (Quality scales)                   │
│  ├─ One-Hot: 110 features (Nominal cats)                    │
│  ├─ Target Encoding: 2 features (Cross-fit K=5)            │
│  └─ StandardScaler: All 176 features                        │
└──────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────┐
│  FINAL: PRODUCTION-READY                                     │
│  ├─ 1,239 × 173 (train)                                     │
│  ├─ 219 × 173 (test)                                        │
│  ├─ 0 nulls, 0 logical errors                               │
│  └─ Zero leakage, fully reproducible                        │
└──────────────────────────────────────────────────────────────┘
```

## 2. Preprocessing (Step 1)

#### Sửa logic MasVnrType/MasVnrArea:

**Implementation:**

```python
def fix_masonry_veneer_logic(self):
	original_len = len(self.df)

	# Get mode của MasVnrType (for Case 2)
	mode_type = self.df['MasVnrType'].mode()[0]

	# Case 1: Area=0, Type≠NULL → DELETE
	case1_mask = (self.df['MasVnrArea'] == 0) & (self.df['MasVnrType'].notna())
	case1_count = case1_mask.sum()
	self.df = self.df[~case1_mask]

	# Case 2: Area>0, Type=NULL → FILL mode
	case2_mask = (self.df['MasVnrArea'] > 0) & (self.df['MasVnrType'].isna())
	case2_count = case2_mask.sum()
	self.df.loc[case2_mask, 'MasVnrType'] = mode_type

	# Case 3: Both NULL → 'None'
	case3_mask = (self.df['MasVnrArea'].isna()) | (self.df['MasVnrType'].isna())
	case3_count = case3_mask.sum()
	if case3_count > 0:
		self.df.loc[case3_mask, 'MasVnrType'] = 'None'
		self.df.loc[case3_mask, 'MasVnrArea'] = 0

	deleted = original_len - len(self.df)

	return self
```

=> Xoá 2 dòng mâu thuẫn; còn **1458 × 81**.

| **Before**                        | **After**      | **Action**       |
| --------------------------------- | -------------- | ---------------- |
| Id=689: Area=0, Type='BrkFace' ❌ | DELETED        | Logical error    |
| Id=1242: Area=0, Type='Stone' ❌  | DELETED        | Logical error    |
| Id=625: Area=288, Type=NULL ✅    | Type='BrkFace' | Filled with mode |

#### Fill missing values có chủ đích:

Danh mục (Categories) → `'None'`; biến đếm/diện tích (Numerics) → `0`; số khác → **median**.

**Implementation:**

```python
def fill_missing_values(self):
	total_nulls_before = self.df.isnull().sum().sum()

	# Categorical columns → 'None'
	categorical_cols = self.df.select_dtypes(include=['object']).columns
	cat_nulls = self.df[categorical_cols].isnull().sum().sum()

	if cat_nulls > 0:
		for col in categorical_cols:
			if self.df[col].isnull().sum() > 0:
				self.df[col] = self.df[col].fillna('None')

	# Numeric columns: count/area → 0, others → median
	numeric_cols = self.df.select_dtypes(include=[np.number]).columns

	# Count/area columns → 0
	count_area_cols = ['GarageCars', 'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2',
					'BsmtUnfSF', 'TotalBsmtSF', 'FullBath', 'HalfBath',
					'BedroomAbvGr', 'KitchenAbvGr', 'WoodDeckSF', 'OpenPorchSF',
					'EnclosedPorch', 'ScreenPorch', '3SsnPorch', 'PoolArea',
					'MasVnrArea', '2ndFlrSF', 'BsmtFullBath', 'BsmtHalfBath']

	for col in count_area_cols:
		if col in self.df.columns and self.df[col].isnull().sum() > 0:
			self.df[col] = self.df[col].fillna(0)

	count_area_nulls = sum(self.df[col].isnull().sum() \
		for col in count_area_cols if col in self.df.columns)

	# Other numeric → median
	other_numeric_nulls = self.df[numeric_cols].isnull().sum().sum()
	if other_numeric_nulls > 0:
		for col in numeric_cols:
			if self.df[col].isnull().sum() > 0:
				median_val = self.df[col].median()
				self.df[col] = self.df[col].fillna(median_val)

	total_nulls_after = self.df.isnull().sum().sum()

	return self
```

#### Garage consistency:

Khóa chặt logic nếu `GarageArea=0` thì toàn bộ thuộc tính garage = `'None'`.

**Implementation:**

```python
def fix_garage_logic(self):
	garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

	# Find rows without garage
	no_garage_mask = self.df['GarageArea'] == 0
	no_garage_count = no_garage_mask.sum()

	# Set 'None' for rows without garage
	for col in garage_cols:
		if col in self.df.columns:
			self.df.loc[no_garage_mask, col] = 'None'

	# Fill remaining nulls với mode
	for col in garage_cols:
		if col in self.df.columns and self.df[col].isnull().sum() > 0:
			mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > \
															0 else 'None'
			null_count = self.df[col].isnull().sum()
			self.df[col] = self.df[col].fillna(mode_val)

	return self
```

#### Chia 85/15 ngay sau khi dữ liệu sạch cơ bản

Khóa test, chia sớm ngăn leakage cho mọi bước sau.

**Implementation:**

```python
def run_preprocessing(self):
	from src.Preprocessing import Preprocessor
	from sklearn.model_selection import train_test_split

	# Load raw data
	df_raw = pd.read_csv(self.raw_data_path)

	# Run preprocessing pipeline
	preprocessor = Preprocessor(df_raw)
	df_clean = (preprocessor
				.fix_masonry_veneer_logic()
				.fill_missing_values()
				.fix_garage_logic()
				.get_summary()
				.get_dataframe())

	# Save cleaned data
	preprocessed_path = self.processed_dir / 'train_preprocessed.csv'
	df_clean.to_csv(preprocessed_path, index=False)

	# Split into train/test (85/15)
	target = df_clean['SalePrice']
	X = df_clean.drop('SalePrice', axis=1)

	X_train, X_test, y_train, y_test = train_test_split(
		X, target, test_size=0.15, random_state=42
	)

	train_df = pd.concat([X_train, y_train], axis=1)
	test_df = pd.concat([X_test, y_test], axis=1)

	train_path = self.processed_dir / 'train_data.csv'
	test_path = self.processed_dir / 'test_data.csv'

	train_df.to_csv(train_path, index=False)
	test_df.to_csv(test_path, index=False)

	return True
```

#### Kết quả Preprocessing:

```
✅ Shape: 1,460 → 1,458 (deleted 2 logical errors)
✅ Nulls: 6,940 → 0 (100% clean)
✅ Garage consistency: Fixed 81 rows
✅ Train/Test: Split early (no leakage)
```

## 3. Feature Engineering (Step 2)

**Ý tưởng cốt lõi:** với cặp biến dễ **trùng thông tin** (multicollinearity), ta “**partial-out**” phần đã giải thích bởi biến cơ sở, chỉ giữ **phần dư độc lập** (residual). Ta làm cho biến mới **ít phụ thuộc** vào biến gốc, giúp mô hình học “tín hiệu thật” thay vì lặp lại cùng một thông tin.

#### Garage Features

```
├─ Tạo: GarageAreaPerCar = GarageArea / GarageCars (Quy mô Garage)
├─ Tạo: HasGarage (binary flag)
└─ Bỏ: GarageCars (multicollinear với GarageArea)
```

**Implementation:**

```python
def engineer_garage_features(self):
	# Feature 1: GarageAreaPerCar
	self.df['GarageAreaPerCar'] = np.where(
		self.df['GarageCars'] > 0,
		self.df['GarageArea'] / self.df['GarageCars'],
		0
	)
	self.new_features.append('GarageAreaPerCar')

	# Feature 2: HasGarage
	self.df['HasGarage'] = (self.df['GarageArea'] > 0).astype(int)
	self.new_features.append('HasGarage')

	# Drop GarageCars
	if 'GarageCars' in self.df.columns:
		self.df = self.df.drop('GarageCars', axis=1)

	return self
```

**Định nghĩa**:

- **GarageAreaPerCar**: Đo “độ rộng rãi” của garage. ~250–300 sqft/car = chuẩn; 400–500+ = rộng/luxury.
- **HasGarage**: 1 nếu nhà có garage (GarageArea > 0), ngược lại 0.
  **Thay thế**: Drop `GarageCars` (trùng thông tin với `GarageArea`); dùng `GarageAreaPerCar` + `HasGarage` để tách scale và existence.

#### Area Features

```
├─ Tạo: AvgRoomSize = GrLivArea / TotRmsAbvGrd
└─ Bỏ: TotRmsAbvGrd (multicollinear với GrLivArea)
```

**Implementation**:

```python
def engineer_area_features(self):
	# Feature 1: AvgRoomSize
	self.df['AvgRoomSize'] = np.where(
		self.df['TotRmsAbvGrd'] > 0,
		self.df['GrLivArea'] / self.df['TotRmsAbvGrd'],
		0
	)
	self.new_features.append('AvgRoomSize')

	# Drop TotRmsAbvGrd
	if 'TotRmsAbvGrd' in self.df.columns:
		self.df = self.df.drop('TotRmsAbvGrd', axis=1)

	return self
```

**Định nghĩa:** Phản ánh cảm giác rộng–chật của không gian sống (≈200 = vừa; 300+ = rộng rãi/luxury).

**Thay thế:** Drop `TotRmsAbvGrd` (corr cao với `GrLivArea`), giữ `GrLivArea` (scale) + `AvgRoomSize` (efficiency).

#### Basement Features

```
├─ Tạo: HasBasement (binary flag)
├─ Tạo: BasementResid (orthogonalized với 1stFlrSF + HasBasement)
└─ Bỏ: TotalBsmtSF (multicollinear với 1stFlrSF)
```

**Implementation:**

```python
def engineer_basement_features(self):
	# Feature 1: HasBasement
	self.df['HasBasement'] = (self.df['TotalBsmtSF'] > 0).astype(int)
	self.new_features.append('HasBasement')

	# Feature 2: BasementResid (orthogonalized)
	X_basement = self.df[['1stFlrSF', 'HasBasement']].values
	y_bsmt = self.df['TotalBsmtSF'].values

	# Dùng hồi quy tuyến tính để loại bỏ phần giải thích được của TotalBsmtSF
	# dựa trên 1stFlrSF và HasBasement
	model = LinearRegression()
	model.fit(X_basement, y_bsmt)

	# BasementResid là phần dư sau hồi quy: y_bsmt - model.predict(X_basement),
	# giúp orthogonal hóa đặc trưng diện tích hầm
	self.df['BasementResid'] = y_bsmt - model.predict(X_basement)
	self.new_features.append('BasementResid')

	# Drop TotalBsmtSF
	if 'TotalBsmtSF' in self.df.columns:
		self.df = self.df.drop('TotalBsmtSF', axis=1)

	return self
```

**Định nghĩa:**

- Orthogonalization (trực giao hóa): tạo residual để loại phụ thuộc tuyến tính → tín hiệu độc lập, giảm VIF.
- BasementResid > 0: Basement lớn hơn kỳ vọng; < 0: nhỏ hơn kỳ vọng.
  **Thay thế:** Drop `TotalBsmtSF`; dùng `HasBasement` + `BasementResid` (r với `1stFlrSF` từ 0.815 → 0.000).

#### Age Features

```
├─ Tạo: HouseAge (năm từ khi xây dựng)
├─ Tạo: GarageLag (garage construction lag)
├─ Tạo: GarageSameAsHouse (binary flag)
└─ Bỏ: YearBuilt, GarageYrBlt (redundant)
```

**Implementation:**

```python
def engineer_age_features(self):
	# Feature 1: HouseAge
	self.df['HouseAge'] = self.df['YrSold'] - self.df['YearBuilt']
	self.new_features.append('HouseAge')

	# Feature 2: GarageLag
	self.df['GarageLag'] = self.df['GarageYrBlt'] - self.df['YearBuilt']
	self.new_features.append('GarageLag')

	# Feature 3: GarageSameAsHouse
	self.df['GarageSameAsHouse'] = (self.df['GarageYrBlt'] == \
		self.df['YearBuilt']).astype(int)
	self.new_features.append('GarageSameAsHouse')

	# Drop raw year features
	cols_to_drop = []
	for col in ['YearBuilt', 'GarageYrBlt', 'GarageAge']:
		if col in self.df.columns:
			self.df = self.df.drop(col, axis=1)
			cols_to_drop.append(col)

	return self
```

**Định nghĩa**: Dùng tuổi tương đối thay vì năm tuyệt đối (1950 vs 2000). `GarageLag` là chỉ báo renovation/upgrade.
**Thay thế**: Drop `YearBuilt`, `GarageYrBlt`; dùng `HouseAge`, `GarageLag`, `GarageSameAsHouse`.

#### Quality Features

```
├─ Fireplace: HasFireplace + ExtraFireplaces
├─ Masonry: HasMasonryVeneer + MasVnrAreaResid (orthogonalized)
├─ Second Floor: Has2ndFlr + SecondFlrShare_resid (orthogonalized)
└─ Bỏ: Raw features (highly correlated)
```

**Implementation:**

```python
def engineer_quality_features(self):
	# ===== FIREPLACE =====
	self.df['HasFireplace'] = (self.df['Fireplaces'] > 0).astype(int)
	self.new_features.append('HasFireplace')
	self.df['ExtraFireplaces'] = np.maximum(self.df['Fireplaces'] - 1, 0)
	self.new_features.append('ExtraFireplaces')

	if 'Fireplaces' in self.df.columns:
		self.df = self.df.drop('Fireplaces', axis=1)

	# ===== MASONRY VENEER =====
	self.df['MasVnrArea'] = self.df['MasVnrArea'].fillna(0)
	self.df['HasMasonryVeneer'] = (self.df['MasVnrArea'] > 0).astype(int)
	self.new_features.append('HasMasonryVeneer')

	# MasVnrAreaResid (orthogonalized)
	X_masonry = self.df[['HasMasonryVeneer', 'OverallQual']].values
	y_masonry = self.df['MasVnrArea'].values

	model_m = LinearRegression()
	model_m.fit(X_masonry, y_masonry)

	self.df['MasVnrAreaResid'] = y_masonry - model_m.predict(X_masonry)
	self.new_features.append('MasVnrAreaResid')

	if 'MasVnrArea' in self.df.columns:
		self.df = self.df.drop('MasVnrArea', axis=1)

	# ===== SECOND FLOOR =====
	self.df['Has2ndFlr'] = (self.df['2ndFlrSF'] > 0).astype(int)
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
	self.new_features.append('SecondFlrShare_resid')

	if '2ndFlrSF' in self.df.columns:
		self.df = self.df.drop('2ndFlrSF', axis=1)

	return self
```

**Định nghĩa:**

- `HasFireplace` + `ExtraFireplaces` - model học riêng “có/không” (comfort) và “bao nhiêu thêm” (luxury).
- `MasVnrAreaResid` = veneer “dư” sau khi đã tính đến quality và việc có veneer; giữ tín hiệu độc lập.
- `Has2ndFlr` + `SecondFlrShare_resid` - tách “có tầng 2” và “tỷ trọng tầng 2” khỏi quy mô tổng, tránh giả định tuyến tính đơn giản.
  **Thay thế:** Drop `Fireplaces`, `MasVnrArea`, `2ndFlrSF`t

**Kết quả:**

> - Efficiency metric: Tỷ lệ hiệu suất (ví dụ diện tích/xe, diện tích/phòng).
> - Orthogonalization/Residual: Phần sai lệch không giải thích bởi biến nền tảng → giảm multicollinearity.
> - Binary flag: Biến 0/1 biểu diễn sự tồn tại của đặc điểm.

![Feature Engineering Illustration](images/Pasted%20image%2020251028142135.png)

## 4. Transformation (Step 3)

#### Target Transformation

**Implementation**:

```python
def _transform_target(self):
	if 'SalePrice' in self.train_data.columns:
		# Lưu giá trị ban đầu
		y_train = self.train_data['SalePrice'].values
		y_test = self.test_data['SalePrice'].values

		# Dùng log1p để giảm skew
		y_train_transformed = np.log1p(y_train)
		y_test_transformed = np.log1p(y_test)

		# Ghi đè dữ liệu
		self.train_data['SalePrice'] = y_train_transformed
		self.test_data['SalePrice'] = y_test_transformed

	return self
```

```
├─ SalePrice → log1p(SalePrice)
├─ Skewness: 2.009 → 0.205 (giảm 89.8%)
└─ Kết quả: Nearly symmetric distribution ✓
```

![Feature Engineering Example 1](images/Pasted%20image%2020251028142603.png)
![Feature Engineering Example 2](images/Pasted%20image%2020251028142715.png)

#### Feature Transformations

**Implementation:**

```python
def _bin_kitchen_abvgr(self):
	# Hàm phân nhóm KitchenAbvGr thành 3 bin:
	# - 0 và 1: bin 0
	# - 2: bin 1
	# - >=3: bin 2
	def bin_kitchen(x):
		if x <= 1:
			return 0
		elif x == 2:
			return 1
		else:
			return 2

	# Tạo cột binned mới
	self.train_data['KitchenAbvGr_Binned'] = \
		self.train_data['KitchenAbvGr'].apply(bin_kitchen)
	self.test_data['KitchenAbvGr_Binned'] = \
		self.test_data['KitchenAbvGr'].apply(bin_kitchen)

	# Tạo cờ "HasMultiKitchen": 1 nếu có ≥2 kitchen
	self.train_data['HasMultiKitchen'] = \
		(self.train_data['KitchenAbvGr'] >= 2).astype(int)
	self.test_data['HasMultiKitchen'] = \
		(self.test_data['KitchenAbvGr'] >= 2).astype(int)

	# Xóa cột cũ KitchenAbvGr
	self.train_data = self.train_data.drop('KitchenAbvGr', axis=1)
	self.test_data = self.test_data.drop('KitchenAbvGr', axis=1)

	return self

def _transform_features_log(self):
	# Chọn ra những đặc trưng áp dụng log1p
	log_features = [f for f, s in self.feature_strategy.items() if s == 'log']

	for feat in log_features:
		if feat in self.train_data.columns:
			# Skew trước biến đổi
			original_skew = skew(self.train_data[feat].dropna())

			# Biến đổi bằng log1p, ghi sang cột mới
			self.train_data[f'{feat}_log'] = np.log1p(self.train_data[feat])
			self.test_data[f'{feat}_log'] = np.log1p(self.test_data[feat])

			# Skew sau biến đổi
			transformed_skew = skew(self.train_data[f'{feat}_log'].dropna())

			# Loại bỏ đặc trưng gốc (chỉ giữ đặc trưng mới)
			self.train_data = self.train_data.drop(feat, axis=1)
			self.test_data = self.test_data.drop(feat, axis=1)

	return self

def _transform_features_yeo_johnson(self):
	from sklearn.preprocessing import PowerTransformer

	# Chọn ra đặc trưng cần biến đổi Yeo-Johnson hoặc log1p cho zero-inflated
	yj_features = [f for f, s in self.feature_strategy.items()
	if s in ['yeo_johnson', 'log_zero_inflated']]
	# Chỉ lấy những cột hiện diện thực tế trong data
	numeric_df_train = self.train_data[[f for f in \
		yj_features if f in self.train_data.columns]].copy()
	numeric_df_test = self.test_data[[f for f in \
		yj_features if f in self.test_data.columns]].copy()

	# Tạo transformer (không chuẩn hóa mean/std)
	pt = PowerTransformer(method='yeo-johnson', standardize=False)

	# Fit transformer trên train data
	pt.fit(numeric_df_train)

	# Áp dụng biến đổi trên cả hai tập train/test
	train_transformed = pt.transform(numeric_df_train)
	test_transformed = pt.transform(numeric_df_test)

	# Thêm cột mới và lưu lại thông tin biến đổi
	for idx, feat in enumerate([f for f in \
			yj_features if f in self.train_data.columns]):
		original_skew = skew(numeric_df_train[feat].dropna())
		transformed_skew = skew(train_transformed[:, idx])

		self.train_data[f'{feat}_yj'] = train_transformed[:, idx]
		self.test_data[f'{feat}_yj'] = test_transformed[:, idx]

		# Xóa đặc trưng gốc
		if feat in self.train_data.columns:
			self.train_data = self.train_data.drop(feat, axis=1)
		if feat in self.test_data.columns:
			self.test_data = self.test_data.drop(feat, axis=1)

	return self
```

```
├─ Log1p: 15 features (tất cả positive values)
├─ Yeo-Johnson: 9 features (zero-inflated/negative)
├─ Binning: KitchenAbvGr → KitchenAbvGr_Binned
└─ No transform: Binary flags + residuals (orthogonal)
```

**Kết quả:**
![Feature Transform After Log1p](images/Pasted%20image%2020251028144755.png)

**Log1p**: Phân phối lệch phải (GrLivArea, LotArea, 1stFlrSF, AvgRoomSize) trở nên đối xứng hơn rõ rệt; đuôi phải ngắn lại,

![Feature Transform After Log for Other Features](images/Pasted%20image%2020251028145402.png)

**Yeo-Johnson:** Các biến zero/đếm/âm (GarageArea, FullBath, HouseAge, GarageLag) trơn tru hơn, giảm spike tại 0; phân phối gần Gaussian hơn.

![Feature Transform After Yeo-Johnson](images/Pasted%20image%2020251028145413.png)

**Binning**: KitchenAbvGr → KitchenAbvGr_Binned gom 0–1 thành bin 0, 2 thành bin 1, 3+ thành bin 2; mean SalePrice theo bin xác nhận insight: multi-kitchen thường rẻ hơn trong Ames (duplex/thuê).

**Lí do chọn KitchenAbvGr để Binning: Extreme Imbalance (95% + 5% + 0.1%)**

- One-hot encoding → 3-4 sparse columns với 95% zeros (lãng phí)
- Binning → 3 meaningful groups (baseline, duplex, rare)

| **Bin** | **Original** | **Count**   | **Mean Price** | **Meaning**         |
| ------- | ------------ | ----------- | -------------- | ------------------- |
| 0       | 0-1 kitchen  | 1,178 (95%) | $169,211       | Standard home       |
| 1       | 2 kitchens   | 60 (5%)     | $125,530       | Duplex/Multi-family |
| 2       | 3+ kitchens  | 1 (0.08%)   | $113,000       | Rare/luxury         |

💡 **Counterintuitive:** More kitchens = LOWER price trong dataset này! → Vì duplex thường ở neighborhoods giá thấp hơn single-family homes ở Ames, Iowa.

**Kết quả:**

- Đã xử lý skewness cho 24 biến đầu vào
- Biến mục tiêu phân phối gần đối xứng (skewness xấp xỉ 0.2)
- Chiến lược cross-fit đảm bảo không rò rỉ dữ liệu huấn luyện
- Quá trình biến đổi luôn có thể thực hiện lại được dễ dàng

## 5. Encoding Strategies (Step 4)

### 5.1 Thách thức: 43 Features Categorical

**Phân tích theo Cardinality:**

- Cardinality thấp (≤5): 27 features
- Cardinality trung bình (6-10): 13 features
- Cardinality cao (>10): 3 features → `Neighborhood`, `Exterior1st`, `Exterior2nd`

**Vấn đề:** One-hot encoding tất cả 43 features sẽ tạo ra quá nhiều cột sparse!

### 5.2 Ordinal Encoding (17 features)

```
├─ Quality scales: ExterQual, KitchenQual, BsmtQual, etc.
├─ Mapping: Ex (Xịn) > Gd (Good) > TA (Trung bình) > Fa (Tạm) > Po (Poor)
├─ Finish levels: GarageFinish (Hoàn thành), BsmtFinType1/2 (Mới xong 1/2)
└─ Shape/Slope: LotShape, LandSlope, PavedDrive (liên quan tới lô đất)
```

**Implementation**:

```python
from sklearn.preprocessing import OrdinalEncoder

# ----- OrdinalEncoder -----
# Ta cần đưa thứ tự category cho từng cột ordinal
# sklearn.OrdinalEncoder muốn categories=[list_for_col1, list_for_col2, ...]
ordinal_categories = []
for col in ordinal_cols:
	mapping = self.ordinal_mappings[col] # ex: {'None': -1, 'Po':0, ...}
	# sắp xếp key theo giá trị rank tăng dần
	ordered_levels = sorted(mapping.keys(), key=lambda k: mapping[k])

ordinal_categories.append(ordered_levels)
ordinal_encoder = OrdinalEncoder(
	categories=ordinal_categories,
	handle_unknown='use_encoded_value',
	unknown_value=-1,
	encoded_missing_value=-1,
)
```

![Ordinal Encoding Example](images/Pasted%20image%2020251028162143.png)
**Bước Nhảy Giá: TA → Ex = +154%**

- TA: $145.246 → Ex: $368.929
- **Chênh lệch: $223.683 (+154%)**
- **Giải thích:** Nhà có chất lượng ngoài thất xuất sắc bán được với giá **gấp 2,5 lần** so với nhà chất lượng bình thường!

### 5.3 Target Encoding (2 features)

**Problem:** Cardinality Cao

- **Neighborhood:** 25 loại duy nhất
- **Exterior2nd:** 16 loại duy nhất
- **Tổng:** 41 loại → One-hot sẽ tạo 39 cột sparse!
  **Solution:** Target Encoding với Cross-fit K-fold (K=5)

```python
from sklearn.preprocessing import TargetEncoder
from sklearn.impute import SimpleImputer

# ----- TargetEncoder -----
# Với TargetEncoder, ta cũng impute most_frequent trước
# TargetEncoder trong sklearn có tham số cv=5 mặc định để cross-fit
# và fit_transform() sẽ tự dùng cross-fitting để tránh leakage.
# Khi Pipeline.fit_transform chạy cho train, nó sẽ gọi đúng logic này.
tgt_pipe = Pipeline(steps=[
	('imputer', SimpleImputer(strategy='most_frequent')),
	('tgt', TargetEncoder(
		cv=5, # cross-fit K-fold =5
		smooth='auto', # smoothing shrink về global mean
		random_state=0,
	)),
])
```

**Cách hoạt động tự động:**

1. sklearn chia train thành 5 folds
2. Với mỗi fold, tính mean từ 4 folds KHÁC
3. Áp dụng vào fold hiện tại
   → Mỗi fold KHÔNG BAO GIỜ nhìn thấy target của chính nó!
   → **Zero leakage guaranteed!** ✅

**Kết quả:**

- **Tương quan Neighborhood_target_enc:** r = +0.7397 (feature mạnh nhất #2!)
- **Tương quan Exterior2nd_target_enc:** r = +0.3965

### 5.4 One-hot Encoding (24 features -> 110 cột)

**Implementation:**

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# ----- OneHotEncoder -----
# Nhiều cột nominal có NA => Imputer(most_frequent) trước OHE
ohe_pipe = Pipeline(steps=[
	('imputer', SimpleImputer(strategy='most_frequent')),
	('ohe', OneHotEncoder(
		handle_unknown='ignore',
		drop='first', # tránh multicollinearity quá mạnh
		sparse_output=False, # output dense để scaler xử lý được
	)),
])
```

**Xử lý Sparse matrices:**

- Hầu hết cột one-hot là sparse (nhiều zeros)
- Sklearn xử lý hiệu quả với sparse matrices
- Lưu trữ và tính toán tiết kiệm bộ nhớ

![One-hot Encoding Example](images/Pasted%20image%2020251028172509.png)

### 5.5 Feature Scaling

**Implementation:**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline.Pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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
```

**Ứng Dụng StandardScaler**

```
num__LowQualFinSF | mean = 0.000, std = 1.000
num__3SsnPorch | mean = 0.000, std = 1.000
num__PoolArea | mean = 0.000, std = 1.000
num__OverallQual_log | mean = -0.000, std = 1.000
```

**Tất cả 172 features được scale thành:** mean ≈ 0, std ≈ 1

**Target không được scale:**

- **SalePrice:** mean = 12.024, std = 0.397 ✅ **KHÔNG được scale!**
- **Lý do:**
  1.  ✅ `log1p` đã xử lý skewness
  2.  ✅ Giữ thang log → dễ đảo ngược (`expm1`)
  3.  ✅ Scale target là tùy chọn cho linear models
  4.  ✅ Khả năng giải thích: log(price) dễ hơn các đơn vị chuẩn hóa

![Scaled Feature Example](images/Pasted%20image%2020251028173458.png)

## 6. Outlier Analysis

**Phân tích:** Phát hiện outliers toàn diện sau transformation
**Quyết định:** ✅ GIỮ TẤT CẢ OUTLIERS

#### Outliers trong Target Variable (SalePrice)

![Outlier Target Variable](images/Pasted%20image%2020251028175257.png)

```
├─ Skewness: 0.205 (Độ lệch nhỏ - phân phối gần đối xứng ✔️)
├─ Số mẫu nằm ngoài khoảng IQR (Interquartile Range): 56 (chiếm khoảng 4.5%)
├─ Số mẫu có z-score > 3: 21 (chiếm 1.7%)
└─ Nhận định: Các outlier này phản ánh sự đa dạng tự nhiên của giá nhà, không phải là lỗi hoặc bất thường cần loại bỏ. Do vậy, giữ nguyên tất cả.
```

#### Outliers trong các Feature

![Outlier Features](images/Pasted%20image%2020251028180305.png)

```
├─ Các biến cờ nhị phân (binary flags): 0% outliers (7 feature) – không có giá trị bất thường
├─ 24 feature không có outlier
├─ 12 feature có rất ít outlier (0-5%) ✔️ – chấp nhận được
├─ 5 feature có lượng outlier vừa phải (5-10%) ⚠ – cần lưu ý nhưng hợp lý
└─ 2 feature có tỷ lệ outlier cao (>10%) ✔️ – có lý do giải thích rõ ràng
```

#### Giải thích cụ thể với các feature có nhiều outlier:

- **MasVnrAreaResid** (17.6%): Phần dư (sai số) giữa diện tích ốp tường đá thực tế và giá trị dự đoán theo các tiêu chí còn lại – giá trị lớn bất thường thể hiện các căn nhà có phần ốp tường khác biệt hẳn so với xu hướng chung.
- **BasementResid** (17.1%): Phần dư (sai số) liên quan đến diện tích hầm (basement) – outlier nghĩa là nhà đó có hầm lớn/nhỏ bất thường so với các đặc điểm khác.
- **GarageAreaPerCar** (9.3%): Diện tích gara chia cho số chỗ để xe – outlier xuất hiện khi 1 chỗ nhưng gara lại rất rộng (rất “thừa”, thiết kế lạ) hoặc ngược lại.
- **OverallCond** (8.4%): Điểm đánh giá tổng thể về điều kiện căn nhà (thang bậc 1-9) – các điểm cực kỳ cao hoặc thấp thường là outlier, phản ánh bất thường về chất lượng.
- **LotFrontage** (8.0%): Chiều rộng mặt tiền đất – những lô có mặt tiền rất rộng (nhà góc, biệt thự) hoặc cực hẹp sẽ bị xem là outlier.
- **MSSubClass** (7.1%): Phân loại kiểu nhà theo mã số xây dựng – một số mã ít xuất hiện có thể tạo thành outlier do hiếm thấy trên thị trường.

#### Tại sao giữ lại toàn bộ outlier?

- Các thuật toán regularization như Ridge/Lasso được thiết kế để giảm ảnh hưởng tiêu cực của outlier lên model. L2 penalty (Ridge) sẽ thu nhỏ tác động các điểm bất thường một cách mềm dẻo, L1 penalty (Lasso) thậm chí có thể loại bỏ hoàn toàn feature nhiễu nếu cần.
- Nếu loại bỏ các outlier này, ta sẽ mất một phần thông tin thực tế liên quan tới sự đa dạng hoặc trường hợp đặc biệt của thị trường nhà đất.
- Số lượng sample là 1239 – nếu giữ lại tất cả sẽ tận dụng tối đa dữ liệu.
- Quá trình Cross-validation sẽ tự động chọn tham số α (hệ số điều chỉnh mức độ regularization) sao cho phù hợp nhất với cấu trúc dữ liệu thực, kể cả khi tồn tại outlier.

# VII. Model Selection: Tìm Kiếm Model Tốt Nhất

## 1. Tại Sao Cần Model Selection?

**Vấn đề**: Không có model nào là hoàn hảo cho mọi bài toán. Mỗi model có điểm mạnh và điểm yếu riêng:

- **Linear Models (Ridge, Lasso)**: Đơn giản, dễ giải thích nhưng có thể không bắt được pattern phức tạp
- **Tree-based (LightGBM, XGBoost)**: Mạnh mẽ, chính xác cao nhưng khó giải thích
- **Regularized Models**: Cân bằng giữa độ chính xác và khả năng giải thích

**Giải pháp**: So sánh **6 models khác nhau** để tìm ra model phù hợp nhất.

**Chiến lược trong project này:**

1. ✅ Train 6 models: Ridge, Lasso, ElasticNet, Huber, LightGBM, XGBoost
2. ✅ Tune hyperparameters cho từng model
3. ✅ Đánh giá bằng 5-fold Cross-Validation
4. ✅ So sánh performance và chọn model tốt nhất

## 2. Metrics: Đánh Giá Model Tốt Hay Không?

Chúng ta dùng 3 metrics chính để đánh giá:

### RMSE (Root Mean Squared Error)

```
RMSE = √[Σ(y_thực - y_dự_đoán)² / n]
```

- **Ý nghĩa**: Sai số trung bình (càng thấp càng tốt)
- **Ưu điểm**: Phạt nặng các lỗi lớn (outliers có ảnh hưởng nhiều)
- **Ví dụ**: RMSE = 0.125 nghĩa là sai số trung bình khoảng 0.125 (trong scale log)

### MAE (Mean Absolute Error)

```
MAE = Σ|y_thực - y_dự_đoán| / n
```

- **Ý nghĩa**: Sai số tuyệt đối trung bình (càng thấp càng tốt)
- **Ưu điểm**: Không bị ảnh hưởng quá nhiều bởi outliers
- **Ví dụ**: MAE = 0.084 nghĩa là sai số trung bình 0.084

### R² Score (R-squared)

```
R² = 1 - (SS_res / SS_tot)
```

- **Ý nghĩa**: Tỷ lệ variance được giải thích bởi model (càng cao càng tốt, tối đa = 1.0)
- **Ví dụ**: R² = 0.906 nghĩa là model giải thích được 90.6% sự biến thiên của giá nhà
- **Ưu điểm**: Dễ hiểu, có thể so sánh giữa các models

### Cross-Validation Score

**Vấn đề**: Nếu chỉ train/test một lần, kết quả có thể "may mắn" hoặc "không may"

**Giải pháp**: **5-Fold Cross-Validation**

```
Dữ liệu được chia thành 5 phần:
┌─────┬─────┬─────┬─────┬─────┐
│  1  │  2  │  3  │  4  │  5  │
└─────┴─────┴─────┴─────┴─────┘

Lần 1: Train trên 2,3,4,5 → Test trên 1
Lần 2: Train trên 1,3,4,5 → Test trên 2
...
Lần 5: Train trên 1,2,3,4 → Test trên 5

→ Tính trung bình 5 kết quả
```

**Tại sao dùng CV?**

- ✅ Kiểm tra model có **overfitting** không
- ✅ Đánh giá **stability** của model
- ✅ Tìm hyperparameters tốt nhất một cách **khách quan**

## 3. Hyperparameter Tuning Strategy

### Chiến Lược 1: Grid Search (Cho Linear Models)

**Ý tưởng**: Thử **TẤT CẢ** các combinations của hyperparameters

**Ví dụ với Ridge:**

```python
ridge_params = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 50, 100]
}
# 7 giá trị alpha × 5 CV folds = 35 lần train

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

search = GridSearchCV(
    Ridge(random_state=42),
    ridge_params,
    cv=5,                           # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
search.fit(X_train, y_train)

print(f"Best alpha: {search.best_params_['alpha']}")
# Output: Best alpha: 100
```

**Tại sao dùng Grid Search cho Linear Models?**

- ✅ Không gian tham số nhỏ (1-2 tham số)
- ✅ Có thể thử hết → Tìm được **global optimum**
- ✅ Không tốn quá nhiều thời gian

### Chiến Lược 2: Randomized Search (Cho Tree Models)

**Vấn đề**: Tree models có **không gian tham số rất lớn**

**Ví dụ với LightGBM:**

```python
lgb_params = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],      # 4 giá trị
    'num_leaves': [31, 50, 100, 200],             # 4 giá trị
    'max_depth': [3, 5, 7, 10],                   # 4 giá trị
    'min_child_samples': [20, 50, 100],           # 3 giá trị
    'subsample': [0.8, 0.9, 1.0],                 # 3 giá trị
    'colsample_bytree': [0.8, 0.9, 1.0]          # 3 giá trị
}
# Tổng: 4×4×4×3×3×3 = 4,320 combinations!
# Grid Search: 4,320 × 5 CV = 21,600 lần train → QUÁ NHIỀU!
```

**Giải pháp**: Randomized Search - Chỉ thử **30 combinations ngẫu nhiên**

```python
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb

search = RandomizedSearchCV(
    lgb.LGBMRegressor(random_state=42, verbose=-1),
    lgb_params,
    n_iter=30,                     # Chỉ thử 30 combinations
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)
search.fit(X_train, y_train)

print(f"Best params: {search.best_params_}")
# Output: {
#     'subsample': 0.9,
#     'num_leaves': 200,
#     'min_child_samples': 20,
#     'max_depth': 3,
#     'learning_rate': 0.1,
#     'colsample_bytree': 0.8
# }
```

**Tại sao dùng Randomized Search cho Tree Models?**

- ✅ Không gian tham số **rất lớn** (6 tham số)
- ✅ Grid Search sẽ tốn quá nhiều thời gian
- ✅ Randomized Search thường tìm được vùng tốt với ít iterations hơn

### So Sánh Grid Search vs Randomized Search

| **Tiêu chí**           | **Grid Search**        | **Randomized Search** |
| ---------------------- | ---------------------- | --------------------- |
| **Không gian tham số** | Nhỏ (1-2 tham số)      | Lớn (5+ tham số)      |
| **Số lần thử**         | Tất cả combinations    | Chỉ một phần (n_iter) |
| **Kết quả**            | Global optimum         | Good enough           |
| **Thời gian**          | Chậm với nhiều tham số | Nhanh hơn             |
| **Khi nào dùng?**      | Linear models          | Tree-based models     |

## 4. Kết Quả So Sánh 6 Models

Sau khi train tất cả 6 models với hyperparameter tuning, đây là kết quả:

| **Rank** | **Model**      | **RMSE**   | **MAE**    | **R²**     | **CV Score** |
| -------- | -------------- | ---------- | ---------- | ---------- | ------------ |
| 🥇 **1** | **LightGBM**   | **0.1249** | **0.0839** | **0.9058** | **0.01768**  |
| 🥈 **2** | **Lasso**      | 0.1258     | 0.0859     | 0.9045     | 0.02043      |
| 🥉 **3** | **ElasticNet** | 0.1276     | 0.0879     | 0.9017     | 0.02020      |
| 4️⃣       | **XGBoost**    | 0.1288     | 0.0854     | 0.8998     | 0.01825      |
| 5️⃣       | **Ridge**      | 0.1329     | 0.0883     | 0.8933     | 0.02222      |
| 6️⃣       | **Huber**      | 0.1901     | 0.0897     | 0.7820     | 0.04617      |

### 📈 Visualization

**Dashboard So Sánh Chi Tiết:**

![Model Comparison Dashboard](models/model_comparison.png)

Biểu đồ này hiển thị 6 góc nhìn khác nhau:

1. **RMSE Comparison**: LightGBM có RMSE thấp nhất
2. **R² Comparison**: LightGBM có R² cao nhất (90.58%)
3. **MAE Comparison**: LightGBM có MAE thấp nhất
4. **CV Score Comparison**: LightGBM ổn định nhất (CV Score = 0.01768)
5. **All Metrics Comparison**: So sánh tất cả metrics đã normalized
6. **Ranking Heatmap**: LightGBM đứng đầu tất cả metrics

**Tóm Tắt Nhanh:**

![Model Summary](models/model_summary.png)

- **Bên trái**: RMSE và R² được hiển thị cạnh nhau cho tất cả models
- **Bên phải**: LightGBM được highlight màu xanh lá - là best model

**Residuals Plot & Actual vs Predicted:**

![Model Residuals and Predictions](models/model_residuals.png)

**Giải thích:**

- **Bên trái (Residuals Plot)**:

  - ✅ Points phân bố **ngẫu nhiên** quanh 0 → Model không bias
  - ✅ Không có pattern rõ ràng → Model không thiếu features

- **Bên phải (Actual vs Predicted)**:
  - ✅ Points nằm **gần đường đỏ** (perfect prediction) → Model dự đoán chính xác
  - ✅ Points phân bố **đều 2 bên** → Model không bias

**Feature Importance Plot:**

![Feature Importance](models/model_feature_importance.png)

**Top 5 Features quan trọng nhất của LightGBM:**

1. **OverallQual** - Chất lượng tổng thể (quan trọng nhất!)
2. **Neighborhood_target_enc** - Khu vực (target encoding thành công)
3. **GrLivArea_log** - Diện tích sống (sau log transform)
4. **GarageArea_yj** - Diện tích garage
5. **1stFlrSF_log** - Diện tích tầng 1

**Insights:**

- ✅ **Chất lượng và Vị trí** là 2 yếu tố quan trọng nhất
- ✅ Feature engineering thành công: Neighborhood được target encoding → rất quan trọng
- ✅ Các features mới (residuals, transformed features) cũng có vai trò

### 🔍 Nhận Xét Tổng Quan

1. **🥇 LightGBM thắng áp đảo**

   - RMSE thấp nhất: 0.1249
   - R² cao nhất: 0.9058 (giải thích 90.58% variance)
   - CV Score thấp nhất: 0.01768 (ổn định nhất)

2. **🥈 Lasso đứng thứ 2, gần như ngang LightGBM!**

   - Chênh lệch RMSE chỉ 0.0009 (rất nhỏ - chỉ 0.7%!)
   - R² = 0.9045 (gần như LightGBM)
   - Ưu điểm: Interpretable (có thể xem feature coefficients)

3. **Tree-based models (LightGBM, XGBoost) tốt hơn linear models**

   - LightGBM và XGBoost đều top 4
   - Chứng tỏ dữ liệu có pattern phức tạp, cần model mạnh để bắt được

4. **Linear models vẫn rất tốt**

   - Lasso và ElasticNet đứng top 3
   - Phù hợp làm baseline hoặc khi cần interpretability

5. **Huber kém nhất**
   - R² = 0.7820 (thấp hơn nhiều)
   - Có thể do robust loss không phù hợp với dữ liệu đã clean

## 5. Phân Tích Chi Tiết Best Model: LightGBM

### Best Parameters

```python
{
    'subsample': 0.9,           # Dùng 90% samples mỗi tree (tránh overfitting)
    'num_leaves': 200,          # Tối đa 200 lá (độ phức tạp)
    'min_child_samples': 20,    # Tối thiểu 20 samples mỗi lá (tránh overfitting)
    'max_depth': 3,             # Độ sâu tối đa = 3 (cây nông, generalization tốt)
    'learning_rate': 0.1,       # Tốc độ học = 0.1 (vừa phải)
    'colsample_bytree': 0.8     # Dùng 80% features mỗi tree (tăng diversity)
}
```

### Tại Sao LightGBM Tốt Nhất?

1. **Xử lý Feature Interactions tốt**

   - Dữ liệu nhà đất có nhiều interactions phức tạp:
     - `OverallQual × Neighborhood`: Nhà chất lượng tốt ở khu tốt = giá rất cao
     - `GrLivArea × HouseAge`: Nhà lớn cũ = giá thấp hơn nhà lớn mới
   - Tree-based models tự động bắt được các interactions này

2. **Hyperparameters được tune tốt**

   - `max_depth=3`: Cây nông → Tránh overfitting
   - `subsample=0.9`, `colsample_bytree=0.8`: Thêm regularization
   - Balance tốt giữa **bias** (độ chính xác) và **variance** (overfitting)

3. **Performance vượt trội**
   - R² = 0.9058: Giải thích được **90.58% variance** trong giá nhà
   - CV Score = 0.01768: Thấp nhất → Model ổn định nhất

**Trade-off:**

- ❌ **Interpretability**: Khó giải thích (black box) → Cần SHAP/LIME để explain
- ❌ **Resource**: Cần nhiều RAM/CPU hơn linear models
- ✅ **Accuracy**: Tốt nhất trong 6 models

### So Sánh với Lasso (Model #2)

| **Tiêu chí**            | **LightGBM**             | **Lasso**       |
| ----------------------- | ------------------------ | --------------- |
| ✅ **Accuracy**         | RMSE = 0.1249 (tốt nhất) | RMSE = 0.1258   |
| ✅ **Stability**        | CV = 0.01768 (thấp nhất) | CV = 0.02043    |
| ✅ **Generalization**   | Test ≈ CV                | Test ≈ CV       |
| ⚠️ **Interpretability** | Thấp (black box)         | Cao (xem hệ số) |

**Khi nào nên dùng Lasso thay LightGBM?**

✅ **Nên dùng Lasso khi:**

- Cần **explainability** (stakeholders muốn hiểu tại sao model dự đoán như vậy)
- Deploy trên **resource hạn chế** (edge devices, mobile apps)
- Cần **baseline đơn giản** trước khi thử ensemble

❌ **Nên dùng LightGBM khi:**

- Ưu tiên **accuracy** cao nhất
- Có đủ resource
- Có thể dùng SHAP/LIME để explain

## 6. Kết Luận: Chọn LightGBM

### Quyết Định Cuối Cùng

**✅ Chọn LightGBM vì:**

1. **Accuracy cao nhất**: RMSE 0.1249, R² 0.9058
2. **Stability tốt nhất**: CV Score 0.01768 (thấp nhất)
3. **Generalization tốt**: Test performance gần CV performance
4. **Interpretability có thể bù**: Dùng SHAP/LIME để explain predictions

**Files đã lưu:**

- ✅ `models/best_model.pkl`: Model đã train
- ✅ `models/best_model_features.json`: Danh sách features
- ✅ `models/best_model_config.json`: Configuration

### Bài Học Quan Trọng

- 🎯 **Không có model nào là hoàn hảo**: Mỗi model có trade-off riêng
- 🎯 **Tune hyperparameters quan trọng**: Cùng một model, tune khác → kết quả khác
- 🎯 **So sánh nhiều models**: Đừng chỉ train 1 model, hãy so sánh nhiều models
- 🎯 **Accuracy không phải tất cả**: Cần cân nhắc interpretability, resource, v.v.

**Next Steps:**

1. **Deploy vào Production**: Load model từ `best_model.pkl`, predict giá nhà mới
2. **Explain Predictions**: Dùng SHAP values để giải thích
3. **Monitor Performance**: Theo dõi model qua thời gian, retrain khi có data mới
