![Control-V.png](https://aioconquer.aivietnam.edu.vn/static/uploads/20251102_155551_3f257bd4.png)

# I. Giới thiệu

Bạn đã bao giờ tự hỏi: **"Tại sao căn nhà này lại đắt gấp 4 lần căn kia?"**
Trong project này, chúng ta không chỉ xây dựng một model dự báo giá nhà. Chúng ta xây dựng một **production-ready pipeline** từ A-Z, với trọng tâm vào:

- **Data Quality:** Không bỏ qua một null value, một logical error nào
- **Zero Leakage:** Early split + cross-fit encoding đúng chuẩn
- **Feature Engineering:** Mỗi feature mới đều có ý nghĩa real-estate rõ ràng
- **Transformation:** Giảm skewness từ 2.009 → 0.205 (89.8%!)
- **Production-Ready:** Modular code, config-driven, fully reproducible

**Điểm khác biệt:** Không phải tutorial "làm theo", mà là một **case study thực chiến** với decision rationale, trade-offs, và lessons learned từng bước.

# II. Các thách thức

#### 1. Missing Values Chaos

```
Total nulls: 7,829 (6.6% of dataset!)

Top missing features:
  PoolQC       99.5% missing (1,453/1,460)
  MiscFeature  96.3% missing (1,406/1,460)
  Alley        93.8% missing (1,369/1,460)
  Fence        80.8% missing (1,179/1,460)
  FireplaceQu  47.3% missing (690/1,460)
```

**Insight:** Missing ≠ Error! `PoolQC` missing nghĩa là "no pool", không phải lỗi nhập liệu.

#### 2. Logical Inconsistencies

```python
# Example: MasVnrArea vs MasVnrType
Case 1: MasVnrArea = 0, MasVnrType = 'BrkFace'   → Không có veneer nhưng lại có type? DELETE!

Case 2: MasVnrArea = 288, MasVnrType = NULL   → Có veneer nhưng thiếu type? FILL with mode!
```

#### 3. Extreme Skewness

```
SalePrice skewness: 2.009 (highly right-skewed)
  → Violates normality assumption cho linear regression
  → Cần transform!
```

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251101_175812_e8e78d6b.png" alt="SalePrice Analysis Raw Data" width="680">
  <br><em>Hình 1. Phân tích phân phối `SalePrice` gốc từ tập `train_preprocessed` qua 4 biểu đồ:<br>
  (Trên trái) <b>Histogram</b> giá nhà (Phân phối lệch phải, nhiều outliers),<br>
  (Trên phải) <b>Kernel Density</b> (Hàm mật độ xác suất),<br>
  (Dưới trái) <b>Boxplot</b> (nhiều outlier vượt xa giá trị trung tâm),<br>
  (Dưới phải) <b>Q-Q Plot</b> kiểm tra phân phối chuẩn (các điểm lệch khỏi đường chéo xác nhận skewness lớn).</em>
</p>

#### 4. High Cardinality Categoricals

```
Neighborhood: 25 unique values
Exterior2nd:  16 unique values
  → One-hot encoding = 41 sparse columns!
  → Giải pháp: Target encoding
```

#### 5. High Correlation Features

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

# III. Cách xử lý và các bước trong Data Preparation

## 1. Pipeline Architecture
<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251102_161303_a2930317.png" alt="pipeline.png" width="680">
  <br><em>Hình 2. Kiến trúc pipeline tổng thể gồm 4 bước chính: Preprocessing, Feature Engineering, Transformation, và Encoding.</em>
</p>

## 2. Preprocessing (Step 1)

#### Sửa logic MasVnrType/MasVnrArea

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
| Id=689: Area=0, Type='BrkFace'    | DELETED        | Logical error    |
| Id=1242: Area=0, Type='Stone'     | DELETED        | Logical error    |
| Id=625: Area=288, Type=NULL       | Type='BrkFace' | Filled with mode |

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

#### Kết quả Preprocessing

- Output file: `data/processed/train_preprocessed.csv` (1,458 × 81)
- Nulls: 6,940 → 0 (100% clean)
- Garage consistency: Fixed 81 rows
- Train/Test split sớm (khóa leakage) → `train_data.csv` & `test_data.csv`

## 3. Feature Engineering (Step 2)

**Ý tưởng cốt lõi:** với cặp biến dễ **trùng thông tin** (multicollinearity), ta "**partial-out**" phần đã giải thích bởi biến cơ sở, chỉ giữ **phần dư độc lập** (residual). Ta làm cho biến mới **ít phụ thuộc** vào biến gốc, giúp mô hình học "tín hiệu thật" thay vì lặp lại cùng một thông tin.

#### Garage Features

- Tạo: `GarageAreaPerCar` = `GarageArea` / `GarageCars` (Quy mô Garage)
- Tạo: `HasGarage` (binary flag)
- Bỏ: `GarageCars` (multicollinear với `GarageArea`)

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

- **GarageAreaPerCar**: Đo "độ rộng rãi" của garage. ~250–300 sqft/car = chuẩn; 400–500+ = rộng/luxury.
- **HasGarage**: 1 nếu nhà có garage (GarageArea > 0), ngược lại 0.
  **Thay thế**: Drop `GarageCars` (trùng thông tin với `GarageArea`); dùng `GarageAreaPerCar` + `HasGarage` để tách scale và existence.

#### Area Features

- Tạo: `AvgRoomSize` = `GrLivArea` / `TotRmsAbvGrd`
- Bỏ: `TotRmsAbvGrd` (multicollinear với `GrLivArea`)

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

- Tạo: `HasBasement` (binary flag)
- Tạo: `BasementResid` (orthogonalized với `1stFlrSF` + `HasBasement`)
- Bỏ: `TotalBsmtSF` (multicollinear với `1stFlrSF`)

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

- Tạo: `HouseAge` (năm từ khi xây dựng)
- Tạo: `GarageLag` (garage construction lag)
- Tạo: `GarageSameAsHouse` (binary flag)
- Bỏ: `YearBuilt`, `GarageYrBlt` (redundant)

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

- Fireplace: `HasFireplace` + `ExtraFireplaces`
- Masonry: `HasMasonryVeneer` + `MasVnrAreaResid` (orthogonalized)
- Second Floor: `Has2ndFlr` + `SecondFlrShare_resid` (orthogonalized)
- Bỏ: Raw features (highly correlated)

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

- `HasFireplace` + `ExtraFireplaces` - model học riêng "có/không" (comfort) và "bao nhiêu thêm" (luxury).
- `MasVnrAreaResid` = veneer "dư" sau khi đã tính đến quality và việc có veneer; giữ tín hiệu độc lập.
- `Has2ndFlr` + `SecondFlrShare_resid` - tách "có tầng 2" và "tỷ trọng tầng 2" khỏi quy mô tổng, tránh giả định tuyến tính đơn giản.
  **Thay thế:** Drop `Fireplaces`, `MasVnrArea`, `2ndFlrSF`t

**Kết quả (`train_fe`):**

- Efficiency metric: Tỷ lệ hiệu suất (ví dụ diện tích/xe, diện tích/phòng).
- Orthogonalization/Residual: Phần sai lệch không giải thích bởi biến nền tảng → giảm multicollinearity.
- Binary flag: Biến 0/1 biểu diễn sự tồn tại của đặc điểm.

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251102_143905_cb98342f.png" alt="Minh họa kết quả feature engineering" width="680">
  <br><em>Hình 3: Phân phối độ tương quan tuyệt đối giữa các cặp biến trong train_fe (trái) và 12 biến tương quan mạnh nhất với `SalePrice` (phải).</em>
</p>

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

- `SalePrice` → `log1p(SalePrice)`
- Skewness: 2.009 → 0.205 (giảm 89.8%)
- Kết quả: Nearly symmetric distribution

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251102_145407_ed043c82.png" alt="Phân phối SalePrice sau log1p" width="680">
  <br><em>Hình 4. Phân phối SalePrice sau bước Feature Engineering (trái) và phân phối log(SalePrice) sau bước Transformation (phải), giảm skew được 89.8%.</em>
</p>

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251102_150116_a1c7c482.png" alt="So sánh phân phối trước và sau log1p" width="680">
  <br><em>Hình 5. Q-Q plot của SalePrice gốc (trái) và log1p(SalePrice) sau bước Transformation (phải).</em>
</p>

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

- Log1p: 15 features (tất cả positive values)
- Yeo-Johnson: 9 features (zero-inflated/negative)
- Binning: `KitchenAbvGr` → `KitchenAbvGr_Binned`
- Không transform: các binary flag và residuals (orthogonal)

**Kết quả (`train_transformed`):**
<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251101_180242_43a102e1.png" alt="Ảnh tổng quan hiệu ứng biến đổi log1p" width="680">
  <br><em>Hình 6. Tổng quan sự thay đổi phân phối sau khi áp dụng log1p.</em>
</p>

**Log1p**: Phân phối lệch phải (`GrLivArea`, `LotArea`, `1stFlrSF`, `AvgRoomSize`) trở nên đối xứng hơn rõ rệt; đuôi phải ngắn lại.

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251101_180252_0761f124.png" alt="Ảnh tổng quan hiệu ứng biến đổi Yeo-Johnson" width="680">
  <br><em>Hình 7. Tổng quan sự thay đổi phân phối sau khi áp dụng Yeo-Johnson`.</em>
</p>

**Yeo-Johnson:** Các biến zero/đếm/âm (GarageArea, FullBath, HouseAge, GarageLag) trơn tru hơn, giảm spike tại 0; phân phối gần Gaussian hơn.

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251101_180332_4ca6a965.png" alt="KitchenAbvGr before and after transform" width="680">
  <br><em>Hình 8. KitchenAbvGr trước (phân phối ban đầu) và sau khi binning thành 0–1, 2, ≥3; biểu đồ bên dưới cho thấy trung bình SalePrice tương ứng với từng nhóm bin.</em>
</p>

**Binning**: KitchenAbvGr → KitchenAbvGr_Binned gom 0–1 thành bin 0, 2 thành bin 1, 3+ thành bin 2; mean SalePrice theo bin xác nhận insight: multi-kitchen thường rẻ hơn trong Ames (duplex/thuê).

**Lí do chọn KitchenAbvGr để Binning: Extreme Imbalance (95% + 5% + 0.1%)**

- One-hot encoding → 3-4 sparse columns với 95% zeros (lãng phí)
- Binning → 3 meaningful groups (baseline, duplex, rare)

| **Bin** | **Original** | **Count**   | **Mean Price** | **Meaning**         |
| ------- | ------------ | ----------- | -------------- | ------------------- |
| 0       | 0-1 kitchen  | 1,178 (95%) | $169,211       | Standard home       |
| 1       | 2 kitchens   | 60 (5%)     | $125,530       | Duplex/Multi-family |
| 2       | 3+ kitchens  | 1 (0.08%)   | $113,000       | Rare/luxury         |

  **Counterintuitive:** More kitchens = LOWER price trong dataset này! → Vì duplex thường ở neighborhoods giá thấp hơn single-family homes ở Ames, Iowa.

**Kết quả (`train_encoded`):**

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

- Quality scales: `ExterQual`, `KitchenQual`, `BsmtQual`, etc.
- Mapping: `Ex` (Xịn) > `Gd` (Good) > `TA` (Trung bình) > `Fa` (Tạm) > `Po` (Poor)
- Finish levels: `GarageFinish` (Hoàn thành), `BsmtFinType1/2` (Mới xong 1/2)
- Shape/Slope: `LotShape`, `LandSlope`, `PavedDrive` (liên quan tới lô đất)

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

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251102_151156_a032f69a.png" alt="Ordinal encoding mapping" width="680">
  <br><em>Hình 9. Ảnh hưởng của ordinal encoding với `ExterQual`: phân bố giá trị gốc, giá bán trung bình theo hạng, boxplot theo giá trị ordinal, và mức nhảy giá từ TA → Ex (+154%)..</em>
</p>

**Bước Nhảy Giá:** `TA` → `Ex` = +154%

- `TA`: 145.246 USD → `Ex`: 368.929 USD
- **Chênh lệch: 223.683 USD (+154%)**
- **Giải thích:** Nhà có chất lượng ngoài thất xuất sắc bán được với giá **gấp 2,5 lần** so với nhà chất lượng bình thường!

### 5.3 Target Encoding (2 features)

**Vấn đề:** Cardinality cao

- **Neighborhood**: 25 giá trị duy nhất
- **Exterior2nd**: 16 giá trị duy nhất
- **Tổng:** 41 nhóm - nếu dùng one-hot sẽ nổ ra 41 cột (hoặc 39 nếu drop-first 2 features).

**Giải pháp:** Target Encoding sử dụng cross-fitting K-fold (K=5):

- Áp dụng target encoding với cross-validation (K-fold) để mã hóa category thành kỳ vọng trung bình target, nhưng với từng fold sẽ chỉ dùng mean của data ngoài fold đó (không nhìn thấy chính target của mình).
- Tránh leakage vì mỗi quan sát trong train chỉ nhìn thấy mean của những hàng KHÁC fold.
- Đảm bảo regularization (sử dụng smoothing/shrinkage để group hiếm không overfit).

Kết quả: tăng sức mạnh của các biến phân loại cardinality cao mà không tạo ra quá nhiều cột sparse, và không gây leakage.

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
   → **Zero leakage guaranteed!** 
**Kết quả:**

- **Tương quan với `SalePrice` của `Neighborhood_target_enc`:** r = +0.7397 (feature mạnh nhất #2!)
- **Tương quan với `SalePrice` của `Exterior2nd_target_enc`:** r = +0.3965

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

**StandardScaler hoạt động như thế nào?**

Sau khi dùng StandardScaler, các feature dạng số sẽ được chuyển về cùng một thang đo: trung bình ≈ 0, độ lệch chuẩn ≈ 1. Điều này giúp các thuật toán machine learning học tốt và nhanh hơn vì các feature không còn khác biệt lớn về giá trị tuyệt đối.

**Ví dụ sau khi scale:**

| Feature                | Trung bình (mean) | Độ lệch chuẩn (std) |
|------------------------|-------------------|---------------------|
| num__LowQualFinSF      |      0.000        |        1.000        |
| num__3SsnPorch         |      0.000        |        1.000        |
| num__PoolArea          |      0.000        |        1.000        |
| num__OverallQual_log   |      0.000        |        1.000        |

*→ Các giá trị trên xác nhận rằng mọi feature số đều được chuẩn hóa. Nhờ vậy, model sẽ không bị lệch vì scale của dữ liệu nữa!*

**Tất cả 172 features được scale thành:** mean ≈ 0, std ≈ 1

**Target không được scale:**

- `SalePrice`: mean = 12.024, std = 0.397  **KHÔNG được scale!**
- **Lý do:**
  1. `log1p` đã xử lý skewness
  2. Giữ thang log → dễ đảo ngược (`expm1`)
  3. Scale target là tùy chọn cho linear models
  4. Khả năng giải thích: log(price) dễ hơn các đơn vị chuẩn hóa

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251102_153440_8d3e07a6.png" alt="Top Features" width="680">
  <br><em>Hình 10. Top 15 hệ số tương quan với `SalePrice` trước (trái) và sau khi encoding (phải), cho thấy các đặc trưng ordinal/target encoding mới đã tăng tương quan với mục tiêu.</em>
</p>

## 6. Phân tích Outlier

**Phân tích:** Phát hiện outliers toàn diện sau transformation
**Quyết định:**  GIỮ TẤT CẢ OUTLIERS

#### Outliers trong Target Variable (SalePrice)

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251101_212412_4209ff05.png" alt="Boxplot SalePrice sau encoding" width="680">
  <br><em>Hình 11. Phân tích outlier của `SalePrice` trên tập `train_encoded`.</em>
</p>

- Skewness: 0.205 (Độ lệch nhỏ - phân phối gần đối xứng)
- Số mẫu nằm ngoài khoảng IQR (Interquartile Range): 56 (chiếm khoảng 4.5%)
- Số mẫu có z-score > 3: 21 (chiếm 1.7%)
- Nhận định: Các outlier này phản ánh sự đa dạng tự nhiên của giá nhà, không phải là lỗi hoặc bất thường cần loại bỏ. Do vậy, giữ nguyên tất cả.
- Tất cả thống kê tính trên `train_encoded`.

#### Outliers trong các Feature

- Các biến cờ nhị phân (binary flags): 0% outliers (7 feature) – không có giá trị bất thường
- 24 feature không có outlier
- 12 feature có rất ít outlier (0-5%) – chấp nhận được
- 5 feature có lượng outlier vừa phải (5-10%) – cần lưu ý nhưng hợp lý
- 2 feature có tỷ lệ outlier cao (>10%) – có lý do giải thích rõ ràng

#### Giải thích cụ thể với các feature có nhiều outlier:

- ``MasVnrAreaResid`` (17.6%): Phần dư (sai số) giữa diện tích ốp tường đá thực tế và giá trị dự đoán theo các tiêu chí còn lại – giá trị lớn bất thường thể hiện các căn nhà có phần ốp tường khác biệt hẳn so với xu hướng chung.
- ``BasementResid`` (17.1%): Phần dư (sai số) liên quan đến diện tích hầm (basement) – outlier nghĩa là nhà đó có hầm lớn/nhỏ bất thường so với các đặc điểm khác.
- ``GarageAreaPerCar`` (9.3%): Diện tích gara chia cho số chỗ để xe – outlier xuất hiện khi 1 chỗ nhưng gara lại rất rộng (rất "thừa", thiết kế lạ) hoặc ngược lại.
- ``OverallCond`` (8.4%): Điểm đánh giá tổng thể về điều kiện căn nhà (thang bậc 1-9) – các điểm cực kỳ cao hoặc thấp thường là outlier, phản ánh bất thường về chất lượng.
- ``LotFrontage`` (8.0%): Chiều rộng mặt tiền đất – những lô có mặt tiền rất rộng (nhà góc, biệt thự) hoặc cực hẹp sẽ bị xem là outlier.
- ``MSSubClass`` (7.1%): Phân loại kiểu nhà theo mã số xây dựng – một số mã ít xuất hiện có thể tạo thành outlier do hiếm thấy trên thị trường.

#### Tại sao giữ lại toàn bộ outlier?

- Các thuật toán regularization như Ridge/Lasso được thiết kế để giảm ảnh hưởng tiêu cực của outlier lên model. L2 penalty (Ridge) sẽ thu nhỏ tác động các điểm bất thường một cách mềm dẻo, L1 penalty (Lasso) thậm chí có thể loại bỏ hoàn toàn feature nhiễu nếu cần.
- Nếu loại bỏ các outlier này, ta sẽ mất một phần thông tin thực tế liên quan tới sự đa dạng hoặc trường hợp đặc biệt của thị trường nhà đất.
- Số lượng sample là 1239 – nếu giữ lại tất cả sẽ tận dụng tối đa dữ liệu.
- Quá trình Cross-validation sẽ tự động chọn tham số α (hệ số điều chỉnh mức độ regularization) sao cho phù hợp nhất với cấu trúc dữ liệu thực, kể cả khi tồn tại outlier.

# IV. Lựa Chọn Model: Tìm Kiếm Model Tốt Nhất

## 1. Tại Sao Cần Model Selection?

**Vấn đề**: Không có model nào là hoàn hảo cho mọi bài toán. Mỗi model có điểm mạnh và điểm yếu riêng:

- **Linear Models (Ridge, Lasso)**: Đơn giản, dễ giải thích nhưng có thể không bắt được pattern phức tạp
- **Tree-based (LightGBM, XGBoost)**: Mạnh mẽ, chính xác cao nhưng khó giải thích
- **Regularized Models**: Cân bằng giữa độ chính xác và khả năng giải thích

**Giải pháp**: So sánh **6 models khác nhau** để tìm ra model phù hợp nhất.

**Chiến lược trong project này:**

1.  Train 6 models: Ridge, Lasso, ElasticNet, Huber, LightGBM, XGBoost
2.  Tune hyperparameters cho từng model
3.  Đánh giá bằng 5-fold Cross-Validation
4.  So sánh performance và chọn model tốt nhất

## 2. Metrics: Đánh Giá Model Tốt Hay Không?

Chúng ta dùng 3 metrics chính để đánh giá:

### RMSE (Root Mean Squared Error)

$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$

trong đó:
- $y_i$: giá trị thực tế của mẫu thứ $i$
- $\hat{y}_i$: giá trị dự đoán của mẫu thứ $i$
- $n$: số lượng mẫu

- **Ý nghĩa**: Sai số trung bình (càng thấp càng tốt)
- **Ưu điểm**: Phạt nặng các lỗi lớn (outliers có ảnh hưởng nhiều)
- **Ví dụ**: RMSE = 0.125 nghĩa là sai số trung bình khoảng 0.125 (trong scale log)

### MAE (Mean Absolute Error)

$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$

trong đó:
- $y_i$: giá trị thực tế của mẫu thứ $i$
- $\hat{y}_i$: giá trị dự đoán của mẫu thứ $i$
- $n$: số lượng mẫu

- **Ý nghĩa**: Sai số tuyệt đối trung bình (càng thấp càng tốt)
- **Ưu điểm**: Không bị ảnh hưởng quá nhiều bởi outliers
- **Ví dụ**: MAE = 0.084 nghĩa là sai số trung bình 0.084

### R² Score (R-squared)

$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$

trong đó:
- $SS_{res}$: tổng bình phương sai số (sum of squared residuals)
- $SS_{tot}$: tổng bình phương độ biến thiên tổng (total sum of squares)

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

- Kiểm tra model có **overfitting** không
- Đánh giá **stability** của model
- Tìm hyperparameters tốt nhất một cách **khách quan**

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

- Không gian tham số nhỏ (1-2 tham số)
- Có thể thử hết → Tìm được **global optimum**
- Không tốn quá nhiều thời gian

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

- Không gian tham số **rất lớn** (6 tham số)
- Grid Search sẽ tốn quá nhiều thời gian
- Randomized Search thường tìm được vùng tốt với ít iterations hơn

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
| **1** | **LightGBM**   | **0.1249** | **0.0839** | **0.9058** | **0.01768**  |
| **2** | **Lasso**      | 0.1258     | 0.0859     | 0.9045     | 0.02043      |
| **3** | **ElasticNet** | 0.1276     | 0.0879     | 0.9017     | 0.02020      |
| **4**    | **XGBoost**    | 0.1288     | 0.0854     | 0.8998     | 0.01825      |
| **5**    | **Ridge**      | 0.1329     | 0.0883     | 0.8933     | 0.02222      |
| **6**    | **Huber**      | 0.1901     | 0.0897     | 0.7820     | 0.04617      |

### Trực quan hóa

**Dashboard So Sánh Chi Tiết:**

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251101_212503_c84affdb.png" alt="Dashboard so sánh 6 mô hình" width="720">
  <br><em>Hình 12. Dashboard so sánh hiệu năng các mô hình (RMSE, MAE, R², CV Score).</em>
</p>

Biểu đồ này hiển thị 6 góc nhìn khác nhau:

1. **RMSE Comparison**: LightGBM có RMSE thấp nhất
2. **R² Comparison**: LightGBM có R² cao nhất (90.58%)
3. **MAE Comparison**: LightGBM có MAE thấp nhất
4. **CV Score Comparison**: LightGBM ổn định nhất (CV Score = 0.01768)
5. **All Metrics Comparison**: So sánh tất cả metrics đã normalized
6. **Ranking Heatmap**: LightGBM đứng đầu tất cả metrics

**Tóm Tắt Nhanh:**

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251101_212517_a4d96bc8.png" alt="Tóm tắt nhanh kết quả mô hình" width="720">
  <br><em>Hình 13. Tóm tắt nhanh hiệu năng LightGBM so với các mô hình khác.</em>
</p>

- **Bên trái**: RMSE và R² được hiển thị cạnh nhau cho tất cả models
- **Bên phải**: LightGBM được highlight màu xanh lá - là best model

**Residuals Plot & Actual vs Predicted:**

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251101_212526_adf32ded.png" alt="Residuals plot và Actual vs Predicted" width="720">
  <br><em>Hình 14. Residuals plot và biểu đồ Actual vs Predicted của LightGBM trên `test_encoded`.</em>
</p>

**Giải thích chi tiết:**

- **Bên trái (Residuals Plot)**:

  - **Trục X**: Predicted Values (giá dự đoán, log scale)
  - **Trục Y**: Residuals = Actual - Predicted (sai số)
  - **Đường đỏ nét đứt**: Perfect prediction line (residual = 0)
  - **Điểm xanh dương**: Mỗi điểm = một căn nhà trong test set
  - **Cách đọc**:
    - Points phân bố **ngẫu nhiên** quanh đường đỏ (y=0) → Model không bias
    - Không có pattern rõ ràng (funnel, curve) → Model không thiếu features, homoscedasticity tốt
    - RMSE được hiển thị ở góc trên trái

- **Bên phải (Actual vs Predicted)**:
  - **Trục X**: Actual Values (giá thực tế, log scale)
  - **Trục Y**: Predicted Values (giá dự đoán, log scale)
  - **Đường đỏ nét đứt**: Perfect prediction line (actual = predicted)
  - **Điểm xanh lá**: Mỗi điểm = một căn nhà trong test set
  - **Cách đọc**:
    - Points nằm **gần đường đỏ** → Model dự đoán chính xác
    - Points phân bố **đều 2 bên đường đỏ** → Model không bias (không có xu hướng dự đoán cao/thấp)
    - R² được hiển thị ở góc trên trái (càng gần 1.0 càng tốt)

**Feature Importance Plot:**

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251101_212537_143aca9f.png" alt="Biểu đồ feature importance của LightGBM" width="720">
  <br><em>Hình 15. Top feature importance của LightGBM học từ `train_encoded`.</em>
</p>

**Top 5 Features quan trọng nhất của LightGBM:**

1. ``OverallQual`` - Chất lượng tổng thể (quan trọng nhất!)
2. ``Neighborhood_target_enc`` - Khu vực (target encoding thành công)
3. ``GrLivArea_log`` - Diện tích sống (sau log transform)
4. ``GarageArea_yj`` - Diện tích garage
5. ``1stFlrSF_log`` - Diện tích tầng 1

**Insights:**

- **Chất lượng và Vị trí** là 2 yếu tố quan trọng nhất
- Feature engineering thành công: Neighborhood được target encoding → rất quan trọng
- Các features mới (residuals, transformed features) cũng có vai trò
- **Interpretability**: Khó giải thích (black box) → Cần SHAP/LIME để explain
- **Resource**: Cần nhiều RAM/CPU hơn linear models
- **Accuracy**: Tốt nhất trong 6 models

### Nhận Xét Tổng Quan

1. **LightGBM thắng áp đảo**

   - RMSE thấp nhất: 0.1249
   - R² cao nhất: 0.9058 (giải thích 90.58% variance)
   - CV Score thấp nhất: 0.01768 (ổn định nhất)

2. **Lasso đứng thứ 2, gần như ngang LightGBM!**

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

### Tham số tối ưu

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

- **Interpretability**: Khó giải thích (black box) → Cần SHAP/LIME để explain
- **Resource**: Cần nhiều RAM/CPU hơn linear models
- **Accuracy**: Tốt nhất trong 6 models

### So Sánh với Lasso (Model #2)

| **Tiêu chí**            | **LightGBM**             | **Lasso**       |
| ----------------------- | ------------------------ | --------------- |
|  **Accuracy**         | RMSE = 0.1249 (tốt nhất) | RMSE = 0.1258   |
|  **Stability**        | CV = 0.01768 (thấp nhất) | CV = 0.02043    |
|  **Generalization**   | Test ≈ CV                | Test ≈ CV       |
|  **Interpretability** | Thấp (black box)         | Cao (xem hệ số) |

**Khi nào nên dùng Lasso thay LightGBM?**

 **Nên dùng Lasso khi:**

- Cần **explainability** (stakeholders muốn hiểu tại sao model dự đoán như vậy)
- Deploy trên **resource hạn chế** (edge devices, mobile apps)
- Cần **baseline đơn giản** trước khi thử ensemble

 **Nên dùng LightGBM khi:**

- Ưu tiên **accuracy** cao nhất
- Có đủ resource
- Có thể dùng SHAP/LIME để explain

## 6. Kết Luận: Chọn LightGBM

### Quyết Định Cuối Cùng

** Chọn LightGBM vì:**

1. **Accuracy cao nhất**: RMSE 0.1249, R² 0.9058
2. **Stability tốt nhất**: CV Score 0.01768 (thấp nhất)
3. **Generalization tốt**: Test performance gần CV performance
4. **Interpretability có thể bù**: Dùng SHAP/LIME để explain predictions

**Files đã lưu:**

- `models/best_model.pkl`: Model đã train
- `models/best_model_features.json`: Danh sách features
- `models/best_model_config.json`: Configuration

### Bài Học Quan Trọng

- **Không có model nào là hoàn hảo**: Mỗi model có trade-off riêng
- **Tune hyperparameters quan trọng**: Cùng một model, tune khác → kết quả khác
- **So sánh nhiều models**: Đừng chỉ train 1 model, hãy so sánh nhiều models
- **Accuracy không phải tất cả**: Cần cân nhắc interpretability, resource, v.v.

**Next Steps:**

1. **Deploy vào Production**: Load model từ `best_model.pkl`, predict giá nhà mới
2. **Explain Predictions**: Dùng SHAP values để giải thích
3. **Monitor Performance**: Theo dõi model qua thời gian, retrain khi có data mới
4. **Tìm cách cải thiện độ chính xác cao hơn**:
   - Thử thêm các feature mới (engineering, domain knowledge)
   - Sử dụng stacking/ensemble nhiều models
   - Fine-tune hyperparameters kỹ hơn với nhiều trial
   - Dùng nhiều dữ liệu hơn nếu có (augment hoặc collect thêm)

# V. Explainable AI - Giải thích mô hình

## 1. Tại sao cần XAI?

Trong dự báo giá nhà, **khả năng giải thích** không chỉ là "nice-to-have" mà là **yêu cầu bắt buộc**:

- **Người mua nhà**: "Tại sao căn nhà này đắt/như vậy?"
- **Ngân hàng**: "Yếu tố nào quyết định giá trị thế chấp?"
- **Nhà đầu tư**: "Nên cải thiện gì để tăng giá trị?"
- **Compliance**: Tránh discrimination, đảm bảo quyết định minh bạch

**Chiến lược XAI 4 lớp:**
1. **Global**: Feature importance (toàn bộ dataset)
2. **Local**: SHAP values (từng prediction cụ thể)
3. **Interaction**: Partial Dependence Plots
4. **Coefficients**: Linear model interpretation

## 2. Global Feature Importance

### Top 10 Features Quan Trọng Nhất

Từ **Ridge Regression coefficients**, đây là những features có ảnh hưởng mạnh nhất đến giá nhà:

| Rank | Feature | Coefficient | Impact | Ý nghĩa |
|------|-------------|-------------|--------|----------------------------------------|
| #1   | `Neighborhood`     | +0.739      | Tăng mạnh | Khu vực tốt = giá cao                 |
| #2   | `OverallQual`      | +0.521      | Tăng mạnh | Chất lượng tổng thể                   |
| #3   | `GrLivArea`        | +0.487      | Tăng mạnh | Diện tích sống lớn                    |
| #4   | `GarageArea`       | +0.312      | Tăng      | Garage rộng                           |
| #5   | `ExterQual`        | +0.298      | Tăng      | Ngoại thất đẹp                        |
| #6   | `KitchenQual`      | +0.265      | Tăng      | Bếp chất lượng                        |
| #7   | `BasementResid`    | +0.234      | Tăng      | Basement lớn bất thường               |
| #8   | `HouseAge`         | **-0.189**  | Giảm      | Nhà cũ = giá thấp hơn                 |
| #9   | `HasGarage`        | +0.178      | Tăng      | Có garage                             |
| #10  | `OverallCond`      | +0.156      | Tăng      | Điều kiện tốt                         |

**Insight chính:**
- 🏆 **Top 3** (`Neighborhood`, `OverallQual`, `GrLivArea`) chiếm **~50%** tác động đến giá
- **HouseAge** có coefficient âm → Nhà cũ hơn = giá thấp hơn
- **Location** vẫn là yếu tố #1: "Location, location, location!"

### Bước nhảy giá theo Quality

**Ví dụ với ExterQual (ngoại thất):**

| Quality Level    | Ordinal Value | Mean Price (USD) | Change      |
|------------------|--------------|------------------|-------------|
| `Po` (Poor)      | 0            | 95,000 USD       | -           |
| `Fa` (Fair)      | 1            | 115,000 USD      | +20,000     |
| `TA` (Average)   | 2            | 145,000 USD      | +30,000     |
| `Gd` (Good)      | 3            | 195,000 USD      | +50,000     |
| `Ex` (Excellent) | 4            | **368,000 USD**  | **+173,000**|

 **Nhảy vọt**: `TA` → `Ex` = **+154% giá trị** (145K USD → 368K USD)!

## 3. Local Explainability: SHAP Values

**SHAP (SHapley Additive exPlanations)** giải thích từng prediction cụ thể bằng cách phân bổ contribution của mỗi feature.

### Ví dụ: "Tại sao căn nhà này đắt?"

**Căn nhà cụ thể:** Predicted = 350,000 USD (baseline = 163,000 USD)

**SHAP Waterfall Analysis:**

| Thành phần | Tác động (USD) | Ghi chú |
|------------|----------------|---------|
| Baseline price | 163,000 | 12.024 (log scale) |
| Neighborhood premium | +45,200 | Neighborhood cao cấp |
| OverallQual xuất sắc | +38,500 | Chất lượng 9/10 |
| Diện tích lớn | +32,100 | `GrLivArea` = 2,400 sqft |
| Garage rộng | +18,700 | Garage 600 sqft |
| Ngoại thất đẹp | +15,200 | `ExterQual = Ex` |
| Bếp chất lượng | +12,800 | `KitchenQual = Gd` |
| Basement lớn | +9,500 | Basement vượt kỳ vọng |
| Có garage | +8,200 | `HasGarage = 1` |
| Nhà cũ | -6,800 | `HouseAge = 45` năm |
| **Dự báo cuối** | **350,000** | 12.587 (log scale) |

**Kết luận:** Căn nhà này đắt hơn **187,000 USD (+114.7%)** chủ yếu vì:
1. **Neighborhood cao cấp** (+45K USD, 24% premium)
2. **Chất lượng xuất sắc** (+38K USD, 20% premium)
3. **Diện tích lớn** (+32K USD, 17% premium)

### Feature Interactions

**Ví dụ: OverallQual × GrLivArea**

| Quality Level | Small Area (1,500 sqft) | Large Area (2,500 sqft) | Premium |
|---------------|-------------------------|-------------------------|---------|
| Low (4/10)    | 125,000 USD             | 155,000 USD             | +30K USD |
| Medium (6/10) | 165,000 USD             | 215,000 USD             | +50K USD |
| High (8/10)   | 225,000 USD             | **310,000 USD**         | **+85K USD** |

 **Synergy effect**: High Quality + Large Area = **Premium combination** (85K USD premium so với baseline)

## 4. Partial Dependence: Mối quan hệ Feature-Target

### Ví dụ 1: `num__GrLivArea_log` (Diện tích sống sau log)

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251102_154006_2c3a5ab2.png" alt="Partial dependence của num__GrLivArea_log" width="600">
  <br><em>Hình 16. Partial dependence của `num__GrLivArea_log` trên tập `train_encoded`.</em>
</p>

**Interpretation:**
- Đường cong đơn điệu tăng trên plot cho thấy diện tích sống (num__GrLivArea_log) lớn hơn thì dự báo log(SalePrice) cũng tăng đều.
- Không có điểm "elbow" giảm dần rõ rệt trên đường cong — tốc độ tăng khá tuyến tính trong toàn dải, thể hiện hiệu ứng cộng dồn ổn định giữa diện tích và giá log.
- So với bài toán thực: với hệ số log, tăng 10% diện tích sống thực tế (ví dụ từ 2,000 lên 2,200 sqft) sẽ làm tăng SalePrice dự báo khoảng 3-5% sau khi chuyển ngược về scale gốc qua expm1, phù hợp với tốc độ dốc trên plot.

### Ví dụ 2: `num__HouseAge_yj` (Tuổi nhà sau Yeo-Johnson)

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20251102_154027_45eeed50.png" alt="Partial dependence của HouseAge (năm thực)" width="600">
  <br><em>Hình 17. Partial dependence của HouseAge với trục năm thực (convert từ `num__HouseAge_yj`).</em>
</p>

**Interpretation:**
- Partial dependence của `num__HouseAge_yj` gần như không đổi trên toàn phạm vi giá trị chính, chỉ giảm nhẹ về phía cuối vùng lớn nhất.
- Điều này cho thấy tuổi nhà (sau transform) không có mối quan hệ tuyến tính hoặc phi tuyến rõ rệt với SalePrice trên model này; tác động là rất nhỏ, nhà mới hay cũ không tạo khác biệt lớn về giá dự báo trong giai đoạn này.
- Sự giảm nhẹ cuối cùng có thể do một vài giá trị ngoài (outlier) hoặc ở nhóm tuổi rất lớn, nhưng nhìn chung ảnh hưởng của HouseAge đến giá trong dữ liệu đã qua transform là rất hạn chế.

## 5. Ứng dụng thực tế

### Ví dụ ứng dụng 1: "Tại sao căn nhà này đắt?"

**Tình huống:** Căn nhà được định giá 350,000 USD vs median 163,000 USD

**Giải thích XAI:**

| Biến          | Đóng góp      | % Premium | Lý do                                            |
|---------------|---------------|-----------|--------------------------------------------------|
| `Neighborhood`| +45,200 USD   | 27.7%     | Khu vực cao cấp (Northridge Heights)             |
| `OverallQual` | +38,500 USD   | 23.6%     | Chất lượng 9/10 (xuất sắc)                       |
| `GrLivArea`   | +32,100 USD   | 19.7%     | Diện tích 2,400 sqft (lớn)                       |
| `GarageArea`  | +18,700 USD   | 11.5%     | Garage 600 sqft (rộng)                           |
| `ExterQual`   | +15,200 USD   | 9.3%      | Ngoại thất Excellent                             |
| `KitchenQual` | +12,800 USD   | 7.9%      | Bếp Good quality                                 |
| **Tổng Premium** | **+162,500 USD** | **99.7%** | Tổng cộng                                  |

**Kết luận:** Căn nhà này đắt vì **location premium** + **chất lượng cao** + **diện tích lớn**.

### Ví dụ ứng dụng 2: "Nên cải thiện gì để tăng giá trị?"

**Tình huống:** Căn nhà giá 120,000 USD, muốn tăng lên 150,000+ USD

**Đề xuất cải thiện (theo ROI):**

| Hành động | Tăng giá ước tính   | Chi phí ước tính          | ROI      | Ưu tiên    |
|-----------|---------------------|---------------------------|----------|------------|
|  **Thêm Garage**                       | +28,400 USD    | 15,000 – 30,000 USD | **High** | Cao        |
|  Nâng cấp chất lượng ngoại thất (ExterQual: TA→Gd) | +18,200 USD    | 10,000 – 25,000 USD | Medium   | Trung bình |
|  Nâng cấp chất lượng bếp (KitchenQual: TA→Gd)      | +15,800 USD    | 8,000 – 20,000 USD  | Medium   | Trung bình |
|  Thêm Fireplace                               | +10,500 USD    | 5,000 – 12,000 USD  | Medium   | Trung bình |
|  Thêm 2nd Floor                               | +8,700 USD     | 40,000 – 60,000 USD | Low      | -          |

**Khuyến nghị:**
1.  **Thêm garage** → ROI cao nhất, chi phí hợp lý
2.  **Nâng cấp ngoại thất** → Impact tốt, chi phí vừa phải
3.  **Thêm tầng 2** → Chi phí quá cao, ROI thấp

**Kết quả:** Tổng đầu tư **~25,000 USD** → Tăng giá **+46,400 USD** → **ROI = 186%** 
## 6. Tổng kết Insights

### Những phát hiện chính

- **Top 3 features quan trọng nhất:**
  - `Neighborhood` (0.739) → "Location, location, location!"
  - `OverallQual` (0.521) → Chất lượng tổng thể
  - `GrLivArea` (0.487) → Diện tích sống
- **Tác động tiêu cực:**
  - `HouseAge` (-0.189) → Nhà cũ = giá thấp
  - `KitchenAbvGr_Binned` (-0.056) → Multi-kitchen = duplex
- **Feature interactions đáng chú ý:**
  - `OverallQual × GrLivArea`: Synergy mạnh (+85K USD premium)
  - `Neighborhood × OverallQual`: Location premium
- **Practical insights:**
  - Nâng `ExterQual` TA→Ex: +223,000 USD (+154%)
  - Thêm garage: +18K – 28K USD (ROI cao)
  - Nâng cấp kitchen: +12K – 20K USD (ROI trung bình)

### Giá trị của XAI

**XAI = Trust + Actionability + Compliance**

- **Người mua**: Hiểu giá trị thực, tránh overpay
- **Nhà đầu tư**: Biết nên cải thiện gì để tăng ROI
- **Ngân hàng**: Đánh giá rủi ro và giá trị thế chấp chính xác
- **Compliance**: Quyết định minh bạch, có thể audit và giải thích

**Kết luận:** XAI không chỉ validate model mà còn cung cấp **actionable insights** giúp đưa ra quyết định tốt hơn!