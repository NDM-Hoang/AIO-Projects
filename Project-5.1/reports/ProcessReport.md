# DỰ BÁO GIÁ NHÀ - BÁO CÁO TIẾN ĐỘ DỰ ÁN

**Dự án:** Kỹ thuật hồi quy nâng cao cho dự báo giá nhà
**Ngày:** 2025-10-24
**Trạng thái:** ✅ Hoàn thành tiền xử lý - Sẵn sàng cho mô hình hóa

---

## 📊 TỔNG QUAN DỰ ÁN

### Mục tiêu
Xây dựng pipeline machine learning mạnh mẽ để dự báo giá nhà sử dụng kỹ thuật hồi quy nâng cao với regularization.

### Dataset
- **Nguồn:** Kaggle 'House Prices: Advanced Regression Techniques'
- **Gốc:** 1460 mẫu × 81 features
- **Cuối cùng:** 1239 train × 177 features, 219 test × 177 features

### Phương pháp
Hồi quy + Regularization (Ridge/Lasso/ElasticNet) với pipeline tiền xử lý toàn diện.

---

## ✅ CÁC GIAI ĐOẠN ĐÃ HOÀN THÀNH

### 1. TIỀN XỬ LÝ (✅ HOÀN THÀNH)

**File:** `src/Preprocessing.py`
**Input:** 1460 × 81 (dữ liệu gốc)
**Output:** 1458 × 81 (dữ liệu sạch) + chia 85/15

#### BƯỚC 0: Sửa logic MasVnrType & MasVnrArea
```
├─ Case 1: Area=0, Type≠NULL → XÓA (2 dòng)
├─ Case 2: Area>0, Type=NULL → ĐIỀN mode (5 dòng)
└─ Case 3: Both NULL → 'None' (867 dòng)
Kết quả: 1460 → 1458 dòng (xóa 2 dòng)
```

#### BƯỚC 1: Điền giá trị thiếu (6940 nulls)
```
├─ Categorical nulls → 'None'
├─ Count/Area features → 0
└─ Other numeric → median
Kết quả: 0 null values ✓
```

#### BƯỚC 2: Sửa logic Garage consistency
```
├─ Nếu GarageArea=0: Set type/finish/qual/cond='None' (81 dòng)
└─ Điền nulls còn lại với mode
Kết quả: Logic consistency ✓
```

#### Chia Train/Test (85/15)
```
├─ Train: 1239 mẫu (85%)
├─ Test: 219 mẫu (15%)
└─ Random state: 42 (có thể tái lập)
```

**Thành tựu chính:**
✅ 0 null values trong dữ liệu cuối
✅ Logic consistency đã sửa
✅ Early split ngăn chặn data leakage
✅ Dữ liệu sạch, sẵn sàng cho feature engineering

### 2. FEATURE ENGINEERING (✅ HOÀN THÀNH)

**File:** `src/FeatureEngineering.py`
**Input:** 1239 × 81 (sau tiền xử lý)
**Output:** 1239 × 87 (6 features mới)

#### Garage Features
```
├─ Tạo: GarageAreaPerCar (metric hiệu quả)
├─ Tạo: HasGarage (binary flag)
└─ Bỏ: GarageCars (multicollinear với GarageArea)
```

#### Area Features
```
├─ Tạo: AvgRoomSize = GrLivArea / TotRmsAbvGrd
└─ Bỏ: TotRmsAbvGrd (multicollinear với GrLivArea)
```

#### Basement Features
```
├─ Tạo: HasBasement (binary flag)
├─ Tạo: BasementResid (orthogonalized với 1stFlrSF + HasBasement)
└─ Bỏ: TotalBsmtSF (multicollinear với 1stFlrSF)
```

#### Age Features
```
├─ Tạo: HouseAge (năm từ khi xây dựng)
├─ Tạo: GarageLag (garage construction lag)
├─ Tạo: GarageSameAsHouse (binary flag)
└─ Bỏ: YearBuilt, GarageYrBlt (redundant)
```

#### Quality Features
```
├─ Fireplace: HasFireplace + ExtraFireplaces
├─ Masonry: HasMasonryVeneer + MasVnrAreaResid (orthogonalized)
├─ Second Floor: Has2ndFlr + SecondFlrShare_resid (orthogonalized)
└─ Bỏ: Raw features (highly correlated)
```

**Thành tựu chính:**
✅ Giảm multicollinearity đáng kể
✅ Tạo derived features có thể giải thích
✅ Orthogonalized residuals cho independence
✅ Binary flags cho patterns có/không có

### 3. TRANSFORMATION (✅ HOÀN THÀNH)

**File:** `src/Transformation.py`
**Input:** 1239 × 87 (sau FE)
**Output:** 1239 × 88 (1 feature mới: KitchenAbvGr_Binned)

#### Target Transformation
```
├─ SalePrice → log1p(SalePrice)
├─ Skewness: 2.009 → 0.205 (giảm 89.8%)
└─ Kết quả: Nearly symmetric distribution ✓
```

#### Feature Transformations
```
├─ Log1p: 15 features (tất cả positive values)
├─ Yeo-Johnson: 9 features (zero-inflated/negative)
├─ Binning: KitchenAbvGr → KitchenAbvGr_Binned
└─ No transform: Binary flags + residuals (orthogonal)
```

#### Cross-Fit Strategy
```
├─ Fit parameters chỉ trên train set
├─ Apply cùng parameters cho test set
└─ Ngăn chặn data leakage ✓
```

**Kết quả nổi bật:**
✅ Đã xử lý skewness cho 24 biến đầu vào
✅ Biến mục tiêu phân phối gần đối xứng (skewness xấp xỉ 0.2)
✅ Chiến lược cross-fit đảm bảo không rò rỉ dữ liệu huấn luyện
✅ Quá trình biến đổi luôn có thể thực hiện lại được dễ dàng

### 4. ENCODING (✅ HOÀN THÀNH)

**File:** `src/Encoding.py`
**Input:** 1239 × 88 (sau transformation)
**Output:** 1239 × 177 (89 features encoded mới)

#### Ordinal Encoding (17 features)
```
├─ Quality scales: ExterQual, KitchenQual, BsmtQual, etc.
├─ Mapping: Ex > Gd > TA > Fa > Po
├─ Finish levels: GarageFinish, BsmtFinType1/2
└─ Shape/Slope: LotShape, LandSlope, PavedDrive
```

#### One-Hot Encoding (24 → 114 features)
```
├─ Nominal categoricals: MSZoning, Exterior1st, Condition1, etc.
├─ Low/medium cardinality features
└─ Generated 114 binary features
```

#### Target Encoding (2 features)
```
├─ Neighborhood: 25 categories → 1D feature
├─ Exterior2nd: 16 categories → 1D feature
└─ Cross-fit strategy ngăn chặn leakage
```

#### Feature Scaling
```
├─ StandardScaler áp dụng cho tất cả 176 features
├─ Mean=0, Std=1 cho tất cả features
└─ Sẵn sàng cho regularization models
```

**Thành tựu chính:**
✅ 177 total features (176 + target)
✅ Tất cả features numeric (sẵn sàng cho sklearn)
✅ Cross-fit encoding ngăn chặn leakage
✅ Proper scaling cho regularization

### 5. INTERACTION FEATURES (✅ BỎ QUA - CÓ LÝ DO)

**Quyết định:** Không tạo Interaction Features explicit

#### Lý do bỏ qua:
```
├─ Tree-based models (LightGBM, XGBoost) tự động học interactions
├─ 177 features đã đủ phức tạp (có thể overfitting)
├─ Regularization models (Ridge/Lasso) ít cần explicit interactions
├─ Cross-validation sẽ tìm optimal complexity
└─ Có thể thêm selective interactions sau nếu cần
```

#### Phân tích chi tiết:
```
├─ Current features: 177 (đã comprehensive)
├─ Potential interactions: 177×176/2 = 15,576 pairs
├─ Risk: Curse of dimensionality + overfitting
├─ Alternative: Let models learn implicit interactions
└─ Future: Có thể thêm domain-specific interactions
```

#### Các loại interactions có thể xem xét sau:
```
├─ Area × Quality: GrLivArea × OverallQual
├─ Age × Condition: HouseAge × OverallCond
├─ Location × Quality: Neighborhood × OverallQual
├─ Garage × Basement: HasGarage × HasBasement
└─ Binary flags: HasFireplace × HasGarage
```

#### Khi nào nên thêm Interaction Features:
```
├─ Nếu model performance plateau
├─ Nếu cross-validation cho thấy underfitting
├─ Nếu domain knowledge suggests specific interactions
├─ Nếu feature importance analysis reveals patterns
└─ Nếu ensemble methods cần explicit interactions
```

**Kết luận:** ✅ Bỏ qua Interaction Features cho giai đoạn này
- Focus vào regularization tuning trước
- Models sẽ học implicit interactions
- Có thể thêm selective interactions sau nếu cần

### 6. PHÁT HIỆN OUTLIERS (✅ HOÀN THÀNH)

**Phân tích chi tiết outliers:**  
Sau khi đã transform dữ liệu, việc phát hiện và xử lý các giá trị ngoại lai (outliers) giúp đảm bảo model không bị ảnh hưởng bởi các điểm bất thường. Dưới đây là tổng kết kết quả phát hiện outlier và giải thích rõ ràng các quyết định giữ lại toàn bộ các điểm này.

#### Outliers trong Target Variable (SalePrice)
```
├─ Skewness: 2.009 (Độ lệch nhỏ - phân phối gần đối xứng ✔️)
├─ Số mẫu có z-score > 3: 21 (chiếm 1.7%) ← Updated
├─ Số mẫu nằm ngoài khoảng IQR (Interquartile Range): 56 (chiếm 4.5%) ← Updated
├─ Mean: $180,949 | Median: $163,000 | Std: $80,428
└─ Nhận định: Các outlier này phản ánh sự đa dạng tự nhiên của giá nhà, không phải là lỗi hoặc bất thường cần loại bỏ. Do vậy, giữ nguyên tất cả.
```

#### Outliers trong các Feature
```
├─ Các biến cờ nhị phân (binary flags): 0% outliers (6 feature) – không có giá trị bất thường
├─ 24 feature không có outlier
├─ 12 feature có rất ít outlier (0-5%) ✔️ – chấp nhận được
├─ 5 feature có lượng outlier vừa phải (5-10%) ⚠ – cần lưu ý nhưng hợp lý
└─ 2 feature có tỷ lệ outlier cao (>10%) ✔️ – có lý do giải thích rõ ràng
```

#### Giải thích cụ thể (dễ hiểu) về các feature có nhiều outlier:
```
├─ MasVnrAreaResid (17.6%): Phần dư (sai số) giữa diện tích ốp tường đá thực tế và giá trị dự đoán theo các tiêu chí còn lại – giá trị lớn bất thường thể hiện các căn nhà có phần ốp tường khác biệt hẳn so với xu hướng chung.
├─ BasementResid (17.1%): Phần dư (sai số) liên quan đến diện tích hầm (basement) – outlier nghĩa là nhà đó có hầm lớn/nhỏ bất thường so với các đặc điểm khác.
├─ GarageAreaPerCar (9.3%): Diện tích gara chia cho số chỗ để xe – outlier xuất hiện khi 1 chỗ nhưng gara lại rất rộng (rất “thừa”, thiết kế lạ) hoặc ngược lại.
├─ OverallCond (8.4%): Điểm đánh giá tổng thể về điều kiện căn nhà (thang bậc 1-9) – các điểm cực kỳ cao hoặc thấp thường là outlier, phản ánh bất thường về chất lượng.
├─ LotFrontage (8.0%): Chiều rộng mặt tiền đất – những lô có mặt tiền rất rộng (nhà góc, biệt thự) hoặc cực hẹp sẽ bị xem là outlier.
└─ MSSubClass (7.1%): Phân loại kiểu nhà theo mã số xây dựng – một số mã ít xuất hiện có thể tạo thành outlier do hiếm thấy trên thị trường. 
```

#### Tại sao giữ lại toàn bộ outlier?
```
├─ Các thuật toán regularization như Ridge/Lasso được thiết kế để giảm ảnh hưởng tiêu cực của outlier lên model. L2 penalty (Ridge) sẽ thu nhỏ tác động các điểm bất thường một cách mềm dẻo, L1 penalty (Lasso) thậm chí có thể loại bỏ hoàn toàn feature nhiễu nếu cần.
├─ Nếu loại bỏ các outlier này, ta sẽ mất một phần thông tin thực tế liên quan tới sự đa dạng hoặc trường hợp đặc biệt của thị trường nhà đất.
├─ Số lượng sample là 1239 – nếu giữ lại tất cả sẽ tận dụng tối đa dữ liệu.
├─ Quá trình Cross-validation sẽ tự động chọn tham số α sao cho phù hợp nhất với cấu trúc dữ liệu thực, kể cả khi tồn tại outlier.
```

**Kết quả quan trọng:**
- ✅ Đã hoàn thiện phân tích outlier toàn diện, minh bạch
- ✅ Đưa ra quyết định giữ toàn bộ outlier nhờ phương pháp regularization
- ✅ Đã trực quan hóa (3 biểu đồ, giúp giải thích cho người dùng)
- ✅ Báo cáo chi tiết chứng minh logic và lý do rõ ràng


---

## ⏭️ GIAI ĐOẠN TIẾP THEO: MÔ HÌNH HÓA

### Sẵn sàng cho Implementation
```
✅ Train data: data/processed/train_encoded.csv (1239×177)
✅ Test data: data/processed/test_encoded.csv (219×177)
✅ Target: SalePrice (log-transformed)
✅ Tất cả features: Numeric, scaled, no nulls
✅ Outliers: Đã phân tích, quyết định (giữ tất cả)
```

### Các mô hình dự kiến
1. **Ridge Regression (L2)**
   - Tốt nhất cho: Data này với residuals
   - α tuning: [0.001, 0.01, 0.1, 1, 10, 100]
   - Kỳ vọng: Stable coefficients, good generalization

2. **Lasso Regression (L1)**
   - Tốt nhất cho: Feature selection
   - α tuning: [0.001, 0.01, 0.1, 1, 10, 100]
   - Kỳ vọng: Sparse model, interpretable

3. **ElasticNet (L1 + L2)**
   - Tốt nhất cho: Kết hợp benefits
   - α tuning: [0.001, 0.01, 0.1, 1]
   - l1_ratio tuning: [0, 0.5, 1]
   - Kỳ vọng: Best flexibility

### Cross-Validation Strategy
```
├─ 5-Fold CV trên train data
├─ Metric: Negative MSE (maximize)
├─ Grid search over α values
└─ Final evaluation trên test set
```

### Hiệu năng dự kiến
```
├─ Ridge/Lasso: R² ≈ 0.85-0.92 (với proper α)
├─ Robust despite outliers (regularization handles them)
├─ Feature importance: Stable và interpretable
└─ Generalization: Good (cross-validation optimized)
```

---

## 📁 CẤU TRÚC DỰ ÁN

```
Project-5.1/
├── app.py                          # Main orchestration
├── README.md                       # Documentation
├── requirements.txt                # Dependencies
│
├── data/
│   ├── raw/                        # Original dataset
│   │   └── train-house-prices-advanced-regression-techniques.csv
│   ├── interim/                    # Config files
│   │   ├── encoding_config.json
│   │   └── transformation_config.json
│   └── processed/                  # Final data
│       ├── train_encoded.csv      # Sẵn sàng cho modeling
│       └── test_encoded.csv
│
├── src/                            # Source code
│   ├── Preprocessing.py           # ✅ Hoàn thành
│   ├── FeatureEngineering.py      # ✅ Hoàn thành
│   ├── Transformation.py           # ✅ Hoàn thành
│   └── Encoding.py                 # ✅ Hoàn thành
│
├── reports/                        # Analysis reports
│   ├── Outlier_Detection_Report.md
│   ├── OUTLIER_ANALYSIS_SUMMARY.txt
│   └── plots/                      # Visualizations
│       ├── 01_target_distribution.png
│       ├── 02_top_outlier_features.png
│       └── 03_outlier_distribution.png
│
└── models/                         # (Future: trained models)
```

---

## 📊 TÓM TẮT DỮ LIỆU CUỐI CÙNG

| Giai đoạn | Shape | Features | Ghi chú |
|-----------|-------|----------|---------|
| Raw | (1460, 81) | 81 | Original dataset |
| Sau Preprocessing | (1458, 81) | 81 | 2 dòng xóa, 0 nulls |
| Train Split | (1239, 81) | 81 | 85% cho training |
| Test Split | (219, 81) | 81 | 15% holdout |
| Sau FE | (1239, 87) | 87 | +6 derived features |
| Sau Transform | (1239, 88) | 88 | +1 binned feature |
| **Sau Encode** | **(1239, 177)** | **177** | **Sẵn sàng cho modeling** |

### Phân tích Features
```
├─ Ordinal encoded: 17 features
├─ One-Hot encoded: 114 features
├─ Target encoded: 2 features
├─ Numeric (transformed): 43 features
└─ Total: 176 features + target
```

---

## ✅ KIỂM TRA CHẤT LƯỢNG

### Data Quality Checks
```
✅ No null values trong dữ liệu cuối
✅ Tất cả features numeric (sẵn sàng cho sklearn)
✅ No data leakage (early split + cross-fit)
✅ Skewness reduced (target nearly symmetric)
✅ Multicollinearity addressed (orthogonalized residuals)
✅ Outliers analyzed (quyết định: giữ tất cả)
✅ Proper scaling (StandardScaler applied)
```

### Pipeline Robustness
```
✅ Modular design (separate files cho mỗi step)
✅ Reproducible (random seeds, saved configs)
✅ Cross-fit strategy (no leakage)
✅ Error handling (try/catch blocks)
✅ Progress tracking (detailed logging)
```

---

## 🎯 THÀNH TỰU

### Thành tựu kỹ thuật
```
✅ 1460 → 1239 train samples (15% test holdout)
✅ 81 → 177 features (comprehensive encoding)
✅ 6940 → 0 null values (complete preprocessing)
✅ Multicollinearity significantly reduced
✅ Skewness: 2.009 → 0.205 (giảm 89.8%)
✅ Tất cả features sẵn sàng cho regularization models
```

### Thành tựu quy trình
```
✅ Early train/test split ngăn chặn leakage
✅ Cross-fit strategy cho tất cả transformations
✅ Comprehensive outlier analysis
✅ Modular, maintainable code structure
✅ Detailed documentation và reports
```

---

## 🚀 SẴN SÀNG CHO MÔ HÌNH HÓA

**Trạng thái:** ✅ Tất cả giai đoạn tiền xử lý hoàn thành

**Bước tiếp theo:**
1. Implement Ridge/Lasso/ElasticNet models
2. Hyperparameter tuning với cross-validation
3. Model evaluation và comparison
4. Feature importance analysis
5. Final predictions trên test set

**Timeline dự kiến:**
- Model implementation: 1-2 giờ
- Hyperparameter tuning: 2-3 giờ
- Evaluation và analysis: 1 giờ
- **Tổng: 4-6 giờ cho giai đoạn modeling hoàn chỉnh**

---

**Báo cáo được tạo:** 2025-10-24
**Trạng thái dự án:** ✅ Hoàn thành tiền xử lý - Sẵn sàng cho mô hình hóa
**Giai đoạn tiếp theo:** Implementation Regression + Regularization