# Hướng Dẫn Sử Dụng Explainable AI (XAI) cho Model Dự Đoán Giá Nhà

## Tổng Quan

Dự án đã được tích hợp **SHAP (SHapley Additive exPlanations)** để giải thích cách các model machine learning đưa ra dự đoán giá nhà.

## Cài Đặt

1. **Cài đặt SHAP:**
```bash
pip install shap
```

Hoặc cài đặt tất cả dependencies:
```bash
pip install -r requirements.txt
```

## Cấu Trúc Module

### 1. `src/Explainability.py`
Module chính cung cấp các tính năng explainability:
- **ModelExplainer**: Class để giải thích model
  - `get_global_feature_importance()`: Tầm quan trọng tổng thể của features
  - `explain_prediction()`: Giải thích cho một dự đoán cụ thể
  - `get_summary_plot_data()`: Dữ liệu cho summary plot
  - `get_waterfall_plot_data()`: Dữ liệu cho waterfall plot

- **load_explainer_from_artifacts()**: Hàm tiện ích để load explainer từ model artifacts

### 2. Tích Hợp vào Streamlit

Ứng dụng Streamlit đã có các tính năng:

#### **Trang Predict (Dự Đoán)**
- Sau khi dự đoán, bạn sẽ thấy phần "🔍 Giải thích dự đoán"
- Hiển thị top 15 features ảnh hưởng đến dự đoán
- Visualization với màu sắc:
  - 🟢 Xanh lá: Feature làm tăng giá
  - 🔴 Đỏ: Feature làm giảm giá
- Bảng chi tiết với SHAP values và phần trăm đóng góp

#### **Trang Model Explainability**
- **Global Feature Importance**: 
  - Top N features quan trọng nhất trong model
  - Visualization và bảng dữ liệu
  - Thống kê tổng quan
  
- **Giải thích về SHAP**: Hướng dẫn cách đọc và hiểu SHAP values

## Cách Sử Dụng

### 1. Xem Global Feature Importance

1. Chạy ứng dụng Streamlit:
```bash
streamlit run streamlit_app.py
```

2. Vào trang **"Model Explainability"** ở sidebar
3. Điều chỉnh số lượng features muốn xem (10-50)
4. Xem visualization và bảng dữ liệu

### 2. Xem Local Explanation (Giải thích dự đoán)

1. Vào trang **"Predict"**
2. Nhập thông tin căn nhà
3. Nhấn "Predict Price"
4. Scroll xuống phần "🔍 Giải thích dự đoán" để xem:
   - Features nào làm tăng/giảm giá
   - Mức độ ảnh hưởng của từng feature
   - Base value và tổng contribution

### 3. Sử Dụng Trong Code

```python
from src.Explainability import load_explainer_from_artifacts

# Load explainer
explainer = load_explainer_from_artifacts()

# Global importance
importance = explainer.get_global_feature_importance(max_features=20)
print(importance)

# Local explanation
explanation = explainer.explain_prediction(X_instance)
print(explanation)
```

## Hiểu Về SHAP Values

### Global Feature Importance
- **Mean Absolute SHAP Value**: Giá trị trung bình |SHAP| trung bình qua nhiều samples
- Giá trị cao hơn = feature quan trọng hơn trong model tổng thể

### Local Explanation
- **Base Value**: Giá trị baseline (trung bình của model)
- **SHAP Values**: Đóng góp của từng feature cho dự đoán cụ thể
  - SHAP > 0: Feature làm tăng giá dự đoán
  - SHAP < 0: Feature làm giảm giá dự đoán
- **Prediction**: = Base Value + Sum(SHAP values)

### Ví Dụ
Giả sử:
- Base Value: 11.5 (log scale)
- OverallQual SHAP: +0.3 (tăng giá)
- LotArea SHAP: +0.1 (tăng giá)
- Age SHAP: -0.2 (giảm giá)
- Prediction: 11.5 + 0.3 + 0.1 - 0.2 = 11.7

## Model Types Được Hỗ Trợ

1. **Linear Models** (Lasso, Ridge, ElasticNet)
   - Sử dụng `LinearExplainer` (nhanh và chính xác)

2. **Tree Models** (XGBoost, LightGBM)
   - Sử dụng `TreeExplainer` (nhanh và chính xác)

3. **Other Models**
   - Fallback sang `KernelExplainer` (chậm hơn nhưng hoạt động với mọi model)

## Lưu Ý

1. **Hiệu Suất**: 
   - Tree explainer và Linear explainer rất nhanh
   - Kernel explainer có thể chậm với dataset lớn

2. **Memory**:
   - SHAP có thể tốn bộ nhớ với dataset lớn
   - Module tự động sample dữ liệu khi cần

3. **Độ Chính Xác**:
   - TreeExplainer và LinearExplainer cho kết quả chính xác
   - KernelExplainer là approximation (nhưng vẫn rất tốt)

## Troubleshooting

### Lỗi: "SHAP not available"
```bash
pip install shap
```

### Lỗi: "Could not load explainer"
- Đảm bảo đã train model: `python app.py --step model`
- Kiểm tra các file trong `models/`:
  - `best_model.pkl`
  - `best_model_features.json`
  - `best_model_config.json`
- Kiểm tra `data/processed/train_encoded.csv` tồn tại

### Lỗi khi explain prediction
- Đảm bảo input features đúng format và có đầy đủ columns
- Kiểm tra feature names khớp với training data

## Tài Liệu Tham Khảo

- [SHAP Documentation](https://shap.readthedocs.io/)
- [SHAP GitHub](https://github.com/slundberg/shap)
- Paper: ["A Unified Approach to Interpreting Model Predictions"](https://arxiv.org/abs/1705.07874)

## Các Tính Năng Mở Rộng (Có Thể Thêm Sau)

- Waterfall plots cho từng prediction
- Partial dependence plots
- Interaction effects
- LIME integration (alternative to SHAP)
- Explanation comparison giữa các models

