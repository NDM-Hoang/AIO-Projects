import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Ensure src package is importable for intra-module imports like `from src.FeatureEngineering ...`
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.Serving import load_artifacts, prepare_single_record, predict_single, run_full_processing  # noqa: E402
from src.Explainability import load_explainer_from_artifacts, SHAP_AVAILABLE  # noqa: E402


st.set_page_config(page_title="House Price Demo", page_icon="🏠", layout="wide")


@st.cache_resource(show_spinner=False)
def _cached_artifacts():
    return load_artifacts()


@st.cache_resource(show_spinner=False)
def _cached_explainer():
    """Load and cache the model explainer."""
    if not SHAP_AVAILABLE:
        return None
    try:
        return load_explainer_from_artifacts(
            models_dir=str(ROOT / "models"),
            processed_dir=str(ROOT / "data" / "processed")
        )
    except Exception as e:
        st.warning(f"Could not load explainer: {e}")
        return None


def page_predict():
    st.header("🏠 House Price Prediction")
    st.caption("Nhập thông tin căn nhà để dự báo giá.")

    model, feature_names, transform_config, defaults = _cached_artifacts()

    with st.form("predict_form"):
        st.subheader("Location & Lot")
        col1, col2, col3 = st.columns(3)
        with col1:
            MSZoning = st.selectbox("MSZoning", ["RL", "RM", "FV", "RH", "C (all)"])
            Neighborhood = st.text_input("Neighborhood", value="NAmes")
            LotArea = st.number_input("LotArea", min_value=0, value=int(defaults.get("LotArea", 8450) or 8450))
        with col2:
            LotFrontage = st.number_input("LotFrontage", min_value=0, value=int(defaults.get("LotFrontage", 65) or 65))
            LotShape = st.selectbox("LotShape", ["Reg", "IR1", "IR2", "IR3"]) 
            LandContour = st.selectbox("LandContour", ["Lvl", "Bnk", "HLS", "Low"]) 
        with col3:
            OverallQual = st.slider("OverallQual", 1, 10, int(defaults.get("OverallQual", 5) or 5))
            OverallCond = st.slider("OverallCond", 1, 10, int(defaults.get("OverallCond", 5) or 5))

        st.subheader("Area & Rooms")
        col4, col5, col6 = st.columns(3)
        with col4:
            GrLivArea = st.number_input("GrLivArea", min_value=0, value=int(defaults.get("GrLivArea", 1460) or 1460))
            TotRmsAbvGrd = st.number_input("TotRmsAbvGrd", min_value=0, value=int(defaults.get("TotRmsAbvGrd", 6) or 6))
        with col5:
            BedroomAbvGr = st.number_input("BedroomAbvGr", min_value=0, value=int(defaults.get("BedroomAbvGr", 3) or 3))
            FullBath = st.number_input("FullBath", min_value=0, value=int(defaults.get("FullBath", 2) or 2))
        with col6:
            HalfBath = st.number_input("HalfBath", min_value=0, value=int(defaults.get("HalfBath", 1) or 1))
            KitchenAbvGr = st.number_input("KitchenAbvGr", min_value=0, value=int(defaults.get("KitchenAbvGr", 1) or 1))

        st.subheader("Basement")
        col7, col8 = st.columns(2)
        with col7:
            TotalBsmtSF = st.number_input("TotalBsmtSF", min_value=0, value=int(defaults.get("TotalBsmtSF", 856) or 856))
        with col8:
            FirstFlrSF = st.number_input("1stFlrSF", min_value=0, value=int(defaults.get("1stFlrSF", 856) or 856))

        st.subheader("Garage")
        col9, col10, col11 = st.columns(3)
        with col9:
            GarageArea = st.number_input("GarageArea", min_value=0, value=int(defaults.get("GarageArea", 548) or 548))
        with col10:
            GarageCars = st.number_input("GarageCars", min_value=0, value=int(defaults.get("GarageCars", 2) or 2))
        with col11:
            GarageYrBlt = st.number_input("GarageYrBlt", min_value=1800, max_value=2050, value=int(defaults.get("GarageYrBlt", 2000) or 2000))

        st.subheader("Quality (Ordinal)")
        col12, col13, col14 = st.columns(3)
        with col12:
            ExterQual = st.selectbox("ExterQual", ["Po", "Fa", "TA", "Gd", "Ex"], index=2)
            KitchenQual = st.selectbox("KitchenQual", ["Po", "Fa", "TA", "Gd", "Ex"], index=2)
        with col13:
            BsmtQual = st.selectbox("BsmtQual", ["None", "Po", "Fa", "TA", "Gd", "Ex"], index=3)
            HeatingQC = st.selectbox("HeatingQC", ["Po", "Fa", "TA", "Gd", "Ex"], index=2)
        with col14:
            FireplaceQu = st.selectbox("FireplaceQu", ["None", "Po", "Fa", "TA", "Gd", "Ex"], index=0)
            GarageQual = st.selectbox("GarageQual", ["None", "Po", "Fa", "TA", "Gd", "Ex"], index=2)

        submitted = st.form_submit_button("Predict Price", type="primary")

    if submitted:
        input_dict = {
            "MSZoning": MSZoning,
            "Neighborhood": Neighborhood,
            "LotArea": LotArea,
            "LotFrontage": LotFrontage,
            "LotShape": LotShape,
            "LandContour": LandContour,
            "OverallQual": OverallQual,
            "OverallCond": OverallCond,
            "GrLivArea": GrLivArea,
            "TotRmsAbvGrd": TotRmsAbvGrd,
            "BedroomAbvGr": BedroomAbvGr,
            "FullBath": FullBath,
            "HalfBath": HalfBath,
            "KitchenAbvGr": KitchenAbvGr,
            "TotalBsmtSF": TotalBsmtSF,
            "1stFlrSF": FirstFlrSF,
            "GarageArea": GarageArea,
            "GarageCars": GarageCars,
            "GarageYrBlt": GarageYrBlt,
            "ExterQual": ExterQual,
            "KitchenQual": KitchenQual,
            "BsmtQual": BsmtQual,
            "HeatingQC": HeatingQC,
            "FireplaceQu": FireplaceQu,
            "GarageQual": GarageQual,
            # Minimal set; Serving will fill other fields by defaults
        }
        with st.spinner("Predicting..."):
            try:
                result = predict_single(input_dict)
                st.success("Prediction complete")
                
                col_pred, col_info = st.columns([2, 1])
                with col_pred:
                    st.metric("Predicted Price", f"${result['prediction']:,.0f}")
                with col_info:
                    st.metric("Prediction (log scale)", f"{result['prediction_log']:.4f}")
                
                # Show explanation if available
                explainer = _cached_explainer()
                if explainer is not None:
                    with st.expander("🔍 Giải thích dự đoán (Explain Prediction)", expanded=True):
                        try:
                            # Prepare the input for explanation
                            model, feature_names, transform_config, defaults = _cached_artifacts()
                            df_raw = prepare_single_record(input_dict, defaults)
                            df_encoded = run_full_processing(df_raw)
                            X_instance = df_encoded.drop(columns=["SalePrice"], errors="ignore")
                            
                            # Ensure correct column order
                            for col in feature_names:
                                if col not in X_instance.columns:
                                    X_instance[col] = 0.0
                            X_instance = X_instance[feature_names]
                            # Ensure X_instance is a DataFrame, not Series
                            if isinstance(X_instance, pd.Series):
                                X_instance = X_instance.to_frame().T
                            
                            # Get explanation
                            explanation = explainer.explain_prediction(X_instance)
                            
                            # Show top contributing features
                            contributions_df = pd.DataFrame(explanation['feature_contributions'])
                            top_contributions = contributions_df.head(15)
                            
                            st.subheader("📊 Top Features Ảnh Hưởng Đến Dự Đoán")
                            
                            # Create visualization
                            fig, ax = plt.subplots(figsize=(10, 8))
                            top_15 = top_contributions.head(15).sort_values('shap_value', ascending=True)
                            
                            colors = ['#ff4444' if x < 0 else '#44ff44' for x in top_15['shap_value']]
                            y_pos = np.arange(len(top_15))
                            
                            ax.barh(y_pos, top_15['shap_value'], color=colors, alpha=0.7)
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(top_15['feature'], fontsize=9)
                            ax.set_xlabel('SHAP Value (Contribution to Prediction)', fontsize=11)
                            ax.set_title('Feature Contributions to Prediction', fontsize=12, fontweight='bold')
                            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                            ax.grid(axis='x', alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Show base value and prediction breakdown
                            st.subheader("📈 Chi Tiết Dự Đoán")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Base Value", f"{explanation['base_value']:.4f}")
                            with col2:
                                total_contribution = sum(explanation['shap_values'])
                                st.metric("Total Contribution", f"{total_contribution:.4f}")
                            with col3:
                                st.metric("Final Prediction", f"{explanation['prediction']:.4f}")
                            
                            # Show feature contributions table
                            st.subheader("📋 Bảng Đóng Góp Của Từng Feature")
                            display_df = top_contributions[['feature', 'shap_value']].copy()
                            display_df['contribution_%'] = (display_df['shap_value'] / abs(display_df['shap_value']).sum() * 100).round(2)
                            display_df.columns = ['Feature', 'SHAP Value', 'Contribution %']
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                            
                        except Exception as e:
                            st.warning(f"Không thể tạo giải thích: {e}")
                            import traceback
                            with st.expander("Chi tiết lỗi"):
                                st.code(traceback.format_exc())
                
                with st.expander("Chi tiết kỹ thuật"):
                    st.write({k: v for k, v in result.items()})
            except Exception as e:
                st.error(str(e))
                import traceback
                with st.expander("Chi tiết lỗi"):
                    st.code(traceback.format_exc())


def page_model_info():
    st.header("📘 Model Info")
    cfg_path = ROOT / "models" / "best_model_config.json"
    if cfg_path.exists():
        st.json(pd.read_json(cfg_path).to_dict())
    else:
        st.info("Config not found. Train the pipeline first.")



def page_explain():
    """Page for model explainability - global feature importance."""
    st.header("🔍 Model Explainability (XAI)")
    st.caption("Tìm hiểu cách model đưa ra dự đoán thông qua SHAP values")
    
    if not SHAP_AVAILABLE:
        st.error("⚠️ SHAP chưa được cài đặt. Vui lòng cài đặt bằng lệnh: `pip install shap`")
        st.code("pip install shap")
        return
    
    explainer = _cached_explainer()
    if explainer is None:
        st.error("⚠️ Không thể load model explainer. Đảm bảo đã train model và có đầy đủ artifacts.")
        return
    
    st.info(f"📊 Model hiện tại: **{explainer.model_name}**")
    
    # Global Feature Importance
    st.subheader("📊 Global Feature Importance")
    st.write("""
    Bảng dưới đây hiển thị tầm quan trọng tổng thể của các features trong model. 
    Các features có importance cao hơn có ảnh hưởng lớn hơn đến dự đoán giá nhà.
    """)
    
    max_features = st.slider("Số lượng features hiển thị", 10, 50, 20)
    
    with st.spinner("Đang tính toán feature importance..."):
        importance_df = explainer.get_global_feature_importance(max_features=max_features)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, max(8, max_features * 0.4)))
    top_features = importance_df.head(max_features)
    
    ax.barh(range(len(top_features)), top_features['importance_abs'], 
            color=plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(top_features))))
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=10)
    ax.set_xlabel('Mean Absolute SHAP Value (Importance)', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {max_features} Most Important Features', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Table view
    st.subheader("📋 Bảng Feature Importance")
    display_df = importance_df[['feature', 'importance', 'importance_abs']].copy()
    display_df.columns = ['Feature', 'Mean SHAP Value', 'Importance (Absolute)']
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Summary statistics
    st.subheader("📈 Thống Kê")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tổng số features", len(explainer.feature_names))
    with col2:
        st.metric("Features hiển thị", len(importance_df))
    with col3:
        top_5_importance = importance_df.head(5)['importance_abs'].sum()
        total_importance = importance_df['importance_abs'].sum()
        percentage = (top_5_importance / total_importance * 100) if total_importance > 0 else 0
        st.metric("Top 5 features chiếm", f"{percentage:.1f}%")
    
    # Explanation of SHAP
    with st.expander("📖 Giải thích về SHAP Values"):
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** là một phương pháp để giải thích output của các model machine learning.
        
        **Global Feature Importance:**
        - Hiển thị tầm quan trọng tổng thể của mỗi feature
        - Tính bằng giá trị trung bình của |SHAP value| qua nhiều mẫu
        - Giá trị cao hơn = feature quan trọng hơn
        
        **Local Explanation (trong trang Predict):**
        - Giải thích cho từng dự đoán cụ thể
        - Cho biết mỗi feature đóng góp bao nhiêu vào dự đoán này
        - SHAP value > 0: feature làm tăng giá dự đoán
        - SHAP value < 0: feature làm giảm giá dự đoán
        
        **Cách đọc:**
        - Base value: Giá trị trung bình của model (baseline)
        - SHAP values: Đóng góp của từng feature
        - Prediction = Base value + Tổng SHAP values
        """)


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=["Predict", "Model Explainability", "Model Info"],
        index=0,
    )

    if page == "Predict":
        page_predict()
    elif page == "Model Explainability":
        page_explain()
    elif page == "Model Info":
        page_model_info()



if __name__ == "__main__":
    main()


