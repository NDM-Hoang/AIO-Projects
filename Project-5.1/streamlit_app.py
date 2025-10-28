import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure src is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from src.Serving import load_artifacts, prepare_single_record, predict_single  # noqa: E402


st.set_page_config(page_title="House Price Demo", page_icon="🏠", layout="wide")


@st.cache_resource(show_spinner=False)
def _cached_artifacts():
    return load_artifacts()


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
            LotArea = st.number_input("LotArea", min_value=0, value=int(defaults.get("LotArea", 8450)))
        with col2:
            LotFrontage = st.number_input("LotFrontage", min_value=0, value=int(defaults.get("LotFrontage", 65)))
            LotShape = st.selectbox("LotShape", ["Reg", "IR1", "IR2", "IR3"]) 
            LandContour = st.selectbox("LandContour", ["Lvl", "Bnk", "HLS", "Low"]) 
        with col3:
            OverallQual = st.slider("OverallQual", 1, 10, int(defaults.get("OverallQual", 5)))
            OverallCond = st.slider("OverallCond", 1, 10, int(defaults.get("OverallCond", 5)))

        st.subheader("Area & Rooms")
        col4, col5, col6 = st.columns(3)
        with col4:
            GrLivArea = st.number_input("GrLivArea", min_value=0, value=int(defaults.get("GrLivArea", 1460)))
            TotRmsAbvGrd = st.number_input("TotRmsAbvGrd", min_value=0, value=int(defaults.get("TotRmsAbvGrd", 6)))
        with col5:
            BedroomAbvGr = st.number_input("BedroomAbvGr", min_value=0, value=int(defaults.get("BedroomAbvGr", 3)))
            FullBath = st.number_input("FullBath", min_value=0, value=int(defaults.get("FullBath", 2)))
        with col6:
            HalfBath = st.number_input("HalfBath", min_value=0, value=int(defaults.get("HalfBath", 1)))
            KitchenAbvGr = st.number_input("KitchenAbvGr", min_value=0, value=int(defaults.get("KitchenAbvGr", 1)))

        st.subheader("Basement")
        col7, col8 = st.columns(2)
        with col7:
            TotalBsmtSF = st.number_input("TotalBsmtSF", min_value=0, value=int(defaults.get("TotalBsmtSF", 856)))
        with col8:
            FirstFlrSF = st.number_input("1stFlrSF", min_value=0, value=int(defaults.get("1stFlrSF", 856)))

        st.subheader("Garage")
        col9, col10, col11 = st.columns(3)
        with col9:
            GarageArea = st.number_input("GarageArea", min_value=0, value=int(defaults.get("GarageArea", 548)))
        with col10:
            GarageCars = st.number_input("GarageCars", min_value=0, value=int(defaults.get("GarageCars", 2)))
        with col11:
            GarageYrBlt = st.number_input("GarageYrBlt", min_value=1800, max_value=2050, value=int(defaults.get("GarageYrBlt", 2000)))

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
                st.metric("Predicted Price", f"${result['prediction']:,.0f}")
                with st.expander("Details"):
                    st.write({k: v for k, v in result.items()})
            except Exception as e:
                st.error(str(e))


def page_model_info():
    st.header("📘 Model Info")
    cfg_path = ROOT / "models" / "best_model_config.json"
    if cfg_path.exists():
        st.json(pd.read_json(cfg_path).to_dict())
    else:
        st.info("Config not found. Train the pipeline first.")


def page_data_preview():
    st.header("📄 Data Preview")
    train_path = ROOT / "data" / "processed" / "train_data.csv"
    if train_path.exists():
        df = pd.read_csv(train_path).head(10)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No processed data found. Run pipeline.")

    report_path = ROOT / "reports" / "ProcessReport.md"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=["Predict", "Model Info", "Data Preview"],
        index=0,
    )

    if page == "Predict":
        page_predict()
    elif page == "Model Info":
        page_model_info()
    else:
        page_data_preview()


if __name__ == "__main__":
    main()


