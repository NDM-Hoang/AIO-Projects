<!-- b88b0110-772e-41d6-ae56-cd4938b05e2b b5895be2-a461-4e1f-8b9a-7efec6db02db -->
# Streamlit House Price Demo App

## Goals

- Train and persist the best model (Lasso), then serve predictions via a Streamlit UI.
- Single-record prediction via a grouped form; auto-fill unspecified fields with train defaults.
- Reuse existing preprocessing/FE/transform/encoding logic to avoid leakage.

## Changes

### 1) Persist best model after training

- Update `app.py` modeling step to refit the best model on training data with its best params and save artifacts:
- `models/best_model.pkl` (scikit-learn estimator)
- `models/best_model_features.json` (ordered feature names used for training)
- Keep `models/best_model_config.json` (already saved)

### 2) Add lightweight inference helpers

- Create `src/Serving.py` with:
- `load_artifacts()` to load model, feature list, configs in `data/interim/*` and defaults computed from `data/processed/train_data.csv`.
- `prepare_single_record(raw_input: Dict) -> pd.DataFrame` to build a one-row raw dataframe (81 cols), fill missing with defaults.
- `run_full_processing(df_raw_one_row) -> pd.DataFrame` to apply `Preprocessing` → `FeatureEngineering` → `Transformation` → `Encoding` using configs so that output matches training schema.
- `predict_single(input_dict) -> float` to return predicted SalePrice (inverse log if needed) and optionally show intermediate values.

### 3) Build Streamlit UI

- Add `streamlit_app.py`:
- Sidebar: navigation (Predict, Model Info, Data Preview).
- Predict page: grouped form sections (Location & Lot, House Quality, Area, Basement, Garage); sensible defaults; on submit -> call `predict_single` and display predicted price.
- Model Info: read `models/best_model_config.json`, show metrics and params; optionally preview top features.
- Data Preview: show few rows from `data/processed/train_data.csv` and pipeline summary from `reports/ProcessReport.md` if exists.
- Use `st.cache_resource` for artifacts and `st.cache_data` for defaults/data.

### 4) Dependencies & run

- Add `streamlit` to `requirements.txt`.
- How to run:
- `python app.py --step all` (build datasets and train; persists model)
- `streamlit run streamlit_app.py`

## Notes

- Target `SalePrice` was log-transformed; ensure inverse-transform for final price in UI.
- If LightGBM/XGBoost are unavailable, linear model persists as planned.
- If any processed files are missing, guide user to run `python app.py --step all`.

### To-dos

- [ ] Refit best model with best params and save .pkl and features list
- [ ] Create src/Serving.py for loading artifacts and single-record processing
- [ ] Implement streamlit_app.py with grouped form and result display
- [ ] Add streamlit to requirements; document run commands