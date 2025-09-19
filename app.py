import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from difflib import get_close_matches

# ⚡ Page setup
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

# ==========================
# Main app heading
# ==========================
st.title("📊 Customer Churn Predictor")
st.markdown("Predict telecom customer churn with a **Random Forest model** 🚀")

# ==========================
# ⚡ Load model once
# ==========================
@st.cache_resource
def load_model():
    import requests, zipfile, io, os
    
    # Download zip from GitHub (replace with your raw file link)
    url = "https://github.com/your-username/your-repo/raw/main/random_forest_pipeline.zip"
    r = requests.get(url)

    # Unzip into a folder
    extract_path = "unzipped_model"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(extract_path)

    # Auto-detect the .pkl file
    files = os.listdir(extract_path)
    pkl_files = [f for f in files if f.endswith(".pkl")]
    if not pkl_files:
        raise FileNotFoundError("No .pkl file found in the extracted zip!")
    
    model_path = os.path.join(extract_path, pkl_files[0])  # pick the first .pkl
    return joblib.load(model_path)

@st.cache_data
def process_csv(file):
    return pd.read_csv(file)

model = load_model()

# ==========================
# Sidebar instructions
# ==========================
st.sidebar.header("ℹ️ How to Use This App")
st.sidebar.markdown("""
1. Choose **Single Customer** to enter one customer's details manually.  
2. Choose **Batch Upload** to upload a CSV file with multiple customers.  
3. The app will automatically map columns even if names differ slightly.  
4. View predictions instantly and download results if needed.  
""")

# ==========================
# Infer columns from pipeline
# ==========================
pipeline_columns = model.named_steps["preprocess"].feature_names_in_
numeric_cols = [col for col in pipeline_columns if col in ["tenure", "MonthlyCharges", "TotalCharges"]]
categorical_cols = [col for col in pipeline_columns if col not in numeric_cols]

# ==========================
# Column mapping function
# ==========================
def map_columns(uploaded_cols, pipeline_cols):
    mapping = {}
    for col in pipeline_cols:
        match = get_close_matches(col, uploaded_cols, n=1, cutoff=0.6)
        if match:
            mapping[match[0]] = col
    return mapping

# ==========================
# Input Mode
# ==========================
mode = st.radio("Choose Input Mode:", ["Single Customer", "Batch Upload"])

# ==========================
# Single Customer Prediction
# ==========================
if mode == "Single Customer":
    st.subheader("🔍 Enter Customer Details")

    # Numeric inputs
    inputs_num = {}
    for col in numeric_cols:
        if col == "tenure":
            inputs_num[col] = st.number_input(f"{col} (months)", min_value=0, max_value=72, value=12)
        else:
            inputs_num[col] = st.number_input(f"{col} ($)", min_value=0.0, max_value=20000.0, value=100.0)

    # Predefined categorical options
    cat_options = {
        "gender": ["Male", "Female"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "YesNo": ["Yes", "No"]
    }

    # Categorical inputs (skip customerID)
    inputs_cat = {}
    for col in categorical_cols:
        if col == "customerID":
            continue
        if col in cat_options:
            inputs_cat[col] = st.selectbox(col, cat_options[col])
        else:
            inputs_cat[col] = st.selectbox(col, cat_options["YesNo"])

    if st.button("Predict Churn"):
        input_df = pd.DataFrame([{**inputs_num, **inputs_cat}])
        for col in pipeline_columns:
            if col not in input_df.columns:
                input_df[col] = 0 if col in numeric_cols else "No"

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ This customer is **likely to churn**. (Risk: {probability:.2%})")
        else:
            st.success(f"✅ This customer is **likely to stay**. (Churn Risk: {probability:.2%})")

# ==========================
# Batch Upload Prediction
# ==========================
else:
    st.subheader("📂 Upload a CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        data = process_csv(uploaded_file)
        st.success("✅ File Uploaded Successfully!")
        st.dataframe(data.head(50))

        # Fuzzy mapping
        col_mapping = map_columns(data.columns, pipeline_columns)
        data = data.rename(columns=col_mapping)

        # Sidebar: Column mapping preview
        st.sidebar.subheader("📋 Column Mapping Preview")
        mapping_preview = pd.DataFrame(list(col_mapping.items()), columns=["Uploaded Column", "Mapped Column"])
        st.sidebar.dataframe(mapping_preview)

        # Remove old prediction columns to avoid duplicates
        for col in ["Churn Prediction", "Churn Probability"]:
            if col in data.columns:
                data = data.drop(columns=[col])

        # Add missing columns
        for col in pipeline_columns:
            if col not in data.columns:
                data[col] = 0 if col in numeric_cols else "No"

        # Ensure numeric columns
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

        # Predict
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]
        data["Churn Prediction"] = ["Churn" if p == 1 else "Stay" for p in predictions]
        data["Churn Probability"] = probabilities

        st.subheader("📊 Prediction Results (Top 100 Preview)")
        st.dataframe(data.head(100))

        # Sidebar: Top-risk customers including customerID
        st.sidebar.subheader("🔥 Top 10 High-Risk Customers")
        top_risk = data.sort_values("Churn Probability", ascending=False).head(10)

        # Columns to display in sidebar
        sidebar_cols = ["Churn Probability", "Churn Prediction"]
        if "customerID" in data.columns:
            sidebar_cols.insert(0, "customerID")  # Add customerID at the front

        # Optionally add a few extra columns
        extra_cols = [
            col for col in data.columns
            if col not in list(pipeline_columns) + ["Churn Prediction", "Churn Probability", "customerID"]
        ][:3]
        sidebar_cols += extra_cols

        st.sidebar.dataframe(top_risk[sidebar_cols])

        # Plot
        plot_data = data if len(data) <= 5000 else data.sample(5000, random_state=42)
        fig = px.histogram(
            plot_data,
            x="Churn Probability",
            color="Churn Prediction",
            nbins=20,
            title="Churn Probability Distribution",
            color_discrete_map={"Churn": "red", "Stay": "green"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download full results
        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button("📥 Download Full Predictions", csv, "predictions.csv", "text/csv")
