import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model
model = joblib.load("../models/decision_tree_model.pkl")

# Title
st.title("🔍 HR Analytics – Employee Attrition Predictor")

# Upload HR data
uploaded_file = st.file_uploader("Upload HR CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Preview of Uploaded Data", df.head())

    # Make predictions
    if st.button("Predict Attrition"):
        predictions = model.predict(df)
        df['Attrition_Predicted'] = predictions
        st.success("Prediction completed ✅")
        st.write(df[['Attrition_Predicted']])

        # SHAP Explanation
        st.subheader("🔎 SHAP Explanation for First Prediction")
        explainer = shap.Explainer(model, df)
        shap_values = explainer(df)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(bbox_inches='tight')

        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Prediction Results", csv, "predictions.csv", "text/csv")

else:
    st.info("Please upload a valid HR dataset (CSV format).")

