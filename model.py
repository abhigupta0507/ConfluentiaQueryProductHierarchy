import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import xgboost as xgb
import io

# -----------------------------
# Load embedding model & trained classifier
# -----------------------------
@st.cache_resource
def load_model():
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model("xgb_model.json")   # Save your trained model beforehand
    return model, xgb_clf

model, xgb_clf = load_model()


# -----------------------------
# Feature engineering function
# -----------------------------
def make_features(q_emb, c_emb):
    cos_sim = np.array([util.cos_sim(q, c).item() for q, c in zip(q_emb, c_emb)])
    element_product = q_emb * c_emb
    abs_diff = np.abs(q_emb - c_emb)
    feats = np.hstack([cos_sim.reshape(-1, 1), q_emb, c_emb, element_product, abs_diff])
    return feats


# -----------------------------
# Streamlit App
# -----------------------------
st.title("📊 Query-Category Matching Inference App")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataframe
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Ensure required columns exist
    if {"origin_query", "category_path"} <= set(df.columns):
        if st.button("Run Inference"):
            with st.spinner("Encoding and predicting..."):
                # Encode
                queries = model.encode(df["origin_query"].tolist(), show_progress_bar=True, batch_size=256)
                cats = model.encode(df["category_path"].tolist(), show_progress_bar=True, batch_size=256)

                # Features
                X_test = make_features(queries, cats)

                # Predict
                preds = xgb_clf.predict(X_test)

                # Add predictions to dataframe
                result_df = df.copy()
                result_df["prediction"] = preds

                st.subheader("Results")
                st.dataframe(result_df.head())

                # Download button
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
    else:
        st.error("CSV must contain 'origin_query' and 'category_path' columns.")