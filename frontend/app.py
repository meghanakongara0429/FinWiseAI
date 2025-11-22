import streamlit as st
import requests
import pandas as pd
import base64
import io

# Backend URL
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="FinWiseAI", layout="wide")

st.title("💳 FinWiseAI – Smart Transaction Categorizer")
st.write("AI-powered financial transaction classification with Explainability + Hybrid Rule + ML Engine.")

# -----------------------------------------------
# Helper: Colored badge
# -----------------------------------------------
def render_badge(text, color):
    return f"""
    <span style="
        background-color:{color};
        padding:4px 10px;
        border-radius:6px;
        color:white;
        font-weight:600;
        font-size:12px;">
        {text}
    </span>
    """

# -----------------------------------------------
# Single Prediction
# -----------------------------------------------
st.subheader("🔍 Single Transaction Prediction")

single_text = st.text_input("Enter your transaction text:")

if st.button("Predict Category"):
    if single_text.strip() == "":
        st.warning("⚠ Please enter a transaction.")
    else:
        try:
            response = requests.post(f"{API_URL}/predict", json={"text": single_text})
            result = response.json()

            col1, col2 = st.columns(2)

            with col1:
                st.success(f"### 🏷 Predicted Category: **{result['predicted_category']}**")

                if result["source"] == "RULE-BASED":
                    st.markdown(render_badge("RULE-BASED", "#008000"), unsafe_allow_html=True)
                else:
                    st.markdown(render_badge("ML MODEL", "#4169E1"), unsafe_allow_html=True)

                st.write(f"### 🤖 Confidence: `{round(result['confidence'], 3)}`")

            with col2:
                st.info("### 🔍 Explainability Insights")
                if len(result["keywords_triggered"]) > 0:
                    st.write("**Matched Keywords:**")
                    st.write(result["keywords_triggered"])
                else:
                    st.write("No direct keywords found. ML model handled prediction.")

            # Feedback buttons
            st.write("### 🙋 Feedback")
            colA, colB = st.columns(2)

            with colA:
                if st.button("👍 Prediction Correct"):
                    st.success("Thanks for your feedback!")

            with colB:
                if st.button("👎 Prediction Incorrect"):
                    correct_cat = st.selectbox("Select correct category:", [
                        "FOOD", "SHOPPING", "TRAVEL", "BILLS", "MEDICAL",
                        "ENTERTAINMENT", "SALARY", "INVESTMENT", "TRANSFER"
                    ])

                    if st.button("Submit Correction"):
                        fb = {
                            "transaction": single_text,
                            "wrong_prediction": result["predicted_category"],
                            "correct_category": correct_cat
                        }
                        requests.post(f"{API_URL}/feedback", json=fb)
                        st.success("Feedback saved! Model will improve in future updates.")

        except Exception as e:
            st.error("⚠ Backend not reachable. Start FastAPI server.")
            st.exception(e)

# ---------------------------------------------------
# CSV Batch Prediction
# ---------------------------------------------------
st.subheader("📁 Batch Prediction (CSV)")

csv_file = st.file_uploader("Upload CSV with 'transaction' column", type=["csv"])

if csv_file:
    df = pd.read_csv(csv_file)
    st.write("### Preview:")
    st.dataframe(df.head())

    if st.button("Predict CSV"):
        file_bytes = csv_file.getvalue()
        response = requests.post(f"{API_URL}/predict-batch", data=file_bytes)
        results = response.json()

        result_df = pd.DataFrame(results)
        st.success("### 🎉 Categorization Complete!")
        st.dataframe(result_df)

        # Download Button
        csv_output = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Categorized CSV",
            data=csv_output,
            file_name="categorized_transactions.csv",
            mime="text/csv",
        )

# ---------------------------------------------------
# PDF Upload Section
# ---------------------------------------------------
st.subheader("📘 Upload Bank Statement (PDF) – Coming Soon")

st.info("PDF parsing is supported but UI integration coming in next update.")
