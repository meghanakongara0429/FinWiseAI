import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="FinWiseAI - Transaction Categorizer")

st.title("💸 FinWiseAI – Smart Transaction Categorizer")
st.write("Enter any financial transaction to predict its category.")

transaction_text = st.text_input(
    "Enter transaction description:",
    placeholder="Example: Paid electricity bill ₹1200"
)

if st.button("Predict Category"):
    if not transaction_text.strip():
        st.warning("⚠️ Please enter a transaction text.")
    else:
        try:
            # CALL BACKEND USING QUERY PARAMS
            response = requests.post(
                BACKEND_URL,
                params={"text": transaction_text}  # ✔ FIXED
            )

            if response.status_code == 200:
                result = response.json()
                st.success(f"📌 **Predicted Category:** {result['predicted_category']}")
            else:
                st.error("⚠️ Could not connect to backend API.")
                st.write("Error:", response.text)

        except Exception as e:
            st.error("⚠️ Failed to connect to backend.")
            st.write(e)

st.markdown("---")
st.write("Built for GHCI Hackathon 2025 – FinWiseAI")
