import joblib
import pandas as pd
import streamlit as st
from pathlib import Path


MODEL_PATH = Path("models/predict_freight_model.pkl")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Please run train.py first.")
        st.stop()
    return joblib.load(MODEL_PATH)


def main():
    st.set_page_config(page_title="Freight Cost Predictor", page_icon="🚚", layout="wide")
    st.title("🚚 Freight Cost Predictor")
    st.caption("Estimate freight expense from shipment details using your trained regression model.")

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    model = load_model()

    with st.form("prediction_form"):
        st.subheader("Shipment Details")
        col1, col2 = st.columns(2)

        with col1:
            dollars = st.number_input(
                "Invoice Amount (USD)",
                min_value=0.0,
                value=1000.0,
                step=100.0,
                format="%.2f"
            )
            quantity = st.number_input(
                "Quantity",
                min_value=0.0,
                value=1.0,
                step=1.0,
                format="%.0f"
            )

        with col2:
            vendor_number = st.text_input("Vendor Number", value="105")
            approval = st.selectbox("Approval Status", ["Pending", "Approved", "Rejected"], index=1)

        submitted = st.form_submit_button("Predict Freight", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame({"Dollars": [dollars]})
        prediction = model.predict(input_df)[0]

        st.success("Estimated Freight Cost")
        st.metric(label="Predicted Freight", value=f"${prediction:,.2f}")

        with st.container():
            st.markdown("### Input Summary")
            st.write(f"- Vendor Number: {vendor_number}")
            st.write(f"- Quantity: {quantity}")
            st.write(f"- Approval Status: {approval}")

        st.balloons()


if __name__ == "__main__":
    main()
