import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("models/logistic_regression_model.pkl")
preprocessor = joblib.load("data/processed/loan_approval_dataset_preprocessed.pkl")

# Load sample data
df_sample = pd.read_csv("data/cleaned/loan_approval_dataset_cleaned.csv")
df_sample = df_sample.drop(columns=['loan_status'])

# Separate column types
num_columns = df_sample.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_columns = df_sample.select_dtypes(include=['object']).columns.tolist()

 
st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="centered",
)


st.markdown(
    "<h1 style='text-align: center; color: #c9e5ee;'>Loan Approval Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Enter customer details to check the loan approval status using a trained ML model.</p>",
    unsafe_allow_html=True
)
st.markdown("---")


user_data = {}

with st.form("user_input_form"):
    st.subheader("Applicant Details")

    # Categorical Inputs in Two Columns
    cat_col1, cat_col2 = st.columns(2)
    for i, col in enumerate(cat_columns):
        options = df_sample[col].dropna().unique().tolist()
        if i % 2 == 0:
            user_data[col] = cat_col1.selectbox(f"{col}", options)
        else:
            user_data[col] = cat_col2.selectbox(f"{col}", options)

    # Numerical Inputs in Two Columns
    num_col1, num_col2 = st.columns(2)
    for i, col in enumerate(num_columns):
        min_value = int(df_sample[col].min())
        max_value = int(df_sample[col].max())
        mean_value = float(df_sample[col].mean())
        if i % 2 == 0:
            user_data[col] = num_col1.number_input(
                f"{col}", min_value=min_value, max_value=max_value, value=int(mean_value)
            )
        else:
            user_data[col] = num_col2.number_input(
                f"{col}", min_value=min_value, max_value=max_value, value=int(mean_value)
            )

    submitted = st.form_submit_button("Predict")


if submitted:
    try:
        input_df = pd.DataFrame([user_data])
        transformed = preprocessor.transform(input_df)
        prediction = model.predict(transformed)[0]
        probability = model.predict_proba(transformed)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(f"Loan Approved with a probability of **{probability:.2%}**")
        else:
            st.error(f"Loan Not Approved with a probability of **{probability:.2%}**")

    except Exception as e:
        st.error(f"Prediction Failed: {e}")
