import streamlit as st
import pandas as pd
import pickle
import requests
import os
from PIL import Image
import logging

# Configure logging to write to a file in the current directory
logging.basicConfig(
    filename='logs/logfile_UI.txt',  
    level=logging.DEBUG,      
    format='%(asctime)s - %(levelname)s - %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Function to load artifacts
def load_artifact(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError as e:
        logging.error(f"Artifact file not found: {filename}")
        raise

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Adhoc Risk Profiling", "Batch Profiling"])

# Layout: Image on the left, title on the right
col1, col2 = st.columns([1, 3])
with col1:
    image = Image.open('risk-image2.jfif')  # Adjusted path
    st.image(image, use_column_width=True)

with col2:
    st.title("Loan Risk Categorization")
    image = Image.open('risk-image.png')  # Adjusted path

# Navigation logic
if page == "Home":
    st.write("Welcome to the Loan Risk Categorization App.")
    st.write("Use the sidebar to navigate to Adhoc or Batch Profiling.")
elif page == "Adhoc Risk Profiling":
    st.header("Enter customer details:")
    age = st.number_input("Age", min_value=18, max_value=100)
    income = st.number_input("Income", min_value=0)
    employment_type = st.selectbox("Employment Type", ['Salaried', 'Unemployed', 'Self-employed'])
    residence_type = st.selectbox("Residence Type", ['Parental Home', 'Rented', 'Owned'])
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term", min_value=0)
    previous_default = st.selectbox("Previous Default", ['Yes', 'No'])

    if st.button('Predict Risk Category'):
        pipeline = load_artifact('artifacts/data_processing_pipeline.pkl')  # Adjusted path
        model = load_artifact('artifacts/best_classifier.pkl')  # Adjusted path
        label_encoder = load_artifact('artifacts/label_encoder.pkl')  # Adjusted path

        input_df = pd.DataFrame([[age, income, employment_type, residence_type, credit_score, loan_amount, loan_term, previous_default]],
                                columns=['Age', 'Income', 'EmploymentType', 'ResidenceType', 'CreditScore', 'LoanAmount', 'LoanTerm', 'PreviousDefault'])
        logging.info(f"User input data frame created")
        
        transformed_input = pipeline.transform(input_df)
        prediction = model.predict(transformed_input)
        decoded_prediction = label_encoder.inverse_transform(prediction)
        
        st.subheader('Predicted Risk Category:')
        st.write(decoded_prediction[0])
        logging.info(f"Prediction: {decoded_prediction[0]}")
elif page == "Batch Profiling":
    st.header("Batch Profiling")
    uploaded_file = st.file_uploader("Upload your CSV file for batch prediction", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        logging.info(f"Batch file uploaded with {len(df)} records")

        response = requests.post("http://127.0.0.1:8000/batch_predict", json={"data": df.to_dict(orient="list")})

        if response.status_code == 200:
            predictions = response.json()
            output_df = pd.DataFrame(predictions)
            output_folder = 'Data/output'
            os.makedirs(output_folder, exist_ok=True)
            output_file_path = os.path.join(output_folder, 'batch_predictions.csv')
            output_df.to_csv(output_file_path, index=False)
            st.success(f"Batch predictions saved to {output_file_path}")
            logging.info(f"Batch predictions saved to {output_file_path}")
        else:
            st.error("Error during batch prediction.")
            logging.error(f"Batch prediction failed with status code {response.status_code}")
