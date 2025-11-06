# House Price Prediction

This project predicts house prices using a simple machine learning model and provides an interactive Streamlit web application.  
Users can upload their own datasets, train a regression model, and make predictions through a web interface.

## Project Overview

The Streamlit app automatically handles:
- Uploading a CSV dataset
- Detecting numeric and categorical columns
- Training a regression model (Linear Regression)
- Generating a form for prediction with smart numeric ranges
- Removing unnecessary ID or serial columns
- Producing live predictions after model training

## How to Run the Application

1. Clone the repository
   
   git clone https://github.com/<your-username>/House-Price-Prediction.git
   cd House-Price-Prediction/app

2. Install dependencies

   pip install -r requirements.txt

3. Run the Streamlit app

   streamlit run smart_house_price_app.py

4. Open your browser
   
   Visit http://localhost:8501 to access the app

