import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("House Price Prediction App")

# Session state setup
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Drop unwanted columns
    drop_cols = [col for col in data.columns if 'id' in col.lower() or 'serial' in col.lower() or 'index' in col.lower()]
    if drop_cols:
        data = data.drop(columns=drop_cols)
        st.info(f"Dropped columns: {', '.join(drop_cols)}")

    st.session_state.data = data
    st.dataframe(data.head())

# Model training
if st.session_state.data is not None:
    data = st.session_state.data
    target_column = st.selectbox("Select Target Column", data.columns)

    if st.button("Train Model"):
        X = data.drop(columns=[target_column])
        y = data[target_column]

        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R² Score: {r2:.3f}")

        st.session_state.model = model
        st.session_state.features = list(X.columns)
        st.session_state.target = target_column

# Prediction section
st.header("Predict Using Model")

if st.session_state.model is not None and st.session_state.data is not None:
    model = st.session_state.model
    feature_names = st.session_state.features
    data = st.session_state.data
    target_column = st.session_state.target

    st.subheader("Enter feature values:")

    input_data = {}
    for col in data.columns:
        if col == target_column:
            continue
        if 'id' in col.lower() or 'serial' in col.lower() or 'index' in col.lower():
            continue

        col_data = data[col].dropna()

        # Numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            col_lower = col.lower()

            # Floors
            if "floor" in col_lower or "storey" in col_lower:
                selected_value = st.selectbox(
                    f"{col}", 
                    options=[""] + [str(i) for i in range(0, 31)],
                    index=0
                )
                input_data[col] = float(selected_value) if selected_value != "" else np.nan

            # Discrete numeric columns
            elif len(col_data.unique()) <= 10:
                selected_value = st.selectbox(
                    f"{col}", 
                    options=[""] + [str(i) for i in sorted(set(col_data.unique()).union(set(range(0, 11))))],
                    index=0
                )
                input_data[col] = float(selected_value) if selected_value != "" else np.nan

            # Continuous numeric
            else:
                value = st.text_input(f"{col}", value="")
                try:
                    input_data[col] = float(value) if value != "" else np.nan
                except ValueError:
                    st.warning(f"Enter a valid number for {col}")

        # Categorical columns
        else:
            unique_values = list(col_data.unique())
            if len(unique_values) <= 10:
                input_data[col] = st.selectbox(f"{col}", options=[""] + unique_values, index=0)
            else:
                input_data[col] = st.text_input(f"{col}")

    # Predict button
    if st.button("Predict"):
        if any(pd.isna(list(input_data.values()))):
            st.warning("Fill all fields before predicting.")
        else:
            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=feature_names, fill_value=0)

            prediction = model.predict(input_df)[0]
            st.success(f"Predicted {target_column}: {prediction:,.2f}")

else:
    st.info("Upload a dataset and train your model first.")
