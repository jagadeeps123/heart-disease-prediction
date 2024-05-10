import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Define mappings for categorical variables
sex_mapping = {0: 'Female', 1: 'Male'}
cp_mapping = {0: 'typical angina', 1: 'atypical angina', 2: 'non-anginal', 3: 'asymptomatic'}
fbs_mapping = {0: 'FALSE', 1: 'TRUE'}
restecg_mapping = {0: 'normal', 1: 'lv hypertrophy', 2: 'other'}
exang_mapping = {0: 'FALSE', 1: 'TRUE'}
slope_mapping = {0: 'downsloping', 1: 'flat', 2: 'upsloping'}
thal_mapping = {0: 'normal', 1: 'fixed defect', 2: 'reversable defect', 3: 'other'}

# Replace numerical values with words for specified columns
data['sex'] = data['sex'].map(sex_mapping)
data['cp'] = data['cp'].map(cp_mapping)
data['fbs'] = data['fbs'].map(fbs_mapping)
data['restecg'] = data['restecg'].map(restecg_mapping)
data['exang'] = data['exang'].map(exang_mapping)
data['slope'] = data['slope'].map(slope_mapping)
data['thal'] = data['thal'].map(thal_mapping)


# Save Logistic Regression model using pickle
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(logistic, f)

# Streamlit App
st.title('Heart Disease Prediction')

# Input fields for user input
st.sidebar.header('User Input')

# Input fields for specified columns
input_values = {}
input_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
for col in input_columns:
    if col == 'sex':
        input_values[col] = st.sidebar.radio(f'Select {col}', options=list(sex_mapping.values()))
    elif col == 'cp':
        input_values[col] = st.sidebar.radio(f'Select {col}', options=list(cp_mapping.values()))
    elif col == 'fbs':
        input_values[col] = st.sidebar.radio(f'Select {col}', options=list(fbs_mapping.values()))
    elif col == 'restecg':
        input_values[col] = st.sidebar.radio(f'Select {col}', options=list(restecg_mapping.values()))
    elif col == 'exang':
        input_values[col] = st.sidebar.radio(f'Select {col}', options=list(exang_mapping.values()))
    elif col == 'slope':
        input_values[col] = st.sidebar.radio(f'Select {col}', options=list(slope_mapping.values()))
    elif col == 'thal':
        input_values[col] = st.sidebar.radio(f'Select {col}', options=list(thal_mapping.values()))
    else:
        input_values[col] = st.sidebar.number_input(f'Enter {col}', value=0)

# Predict function
def predict(model, input_values):
    # Prepare input data with one-hot encoding
    input_data = pd.DataFrame(columns=X_train.columns, data=[input_values])
    input_data = input_data.fillna(0)  # Fill missing values with zeros
    prediction = model.predict(input_data)
    # Map prediction to meaningful labels
    if prediction[0] == 0:
        return 'No heart disease'
    else:
        return 'Heart disease'



# Load Logistic Regression model using pickle
with open('logistic_model.pkl', 'rb') as f:
    logistic_model = pickle.load(f)

# Prediction
if st.sidebar.button('Predict'):
    prediction = predict(logistic_model, input_values)
    st.write('Prediction:', prediction)
