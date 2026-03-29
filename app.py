import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score

# --- Page Setup ---
st.set_page_config(page_title="Heart Disease Project", layout="wide")
st.title("Heart Disease Diagnostic Dashboard")

# --- Model Training and Validation ---
@st.cache_data
def train_and_validate():
    # Load the data
    df = pd.read_csv('heart.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split for metric calculation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and Train (C=0.1 helps prevent overfitting)
    model = LogisticRegression(solver='liblinear', C=0.1)
    model.fit(X_train_scaled, y_train)
    
    # Get Metrics for the dashboard
    y_pred = model.predict(X_test_scaled)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    return model, scaler, X.columns, cv_scores, precision, recall

model, scaler, feature_names, cv_scores, precision, recall = train_and_validate()

# --- Sidebar Inputs ---
st.sidebar.header("Patient Medical Data")

def get_user_input():
    inputs = {}
    inputs['age'] = st.sidebar.slider("Age", 1, 100, 50)
    inputs['sex'] = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x==1 else "Female")
    inputs['cp'] = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    inputs['trestbps'] = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
    inputs['chol'] = st.sidebar.number_input("Cholesterol", 100, 600, 240)
    inputs['fbs'] = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0, 1])
    inputs['restecg'] = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
    inputs['thalach'] = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
    inputs['exang'] = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    inputs['oldpeak'] = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    inputs['slope'] = st.sidebar.selectbox("ST Slope", [0, 1, 2])
    inputs['ca'] = st.sidebar.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
    inputs['thal'] = st.sidebar.selectbox("Thalassemia", [1, 2, 3])
    return pd.DataFrame(inputs, index=[0])

input_df = get_user_input()

# --- Main Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Diagnostic Result")
    
    if st.button("Run Diagnostic"):
        # Scale input and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]
        
        if prediction[0] == 1:
            st.error(f"Prediction: Heart Disease Detected ({probability:.2%})")
        else:
            st.success(f"Prediction: No Heart Disease ({probability:.2%})")
            
        st.progress(probability)
        
        # --- Local Contribution Analysis ---
        st.write("---")
        st.subheader("Patient-Specific Drivers")
        
        # Contribution = Scaled Value * Weight
        contributions = input_scaled[0] * model.coef_[0]
        contrib_series = pd.Series(contributions, index=feature_names).sort_values()
        
        top_risk = contrib_series.idxmax()
        top_protect = contrib_series.idxmin()
        
        st.write(f"**Primary Risk Driver:** {top_risk}")
        st.write(f"**Primary Protective Factor:** {top_protect}")
        
        with st.expander("Show mathematical breakdown"):
            st.table(contrib_series.sort_values(ascending=False))
            
    else:
        st.info("Adjust values and click Predict.")

with col2:
    st.subheader("2. Model Logic (Global Coefficients)")
    # Show feature importance visually
    importance = pd.Series(model.coef_[0], index=feature_names).sort_values()
    fig, ax = plt.subplots()
    importance.plot(kind='barh', color='#2c7bb6', ax=ax)
    st.pyplot(fig)

# --- Performance Section ---
st.divider()
st.subheader("3. Model Validation Summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("CV Accuracy", f"{cv_scores.mean():.2%}")
m2.metric("Stability (Std)", f"{cv_scores.std():.4f}")
m3.metric("Precision", f"{precision:.2%}")
m4.metric("Recall", f"{recall:.2%}")