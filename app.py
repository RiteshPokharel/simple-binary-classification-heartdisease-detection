import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# --- Page Configuration ---
st.set_page_config(page_title="Heart Disease Analysis", layout="wide")
st.title("Cardiovascular Diagnostic Analysis")
st.markdown("Assess patient heart disease risk using clinical markers and explainable AI logic.")

# --- Data Processing and Model Training ---
@st.cache_data
def train_and_validate():
    df = pd.read_csv('heart.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(solver='liblinear', C=0.1)
    model.fit(X_train_scaled, y_train)
    
    # Validation data for Confusion Matrix and metrics
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    return model, scaler, X.columns, cv_scores, precision, recall, cm

model, scaler, feature_names, cv_scores, precision, recall, cm = train_and_validate()

# --- Patient Data Input Section ---
st.sidebar.header("Patient Clinical Profile")

def get_user_input():
    inputs = {}
    inputs['age'] = st.sidebar.slider("Patient Age", 1, 100, 50)
    
    inputs['sex'] = st.sidebar.selectbox("Biological Sex", [0, 1], 
        format_func=lambda x: "Male" if x == 1 else "Female")
    
    inputs['cp'] = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], 
        format_func=lambda x: {
            0: "Typical Angina (Heart-related)", 
            1: "Atypical Angina", 
            2: "Non-anginal Pain", 
            3: "Asymptomatic (No pain)"
        }[x], help="Type 0 is generally the highest risk indicator.")
    
    inputs['trestbps'] = st.sidebar.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120,
        help="Values below 120 mm Hg are generally considered healthy.")
    
    inputs['chol'] = st.sidebar.number_input("Serum Cholesterol (chol)", 100, 600, 240,
        help="Total cholesterol under 200 mg/dl is usually healthy.")
    
    # Humanized Fasting Blood Sugar labels
    inputs['fbs'] = st.sidebar.selectbox("Fasting Blood Sugar (fbs)", [0, 1], 
        format_func=lambda x: "Higher than 120 mg/dl" if x == 1 else "Lower than 120 mg/dl",
        help="A value lower than 120 mg/dl is generally better for heart health.")
    
    inputs['restecg'] = st.sidebar.selectbox("Resting ECG Results (restecg)", [0, 1, 2], 
        format_func=lambda x: {
            0: "Normal", 
            1: "ST-T Wave Abnormality", 
            2: "Left Ventricular Hypertrophy"
        }[x], help="A 'Normal' result is the healthy baseline.")
    
    inputs['thalach'] = st.sidebar.slider("Maximum Heart Rate (thalach)", 60, 220, 150,
        help="A higher maximum heart rate achieved during stress is often a sign of a stronger heart.")
    
    inputs['exang'] = st.sidebar.selectbox("Exercise Induced Angina (exang)", [0, 1], 
        format_func=lambda x: "Yes" if x == 1 else "No", 
        help="Selecting 'No' is generally a positive indicator.")
    
    inputs['oldpeak'] = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0,
        help="Lower values (closer to 0) are typically healthier.")
    
    inputs['slope'] = st.sidebar.selectbox("ST Slope (slope)", [0, 1, 2], 
        format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x],
        help="'Upsloping' is usually the healthiest response to exercise.")
    
    inputs['ca'] = st.sidebar.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3],
        help="Zero (0) colored vessels is the ideal healthy result.")
    
    inputs['thal'] = st.sidebar.selectbox("Thalassemia Result (thal)", [1, 2, 3], 
        format_func=lambda x: {
            1: "Fixed Defect", 
            2: "Normal", 
            3: "Reversible Defect"
        }[x], help="'Normal' is the healthy state.")
        
    return pd.DataFrame(inputs, index=[0])

user_df = get_user_input()

# --- Result and Analysis Display ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Diagnostic Assessment")
    if st.button("Generate Prediction"):
        input_scaled = scaler.transform(user_df)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]
        
        bias = model.intercept_[0]
        contributions = input_scaled[0] * model.coef_[0]
        total_logit = bias + contributions.sum()
        
        if prediction[0] == 1:
            st.error(f"Detection: Heart Disease Risk Identified ({probability:.2%})")
        else:
            st.success(f"Detection: No Heart Disease Risk Identified ({probability:.2%})")
        
        st.progress(probability)
        
        st.divider()
        st.subheader("Individual Risk Contribution")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Baseline Bias", f"{bias:.4f}")
        m_col2.metric("Total Input Impact", f"{contributions.sum():.4f}")
        m_col3.metric("Final Risk Score", f"{total_logit:.4f}")

        st.markdown("### Feature Breakdown (Value × Weight)")
        contrib_series = pd.Series(contributions, index=feature_names).sort_values(ascending=False)
        
        primary_risk = contrib_series.idxmax()
        primary_protection = contrib_series.idxmin()
        
        st.info(f"The primary driver increasing risk is **{primary_risk}**. The primary factor lowering risk is **{primary_protection}**.")
        st.table(contrib_series)
    else:
        st.info("Adjust the patient profile and click the button to analyze risk.")

with col2:
    st.subheader("Global Model Logic")
    importance = pd.Series(model.coef_[0], index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    importance.plot(kind='barh', color='#1f77b4', ax=ax)
    ax.set_title("How the model weighs each factor overall")
    st.pyplot(fig)

# --- Validation and Technical Summary ---
st.divider()
st.subheader("Model Validation Summary")

metric_col, cm_col = st.columns([1, 1])

with metric_col:
    st.write("Reliability Metrics")
    p1, p2 = st.columns(2)
    p3, p4 = st.columns(2)
    p1.metric("Average Accuracy", f"{cv_scores.mean():.2%}")
    p2.metric("Prediction Stability", f"{cv_scores.std():.4f}")
    p3.metric("Precision", f"{precision:.2%}")
    p4.metric("Recall (Sensitivity)", f"{recall:.2%}")
    st.caption("Recall measures the model's ability to identify truly sick patients.")

with cm_col:
    st.write("Confusion Matrix (Test Data)")
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                xticklabels=['Healthy', 'Sick'], yticklabels=['Healthy', 'Sick'])
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('Actual Label')
    st.pyplot(fig_cm)