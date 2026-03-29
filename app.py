import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, auc

# --- Page Configuration ---
st.set_page_config(page_title="Heart Disease Analysis", layout="wide")
st.title("Cardiovascular Diagnostic Analysis")
st.markdown("Assess patient heart disease risk using clinical markers and explainable AI logic.")

# --- Feature Engineering ---
def engineer_features(df):
    df = df.copy()
    new_features = []

    # 1. Age Group — clinical age risk brackets (AHA guidelines)
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 40, 55, 70, 120],
        labels=[0, 1, 2, 3]
    ).astype(float)
    new_features.append('age_group')

    # 2. Blood Pressure Category — JNC hypertension classification
    df['bp_category'] = pd.cut(
        df['trestbps'],
        bins=[0, 120, 130, 140, 300],
        labels=[0, 1, 2, 3]
    ).astype(float)
    new_features.append('bp_category')

    # 3. Cholesterol Risk Flag — AHA threshold at 240 mg/dl
    df['chol_risk'] = (df['chol'] > 240).astype(int)
    new_features.append('chol_risk')

    # 4. Heart Rate Reserve — Karvonen formula proxy for cardiovascular fitness
    df['hr_reserve'] = (220 - df['age']) - df['thalach']
    new_features.append('hr_reserve')

    # 5. Angina-ST Interaction — compound exercise stress indicator
    df['angina_st_combo'] = df['exang'] * df['oldpeak']
    new_features.append('angina_st_combo')

    return df, new_features


# --- Data Processing and Model Training ---
@st.cache_data
def train_and_validate():
    df = pd.read_csv('heart.csv')
    df_eng, new_features = engineer_features(df)

    X = df_eng.drop('target', axis=1)
    y = df_eng['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    return model, scaler, X.columns.tolist(), new_features, cv_scores, precision, recall, cm, fpr, tpr, roc_auc

model, scaler, feature_names, new_features, cv_scores, precision, recall, cm, fpr, tpr, roc_auc = train_and_validate()

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
        user_eng, _ = engineer_features(user_df)
        user_eng = user_eng[feature_names]

        input_scaled = scaler.transform(user_eng)
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

        contrib_df = contrib_series.reset_index()
        contrib_df.columns = ['Feature', 'Contribution']
        contrib_df['Engineered'] = contrib_df['Feature'].apply(
            lambda x: 'Yes' if x in new_features else 'No'
        )
        st.table(contrib_df)

    else:
        st.info("Adjust the patient profile and click the button to analyze risk.")

with col2:
    st.subheader("Global Model Logic")
    importance = pd.Series(model.coef_[0], index=feature_names).sort_values()
    colors = ['#d62728' if f in new_features else '#1f77b4' for f in importance.index]

    fig, ax = plt.subplots(figsize=(8, 7))
    importance.plot(kind='barh', color=colors, ax=ax)
    ax.set_title("How the model weighs each factor overall")
    blue_patch = mpatches.Patch(color='#1f77b4', label='Original feature')
    red_patch = mpatches.Patch(color='#d62728', label='Engineered feature')
    ax.legend(handles=[blue_patch, red_patch])
    st.pyplot(fig)

# --- Feature Engineering Section ---
st.divider()
st.subheader("Feature Engineering — What Was Done & Why")
st.markdown("Five new clinically-motivated features were derived from the original 13 to give the model richer signal:")

fe_data = {
    "Feature": ["age_group", "bp_category", "chol_risk", "hr_reserve", "angina_st_combo"],
    "Formula": [
        "cut(age, [0,40,55,70,120])",
        "cut(trestbps, [0,120,130,140,300])",
        "(chol > 240).astype(int)",
        "(220 - age) - thalach",
        "exang × oldpeak"
    ],
    "Type": ["Binning", "Binning", "Binary Flag", "Interaction", "Interaction"],
    "Rationale": [
        "CV risk accelerates past 55 — bins capture non-linear age risk better than a raw value",
        "Encodes JNC hypertension stages; 128 and 138 mmHg are numerically close but clinically different",
        "AHA classifies >240 mg/dl as high risk — a threshold LR can't learn cleanly without the flag",
        "Karvonen formula: how far below predicted max HR the patient peaked — low value = poor fitness",
        "Angina + ST depression compound each other; their product captures combined exercise stress"
    ]
}
st.table(pd.DataFrame(fe_data))

# --- Validation and Technical Summary ---
st.divider()
st.subheader("Model Validation Summary")

metric_col, cm_col = st.columns([1, 1])

with metric_col:
    st.write("Reliability Metrics")
    p1, p2 = st.columns(2)
    p3, p4 = st.columns(2)
    p5, _ = st.columns(2)
    p1.metric("Average Accuracy", f"{cv_scores.mean():.2%}")
    p2.metric("Prediction Stability", f"{cv_scores.std():.4f}")
    p3.metric("Precision", f"{precision:.2%}")
    p4.metric("Recall (Sensitivity)", f"{recall:.2%}")
    p5.metric("ROC-AUC", f"{roc_auc:.3f}")
    st.caption("Recall measures the model's ability to identify truly sick patients.")

    st.write("Cross-Validation Fold Scores")
    fig_cv, ax_cv = plt.subplots(figsize=(5, 2.5))
    ax_cv.bar([f'Fold {i+1}' for i in range(len(cv_scores))], cv_scores, color='#1f77b4')
    ax_cv.axhline(cv_scores.mean(), color='red', linewidth=1.5, linestyle='--',
                  label=f'Mean = {cv_scores.mean():.3f}')
    ax_cv.set_ylim(0.7, 1.0)
    ax_cv.legend()
    plt.tight_layout()
    st.pyplot(fig_cv)

with cm_col:
    st.write("Confusion Matrix (Test Data)")
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['Healthy', 'Sick'], yticklabels=['Healthy', 'Sick'])
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('Actual Label')
    st.pyplot(fig_cm)

    st.write("ROC Curve")
    fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
    ax_roc.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'AUC = {roc_auc:.3f}')
    ax_roc.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random baseline')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend()
    plt.tight_layout()
    st.pyplot(fig_roc)