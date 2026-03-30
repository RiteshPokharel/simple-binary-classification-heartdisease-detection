import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class StandardScaler:
    def fit_transform(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1
        return (X - self.mean_) / self.std_

    def transform(self, X):
        return (X - self.mean_) / self.std_


class LogisticRegression:
    def __init__(self, lr=0.1, max_iter=1000, C=1.0):
        self.lr = lr
        self.max_iter = max_iter
        self.C = C

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n, p = X.shape
        self.w = np.zeros(p)
        self.b = 0.0
        for _ in range(self.max_iter):
            y_hat = self._sigmoid(X @ self.w + self.b)
            err = y_hat - y
            self.w -= self.lr * (X.T @ err / n + self.w / self.C)
            self.b -= self.lr * err.mean()
        self.coef_ = np.array([self.w])
        self.intercept_ = np.array([self.b])

    def predict_proba(self, X):
        p1 = self._sigmoid(X @ self.w + self.b)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)




def confusion_matrix_scratch(y_true, y_pred):
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    return np.array([[TN, FP], [FN, TP]])

def precision_scratch(y_true, y_pred):
    cm = confusion_matrix_scratch(y_true, y_pred)
    TP, FP = cm[1,1], cm[0,1]
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def recall_scratch(y_true, y_pred):
    cm = confusion_matrix_scratch(y_true, y_pred)
    TP, FN = cm[1,1], cm[1,0]
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def accuracy_scratch(y_true, y_pred):
    return float((y_true == y_pred).mean())

def roc_curve_scratch(y_true, y_prob):
    thresholds = np.sort(np.unique(y_prob))[::-1]
    P, N = y_true.sum(), len(y_true) - y_true.sum()
    fprs, tprs = [0.0], [0.0]
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        TP = int(((y_pred == 1) & (y_true == 1)).sum())
        FP = int(((y_pred == 1) & (y_true == 0)).sum())
        tprs.append(TP / P if P > 0 else 0.0)
        fprs.append(FP / N if N > 0 else 0.0)
    fprs.append(1.0); tprs.append(1.0)
    return np.array(fprs), np.array(tprs)

def cross_val_score_scratch(X, y, cv=5):
    rng = np.random.default_rng(42)
    classes = np.unique(y)
    class_folds = {}
    for c in classes:
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        class_folds[c] = np.array_split(idx, cv)
    scores = []
    for fold in range(cv):
        val_idx = np.concatenate([class_folds[c][fold] for c in classes])
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_val_s = sc.transform(X_val)
        m = LogisticRegression()
        m.fit(X_tr_s, y_tr)
        scores.append(accuracy_scratch(y_val, m.predict(X_val_s)))
    return np.array(scores)


st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("Heart Disease Risk Predictor")
st.caption("Logistic Regression implemented from scratch with NumPy — no sklearn.")


#LOAD DATA and TRAIN


@st.cache_data
def train():
    df = pd.read_csv('heart.csv')
    X = df.drop('target', axis=1).to_numpy().astype(float)
    y = df['target'].to_numpy().astype(float)
    feature_names = df.drop('target', axis=1).columns.tolist()

    # Stratified train/test split
    rng = np.random.default_rng(42)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        n_test = int(len(idx) * 0.2)
        test_idx.extend(idx[:n_test]); train_idx.extend(idx[n_test:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    cm        = confusion_matrix_scratch(y_test, y_pred)
    precision = precision_scratch(y_test, y_pred)
    recall    = recall_scratch(y_test, y_pred)
    fpr, tpr  = roc_curve_scratch(y_test, y_prob)
    roc_auc = float(np.sum((fpr[1:] - fpr[:-1]) * tpr[1:]))
    cv_scores = cross_val_score_scratch(X_train, y_train)

    return model, scaler, feature_names, cv_scores, precision, recall, cm, fpr, tpr, roc_auc

model, scaler, feature_names, cv_scores, precision, recall, cm, fpr, tpr, roc_auc = train()



st.sidebar.header("Patient Profile")

def get_user_input():
    i = {}
    i['age']      = st.sidebar.slider("Age", 1, 100, 50)
    i['sex']      = st.sidebar.selectbox("Sex", [0,1], format_func=lambda x: "Male" if x else "Female")
    i['cp']       = st.sidebar.selectbox("Chest Pain Type", [0,1,2,3],
                        format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",2:"Non-anginal",3:"Asymptomatic"}[x])
    i['trestbps'] = st.sidebar.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120)
    i['chol']     = st.sidebar.number_input("Cholesterol (mg/dl)", 100, 600, 240)
    i['fbs']      = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0,1], format_func=lambda x: "Yes" if x else "No")
    i['restecg']  = st.sidebar.selectbox("Resting ECG", [0,1,2],
                        format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[x])
    i['thalach']  = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
    i['exang']    = st.sidebar.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "Yes" if x else "No")
    i['oldpeak']  = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
    i['slope']    = st.sidebar.selectbox("ST Slope", [0,1,2],
                        format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
    i['ca']       = st.sidebar.selectbox("Major Vessels (ca)", [0,1,2,3])
    i['thal']     = st.sidebar.selectbox("Thalassemia", [1,2,3],
                        format_func=lambda x: {1:"Fixed Defect",2:"Normal",3:"Reversible Defect"}[x])
    return np.array([[i[f] for f in feature_names]], dtype=float)

user_input = get_user_input()



col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction")
    if st.button("Run Prediction"):
        x_scaled    = scaler.transform(user_input)
        prediction  = model.predict(x_scaled)[0]
        probability = model.predict_proba(x_scaled)[0][1]

        if prediction == 1:
            st.error(f"Heart Disease Risk Detected ({probability:.2%})")
        else:
            st.success(f"No Heart Disease Risk Detected ({probability:.2%})")
        st.progress(float(probability))

        st.divider()
        st.subheader("Feature Contributions")
        st.caption("contribution = scaled_value × learned_weight")

        contributions = x_scaled[0] * model.coef_[0]
        contrib_df = pd.DataFrame({
            'Feature': feature_names,
            'Contribution': contributions
        }).sort_values('Contribution', ascending=False)

        colors = ['#d62728' if v > 0 else '#2ca02c' for v in contrib_df['Contribution']]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(contrib_df['Feature'], contrib_df['Contribution'], color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title("Risk (+) vs Protective (−) Factors")
        ax.set_xlabel("Contribution to risk score")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Set patient values in the sidebar and click Run Prediction.")

with col2:
    st.subheader("Model Weights")
    st.caption("Positive = increases risk, Negative = decreases risk")
    weights = pd.Series(model.coef_[0], index=feature_names).sort_values()
    colors  = ['#d62728' if v > 0 else '#2ca02c' for v in weights]
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.barh(weights.index, weights.values, color=colors)
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.set_title("Learned Weights (global)")
    ax2.set_xlabel("Weight value")
    plt.tight_layout()
    st.pyplot(fig2)


st.divider()
st.subheader("Model Validation")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("CV Accuracy",  f"{cv_scores.mean():.2%}")
m2.metric("CV Std Dev",   f"{cv_scores.std():.4f}")
m3.metric("Precision",    f"{precision:.2%}")
m4.metric("Recall",       f"{recall:.2%}")
m5.metric("ROC-AUC",      f"{roc_auc:.3f}")

v1, v2 = st.columns(2)

with v1:
    st.write("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['Healthy','Sick'], yticklabels=['Healthy','Sick'])
    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)

with v2:
    st.write("ROC Curve")
    fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
    ax_roc.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'AUC = {roc_auc:.3f}')
    ax_roc.plot([0,1],[0,1], color='gray', lw=1, ls='--')
    ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
    ax_roc.legend(); plt.tight_layout()
    st.pyplot(fig_roc)