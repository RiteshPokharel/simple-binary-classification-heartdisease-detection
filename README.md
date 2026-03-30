# Heart Disease Risk Predictor

A binary classification web app built with Streamlit that predicts heart disease risk from clinical markers. Logistic Regression, Standard Scaling, and all evaluation metrics are implemented **from scratch using NumPy only** — no scikit-learn.

---

## Demo

![App Screenshot](screenshot.png)

---

## How It Works

The model is trained on the [UCI Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) using:

- **Logistic Regression** via batch gradient descent with L2 regularization
- **Standard Scaling** (zero mean, unit variance)
- **Stratified train/test split** (80/20)
- **5-fold stratified cross-validation**

All of the following are written from scratch:
- Sigmoid function & gradient descent loop
- StandardScaler
- Confusion matrix, Precision, Recall, Accuracy
- ROC curve & AUC (trapezoidal rule)
- Cross-validation

---

## Project Structure

```
├── app.py          # Main Streamlit app
├── heart.csv       # Dataset (UCI Heart Disease)
└── README.md
```

---

## Setup & Run

### 1. Clone the repo

```bash
git clone https://github.com/RiteshPokharel/simple-binary-classification-heartdisease-detection.git
cd simple-binary-classification-heartdisease-detection
```

### 2. Install dependencies

```bash
pip install streamlit pandas numpy matplotlib seaborn
```

> Requires Python 3.8+

### 3. Run the app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Dataset

The dataset (`heart.csv`) must be in the same folder as `app.py`. It uses the UCI Heart Disease dataset available on Kaggle:

[https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

**Features used (13 original, no engineering):**

| Feature | Description |
|---|---|
| age | Age in years |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0–3) |
| thal | Thalassemia (1 = fixed, 2 = normal, 3 = reversible) |

**Target:** `1` = heart disease present, `0` = absent

---

## Dependencies

| Package | Version |
|---|---|
| streamlit | ≥ 1.30 |
| pandas | ≥ 1.5 |
| numpy | ≥ 1.24 |
| matplotlib | ≥ 3.6 |
| seaborn | ≥ 0.12 |