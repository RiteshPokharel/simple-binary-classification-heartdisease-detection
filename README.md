# Cardiovascular Diagnostic Analysis & Explainable AI
> **A transparent approach to medical machine learning.**

This project is a diagnostic tool built to help users understand the specific clinical factors that drive heart disease risk. By using **Logistic Regression** and **Explainable AI (XAI)** principles, the application breaks down complex calculations into clear, human-centered terms.

---

## **1. Quick Start & Installation**

To get the environment ready and the app running, follow these steps:

**1.1. Clone the repository**
`git clone https://github.com/your-username/heart-disease-dashboard.git`
`cd heart-disease-dashboard`

**1.2. Install all required libraries**
`pip install streamlit pandas scikit-learn matplotlib seaborn`

**1.3. Ensure heart.csv is in the root folder**

**1.4. Launch the application**
`streamlit run app.py`

---

## **2. Key Features**

**2.1. Interactive Patient Profiling:** A dedicated sidebar to input 13 clinical markers including age, resting blood pressure, cholesterol, and ECG results.

**2.2. Explainable AI (XAI) Breakdown:** A real-time table showing exactly how each feature (Value × Weight) contributes to the final risk score.

**2.3. Clinical Health Hints:** Integrated descriptions that provide medical context, such as healthy ranges for blood pressure and fasting blood sugar.

**2.4. Global Model Insights:** A horizontal bar chart illustrating which features the model generally considers most important across the entire population.

**2.5. Technical Validation Panel:** Real-time performance metrics featuring Accuracy, Precision, Recall, and a visual Confusion Matrix.

---

## **3. Technical Implementation & Logic**

### **3.1. Data Source**
The model is trained on the **UCI Heart Disease dataset**, which contains patient records with 13 features and a binary target representing the presence or absence of heart disease.

### **3.2. The Model: Logistic Regression**
We chose **Logistic Regression** for this task because of its high interpretability. In a medical context, being able to audit the "why" behind a prediction is as important as the prediction itself. Every decision in this model can be traced back to a specific coefficient.

### **3.3. Feature Engineering & Scaling**
* **Standard Scaling:** We applied a `StandardScaler` to normalize features. This prevents features with larger numerical scales (like Cholesterol) from dominating the model's coefficients.
* **Label Formatting:** Categorical variables like Chest Pain Type and Thalassemia are mapped to their clinical descriptions for better user understanding.

---

## **4. Understanding the Validation**

To ensure the model is robust and not "overfitting" to specific data, we implemented:

**4.1. 5-Fold Cross-Validation:** This tests the model on five different subsets of the data to ensure the accuracy (typically 82% to 85%) is consistent.

**4.2. Confusion Matrix:** This helps identify the specific types of errors. We focus on **Recall (Sensitivity)** because, in heart disease, a False Negative (missing a sick patient) is far more dangerous than a False Positive (a false alarm).

**4.3. Precision:** Measures how often the model is correct when it predicts heart disease.

---

## **5. Interpretation Guide**

When using the dashboard, the results are broken down into:

**5.1. The Prediction:** A clear success or error message indicating the risk probability percentage.

**5.2. Individual Contribution:** A breakdown showing how each vital sign moved the calculation toward or away from a diagnosis.

**5.3. Primary Driver:** An automated insight identifying the most significant factor for that specific patient.

---

## **6. Disclaimer**

This application is for educational and demonstrative purposes only. It is not intended for use in clinical diagnosis or as a substitute for professional medical advice. All data and predictions should be reviewed by a qualified healthcare professional.