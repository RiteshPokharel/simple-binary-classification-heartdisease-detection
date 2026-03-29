Heart Disease Diagnostic Dashboard
Most AI models act as black boxes. They provide an answer but rarely explain the reasoning behind it. This project is a diagnostic tool built to help users understand the specific clinical factors that drive heart disease risk. By using Logistic Regression and Explainable AI principles, the application breaks down complex calculations into clear terms.

Installation and Setup
Clone the repository:

Bash
git clone https://github.com/your-username/heart-disease-dashboard.git
cd heart-disease-dashboard
Install the required libraries:

Bash
pip install streamlit pandas scikit-learn matplotlib seaborn
Run the application:

Bash
streamlit run app.py
Technical Implementation
Data: Based on the UCI Heart Disease dataset (303 records).

Model: Logistic Regression for mathematical transparency.

Preprocessing: Standard Scaling to normalize clinical features.

Interface: Streamlit for a clean and accessible web experience.

Validation Metrics
5-Fold Cross-Validation: Ensures the model is stable across different data subsets.

Confusion Matrix: Visualizes the specific types of errors the model makes.

Recall: Prioritized to minimize the risk of missing patients who are truly sick.

Disclaimer
This application is for educational and demonstrative purposes only. It is not intended for use in clinical diagnosis or as a substitute for professional medical advice.