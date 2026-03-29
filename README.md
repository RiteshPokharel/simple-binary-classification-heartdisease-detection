Heart Disease Diagnostic Dashboard
Bridging the gap between Machine Learning and Medical Logic
Most AI models act as black boxes. They provide an answer but rarely explain the reasoning behind it. This project is a diagnostic tool built to help users understand the specific clinical factors that drive heart disease risk. By using Logistic Regression and Explainable AI principles, the application breaks down complex calculations into clear terms.

Installation and Setup
To run this project locally, follow these steps:

Clone the repository

Bash
git clone https://github.com/your-username/heart-disease-dashboard.git
cd heart-disease-dashboard
Install the required libraries

Bash
pip install streamlit pandas scikit-learn matplotlib seaborn
Run the application

Bash
streamlit run app.py
Technical Implementation
Data: The model is based on the UCI Heart Disease dataset consisting of 303 patient records.

Model: We utilized Logistic Regression to ensure mathematical transparency and interpretability.

Preprocessing: Standard Scaling was applied to normalize clinical features so that different units of measurement do not bias the results.

Interface: Developed using Streamlit for a clean, accessible, and interactive web experience.

Validation Metrics
The model is evaluated using two primary methods to ensure reliability:

5-Fold Cross-Validation: This confirms that the model performance is stable across different subsets of the data.

Confusion Matrix: This provides a visual breakdown of the specific types of errors: such as False Positives or False Negatives: the model makes.

Recall: This metric is prioritized to minimize the risk of missing patients who are truly sick.

Disclaimer
This application is for educational and demonstrative purposes only. It is not intended for use in clinical diagnosis or as a substitute for professional medical advice.