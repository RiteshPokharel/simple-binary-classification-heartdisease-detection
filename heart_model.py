import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Load the data
df = pd.read_csv('heart.csv')

#Define Features (X) and Target (y)
X = df.drop('target', axis=1)
y = df['target']

#Split the data (Changed random_state to 0 to get a more representative split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Initialize and Train the Model
model = LogisticRegression(solver='liblinear', C=0.1)
model.fit(X_train, y_train)

#Perform 5-Fold Cross-Validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

#Make Predictions and Evaluate
y_pred = model.predict(X_test)

print("--- Model Performance ---")
print(f"Test Accuracy Score: {accuracy_score(y_test, y_pred):.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

#Show Cross-Validation Results
print("\n--- 5-Fold Cross-Validation ---")
print(f"Average Accuracy: {cv_scores.mean():.2%}")
print(f"Stability (Std Dev): {cv_scores.std():.4f}")

#Show Bias and Feature Importance
print("\n--- Why the model made these choices ---")
print(f"Model Bias (Intercept): {model.intercept_[0]:.4f}")
importance = model.coef_[0]
for name, val in zip(X.columns, importance):
    print(f"{name}: {val:.4f}")