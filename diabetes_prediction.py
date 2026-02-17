import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

# -------------------------------
# Load Dataset
# -------------------------------
dataset_path = "diabetes.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError("Dataset not found. Place 'diabetes.csv' in the same folder as this script.")

df = pd.read_csv(dataset_path)

# -------------------------------
# Data Cleaning
# -------------------------------
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zero:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# -------------------------------
# Add Synthetic Features
# -------------------------------
np.random.seed(42)
df["HbA1c"] = np.random.uniform(4.5, 12.0, size=len(df))
df["Cholesterol"] = np.random.uniform(150, 280, size=len(df))
df["FamilyHistory"] = np.random.choice([0,1], size=len(df), p=[0.6,0.4])
df["PhysicalActivity"] = np.random.randint(0, 10, size=len(df))
df["DietQuality"] = np.random.randint(1, 11, size=len(df))
df["SmokingStatus"] = np.random.choice([0,1], size=len(df), p=[0.7,0.3])
df["Glucose_Age_Ratio"] = df["Glucose"] / df["Age"]

# One-hot encodings
df["BMI_Category"] = pd.cut(df["BMI"], bins=[0,18.5,25,30,100],
                            labels=["Underweight","Normal","Overweight","Obese"])
df["Age_Group"] = pd.cut(df["Age"], bins=[20,30,40,50,60,100],
                         labels=["20s","30s","40s","50s","60+"])
df = pd.get_dummies(df, columns=["BMI_Category","Age_Group"])  # keep all dummies

X = df.drop(columns='Outcome')
Y = df['Outcome']

# -------------------------------
# Standardization
# -------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

# -------------------------------
# Balance Dataset with SMOTE
# -------------------------------
smote = SMOTE(random_state=42)
X_train, Y_train = smote.fit_resample(X_train, Y_train)

# -------------------------------
# Define Models
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (Linear Kernel)": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7)
}

results = {}
conf_matrices = {}

# -------------------------------
# Hyperparameter Tuning for Random Forest
# -------------------------------
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, Y_train)
best_rf = grid_search.best_estimator_
models["Random Forest (Tuned)"] = best_rf

# -------------------------------
# Train & Evaluate Models
# -------------------------------
for name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, y_pred)
    results[name] = acc
    conf_matrices[name] = confusion_matrix(Y_test, y_pred)

    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(Y_test, y_pred))
    print("-"*50)

# -------------------------------
# Show Plots All At Once
# -------------------------------
plt.figure(figsize=(8,6))
plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='-', color='blue')
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.grid(True)

for name, cm in conf_matrices.items():
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.figure(figsize=(8,6))
importances = best_rf.feature_importances_
feature_names = df.drop(columns='Outcome').columns
sns.barplot(x=importances, y=feature_names, palette="viridis")
plt.title("Feature Importance (Random Forest Tuned)")

plt.figure(figsize=(8,6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(Y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.figure(figsize=(8,6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
        precision, recall, _ = precision_recall_curve(Y_test, y_prob)
        plt.plot(recall, precision, label=name)
plt.title("Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

train_sizes, train_scores, test_scores = learning_curve(best_rf, X, Y, cv=5, n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_mean, 'o-', label="Training Score")
plt.plot(train_sizes, test_mean, 'o-', label="Validation Score")
plt.title("Learning Curve (Best Model)")
plt.xlabel("Training Examples")
plt.ylabel("Score")
plt.legend()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")

plt.show(block=False)

# -------------------------------
# Best Model Selection
# -------------------------------
best_model_name = max(results, key=results.get)
print(f"Best Model: {best_model_name} with accuracy {results[best_model_name]:.4f}")
best_model = models[best_model_name]

# -------------------------------
# Dynamic Helper for Prediction
# -------------------------------
def build_input(preg, glucose, bp, skin, insulin, bmi, dpf, age):
    # Start with a dict of values for the original + synthetic features
    input_dict = {
        "Pregnancies": preg,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
        "HbA1c": 7.0,
        "Cholesterol": 190,
        "FamilyHistory": 1,
        "PhysicalActivity": 3,
        "DietQuality": 6,
        "SmokingStatus": 0,
        "Glucose_Age_Ratio": glucose/age
    }

    # Add dummy columns with 0 for any one-hot features
    for col in df.drop(columns="Outcome").columns:
        if col not in input_dict:
            input_dict[col] = 0

    # Ensure correct order
    ordered_values = [input_dict[col] for col in df.drop(columns="Outcome").columns]
    return np.array(ordered_values).reshape(1, -1)

# -------------------------------
# Interactive Prediction
# -------------------------------
print("\nEnter patient details:")
preg = int(input("Pregnancies: "))
glucose = float(input("Glucose: "))
bp = float(input("Blood Pressure: "))
skin = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = int(input("Age: "))

# Build full input vector
input_data = build_input(preg, glucose, bp, skin, insulin, bmi, dpf, age)

# Scale and predict
std_data = scaler.transform(input_data)
prediction = best_model.predict(std_data)

print("\nPrediction:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
