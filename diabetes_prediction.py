import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE   # install with: pip install imbalanced-learn

# -------------------------------
# Load Dataset Safely
# -------------------------------
dataset_path = "diabetes.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError("Dataset not found. Place 'diabetes.csv' in the same folder as this script.")

diabetes_dataset = pd.read_csv(dataset_path)

# -------------------------------
# Data Cleaning (replace zeros with median values)
# -------------------------------
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zero:
    diabetes_dataset[col] = diabetes_dataset[col].replace(0, np.nan)
    diabetes_dataset[col].fillna(diabetes_dataset[col].median(), inplace=True)

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

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
    "SVM (Linear Kernel)": SVC(kernel='linear'),
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
# Line Chart Comparison
# -------------------------------
plt.figure(figsize=(10,6))
plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='-', color='blue')
plt.title("Model Accuracy Comparison (Line Chart)")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.grid(True)
plt.show()

# -------------------------------
# Confusion Matrix Heatmaps
# -------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15,10))
axes = axes.flatten()

for idx, (name, cm) in enumerate(conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx])
    axes[idx].set_title(name)
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# -------------------------------
# Feature Importance (Random Forest Tuned)
# -------------------------------
importances = best_rf.feature_importances_
feature_names = diabetes_dataset.drop(columns='Outcome').columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=feature_names, palette="viridis")
plt.title("Feature Importance (Random Forest Tuned)")
plt.show()

# -------------------------------
# Best Model Selection
# -------------------------------
best_model_name = max(results, key=results.get)
print(f"Best Model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

best_model = models[best_model_name]

# -------------------------------
# Example Prediction
# -------------------------------
def build_input(preg, glucose, bp, skin, insulin, bmi, dpf, age):
    # Derived features
    hba1c = 7.0
    cholesterol = 190
    family_history = 1
    physical_activity = 3
    diet_quality = 6
    smoking_status = 0
    glucose_age_ratio = glucose / age
    
    # One-hot placeholders (BMI_Category, Age_Group)
    bmi_cat = [0,0,0]   # adjust based on BMI
    age_group = [0,0,0,0]  # adjust based on Age
    
    return np.array([preg, glucose, bp, skin, insulin, bmi, dpf, age,
                     hba1c, cholesterol, family_history, physical_activity,
                     diet_quality, smoking_status,
                     *bmi_cat, *age_group,
                     glucose_age_ratio]).reshape(1, -1)

# Example usage
input_data = build_input(5,166,72,19,175,25.8,0.587,51)
std_data = scaler.transform(input_data)
prediction = best_model.predict(std_data)
