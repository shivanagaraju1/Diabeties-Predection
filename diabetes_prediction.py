# Diabetes Prediction using SVM
# Author: Anchuri
# Description: Machine learning project to predict diabetes using the PIMA Diabetes Dataset.

# -------------------------------
# Importing Dependencies
# -------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# -------------------------------
# Data Collection and Analysis
# -------------------------------
# Load the dataset
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

# Display first 5 rows
print(diabetes_dataset.head())

# Dataset shape
print("Dataset shape:", diabetes_dataset.shape)

# Statistical summary
print(diabetes_dataset.describe())

# Outcome distribution
print(diabetes_dataset['Outcome'].value_counts())

# Grouped mean values by outcome
print(diabetes_dataset.groupby('Outcome').mean())

# Separate features and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print("Features:\n", X)
print("Labels:\n", Y)

# -------------------------------
# Data Standardization
# -------------------------------
scaler = StandardScaler()
scaler.fit(X)

standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['Outcome']

print("Standardized Features:\n", X)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print("Full dataset shape:", X.shape)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# -------------------------------
# Training the Model
# -------------------------------
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# -------------------------------
# Model Evaluation
# -------------------------------
# Accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy score of the training data:", training_data_accuracy)

# Accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy score of the test data:", test_data_accuracy)

# -------------------------------
# Making a Predictive System
# -------------------------------
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Convert to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape for single instance prediction
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize input
std_data = scaler.transform(input_data_reshaped)
print("Standardized input:", std_data)

# Prediction
prediction = classifier.predict(std_data)
print("Prediction:", prediction)

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")