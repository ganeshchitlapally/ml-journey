"""
Titanic Survival Prediction Analysis

This script performs data analysis and machine learning on the Titanic dataset.
It includes data cleaning, feature engineering, and model training with Logistic Regression
and Decision Tree classifiers to predict passenger survival.

Steps:
1. Load and explore the data
2. Clean missing values
3. Engineer new features
4. Prepare data for modeling
5. Train and compare models
6. Tune the Decision Tree model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==============================================================================================================
# STEP 1 - LOAD THE DATA
# ==============================================================================================================
# Load the Titanic dataset from a public GitHub repository
# This dataset contains information about passengers on the Titanic, including survival status
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(" === RAW DATA SHAPE ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n=== MISSING VALUES BEFORE CLEANING ===")
print(df.isnull().sum())

# ==============================================================================================================
# Step 2 - PROPER DATA CLEANING
# ==============================================================================================================
# Handle missing values in the dataset
# Cabin has too many missing values (77%), so we drop it entirely
df = df.drop(columns=["Cabin"])

# Fill missing Age values with median age grouped by passenger class
# This is smarter than using global median because age correlates with class
df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

# Fill missing Embarked values with the most common embarkation point
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print("\n=== MISSING VALUES AFTER CLEANING ===")
print(df.isnull().sum())

# ==============================================================================================================
# Step 3 - FEATURE ENGINEERING
# ==============================================================================================================
# Convert categorical variables to numeric for machine learning algorithms

# Convert Sex to binary: male=0, female=1
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

# Convert Embarked ports to numeric: Southampton=0, Cherbourg=1, Queenstown=2
df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})

# Create FamilySize feature: total family members on board (including passenger)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Create IsAlone feature: 1 if passenger is alone, 0 if with family
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

print("=== NEW FEATURES ADDED ===")
print(df[['SibSp', 'Parch', 'FamilySize', 'IsAlone']].head(10))

# ==============================================================================================================
# Step 4 - PREPARE FEATURES AND SPLIT DATA
# ==============================================================================================================
# Select the features to use for prediction
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']

# Separate features (X) and target variable (y)
X = df[features]
y = df['Survived']

# Split data into training (80%) and testing (20%) sets
# random_state ensures reproducible results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=== TRAINING SET SIE ===")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# ============================================================
# STEP 5 - TRAIN TWO MODELS AND COMPARE
# ============================================================
# Train and evaluate two different machine learning models

# Model 1 - Logistic Regression
# A linear model good for binary classification
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Model 2 - Decision Tree
# A non-linear model that can capture complex patterns
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Compare model performance
print("=== LOGISTIC REGRESSION ===")
print(f"Accuracy: {accuracy_score(y_test, lr_predictions) * 100:.1f}%")
print(classification_report(y_test, lr_predictions))

print("=== DECISION TREE ===")
print(f"Accuracy: {accuracy_score(y_test, dt_predictions) * 100:.1f}%")
print(classification_report(y_test, dt_predictions))

# ============================================================
# STEP 6 - FIX THE DECISION TREE
# ============================================================
# Tune the Decision Tree to prevent overfitting
# Limit depth and require minimum samples for splits
dt_tuned = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)
dt_tuned.fit(X_train, y_train)
dt_tuned_predictions = dt_tuned.predict(X_test)

print("=== TUNED DECISION TREE ===")
print(f"Accuracy: {accuracy_score(y_test, dt_tuned_predictions) * 100:.1f}%")
print(classification_report(y_test, dt_tuned_predictions))

