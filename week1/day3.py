import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==============================================================================================================
# STEP 1 - LOAD THE DATA
# ==============================================================================================================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(" === RAW DATA SHAPE ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n=== MISSING VALUES BEFORE CLEANING ===")
print(df.isnull().sum())

# ==============================================================================================================
# Step 2 - PROPER DATA CLEANING
# ==============================================================================================================

# Drop Cabin - 77% missing vales, soi its better to drop it

df = df.drop(columns=["Cabin"])

# fill age with median grouped by Pclass
# richer passengers tend to be older - smarter than global medan
df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

# Fill Embarked with most comman value
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print("\n=== MISSING VALUES AFTER CLEANING ===")
print(df.isnull().sum())

# ==============================================================================================================
# Step 3 - FEATURE ENGINEERING
# ==============================================================================================================

# Convert Sex to numeric
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

# Convert Embarked to numbers 
df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})

# Create a new feature - FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Create a new feature - Is Alone
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

print("=== NEW FEATURES ADDED ===")
print(df[['SibSp', 'Parch', 'FamilySize', 'IsAlone']].head(10))

# ==============================================================================================================
# Step 4 - PREPARE FEATURES AND SPLIT DATA
# ==============================================================================================================

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']

X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=== TRAINING SET SIE ===")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# ============================================================
# STEP 5 - TRAIN TWO MODELS AND COMPARE
# ============================================================

# Model 1 - Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Model 2 - Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Compare
print("=== LOGISTIC REGRESSION ===")
print(f"Accuracy: {accuracy_score(y_test, lr_predictions) * 100:.1f}%")
print(classification_report(y_test, lr_predictions))

print("=== DECISION TREE ===")
print(f"Accuracy: {accuracy_score(y_test, dt_predictions) * 100:.1f}%")
print(classification_report(y_test, dt_predictions))

# ============================================================
# STEP 6 - FIX THE DECISION TREE
# ============================================================
dt_tuned = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)
dt_tuned.fit(X_train, y_train)
dt_tuned_predictions = dt_tuned.predict(X_test)

print("=== TUNED DECISION TREE ===")
print(f"Accuracy: {accuracy_score(y_test, dt_tuned_predictions) * 100:.1f}%")
print(classification_report(y_test, dt_tuned_predictions))