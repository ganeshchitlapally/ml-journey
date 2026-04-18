import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load the Dataset

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# spt 1 - Keep only useful features
df = df[['Survived','Pclass','Sex','Age','Fare']]

# Step 2 - Handle missing values of Age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Step 3 - Convert Sex from text to numbers( male = 0, female = 1)
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

# Step 4 - Define features and target
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# Step 5 - Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6 - Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7 - Make predictions
predictions = model.predict(X_test)

# Step 8 - Evaluate
print("=== MODEL RESULTS ===")
print(f"Accuracy: {accuracy_score(y_test, predictions)*100:.1f}%")

print("\n=== CLASSIFICATON REPORT ===")
print(classification_report(y_test, predictions))

# Step 9 - Feature Importance
print("\n=== FEATURE IMPORTANCE ===")
feature_names = ['Pclass','Sex','Age','Fare']
coefficients = model.coef_[0]

for feature,coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")

print(model.intercept_)