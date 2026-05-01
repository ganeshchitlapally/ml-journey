import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load and clean data — same as Day 3
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df = df.drop(columns=['Cabin'])
df['Age'] = df.groupby('Pclass')['Age'].transform(
    lambda x: x.fillna(x.median())
)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=== DATA READY ===")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ==============================================================================================================
# STEP 4 - CROSS VALIDATION
# ==============================================================================================================

models = {
    'Logsitic Regression' : LogisticRegression(max_iter=1000),
    'Decision Tree' : DecisionTreeClassifier(max_depth=4, random_state=42),
    'Random Forest' : RandomForestClassifier(n_estimators = 100, random_state=42)
}

print("\n=== CROSS VALIDATION SCORES (5 Fold) ===")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name}")
    print(f"Scores per fold: {scores.round(3)}")
    print(f" Mean Accuracy: {scores.mean():.3f}")
    print(f" Std Dev: {scores.std():.3f}\n")

rf = RandomForestClassifier(n_estimators = 100, random_state=42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

print("=== FEATURE IMPORTANCE ===")
importances = rf.feature_importances_
for feature, importance in zip(features, importances):
    bar = "█" * int(importance * 50)
    print(f"{feature:12} {importance:.4f}  {bar}")