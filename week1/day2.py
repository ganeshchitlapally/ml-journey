import numpy as np
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("=== FIRST 5 ROWS")
print(df.head())

print("\n=== SHAPE OF DATA ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

#survival rate
print("\n=== SURVIVAL RATE ===")
survial_rate = df['Survived'].mean()*100
print(f"Overall survival rate: {survial_rate:.1f}%")

# Survival rate by Gender
print("\n=== SURVIVAL RATE BY GENDER ===")
print(df.groupby('Sex')['Survived'].mean()*100)

# Survival rate by Passenger Class
print("\n=== SURVIVAL RATE BY PASSENGER CLASS ===")
print(df.groupby('Pclass')['Survived'].mean()*100)

# Average age of Survuived vs Not Survived
print("\n=== AVERAGE AGE OF SURVIVED VS NOT SURVIVED ===")
print(df.groupby('Survived')['Age'].mean())