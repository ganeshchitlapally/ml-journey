import numpy as np
import pandas as pd

# create a simple dataset from scratch 

data = { 
    'age': [22, 25, 47, 35, 46, 56, 23, 34, 52,28],
    'salary': [25000, 30000, 80000, 52000, 75000, 95000, 27000, 48000, 88000, 35000],
    'experience_years': [1, 2, 10, 6, 9, 15, 1, 5, 12, 3]

}

df = pd.DataFrame(data)

print("=== DATASET ===")
print(df)

print("\n=== BASIC STATISTICS ===")
print(df.describe())

print("\n=== MY OBSERVATIONS ===")
print(f"Average Salary: {df['salary'].mean():,.0f}")
print(f"Most Experienced Person: {df['experience_years'].max()} years")
print(f"Youngest Person: {df['age'].min()} years old")

correlation = df['experience_years'].corr(df['salary'])
print(f"\n Correlation between Experience and Salary: {correlation:.2f}")
print("This means: more experience = higher salary," if correlation > 0.7 else "Weak correlation")

print(f"The person with the highest salary is {df['salary'].max()}")
print(f"The least Experinenced person is {df['experience_years'].min()} years old")
print(f" the average age of people earning  more than 50000 is {df[df['salary'] > 50000]['age'].mean()}")