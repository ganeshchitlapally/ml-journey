import numpy as np
import pandas as pd

# create a dictionary with employee data - age, salary, and years of experience

data = { 
    'age': [22, 25, 47, 35, 46, 56, 23, 34, 52,28],
    'salary': [25000, 30000, 80000, 52000, 75000, 95000, 27000, 48000, 88000, 35000],
    'experience_years': [1, 2, 10, 6, 9, 15, 1, 5, 12, 3]

}
# Converting the data from a dictionary to a data frame using pd.DataFrame() function
df = pd.DataFrame(data)

print("=== DATASET ===")
print(df)
# Checking the basic Statistics
print("\n=== BASIC STATISTICS ===")
print(df.describe())
# Observations and insights
print("\n=== MY OBSERVATIONS ===")
print(f"Average Salary: {df['salary'].mean():,.0f}")
print(f"Most Experienced Person: {df['experience_years'].max()} years")
print(f"Youngest Person: {df['age'].min()} years old")
# Calculate the correlation between experience and salary
correlation = df['experience_years'].corr(df['salary'])
print(f"\n Correlation between Experience and Salary: {correlation:.2f}")
print("This means: more experience = higher salary," if correlation > 0.7 else "Weak correlation")
# Additional insights
print(f"The person with the highest salary is {df['salary'].max()}")
print(f"The least Experinenced person is {df['experience_years'].min()} years old")
print(f" the average age of people earning  more than 50000 is {df[df['salary'] > 50000]['age'].mean()}")


# Shows the entire row of the highest earner
print(df[df['salary'] == df['salary'].max()])

# Shows the entire row of least experienced
print(df[df['experience_years'] == df['experience_years'].min()])