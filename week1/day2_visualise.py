import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load and prepare data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df[['Pclass', 'Sex', 'Age', 'Fare']].values
y = df['Survived'].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias column
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize weights and bias to zero
weights = np.zeros(X.shape[1])

# Track history
weight_history = []
bias_history = []
loss_history = []

# Training loop - gradient descent
learning_rate = 0.1
epochs = 200

for epoch in range(epochs):
    # Forward pass
    score = X @ weights
    predictions = sigmoid(score)

    # Calculate loss
    loss = -np.mean(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))

    # Calculate gradients
    error = predictions - y
    gradient = X.T @ error / len(y)

    # Update weights
    weights -= learning_rate * gradient

    # Record history
    bias_history.append(weights[0])
    weight_history.append(weights[1:].copy())
    loss_history.append(loss)

# Plot everything
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Plot 1 - Loss over time
axes[0].plot(loss_history, color='red')
axes[0].set_title('Loss Over Time (Should Go Down)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True)

# Plot 2 - Bias over time
axes[1].plot(bias_history, color='blue')
axes[1].set_title('Bias Value Over Time')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Bias')
axes[1].grid(True)

# Plot 3 - Weights over time
weight_history = np.array(weight_history)
labels = ['Pclass', 'Sex', 'Age', 'Fare']
for i, label in enumerate(labels):
    axes[2].plot(weight_history[:, i], label=label)
axes[2].set_title('Weight Values Over Time')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Weight')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('weight_visualization.png')
plt.show()

print("=== FINAL WEIGHTS ===")
print(f"Bias: {bias_history[-1]:.4f}")
for i, label in enumerate(labels):
    print(f"{label}: {weight_history[-1][i]:.4f}")

print("\n=== INITIAL WEIGHTS ===")
print("Bias: 0.0000")
for label in labels:
    print(f"{label}: 0.0000")