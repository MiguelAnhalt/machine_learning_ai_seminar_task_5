import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from MLPClassifier import MLPClassifier

# Load and prepare the Iris dataset
filename = "Iris.csv"
df = pd.read_csv(filename)
df = df.drop("Id", axis=1)

# Extract features and labels

# Normalize the features
x = df.iloc[:, 0:4].values  # Using all 4 features
for i in range(x.shape[1]):
   print(i)
   x[:, i] = (x[:, i] - x[:, i].mean()) / x[:, i].std()

X = x
y = df.iloc[:, 4].values

# Convert string labels to numeric (1, 2, 3)
label_map = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
y = np.array([label_map[label] for label in y])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train the classifier with multiple hidden layers
mlp = MLPClassifier(
    hidden_layer_sizes=[4, 3],  # Two hidden layers
    learning_rate=0.1,         # Higher learning rate
    
    epochs=16000            # More iterations
)


# Train the model
mlp.fit(X_train, y_train)

# Calculate accuracy
train_accuracy = mlp.score(X_train, y_train)
test_accuracy = mlp.score(X_test, y_test)

# Print performance metrics
print("\nMLP Neural Network Performance:")
print(f"Number of features used: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Hidden layer architecture: {mlp.hidden_layer_sizes}")
print(f"Number of iterations: {mlp.epochs}")
print(f"Training Accuracy: {train_accuracy:.2f}%")
print(f"Testing Accuracy: {test_accuracy:.2f}%")
