import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from MLPClassifier import MLPClassifier

# Load and prepare the Dry_Bean_Dataset
filename = "heart-disease.csv"
df = pd.read_csv(filename)
# print(df[df["target"] == 0])
# print(df.info())

# check if there are any missing values
# print(df.isnull().sum())
y = df.iloc[:, -1].values
# print("Unique classes:", np.unique(y))

df_features = df.drop("target", axis=1)

# region Get correlation of the features
# Get correlation of the features
matriz_correlacion = df_features.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
# plt.show()

x = df_features.values # Using all features


# Normalize the features
for i in range(x.shape[1]):
   x[:, i] = (x[:, i] - x[:, i].mean()) / x[:, i].std()

X = x

# endregion

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Create and train the classifier with multiple hidden layers
mlp = MLPClassifier(
    hidden_layer_sizes=[26,2],
    learning_rate=0.1,         # Higher learning rate
    epochs=50000            # More iterations
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
