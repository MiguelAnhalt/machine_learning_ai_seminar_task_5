import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from MLPClassifier import MLPClassifier

# Load and prepare the Dry_Bean_Dataset
filename = "Dry_Bean_Dataset.xlsx"
df = pd.read_excel(filename)
#print(df.head())
# print(df.info())

# check if there are any missing values
# print(df.isnull().sum())

y = df.iloc[:, -1].values
# print("Unique classes:", np.unique(y))

df_features = df.drop("Class", axis=1)

# region Get correlation of the features
# Get correlation of the features
matriz_correlacion = df_features.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
# plt.show()

# region Result of the correlation matrix
        # correlation_1 :Area - Perimeter
        # correlation_2 :Area - MajorAxisLength
        # correlation_3 :Area - MinorAxisLength
        # correlation_4 :Area - ConvexArea
        # correlation_5 :Area - EquivDiameter
        # 
        # correlation_6 :Perimeter - Area
        # correlation_7 :Perimeter - MajorAxisLength
        # correlation_8 :Perimeter - MinorAxisLength
        # correlation_9 :Perimeter - ConvexArea
        # correlation_10 :Perimeter - EquivDiameter
        # 
        # correlation_11 : MajorAxisLength - Area
        # correlation_12 : MajorAxisLength - Perimeter
        # correlation_13 : MajorAxisLength - ConvexArea
        # correlation_14 : MajorAxisLength - EquivDiameter
        # 
        # correlation_15 : MinorAxisLength - Area
        # correlation_16 : MinorAxisLength - Perimeter
        # correlation_17 : MinorAxisLength - ConvexArea
        # correlation_18 : MinorAxisLength - EquivDiameter
        # 
        # correlation_19 : ConvexArea - Area
        # correlation_20 : ConvexArea - Perimeter
        # correlation_21 : ConvexArea - MajorAxisLength
        # correlation_22 : ConvexArea - MinorAxisLength
        # correlation_23 : ConvexArea - EquivDiameter
        # 
        # correlation_24 : EquivDiameter - Area
        # correlation_25 : EquivDiameter - Perimeter
        # correlation_26 : EquivDiameter - MajorAxisLength
        # correlation_27 : EquivDiameter - MinorAxisLength
        # correlation_28 : EquivDiameter - ConvexArea

# endregion

# Extract only the features with high correlation
features_to_use = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "ConvexArea", "EquivDiameter"]
# x = df_features[features_to_use].values # Using only the features with high correlation
x = df_features.values # Using all features


# Normalize the features
for i in range(x.shape[1]):
   x[:, i] = (x[:, i] - x[:, i].mean()) / x[:, i].std()

X = x
# endregion


# Convert string labels to numeric (1, 2, 3)
label_map = {'BARBUNYA': 1, 'BOMBAY': 2, 'CALI': 3, 'DERMASON': 4, 'HOROZ': 5, 'SEKER': 6, 'SIRA': 7}
y = np.array([label_map[label] for label in y])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train the classifier with multiple hidden layers
mlp = MLPClassifier(
    hidden_layer_sizes=[36,18, 7],
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
