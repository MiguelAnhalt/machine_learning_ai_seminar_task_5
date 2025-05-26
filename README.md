# Task 5 MLP

This project implements a Multi-Layer Perceptron (MLP) neural network from scratch to predict heart disease using the Cleveland Heart Disease dataset.

## Project Structure
```
SLP_Example/
└── MLP_Model/
    ├── MLPClassifier.py    # Custom MLP implementation
    └── MLP_Heart.py        # Heart disease prediction script
```

## Datasets

### 1. Cleveland Heart Disease Dataset
The primary dataset used for heart disease prediction, containing 303 samples with 13 features and a binary target variable.

**Source**: UCI Machine Learning Repository - Heart Disease Dataset
**Link**: https://archive.ics.uci.edu/ml/datasets/heart+disease

**Features**:
- age: Age in years
- sex: Gender (1 = male, 0 = female)
- cp: Chest pain type (0-3)
- trestbps: Resting blood pressure
- chol: Serum cholesterol
- fbs: Fasting blood sugar > 120 mg/dl
- restecg: Resting electrocardiographic results
- thalach: Maximum heart rate achieved
- exang: Exercise induced angina
- oldpeak: ST depression induced by exercise
- slope: Slope of the peak exercise ST segment
- ca: Number of major vessels colored by fluoroscopy
- thal: Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)

**Target Variable**:
- target: Presence of heart disease (1 = disease, 0 = no disease)

**Dataset Characteristics**:
- Number of instances: 303
- Number of attributes: 13
- Missing values: None
- Class distribution: Binary (165 no disease, 138 disease)

### 2. Dry Bean Dataset (Reference Implementation)
Used for initial MLP implementation testing and validation.

**Source**: UCI Machine Learning Repository - Dry Bean Dataset
**Link**: https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset

**Features**:
- Area: Area of the bean
- Perimeter: Perimeter of the bean
- MajorAxisLength: Length of the major axis
- MinorAxisLength: Length of the minor axis
- ConvexArea: Area of the convex hull
- EquivDiameter: Equivalent diameter
- Eccentricity: Eccentricity of the bean
- Extent: Extent of the bean
- Solidity: Solidity of the bean
- Roundness: Roundness of the bean
- Compactness: Compactness of the bean
- ShapeFactor1: First shape factor
- ShapeFactor2: Second shape factor
- ShapeFactor3: Third shape factor
- ShapeFactor4: Fourth shape factor

**Target Variable**:
- Class: Type of dry bean (7 classes: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA)

**Dataset Characteristics**:
- Number of instances: 13,611
- Number of attributes: 16
- Missing values: None
- Class distribution: Multi-class (7 different types of beans)

### Dataset Preprocessing
Both datasets underwent the following preprocessing steps:
1. Feature normalization using standardization
2. Train-test split (75% training, 25% testing)
3. Stratified sampling to maintain class distribution
4. One-hot encoding for multi-class classification (Dry Bean dataset)

## Implementation Details

### MLPClassifier
The custom MLP implementation includes:
- Sigmoid activation function
- Backpropagation algorithm
- Multiple hidden layers support
- Early stopping based on accuracy
- One-hot encoding for multi-class classification

### Model Architecture
- Input layer: 13 features
- Hidden layers: [26, 2] neurons
- Output layer: 1 neuron (binary classification)
- Learning rate: 0.1
- Maximum epochs: 50000

## Usage

1. Ensure you have the required dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

2. Place the heart-disease.csv file in the project directory

3. Run the prediction script:
```bash
python SLP_Example/MLP_Model/MLP_Heart.py
```

## Performance
The model's performance is evaluated using:
- Training accuracy
- Testing accuracy
- Learning curve visualization
- Feature correlation analysis

## References
1. Kumar, A., Kumar, P., & Kumar, Y. (2023). Prediction of cardiovascular disease using MLP. AIP Conference Proceedings, 2495(1), 020059. https://doi.org/10.1063/5.0132281

2. Al Bataineh, A., & Manacek, S. (2022). MLP-PSO Hybrid Algorithm for Heart Disease Prediction. Journal of Personalized Medicine, 12(8), 1208. https://doi.org/10.3390/jpm12081208

## Future Improvements
- Implement cross-validation
- Add more sophisticated feature engineering
- Try different activation functions
- Implement regularization techniques
- Add model persistence (save/load functionality)
- Implement hyperparameter tuning
- Add more evaluation metrics (precision, recall, F1-score)

## License
This project is open source and available under the MIT License.

## Author
Miguel Angel Lopez Mejia
