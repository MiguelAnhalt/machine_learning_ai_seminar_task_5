import numpy as np
from typing import List, Tuple

class MLPClassifier:
    def __init__(
        self,
        hidden_layer_sizes: List[int] = [5],
        learning_rate: float = 0.1,  # Increased learning rate
        epochs: int = 16000
    ):
        """
        Multi-layer Perceptron Classifier with flexible architecture.
        
        Parameters:
        -----------
        hidden_layer_sizes : List[int]
            List of integers representing the number of neurons in each hidden layer
        learning_rate : float
            Learning rate for gradient descent
        epochs : int
            Maximum number of iterations
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []

        #Save accuracy
        self.accuracy = []
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid activation function."""
        return x * (1 - x)
    
    def _initialize_weights(self, input_size: int, output_size: int) -> None:
        """Initialize weights and biases for all layers."""
        layer_sizes = [input_size] + self.hidden_layer_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            # Initialize weights using Xavier/Glorot initialization for sigmoid
            # Takes into account both input and output dimensions
                        # Initialize weights with small random values
            # weight = np.random.uniform(-0.1, 0.1, (layer_sizes[i], layer_sizes[i + 1]))
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            bias = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
        """
        General Rule
            For a weight matrix of shape (A, B):
            Rows = neurons in the previous layer
            Columns = neurons in the next layer
            W[i, j] = connection from neuron i in previous layer to neuron j in next layer
        """
    
    def _forward_pass(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Perform forward pass through the network."""
        activations = [X]
        
        for i in range(len(self.weights)):
            # Linear transformation
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                       
            # Apply sigmoid activation
            activation = self.sigmoid(z)
            activations.append(activation)
            
        return activations
    
    def _backward_pass(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray]) -> None:
        """Perform backward pass and update weights."""
        m = X.shape[0]

        # Calculate the error
        error = activations[-1] - y 

        # Iterate through the weights in reverse order
        # Consider the weights as a matrix, the input is the activations of the previous layer
        
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            # y = current weight + learning rate * (input * error )
            # (input * error ) = (derivative of the sigmoid function * error of the output layer)
            input_by_error = np.dot(activations[i].T, error) / m

            # Calculate the bias, by summing the error of the output layer along all the rows or samples
            db = np.sum(error, axis=0, keepdims=True) / m
            
            # y = current weight matrix + learning rate * (derivative of the sigmoid function * error of the output layer)
            # Update weights and biases
            self.weights[i] -= self.learning_rate * input_by_error
            self.biases[i] -= self.learning_rate * db
            
            if i > 0:
                # Compute delta for next layer
                error = np.dot(error, self.weights[i].T) * self.sigmoid_derivative(activations[i])
    
    def _one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        """Convert labels to one-hot encoding."""
        n_classes = len(np.unique(y))
        y_onehot = np.zeros((len(y), n_classes))
        # Convert 1-based labels to 0-based indices
        y_zero_based = y - 1
        y_onehot[np.arange(len(y)), y_zero_based] = 1
    
        # This will be the example output:
        # y = [1, 2, 3]
        # y_onehot = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        return y_onehot
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPClassifier_Multi':
        """
        Fit the model to the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
        """
        # Convert labels to one-hot encoding
        y_onehot = self._one_hot_encode(y)
        
        # Initialize weights
        self._initialize_weights(X.shape[1], y_onehot.shape[1])
        
        # Flag to control training
        should_continue = True
        epoch = 0
                
        # Training loop
        while epoch < self.epochs and should_continue:
            # Forward pass
            activations = self._forward_pass(X)
            
            # Backward pass
            self._backward_pass(X, y_onehot, activations)
            
            # Print progress every 1000 epochs
            if epoch % 1000 == 0:
                current_accuracy = self.score(X, y)
                print(f"Epoch {epoch}, Accuracy: {current_accuracy:.2f}%")
                """
                # Check if accuracy is not improving
                if len(self.accuracy) > 2:
                    if current_accuracy < self.accuracy[-2]:
                        should_continue = False
                        print(f"\nStopping early at epoch {epoch} as accuracy is not improving")
                """
            epoch += 1
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        activations= self._forward_pass(X)
        return activations[-1]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        # Convert 0-based indices back to 1-based labels
        return np.argmax(proba, axis=1) + 1
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the accuracy score on the given test data and labels."""
        predictions = self.predict(X)
        current_accuracy = np.mean(predictions == y) * 100
        self.accuracy.append(current_accuracy)
        return current_accuracy