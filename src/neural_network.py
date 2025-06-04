"""
Neural Network implementation from scratch using only NumPy
"""
import numpy as np


class NeuralNetwork:
    """
    A multi-layer neural network implemented from scratch with advanced features
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001, 
                 dropout_rate=0.5, use_batch_norm=False, momentum=0.9):
        """
        Initialize the neural network
        
        Args:
            input_size (int): Number of input features
            hidden_sizes (list): List of hidden layer sizes
            output_size (int): Number of output classes
            learning_rate (float): Learning rate for gradient descent
            dropout_rate (float): Dropout rate for regularization
            use_batch_norm (bool): Whether to use batch normalization
            momentum (float): Momentum for optimization
        """
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.momentum = momentum
        self.training = True
        
        # Initialize layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        self.velocity_w = []  # For momentum
        self.velocity_b = []  # For momentum
        
        # Batch normalization parameters
        if use_batch_norm:
            self.bn_gamma = []
            self.bn_beta = []
            self.bn_running_mean = []
            self.bn_running_var = []
        
        # Xavier/He initialization with better scaling
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU activations
            fan_in = layer_sizes[i]
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
            
            # Initialize velocity for momentum
            self.velocity_w.append(np.zeros_like(w))
            self.velocity_b.append(np.zeros_like(b))
            
            # Initialize batch normalization parameters
            if use_batch_norm and i < len(layer_sizes) - 2:  # Not for output layer
                self.bn_gamma.append(np.ones((1, layer_sizes[i + 1])))
                self.bn_beta.append(np.zeros((1, layer_sizes[i + 1])))
                self.bn_running_mean.append(np.zeros((1, layer_sizes[i + 1])))
                self.bn_running_var.append(np.ones((1, layer_sizes[i + 1])))
    
    def leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU activation function"""
        return np.maximum(alpha * x, x)
    
    def leaky_relu_derivative(self, x, alpha=0.01):
        """Derivative of Leaky ReLU function"""
        return np.where(x > 0, 1.0, alpha)
    
    def dropout(self, x, rate):
        """Apply dropout during training"""
        if not self.training or rate == 0:
            return x
        keep_prob = 1 - rate
        mask = np.random.binomial(1, keep_prob, x.shape) / keep_prob
        return x * mask
    
    def batch_normalize(self, x, gamma, beta, running_mean, running_var, eps=1e-8):
        """Apply batch normalization"""
        if self.training:
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)
            
            # Update running statistics
            momentum = 0.9
            running_mean[:] = momentum * running_mean + (1 - momentum) * batch_mean
            running_var[:] = momentum * running_var + (1 - momentum) * batch_var
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + eps)
        else:
            # Use running statistics during inference
            x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        
        return gamma * x_norm + beta
    
    def softmax(self, x):
        """Softmax activation function"""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation with dropout and batch normalization
        
        Args:
            X (np.array): Input data of shape (batch_size, input_size)
            
        Returns:
            tuple: (activations, z_values, bn_cache) for backpropagation
        """
        activations = [X]
        z_values = []
        bn_cache = []
        
        current_input = X
        
        # Forward through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Apply batch normalization if enabled
            if self.use_batch_norm:
                z = self.batch_normalize(z, self.bn_gamma[i], self.bn_beta[i], 
                                       self.bn_running_mean[i], self.bn_running_var[i])
                bn_cache.append(z)
            
            # Apply activation function (Leaky ReLU)
            current_input = self.leaky_relu(z)
            
            # Apply dropout
            current_input = self.dropout(current_input, self.dropout_rate)
            
            activations.append(current_input)
        
        # Output layer with softmax (no dropout, no batch norm)
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        z_values.append(z_output)
        output = self.softmax(z_output)
        activations.append(output)
        
        return activations, z_values, bn_cache
    
    def compute_loss(self, y_pred, y_true):
        """
        Compute cross-entropy loss
        
        Args:
            y_pred (np.array): Predicted probabilities
            y_true (np.array): True labels (one-hot encoded)
            
        Returns:
            float: Cross-entropy loss
        """
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
    
    def backward(self, activations, z_values, y_true, bn_cache=None):
        """
        Backward propagation with momentum and batch normalization
        
        Args:
            activations (list): Activations from forward pass
            z_values (list): Z values from forward pass
            y_true (np.array): True labels (one-hot encoded)
            bn_cache (list): Batch normalization cache
            
        Returns:
            tuple: (weight_gradients, bias_gradients)
        """
        m = y_true.shape[0]  # batch size
        
        weight_gradients = []
        bias_gradients = []
        
        # Output layer gradient
        dA = activations[-1] - y_true  # Softmax + cross-entropy derivative
        
        # Backpropagate through all layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients for current layer
            dW = np.dot(activations[i].T, dA) / m
            db = np.mean(dA, axis=0, keepdims=True)
            
            # Add L2 regularization
            l2_reg = 0.001
            dW += l2_reg * self.weights[i]
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            # Compute gradient for previous layer (if not input layer)
            if i > 0:
                dA = np.dot(dA, self.weights[i].T)
                
                # Apply dropout mask (approximation during backprop)
                if self.training and self.dropout_rate > 0:
                    dA *= (1 - self.dropout_rate)
                
                # Apply activation derivative (Leaky ReLU)
                dA = dA * self.leaky_relu_derivative(z_values[i - 1])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """
        Update network parameters using momentum-based gradient descent
        
        Args:
            weight_gradients (list): Weight gradients
            bias_gradients (list): Bias gradients
        """
        for i in range(len(self.weights)):
            # Update velocity with momentum
            self.velocity_w[i] = self.momentum * self.velocity_w[i] + self.learning_rate * weight_gradients[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] + self.learning_rate * bias_gradients[i]
            
            # Update parameters
            self.weights[i] -= self.velocity_w[i]
            self.biases[i] -= self.velocity_b[i]
    
    def train_step(self, X, y):
        """
        Perform one training step
        
        Args:
            X (np.array): Input data
            y (np.array): True labels (one-hot encoded)
            
        Returns:
            float: Loss for this step
        """
        # Set training mode
        self.training = True
        
        # Forward pass
        activations, z_values, bn_cache = self.forward(X)
        
        # Compute loss
        loss = self.compute_loss(activations[-1], y)
        
        # Backward pass
        weight_gradients, bias_gradients = self.backward(activations, z_values, y, bn_cache)
        
        # Update parameters
        self.update_parameters(weight_gradients, bias_gradients)
        
        return loss
    
    def set_training(self, training):
        """Set training mode"""
        self.training = training
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (np.array): Input data
            
        Returns:
            np.array: Predicted class indices
        """
        self.training = False
        activations, _, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X (np.array): Input data
            
        Returns:
            np.array: Prediction probabilities
        """
        self.training = False
        activations, _, _ = self.forward(X)
        return activations[-1]
