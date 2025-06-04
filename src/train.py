"""
Training script for CIFAR-10 image classification neural network
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from neural_network import NeuralNetwork
from data_loader import CIFAR10DataLoader
from utils import plot_loss_curves, save_training_history


def train_model():
    """Train the neural network on CIFAR-10 data"""
    
    print("Loading and preprocessing CIFAR-10 data...")
    
    # Load data
    data_loader = CIFAR10DataLoader()
    X_train, y_train_onehot, X_test, y_test_onehot, y_train_labels, y_test_labels = data_loader.get_preprocessed_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {y_train_onehot.shape[1]}")
    
    # Network architecture - larger capacity for 96% accuracy target
    input_size = X_train.shape[1]  # 32*32*3 = 3072
    hidden_sizes = [1024, 512, 256, 128]  # Larger network for higher accuracy
    output_size = 5  # Five classes
    learning_rate = 0.005  # Lower initial learning rate for stability
    
    # Initialize network with advanced features
    print(f"Initializing large neural network: {input_size} -> {hidden_sizes} -> {output_size}")
    model = NeuralNetwork(
        input_size, 
        hidden_sizes, 
        output_size, 
        learning_rate=learning_rate,
        dropout_rate=0.1,  # Minimal dropout for better convergence
        use_batch_norm=True,  # Batch normalization
        momentum=0.9  # Momentum for optimization
    )
    
    # Training parameters - optimized for high accuracy (96% target)
    epochs = 150  # More epochs for better convergence
    batch_size = 64  # Smaller batches for better gradient estimates
    
    # Learning rate scheduling - more gradual decay
    initial_lr = learning_rate
    lr_decay_factor = 0.98  # Slower decay
    lr_decay_epochs = 15  # More frequent adjustments
    
    # Training history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    print(f"Starting training for {epochs} epochs with batch size {batch_size}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Learning rate decay
        if (epoch + 1) % lr_decay_epochs == 0:
            model.learning_rate *= lr_decay_factor
            print(f"Learning rate decayed to: {model.learning_rate:.6f}")
        
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train_onehot[indices]
        
        # Mini-batch training
        epoch_losses = []
        num_batches = len(X_train) // batch_size
        
        for i in range(0, len(X_train), batch_size):
            end_idx = min(i + batch_size, len(X_train))
            X_batch = X_train_shuffled[i:end_idx]
            y_batch = y_train_shuffled[i:end_idx]
            
            loss = model.train_step(X_batch, y_batch)
            epoch_losses.append(loss)
        
        # Calculate epoch metrics
        avg_train_loss = np.mean(epoch_losses)
        
        # Training accuracy
        train_predictions = model.predict(X_train)
        train_accuracy = np.mean(train_predictions == y_train_labels)
        
        # Validation metrics (on test set)
        test_predictions = model.predict(X_test)
        test_accuracy = np.mean(test_predictions == y_test_labels)
        
        # Validation loss
        test_probs = model.predict_proba(X_test)
        val_loss = model.compute_loss(test_probs, y_test_onehot)
        
        # Store history
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(test_accuracy)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress more frequently for monitoring
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Train Acc: {train_accuracy:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {test_accuracy:.4f} | "
                  f"LR: {model.learning_rate:.6f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # Early stopping if accuracy is high enough
        if test_accuracy >= 0.96:
            print(f"Target accuracy of 96% reached at epoch {epoch+1}")
            break
        
        # Additional early stopping for very high accuracy
        if test_accuracy >= 0.98:
            print(f"Excellent accuracy of 98% reached at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Final training accuracy: {train_accuracies[-1]:.3f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.3f}")
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'epochs': len(train_losses),
        'final_train_acc': train_accuracies[-1],
        'final_val_acc': val_accuracies[-1],
        'training_time': total_time
    }
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save model weights
    model_data = {
        'weights': model.weights,
        'biases': model.biases,
        'architecture': {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'output_size': output_size,
            'learning_rate': initial_lr,
            'dropout_rate': 0.3,
            'use_batch_norm': True,
            'momentum': 0.9
        }
    }
    
    import pickle
    with open('results/trained_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    save_training_history(training_history, 'results/training_history.pkl')
    
    # Plot and save loss curves
    plot_loss_curves(train_losses, val_losses, train_accuracies, val_accuracies, 'results/training_curves.png')
    
    print("Model and training history saved to 'results/' directory")
    
    return model, training_history


if __name__ == "__main__":
    model, history = train_model()
