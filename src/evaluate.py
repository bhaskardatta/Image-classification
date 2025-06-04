"""
Evaluation script for CIFAR-10 image classification neural network
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from neural_network import NeuralNetwork
from data_loader import CIFAR10DataLoader
from utils import plot_confusion_matrix, save_evaluation_report


def load_trained_model(model_path='results/trained_model.pkl'):
    """Load a trained model from file"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    arch = model_data['architecture']
    model = NeuralNetwork(
        arch['input_size'],
        arch['hidden_sizes'], 
        arch['output_size'],
        arch['learning_rate'],
        dropout_rate=arch.get('dropout_rate', 0.0),
        use_batch_norm=arch.get('use_batch_norm', False),
        momentum=arch.get('momentum', 0.9)
    )
    
    model.weights = model_data['weights']
    model.biases = model_data['biases']
    
    return model


def evaluate_model():
    """Evaluate the trained neural network"""
    
    print("Loading trained model...")
    
    # Check if model exists
    if not os.path.exists('results/trained_model.pkl'):
        print("Error: No trained model found. Please run train.py first.")
        return
    
    # Load model
    model = load_trained_model()
    
    print("Loading test data...")
    
    # Load data
    data_loader = CIFAR10DataLoader()
    X_train, y_train_onehot, X_test, y_test_onehot, y_train_labels, y_test_labels = data_loader.get_preprocessed_data()
    
    class_names = data_loader.selected_class_names
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of test samples: {len(X_test)}")
    print(f"Classes: {class_names}")
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate overall accuracy
    accuracy = np.mean(y_pred == y_test_labels)
    print(f"\\nOverall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_labels, y_pred, average=None, labels=[0, 1, 2, 3, 4]
    )
    
    print("\\nPer-Class Metrics:")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>12}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}")
    
    # Calculate macro and weighted averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    print("\\nAverage Metrics:")
    print("-" * 30)
    print(f"Macro avg:    Precision={macro_precision:.3f}, Recall={macro_recall:.3f}, F1={macro_f1:.3f}")
    print(f"Weighted avg: Precision={weighted_precision:.3f}, Recall={weighted_recall:.3f}, F1={weighted_f1:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test_labels, y_pred)
    print("\\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, 'results/confusion_matrix.png')
    
    # Detailed classification report
    print("\\nDetailed Classification Report:")
    print(classification_report(y_test_labels, y_pred, target_names=class_names))
    
    # Per-class accuracy
    print("\\nPer-Class Accuracy:")
    print("-" * 25)
    for i, class_name in enumerate(class_names):
        class_mask = y_test_labels == i
        class_accuracy = np.mean(y_pred[class_mask] == y_test_labels[class_mask])
        print(f"{class_name:>12}: {class_accuracy:.3f} ({class_accuracy*100:.1f}%)")
    
    # Model confidence analysis
    print("\\nModel Confidence Analysis:")
    print("-" * 30)
    max_probs = np.max(y_pred_proba, axis=1)
    print(f"Average confidence: {np.mean(max_probs):.3f}")
    print(f"Min confidence: {np.min(max_probs):.3f}")
    print(f"Max confidence: {np.max(max_probs):.3f}")
    
    # Correct vs incorrect predictions confidence
    correct_mask = y_pred == y_test_labels
    correct_confidence = np.mean(max_probs[correct_mask])
    incorrect_confidence = np.mean(max_probs[~correct_mask])
    
    print(f"Average confidence (correct): {correct_confidence:.3f}")
    print(f"Average confidence (incorrect): {incorrect_confidence:.3f}")
    
    # Save evaluation results
    evaluation_results = {
        'accuracy': accuracy,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.tolist(),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'confidence_stats': {
            'mean': float(np.mean(max_probs)),
            'std': float(np.std(max_probs)),
            'min': float(np.min(max_probs)),
            'max': float(np.max(max_probs)),
            'correct_mean': float(correct_confidence),
            'incorrect_mean': float(incorrect_confidence)
        }
    }
    
    save_evaluation_report(evaluation_results, 'results/evaluation_report.pkl')
    
    # Generate text report
    with open('results/evaluation_report.txt', 'w') as f:
        f.write("CIFAR-10 Neural Network Evaluation Report\\n")
        f.write("=" * 45 + "\\n\\n")
        f.write(f"Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\\n\\n")
        
        f.write("Per-Class Metrics:\\n")
        f.write("-" * 50 + "\\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:>12}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}\\n")
        
        f.write("\\nAverage Metrics:\\n")
        f.write("-" * 30 + "\\n")
        f.write(f"Macro avg:    Precision={macro_precision:.3f}, Recall={macro_recall:.3f}, F1={macro_f1:.3f}\\n")
        f.write(f"Weighted avg: Precision={weighted_precision:.3f}, Recall={weighted_recall:.3f}, F1={weighted_f1:.3f}\\n")
        
        f.write("\\nConfusion Matrix:\\n")
        f.write(str(cm) + "\\n")
        
        f.write("\\nDetailed Classification Report:\\n")
        f.write(classification_report(y_test_labels, y_pred, target_names=class_names))
    
    print("\\nEvaluation complete! Results saved to 'results/' directory.")
    print("Files created:")
    print("- confusion_matrix.png")
    print("- evaluation_report.pkl")
    print("- evaluation_report.txt")
    
    return evaluation_results


if __name__ == "__main__":
    results = evaluate_model()
