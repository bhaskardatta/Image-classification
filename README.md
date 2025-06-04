# CIFAR-10 Image Classification from Scratch

A neural network implementation from scratch for CIFAR-10 image classification using only NumPy. This project demonstrates fundamental deep learning concepts by building a multi-layer perceptron without high-level frameworks.

## 🎯 Project Goals

- Build a neural network using only NumPy (no TensorFlow/PyTorch high-level APIs)
- Implement forward propagation, backpropagation, and gradient descent manually
- Train on 5 classes from CIFAR-10: airplane, automobile, bird, cat, deer
- Achieve good classification performance with comprehensive evaluation metrics

## 📁 Project Structure

```
├── src/
│   ├── neural_network.py    # Core neural network implementation
│   ├── data_loader.py       # CIFAR-10 data loading and preprocessing
│   ├── train.py             # Training script with loss tracking
│   ├── evaluate.py          # Evaluation and metrics calculation
│   └── utils.py             # Utility functions for visualization
├── data/                    # CIFAR-10 dataset (auto-downloaded)
├── results/                 # Training outputs and metrics
├── main.py                  # Complete pipeline script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bhaskardatta/Image-classification.git
   cd Image-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete pipeline**
   ```bash
   python main.py
   ```

That's it! The script will automatically:
- Download the CIFAR-10 dataset
- Train the neural network
- Evaluate performance
- Generate comprehensive reports and visualizations

## 📊 Sample Input/Output

### Input
- **Dataset**: CIFAR-10 (32×32 RGB images)
- **Classes**: 5 selected classes from CIFAR-10
  - 0: airplane ✈️
  - 1: automobile 🚗
  - 2: bird 🐦
  - 3: cat 🐱
  - 4: deer 🦌
- **Training samples**: ~25,000 images
- **Test samples**: ~5,000 images

### Sample Training Output
```
Loading and preprocessing CIFAR-10 data...
Training data shape: (25000, 3072)
Test data shape: (5000, 3072)
Number of features: 3072
Number of classes: 5

Initializing neural network: 3072 -> [512, 256, 128] -> 5
Starting training for 100 epochs with batch size 128
------------------------------------------------------------
Epoch   1/100 | Train Loss: 1.6094 | Train Acc: 0.2012 | Val Loss: 1.6086 | Val Acc: 0.2024
Epoch   5/100 | Train Loss: 1.4234 | Train Acc: 0.3567 | Val Loss: 1.4156 | Val Acc: 0.3612
Epoch  10/100 | Train Loss: 1.2845 | Train Acc: 0.4723 | Val Loss: 1.2934 | Val Acc: 0.4656
...
Epoch 100/100 | Train Loss: 0.8234 | Train Acc: 0.7123 | Val Loss: 0.9456 | Val Acc: 0.6784
------------------------------------------------------------
Training completed in 342.56 seconds
Final training accuracy: 0.712
Final validation accuracy: 0.678
```

### Generated Output Files

After running `python main.py`, the following files are created in the `results/` directory:

1. **trained_model.pkl** - Serialized neural network model
2. **training_history.pkl** - Training metrics and loss curves data
3. **training_curves.png** - Loss and accuracy visualization
4. **evaluation_report.txt** - Detailed performance metrics
5. **confusion_matrix.png** - Classification confusion matrix
6. **analysis_report.txt** - 200-word performance analysis

### Sample Evaluation Report
```
CIFAR-10 Neural Network Evaluation Report
=============================================

Overall Test Accuracy: 0.6394 (63.94%)

Per-Class Metrics:
--------------------------------------------------
    airplane: Precision=0.722, Recall=0.703, F1=0.712, Support=1000
  automobile: Precision=0.823, Recall=0.791, F1=0.807, Support=1000
        bird: Precision=0.514, Recall=0.497, F1=0.505, Support=1000
         cat: Precision=0.548, Recall=0.749, F1=0.633, Support=1000
        deer: Precision=0.634, Recall=0.457, F1=0.531, Support=1000

Average Metrics:
------------------------------
Macro avg:    Precision=0.648, Recall=0.639, F1=0.638
Weighted avg: Precision=0.648, Recall=0.639, F1=0.638
```

## 🏗️ Architecture Details

### Neural Network
- **Input Layer**: 3,072 neurons (32×32×3 flattened RGB images)
- **Hidden Layers**: 3 layers with [512, 256, 128] neurons
- **Output Layer**: 5 neurons (one for each class)
- **Activation Functions**: ReLU (hidden layers), Softmax (output)
- **Loss Function**: Cross-entropy
- **Optimizer**: Mini-batch gradient descent with momentum

### Key Features
- **Manual Implementation**: All forward/backward propagation coded from scratch
- **Batch Processing**: Efficient mini-batch training
- **Weight Initialization**: Xavier/He initialization for stable training
- **Regularization**: Dropout and L2 regularization options
- **Monitoring**: Real-time loss and accuracy tracking

## 🛠️ Advanced Usage

### Training Only
```bash
python src/train.py
```

### Evaluation Only (requires trained model)
```bash
python src/evaluate.py
```

### Customization

**Change Network Architecture** (edit `src/train.py`):
```python
hidden_sizes = [256, 128]           # Smaller network
hidden_sizes = [1024, 512, 256]     # Larger network
```

**Adjust Hyperparameters**:
```python
learning_rate = 0.001    # Learning rate
batch_size = 64          # Batch size
epochs = 50              # Training epochs
```

**Select Different Classes** (edit `src/data_loader.py`):
```python
selected_classes = [0, 1, 2, 3, 4]  # Current: airplane, automobile, bird, cat, deer
selected_classes = [5, 6, 7, 8, 9]  # dog, frog, horse, ship, truck
```

## 📈 Performance Metrics

The evaluation provides comprehensive metrics:
- **Overall Accuracy**: Percentage of correctly classified samples
- **Per-Class Metrics**: Precision, Recall, F1-score for each class
- **Confusion Matrix**: Visual representation of classification results
- **Training Curves**: Loss and accuracy progression during training

## 🔧 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure you're in the project root directory
cd Image-classification
python main.py
```

**Memory Issues**
```python
# Reduce batch size in src/train.py
batch_size = 32  # Instead of 128
```

**Slow Training**
```python
# Reduce epochs or network size
epochs = 50           # Instead of 100
hidden_sizes = [256, 128]  # Smaller network
```

**Dataset Download Issues**
- Ensure stable internet connection
- The script automatically downloads CIFAR-10 (~170MB)
- Check firewall settings if download fails

## 📋 Requirements

- **Python**: 3.7+
- **NumPy**: ≥1.24.0 (core computations)
- **Matplotlib**: ≥3.6.0 (visualizations)
- **Scikit-learn**: ≥1.3.0 (metrics calculation)
- **SciPy**: ≥1.10.0 (dataset handling)
