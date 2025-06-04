"""
CIFAR-10 data loading and preprocessing utilities
"""
import numpy as np
import pickle
import os
import urllib.request
import tarfile


class CIFAR10DataLoader:
    """
    Data loader for CIFAR-10 dataset
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.cifar10_dir = os.path.join(data_dir, 'cifar-10-batches-py')
        
        # CIFAR-10 class names
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # For this project, we'll focus on 5 classes
        self.selected_classes = [0, 1, 2, 3, 4]  # airplane, automobile, bird, cat, deer
        self.selected_class_names = [self.class_names[i] for i in self.selected_classes]
        
        os.makedirs(data_dir, exist_ok=True)
    
    def download_cifar10(self):
        """Download CIFAR-10 dataset if not already present"""
        if os.path.exists(self.cifar10_dir):
            print("CIFAR-10 dataset already exists.")
            return
        
        print("Downloading CIFAR-10 dataset...")
        tar_path = os.path.join(self.data_dir, 'cifar-10-python.tar.gz')
        
        try:
            urllib.request.urlretrieve(self.cifar10_url, tar_path)
            print("Download completed. Extracting...")
            
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(self.data_dir)
            
            os.remove(tar_path)
            print("CIFAR-10 dataset extracted successfully.")
            
        except Exception as e:
            print(f"Error downloading CIFAR-10: {e}")
            raise
    
    def load_batch(self, batch_file):
        """Load a single CIFAR-10 batch file"""
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        
        data = batch[b'data']
        labels = batch[b'labels']
        
        # Reshape data from (10000, 3072) to (10000, 32, 32, 3)
        data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)
        
        return data, np.array(labels)
    
    def load_data(self):
        """
        Load and preprocess CIFAR-10 data
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test) for selected classes
        """
        self.download_cifar10()
        
        # Load training data
        X_train_list = []
        y_train_list = []
        
        for i in range(1, 6):  # data_batch_1 to data_batch_5
            batch_file = os.path.join(self.cifar10_dir, f'data_batch_{i}')
            data, labels = self.load_batch(batch_file)
            X_train_list.append(data)
            y_train_list.append(labels)
        
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        
        # Load test data
        test_batch_file = os.path.join(self.cifar10_dir, 'test_batch')
        X_test, y_test = self.load_batch(test_batch_file)
        
        # Filter for selected classes only
        train_mask = np.isin(y_train, self.selected_classes)
        test_mask = np.isin(y_test, self.selected_classes)
        
        X_train_filtered = X_train[train_mask]
        y_train_filtered = y_train[train_mask]
        X_test_filtered = X_test[test_mask]
        y_test_filtered = y_test[test_mask]
        
        # Remap labels to 0, 1, 2
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(self.selected_classes)}
        y_train_remapped = np.array([label_mapping[label] for label in y_train_filtered])
        y_test_remapped = np.array([label_mapping[label] for label in y_test_filtered])
        
        print(f"Training samples: {len(X_train_filtered)}")
        print(f"Test samples: {len(X_test_filtered)}")
        print(f"Classes: {self.selected_class_names}")
        
        return X_train_filtered, y_train_remapped, X_test_filtered, y_test_remapped
    
    def preprocess_data(self, X_train, X_test):
        """
        Preprocess the image data with advanced normalization
        
        Args:
            X_train (np.array): Training images
            X_test (np.array): Test images
            
        Returns:
            tuple: Preprocessed (X_train, X_test)
        """
        # Flatten images for neural network input
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Normalize pixel values to [0, 1]
        X_train_normalized = X_train_flat.astype(np.float32) / 255.0
        X_test_normalized = X_test_flat.astype(np.float32) / 255.0
        
        # Advanced normalization: zero-center and standardize
        mean = np.mean(X_train_normalized, axis=0)
        std = np.std(X_train_normalized, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
        
        X_train_normalized = (X_train_normalized - mean) / std
        X_test_normalized = (X_test_normalized - mean) / std
        
        return X_train_normalized, X_test_normalized
    
    def one_hot_encode(self, labels, num_classes=5):
        """
        Convert labels to one-hot encoding
        
        Args:
            labels (np.array): Class labels
            num_classes (int): Number of classes
            
        Returns:
            np.array: One-hot encoded labels
        """
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot
    
    def get_preprocessed_data(self):
        """
        Get fully preprocessed data ready for training
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test) preprocessed and ready
        """
        # Load raw data
        X_train_raw, y_train, X_test_raw, y_test = self.load_data()
        
        # Preprocess images
        X_train, X_test = self.preprocess_data(X_train_raw, X_test_raw)
        
        # One-hot encode labels
        y_train_onehot = self.one_hot_encode(y_train)
        y_test_onehot = self.one_hot_encode(y_test)
        
        return X_train, y_train_onehot, X_test, y_test_onehot, y_train, y_test
