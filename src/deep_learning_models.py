"""
Deep Learning Models for fMRI Emotion Classification
2D CNN for connectivity matrix classification
"""

import numpy as np
from typing import Tuple, Dict, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


class ConnectomeCNN:
    """
    2D Convolutional Neural Network for classifying connectivity matrices
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 n_classes: int,
                 architecture: str = 'simple'):
        """
        Initialize the CNN model
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of connectivity matrix (n_regions, n_regions)
        n_classes : int
            Number of emotion classes
        architecture : str
            Model architecture: 'simple', 'deep', 'resnet'
        """
        self.input_shape = input_shape + (1,)  # Add channel dimension
        self.n_classes = n_classes
        self.architecture = architecture
        self.model = None
        self.history = None
        self.label_encoder = LabelEncoder()
        
        self._build_model()
    
    def _build_model(self):
        """Build the CNN architecture"""
        print(f"🏗️  Building {self.architecture} CNN architecture...")
        
        if self.architecture == 'simple':
            self.model = self._build_simple_cnn()
        elif self.architecture == 'deep':
            self.model = self._build_deep_cnn()
        elif self.architecture == 'resnet':
            self.model = self._build_resnet()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        print(f"   ✓ Model built")
        print(f"   ✓ Total parameters: {self.model.count_params():,}")
    
    def _build_simple_cnn(self) -> keras.Model:
        """
        Build a simple CNN architecture
        Good starting point for small datasets
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        return model
    
    def _build_deep_cnn(self) -> keras.Model:
        """
        Build a deeper CNN architecture
        Better for larger datasets
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Conv Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Conv Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        return model
    
    def _build_resnet(self) -> keras.Model:
        """
        Build a ResNet-inspired architecture with skip connections
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial conv
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Residual Block 1
        shortcut = x
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Residual Block 2
        shortcut = layers.Conv2D(64, (1, 1), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Residual Block 3
        shortcut = layers.Conv2D(128, (1, 1), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        # Global pooling and dense
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def compile_model(self, 
                     learning_rate: float = 0.001,
                     optimizer: str = 'adam'):
        """
        Compile the model
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for optimizer
        optimizer : str
            Optimizer type: 'adam', 'sgd', 'rmsprop'
        """
        print(f"⚙️  Compiling model...")
        
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        print(f"   ✓ Model compiled with {optimizer} optimizer (lr={learning_rate})")
    
    def prepare_data(self, 
                    connectomes: np.ndarray, 
                    labels: list,
                    test_size: float = 0.2,
                    val_size: float = 0.1) -> Tuple:
        """
        Prepare data for CNN training
        
        Parameters:
        -----------
        connectomes : np.ndarray
            Array of connectivity matrices (n_samples x n_regions x n_regions)
        labels : list
            List of emotion labels
        test_size : float
            Proportion of test set
        val_size : float
            Proportion of validation set (from training set)
            
        Returns:
        --------
        tuple : (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print(f"\n📊 Preparing data for CNN...")
        
        # Add channel dimension
        X = connectomes[..., np.newaxis]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        y_categorical = to_categorical(y_encoded, num_classes=self.n_classes)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )
        
        print(f"   ✓ Train set: {X_train.shape[0]} samples")
        print(f"   ✓ Validation set: {X_val.shape[0]} samples")
        print(f"   ✓ Test set: {X_test.shape[0]} samples")
        print(f"   ✓ Input shape: {X_train.shape[1:]}")
        print(f"   ✓ Classes: {self.label_encoder.classes_}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, 
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             epochs: int = 50,
             batch_size: int = 32,
             use_callbacks: bool = True) -> keras.callbacks.History:
        """
        Train the CNN model
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training data
        y_train : np.ndarray
            Training labels (one-hot encoded)
        X_val : np.ndarray
            Validation data
        y_val : np.ndarray
            Validation labels (one-hot encoded)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        use_callbacks : bool
            Whether to use callbacks (early stopping, reduce LR)
            
        Returns:
        --------
        keras.callbacks.History : Training history
        """
        print(f"\n🎓 Training CNN for {epochs} epochs...")
        
        callback_list = []
        
        if use_callbacks:
            # Early stopping
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(early_stop)
            
            # Reduce learning rate on plateau
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
            callback_list.append(reduce_lr)
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        print(f"   ✓ Training completed")
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the model on test set
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test data
        y_test : np.ndarray
            Test labels (one-hot encoded)
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        print(f"\n📈 Evaluating CNN...")
        
        # Evaluate
        test_loss, test_acc, test_auc = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        results = {
            'loss': test_loss,
            'accuracy': test_acc,
            'auc': test_auc,
            'predictions': y_pred,
            'true_labels': y_true,
            'probabilities': y_pred_proba
        }
        
        print(f"   ✓ Test Loss: {test_loss:.4f}")
        print(f"   ✓ Test Accuracy: {test_acc:.4f}")
        print(f"   ✓ Test AUC: {test_auc:.4f}")
        
        return results
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        """
        if self.history is None:
            raise ValueError("Model must be trained first")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train')
        axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: Optional[str] = None):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        save_path : str, optional
            Path to save the figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - CNN ({self.architecture})', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Print detailed classification report
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        """
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_,
            digits=4
        )
        print(report)
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        self.model.save(filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"📂 Model loaded from {filepath}")
    
    def summary(self):
        """Print model summary"""
        self.model.summary()


def get_model(model_type, **kwargs):
    """Factory function to get models for PyTorch compatibility"""
    if model_type == 'connectome_cnn':
        import torch.nn as nn
        import torch
        
        class PyTorchConnectomeCNN(nn.Module):
            def __init__(self, num_regions, num_classes, dropout=0.5):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(64)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.bn3 = nn.BatchNorm2d(128)
                
                self.pool = nn.MaxPool2d(2)
                self.dropout = nn.Dropout(dropout)
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                
                self.fc1 = nn.Linear(128, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, num_classes)
                
            def forward(self, x):
                x = self.pool(torch.relu(self.bn1(self.conv1(x))))
                x = self.dropout(x)
                x = self.pool(torch.relu(self.bn2(self.conv2(x))))
                x = self.dropout(x)
                x = torch.relu(self.bn3(self.conv3(x)))
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout(x)
                return self.fc3(x)
        
        return PyTorchConnectomeCNN(kwargs['num_regions'], kwargs['num_classes'], kwargs.get('dropout', 0.5))
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("CNN Model Example")
    print("="*60)
