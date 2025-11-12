"""
Classical Machine Learning Models for fMRI Emotion Classification
Includes SVM, Random Forest, and other sklearn-based models
"""

import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, f1_score, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


class ClassicalMLPipeline:
    """
    Pipeline for training and evaluating classical ML models on fMRI connectomes
    """
    
    def __init__(self, model_type: str = 'svm'):
        """
        Initialize the ML pipeline
        
        Parameters:
        -----------
        model_type : str
            Type of model: 'svm', 'random_forest', 'logistic', 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specified model"""
        print(f"🤖 Initializing {self.model_type} model...")
        
        if self.model_type == 'svm':
            self.model = SVC(
                kernel='linear',
                C=1.0,
                probability=True,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"   ✓ Model initialized")
    
    def prepare_data(self, 
                    connectomes: np.ndarray, 
                    labels: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for classical ML (flatten connectomes)
        
        Parameters:
        -----------
        connectomes : np.ndarray
            Array of connectivity matrices (n_samples x n_regions x n_regions)
        labels : list
            List of emotion labels
            
        Returns:
        --------
        tuple : (X_flattened, y_encoded)
        """
        print(f"\n📊 Preparing data...")
        print(f"   Input shape: {connectomes.shape}")
        
        # Flatten connectomes
        n_samples = connectomes.shape[0]
        X_flattened = connectomes.reshape(n_samples, -1)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        print(f"   ✓ Flattened shape: {X_flattened.shape}")
        print(f"   ✓ Labels: {self.label_encoder.classes_}")
        print(f"   ✓ Label distribution: {np.bincount(y_encoded)}")
        
        return X_flattened, y_encoded
    
    def train_test_split_data(self, 
                              X: np.ndarray, 
                              y: np.ndarray,
                              test_size: float = 0.2,
                              random_state: int = 42) -> Tuple:
        """
        Split data into train and test sets
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        test_size : float
            Proportion of test set
        random_state : int
            Random seed
            
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test)
        """
        print(f"\n🔀 Splitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"   ✓ Train set: {X_train.shape[0]} samples")
        print(f"   ✓ Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the model
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        """
        print(f"\n🎓 Training {self.model_type} model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Training accuracy
        train_acc = self.model.score(X_train_scaled, y_train)
        print(f"   ✓ Training accuracy: {train_acc:.4f}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the model on test set
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
            
        Returns:
        --------
        dict : Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
        
        print(f"\n📈 Evaluating model...")
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC AUC (for binary classification)
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"   ✓ Test Accuracy: {accuracy:.4f}")
        print(f"   ✓ F1 Score: {f1:.4f}")
        print(f"   ✓ ROC AUC: {roc_auc:.4f}")
        
        return results
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """
        Perform cross-validation
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        cv : int
            Number of folds
            
        Returns:
        --------
        dict : Cross-validation results
        """
        print(f"\n🔄 Performing {cv}-fold cross-validation...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        
        results = {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std()
        }
        
        print(f"   ✓ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return results
    
    def hyperparameter_tuning(self, 
                             X_train: np.ndarray, 
                             y_train: np.ndarray,
                             param_grid: Optional[Dict] = None,
                             cv: int = 3) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        param_grid : dict, optional
            Parameter grid for search
        cv : int
            Number of folds
            
        Returns:
        --------
        dict : Best parameters and scores
        """
        print(f"\n🔍 Hyperparameter tuning...")
        
        # Default parameter grids
        if param_grid is None:
            if self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
            elif self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15]
                }
            else:
                print("   ⚠️  No default param_grid for this model")
                return {}
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Grid search
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
        print(f"   ✓ Best parameters: {grid_search.best_params_}")
        print(f"   ✓ Best CV score: {grid_search.best_score_:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray,
                             save_path: Optional[str] = None):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        y_test : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        save_path : str, optional
            Path to save the figure
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {self.model_type.upper()}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def print_classification_report(self, y_test: np.ndarray, y_pred: np.ndarray):
        """
        Print detailed classification report
        
        Parameters:
        -----------
        y_test : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        """
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        
        report = classification_report(
            y_test, y_pred, 
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
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.is_fitted = True
        
        print(f"📂 Model loaded from {filepath}")


def compare_models(connectomes: np.ndarray, 
                  labels: list,
                  test_size: float = 0.2) -> Dict:
    """
    Compare multiple classical ML models
    
    Parameters:
    -----------
    connectomes : np.ndarray
        Array of connectivity matrices
    labels : list
        List of emotion labels
    test_size : float
        Test set proportion
        
    Returns:
    --------
    dict : Comparison results for all models
    """
    model_types = ['svm', 'random_forest', 'logistic', 'gradient_boosting']
    results = {}
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Testing: {model_type.upper()}")
        print(f"{'='*60}")
        
        # Initialize pipeline
        pipeline = ClassicalMLPipeline(model_type=model_type)
        
        # Prepare data
        X, y = pipeline.prepare_data(connectomes, labels)
        
        # Split data
        X_train, X_test, y_train, y_test = pipeline.train_test_split_data(
            X, y, test_size=test_size
        )
        
        # Train
        pipeline.train(X_train, y_train)
        
        # Evaluate
        eval_results = pipeline.evaluate(X_test, y_test)
        
        results[model_type] = {
            'pipeline': pipeline,
            'accuracy': eval_results['accuracy'],
            'f1_score': eval_results['f1_score'],
            'roc_auc': eval_results['roc_auc']
        }
    
    # Print comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'ROC AUC':<12}")
    print("-"*60)
    
    for model_type, res in results.items():
        print(f"{model_type:<20} {res['accuracy']:<12.4f} {res['f1_score']:<12.4f} {res['roc_auc']:<12.4f}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Classical ML Pipeline Example")
    print("="*60)
    
    # Simulate some data
    n_samples = 100
    n_regions = 48
    
    # Random connectomes
    connectomes = np.random.randn(n_samples, n_regions, n_regions)
    labels = ['neutral'] * 50 + ['smiling'] * 50
    
    # Initialize pipeline
    pipeline = ClassicalMLPipeline(model_type='svm')
    
    # Prepare data
    X, y = pipeline.prepare_data(connectomes, labels)
    
    # Split data
    X_train, X_test, y_train, y_test = pipeline.train_test_split_data(X, y)
    
    # Train
    pipeline.train(X_train, y_train)
    
    # Evaluate
    results = pipeline.evaluate(X_test, y_test)
    
    # Confusion matrix
    pipeline.plot_confusion_matrix(y_test, results['predictions'])
    
    # Classification report
    pipeline.print_classification_report(y_test, results['predictions'])
