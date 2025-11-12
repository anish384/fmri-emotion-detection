"""
Complete End-to-End Pipeline for fMRI Emotion Detection
Orchestrates data loading, feature extraction, and model training
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import json
from datetime import datetime

from data_loader import FMRIDataLoader, FMRIPreprocessor
from feature_extraction import ConnectomeExtractor, TrialBasedExtractor
from classical_models import ClassicalMLPipeline, compare_models
from deep_learning_models import ConnectomeCNN


class EmotionDetectionPipeline:
    """
    Complete pipeline for fMRI emotion detection
    """
    
    def __init__(self, 
                 dataset_path: str,
                 task: str = "face",
                 atlas_name: str = "harvard_oxford",
                 output_dir: str = "results"):
        """
        Initialize the pipeline
        
        Parameters:
        -----------
        dataset_path : str
            Path to BIDS dataset
        task : str
            Task name
        atlas_name : str
            Brain atlas to use
        output_dir : str
            Directory to save results
        """
        self.dataset_path = Path(dataset_path)
        self.task = task
        self.atlas_name = atlas_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = FMRIDataLoader(str(self.dataset_path), task=task)
        self.preprocessor = FMRIPreprocessor()
        self.connectome_extractor = ConnectomeExtractor(atlas_name=atlas_name)
        
        # Data storage
        self.connectomes = None
        self.labels = None
        
        print(f"\n{'='*60}")
        print(f"🚀 EMOTION DETECTION PIPELINE INITIALIZED")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset_path.name}")
        print(f"Task: {self.task}")
        print(f"Atlas: {self.atlas_name}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def extract_features_from_dataset(self,
                                     subjects: Optional[List[str]] = None,
                                     sessions: Optional[List[str]] = None,
                                     smooth_fwhm: float = 6.0,
                                     use_trial_based: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Extract connectome features from entire dataset
        
        Parameters:
        -----------
        subjects : list, optional
            List of subject IDs to process (None = all)
        sessions : list, optional
            List of session IDs to process (None = all)
        smooth_fwhm : float
            Smoothing kernel size
        use_trial_based : bool
            Whether to extract trial-level connectomes
            
        Returns:
        --------
        tuple : (connectomes array, labels list)
        """
        print(f"\n{'='*60}")
        print(f"📊 FEATURE EXTRACTION")
        print(f"{'='*60}\n")
        
        # Discover dataset
        data_info = self.data_loader.discover_data()
        
        if subjects is None:
            subjects = data_info['subjects']
        if sessions is None:
            sessions = data_info['sessions']
        
        all_connectomes = []
        all_labels = []
        
        # Process each subject and session
        for subject in subjects:
            for session in sessions:
                print(f"\n{'─'*60}")
                print(f"Processing: {subject} / {session}")
                print(f"{'─'*60}")
                
                try:
                    # Load all runs for this subject/session
                    runs_data = self.data_loader.load_all_runs(subject, session)
                    
                    for bold_img, events_df, run in runs_data:
                        print(f"\n🔄 Run {run}:")
                        
                        # Preprocess
                        if smooth_fwhm > 0:
                            bold_img = self.preprocessor.smooth_image(bold_img, fwhm=smooth_fwhm)
                        
                        if use_trial_based:
                            # Extract trial-level connectomes
                            trial_volumes = self.data_loader.get_trial_volumes(events_df)
                            trial_extractor = TrialBasedExtractor(self.connectome_extractor)
                            
                            connectomes, labels = trial_extractor.extract_trial_connectomes(
                                bold_img, trial_volumes
                            )
                            
                            all_connectomes.extend(connectomes)
                            all_labels.extend(labels)
                        else:
                            # Extract run-level connectome
                            connectome = self.connectome_extractor.extract_connectome_from_image(bold_img)
                            emotions = self.data_loader.extract_emotion_labels(events_df)
                            
                            # Use majority emotion as label
                            from collections import Counter
                            majority_emotion = Counter(emotions).most_common(1)[0][0]
                            
                            all_connectomes.append(connectome)
                            all_labels.append(majority_emotion)
                
                except Exception as e:
                    print(f"⚠️  Error processing {subject}/{session}: {e}")
                    continue
        
        # Convert to arrays
        self.connectomes = np.array(all_connectomes)
        self.labels = all_labels
        
        print(f"\n{'='*60}")
        print(f"✅ FEATURE EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.connectomes)}")
        print(f"Connectome shape: {self.connectomes.shape}")
        print(f"Unique labels: {set(self.labels)}")
        
        # Check if we have data
        if len(self.connectomes) == 0:
            print(f"\n⚠️  WARNING: No data was extracted!")
            print(f"\nPossible reasons:")
            print(f"   1. No BOLD files found for specified subjects/sessions")
            print(f"   2. No event files found")
            print(f"   3. All trials were too short (< 3 volumes)")
            print(f"   4. File naming doesn't match BIDS format")
            print(f"\n💡 Run 'python diagnose_data.py' to identify the issue")
            print(f"{'='*60}\n")
            return self.connectomes, self.labels
        
        print(f"Label distribution: {dict(zip(*np.unique(self.labels, return_counts=True)))}")
        print(f"{'='*60}\n")
        
        # Save features
        self._save_features()
        
        return self.connectomes, self.labels
    
    def _save_features(self):
        """Save extracted features to disk"""
        features_path = self.output_dir / "features.pkl"
        
        with open(features_path, 'wb') as f:
            pickle.dump({
                'connectomes': self.connectomes,
                'labels': self.labels,
                'atlas': self.atlas_name
            }, f)
        
        print(f"💾 Features saved to {features_path}")
    
    def load_features(self, features_path: Optional[str] = None):
        """
        Load previously extracted features
        
        Parameters:
        -----------
        features_path : str, optional
            Path to features file (default: output_dir/features.pkl)
        """
        if features_path is None:
            features_path = self.output_dir / "features.pkl"
        
        with open(features_path, 'rb') as f:
            data = pickle.load(f)
        
        self.connectomes = data['connectomes']
        self.labels = data['labels']
        
        print(f"📂 Features loaded from {features_path}")
        print(f"   Samples: {len(self.connectomes)}")
        print(f"   Shape: {self.connectomes.shape}")
    
    def train_classical_models(self, 
                              test_size: float = 0.2,
                              compare_all: bool = True) -> Dict:
        """
        Train classical ML models
        
        Parameters:
        -----------
        test_size : float
            Test set proportion
        compare_all : bool
            Whether to compare all model types
            
        Returns:
        --------
        dict : Training results
        """
        if self.connectomes is None:
            raise ValueError("Features must be extracted first")
        
        if len(self.connectomes) == 0:
            print(f"\n{'='*60}")
            print(f"⚠️  SKIPPING CLASSICAL ML TRAINING")
            print(f"{'='*60}")
            print(f"No data available. Run 'python diagnose_data.py' to troubleshoot.")
            print(f"{'='*60}\n")
            return {}
        
        print(f"\n{'='*60}")
        print(f"🤖 CLASSICAL ML TRAINING")
        print(f"{'='*60}\n")
        
        if compare_all:
            results = compare_models(self.connectomes, self.labels, test_size=test_size)
            
            # Save best model
            best_model_type = max(results, key=lambda k: results[k]['accuracy'])
            best_pipeline = results[best_model_type]['pipeline']
            
            model_path = self.output_dir / f"best_classical_model_{best_model_type}.pkl"
            best_pipeline.save_model(str(model_path))
            
            return results
        else:
            # Train single model (SVM by default)
            pipeline = ClassicalMLPipeline(model_type='svm')
            X, y = pipeline.prepare_data(self.connectomes, self.labels)
            X_train, X_test, y_train, y_test = pipeline.train_test_split_data(
                X, y, test_size=test_size
            )
            
            pipeline.train(X_train, y_train)
            results = pipeline.evaluate(X_test, y_test)
            
            # Save model
            model_path = self.output_dir / "svm_model.pkl"
            pipeline.save_model(str(model_path))
            
            # Visualizations
            pipeline.plot_confusion_matrix(
                y_test, results['predictions'],
                save_path=str(self.output_dir / "confusion_matrix_svm.png")
            )
            pipeline.print_classification_report(y_test, results['predictions'])
            
            return {'svm': results}
    
    def train_deep_learning_model(self,
                                  architecture: str = 'simple',
                                  epochs: int = 50,
                                  batch_size: int = 32,
                                  learning_rate: float = 0.001,
                                  test_size: float = 0.2,
                                  val_size: float = 0.1) -> Dict:
        """
        Train deep learning CNN model
        
        Parameters:
        -----------
        architecture : str
            CNN architecture type
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate
        test_size : float
            Test set proportion
        val_size : float
            Validation set proportion
            
        Returns:
        --------
        dict : Training results
        """
        if self.connectomes is None:
            raise ValueError("Features must be extracted first")
        
        if len(self.connectomes) == 0:
            print(f"\n{'='*60}")
            print(f"⚠️  SKIPPING DEEP LEARNING TRAINING")
            print(f"{'='*60}")
            print(f"No data available. Run 'python diagnose_data.py' to troubleshoot.")
            print(f"{'='*60}\n")
            return {}
        
        print(f"\n{'='*60}")
        print(f"🧠 DEEP LEARNING TRAINING")
        print(f"{'='*60}\n")
        
        # Get input shape and number of classes
        n_regions = self.connectomes.shape[1]
        n_classes = len(set(self.labels))
        
        # Initialize CNN
        cnn = ConnectomeCNN(
            input_shape=(n_regions, n_regions),
            n_classes=n_classes,
            architecture=architecture
        )
        
        # Compile
        cnn.compile_model(learning_rate=learning_rate)
        
        # Print summary
        cnn.summary()
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = cnn.prepare_data(
            self.connectomes, self.labels,
            test_size=test_size, val_size=val_size
        )
        
        # Train
        history = cnn.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Evaluate
        results = cnn.evaluate(X_test, y_test)
        
        # Visualizations
        cnn.plot_training_history(
            save_path=str(self.output_dir / "training_history_cnn.png")
        )
        cnn.plot_confusion_matrix(
            results['true_labels'], results['predictions'],
            save_path=str(self.output_dir / "confusion_matrix_cnn.png")
        )
        cnn.print_classification_report(results['true_labels'], results['predictions'])
        
        # Save model
        model_path = self.output_dir / f"cnn_model_{architecture}.h5"
        cnn.save_model(str(model_path))
        
        return results
    
    def run_complete_pipeline(self,
                             subjects: Optional[List[str]] = None,
                             sessions: Optional[List[str]] = None,
                             train_classical: bool = True,
                             train_deep_learning: bool = True,
                             smooth_fwhm: float = 6.0) -> Dict:
        """
        Run the complete pipeline from start to finish
        
        Parameters:
        -----------
        subjects : list, optional
            Subjects to process
        sessions : list, optional
            Sessions to process
        train_classical : bool
            Whether to train classical models
        train_deep_learning : bool
            Whether to train deep learning models
        smooth_fwhm : float
            Smoothing kernel size
            
        Returns:
        --------
        dict : Complete results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'dataset': str(self.dataset_path),
            'atlas': self.atlas_name
        }
        
        # Step 1: Extract features
        print("\n" + "🔹"*30)
        print("STEP 1: FEATURE EXTRACTION")
        print("🔹"*30)
        
        self.extract_features_from_dataset(
            subjects=subjects,
            sessions=sessions,
            smooth_fwhm=smooth_fwhm,
            use_trial_based=True
        )
        
        results['n_samples'] = len(self.connectomes)
        results['connectome_shape'] = self.connectomes.shape
        
        # Step 2: Train classical models
        if train_classical:
            print("\n" + "🔹"*30)
            print("STEP 2: CLASSICAL ML")
            print("🔹"*30)
            
            classical_results = self.train_classical_models(compare_all=True)
            
            # Only add results if training succeeded
            if classical_results:
                results['classical'] = {
                    k: {
                        'accuracy': v.get('accuracy', 0.0),
                        'f1_score': v.get('f1_score', 0.0),
                        'roc_auc': v.get('roc_auc', 0.0)
                    }
                    for k, v in classical_results.items()
                }
        
        # Step 3: Train deep learning model
        if train_deep_learning:
            print("\n" + "🔹"*30)
            print("STEP 3: DEEP LEARNING")
            print("🔹"*30)
            
            dl_results = self.train_deep_learning_model(
                architecture='simple',
                epochs=50,
                batch_size=16
            )
            
            # Only add results if training succeeded
            if dl_results and 'accuracy' in dl_results:
                results['deep_learning'] = {
                    'accuracy': dl_results['accuracy'],
                    'auc': dl_results.get('auc', 0.0),
                    'loss': dl_results.get('loss', 0.0)
                }
        
        # Save results summary
        results_path = self.output_dir / "results_summary.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✅ PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*60}\n")
        
        return results


if __name__ == "__main__":
    # Example: Run complete pipeline
    dataset_path = r"c:\Users\Hp\Documents\emotionDetectionFmri\ds003477"
    
    pipeline = EmotionDetectionPipeline(
        dataset_path=dataset_path,
        task="face",
        atlas_name="harvard_oxford",
        output_dir="results"
    )
    
    # Run complete pipeline on one subject
    results = pipeline.run_complete_pipeline(
        subjects=["sub-03"],
        train_classical=True,
        train_deep_learning=True,
        smooth_fwhm=6.0
    )
    
    print("\n📊 Final Results:")
    print(json.dumps(results, indent=2))
