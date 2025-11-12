"""
fMRI Emotion Detection Package

A complete machine learning pipeline for emotion classification from fMRI data.

Modules:
--------
- data_loader: Load and preprocess BIDS-formatted fMRI data
- feature_extraction: Extract functional connectivity features (connectomes)
- classical_models: Classical ML models (SVM, Random Forest, etc.)
- deep_learning_models: Deep learning models (2D CNN)
- pipeline: Complete end-to-end pipeline orchestration
- visualization: Advanced visualization utilities

Quick Start:
-----------
>>> from pipeline import EmotionDetectionPipeline
>>> pipeline = EmotionDetectionPipeline("ds003477", task="face")
>>> results = pipeline.run_complete_pipeline(subjects=["sub-03"])
"""

__version__ = "1.0.0"
__author__ = "fMRI ML Project"

from .data_loader import FMRIDataLoader, FMRIPreprocessor
from .feature_extraction import ConnectomeExtractor, TrialBasedExtractor
from .classical_models import ClassicalMLPipeline, compare_models
from .deep_learning_models import ConnectomeCNN
from .pipeline import EmotionDetectionPipeline
from .visualization import ResultsVisualizer

__all__ = [
    'FMRIDataLoader',
    'FMRIPreprocessor',
    'ConnectomeExtractor',
    'TrialBasedExtractor',
    'ClassicalMLPipeline',
    'compare_models',
    'ConnectomeCNN',
    'EmotionDetectionPipeline',
    'ResultsVisualizer'
]
