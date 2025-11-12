"""
Make Predictions Using the Model
"""

import sys
sys.path.append('src')

import numpy as np
import pickle
from pathlib import Path
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from nilearn.connectome import ConnectivityMeasure
from dataset_config import get_emotion_mapping
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("🔮 EMOTION PREDICTION WITH BALANCED MODEL")
print("="*80 + "\n")

# ============================================================================
# STEP 1: LOAD THE BALANCED MODEL
# ============================================================================

print("📂 Step 1: Loading balanced model...")

model_path = Path("models/model_balanced.pkl")

if not model_path.exists():
    print(f"\n❌ ERROR: Model file not found!")
    print(f"   Expected: {model_path}")
    print(f"\n   Please run first: python retrain_balanced_model.py")
    sys.exit(1)

with open(model_path, 'rb') as f:
    model_package = pickle.load(f)

# Extract components
model = model_package['model']
scaler = model_package['scaler']
selector = model_package['selector']
pca = model_package['pca']
config = model_package['dataset_config']
extractor_config = model_package['extractor_config']
window_config = model_package['window_config']
emotions = model_package['emotions']
performance = model_package['performance']

print(f"   ✅ Model loaded successfully!")
print(f"   Model type: {performance['best_model_name']}")
print(f"   LOSO CV Accuracy: {performance['loso_cv_accuracy']:.4f} ({performance['loso_cv_accuracy']*100:.2f}%)")
print(f"   Emotions: {', '.join(emotions)}")

# ============================================================================
# STEP 2: LOAD TEST DATA
# ============================================================================

print(f"\n📂 Step 2: Loading test data...")

# Load the dataset
dataset_name = model_package['dataset_name']
dataset_path = Path(f"ds003548")  # Hardcoded for ds003548
if not dataset_path.exists():
    print(f"\n❌ ERROR: Dataset not found at {dataset_path}")
    print(f"   Looking for: {dataset_path.absolute()}")
    sys.exit(1)

# Get all functional files
func_files = sorted(dataset_path.glob("sub-*/func/*_bold.nii.gz"))

if len(func_files) == 0:
    print(f"\n❌ ERROR: No functional files found!")
    sys.exit(1)

print(f"   ✅ Found {len(func_files)} functional files")

# Select a few random samples for prediction
np.random.seed(42)
n_samples_to_predict = min(10, len(func_files))
sample_indices = np.random.choice(len(func_files), n_samples_to_predict, replace=False)
sample_files = [func_files[i] for i in sample_indices]

print(f"   📊 Selected {n_samples_to_predict} random samples for prediction")

# ============================================================================
# STEP 3: EXTRACT FEATURES FROM TEST SAMPLES
# ============================================================================

print(f"\n🧠 Step 3: Extracting features from test samples...")

# Import required modules
from feature_extraction import ConnectomeExtractor
import io
import contextlib

# Initialize extractors (same as training)
print(f"   Loading brain atlases: {', '.join(extractor_config['atlases'])}...")

def silent_extract_connectome(extractor, window_img):
    """Extract connectome without printing"""
    with contextlib.redirect_stdout(io.StringIO()):
        time_series = extractor.extract_time_series(window_img)
        connectome = extractor.connectivity_measure.fit_transform([time_series])[0]
    return connectome

extractors = {}
for atlas_name in extractor_config['atlases']:
    extractors[atlas_name] = ConnectomeExtractor(atlas_name=atlas_name)

# Extract features
print(f"   Extracting connectomes...")
test_connectomes = []
test_files_info = []

for i, func_file in enumerate(sample_files):
    try:
        # Extract subject and run info
        parts = func_file.parts
        subject_id = [p for p in parts if p.startswith('sub-')][0]
        run_id = func_file.stem.split('_run-')[1].split('_')[0] if '_run-' in func_file.stem else '01'
        
        # Load BOLD image
        from nilearn import image
        bold_img = image.load_img(func_file)
        
        # Apply windowing
        window_size = window_config['window_size']
        n_volumes = bold_img.shape[3]
        
        # Use middle window
        start_idx = (n_volumes - window_size) // 2
        end_idx = start_idx + window_size
        window_img = image.index_img(bold_img, slice(start_idx, end_idx))
        
        # Extract features from all atlases
        connectome_features = []
        for atlas_name, extractor in extractors.items():
            connectome = silent_extract_connectome(extractor, window_img)
            triu_indices = np.triu_indices_from(connectome, k=1)
            connectome_flat = connectome[triu_indices]
            connectome_features.extend(connectome_flat)
        
        test_connectomes.append(connectome_features)
        
        test_files_info.append({
            'file': func_file.name,
            'subject': subject_id,
            'run': run_id
        })
        
        print(f"   ✓ Sample {i+1}/{n_samples_to_predict}: {subject_id}, run {run_id}")
        
    except Exception as e:
        print(f"   ✗ Error processing {func_file.name}: {e}")
        continue

X_test = np.array(test_connectomes)
print(f"\n   ✅ Extracted {len(X_test)} connectomes")
print(f"   Feature dimensions: {X_test.shape}")

# ============================================================================
# STEP 4: PREPROCESS FEATURES
# ============================================================================

print(f"\n🔧 Step 4: Preprocessing features...")

# Apply same preprocessing as training
X_test_scaled = scaler.transform(X_test)
X_test_selected = selector.transform(X_test_scaled)
X_test_pca = pca.transform(X_test_selected)

print(f"   ✅ Features preprocessed")
print(f"   {X_test.shape[1]} → {X_test_selected.shape[1]} → {X_test_pca.shape[1]} features")

# ============================================================================
# STEP 5: MAKE PREDICTIONS
# ============================================================================

print(f"\n🔮 Step 5: Making predictions...")

# Get predictions
predictions = model.predict(X_test_pca)

# Get prediction probabilities (if available)
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(X_test_pca)
    has_proba = True
else:
    has_proba = False
    print(f"   ⚠️  Model does not support probability predictions")

print(f"   ✅ Predictions complete!")

# ============================================================================
# STEP 6: DISPLAY RESULTS
# ============================================================================

print(f"\n{'='*80}")
print(f"📊 PREDICTION RESULTS")
print(f"{'='*80}\n")

for i in range(len(predictions)):
    info = test_files_info[i]
    pred = predictions[i]
    
    print(f"Sample {i+1}:")
    print(f"   File: {info['file']}")
    print(f"   Subject: {info['subject']}, Run: {info['run']}")
    print(f"   🎭 Predicted Emotion: {pred.upper()}")
    
    if has_proba:
        print(f"   Confidence scores:")
        probs = probabilities[i]
        # Sort by probability
        sorted_indices = np.argsort(probs)[::-1]
        for idx in sorted_indices:
            emotion = emotions[idx]
            prob = probs[idx]
            bar_length = int(prob * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            print(f"      {emotion:>10}: {bar} {prob*100:5.1f}%")
    
    print()

# ============================================================================
# STEP 7: PREDICTION SUMMARY
# ============================================================================

print(f"{'='*80}")
print(f"📈 PREDICTION SUMMARY")
print(f"{'='*80}\n")

from collections import Counter
pred_counts = Counter(predictions)

print(f"Emotion Distribution:")
for emotion in emotions:
    count = pred_counts.get(emotion, 0)
    percentage = (count / len(predictions) * 100) if len(predictions) > 0 else 0
    bar_length = int(percentage / 100 * 40)
    bar = '█' * bar_length + '░' * (40 - bar_length)
    print(f"   {emotion:>10}: {bar} {count:>2}/{len(predictions)} ({percentage:>5.1f}%)")

print(f"\n{'='*80}")
print(f"✅ PREDICTION COMPLETE!")
print(f"{'='*80}\n")

print(f"💡 Model Information:")
print(f"   Model: {performance['best_model_name']}")
print(f"   LOSO CV Accuracy: {performance['loso_cv_accuracy']:.4f} ({performance['loso_cv_accuracy']*100:.2f}%)")
print(f"   Trained on: {performance['n_samples']} samples from {performance['n_subjects']} subjects")
print(f"   Emotions: {', '.join(emotions)}")

print(f"\n{'='*80}\n")
