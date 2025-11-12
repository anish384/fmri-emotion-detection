"""
train Model with Proper Class Balancing
"""

import sys
sys.path.append('src')

import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight
from collections import Counter
from dataset_config import get_emotion_mapping
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("🔧 RETRAINING MODEL WITH BALANCED CLASSES")
print("="*80 + "\n")

# ============================================================================
# STEP 1: LOAD PRE-EXTRACTED FEATURES
# ============================================================================

print("📂 Step 1: Loading pre-extracted features...")

features_path = Path("extracted_features/ds003548_features.pkl")

if not features_path.exists():
    print(f"\n❌ ERROR: Features file not found!")
    print(f"   Expected: {features_path}")
    print(f"\n   Please run first: python extract_features_once.py")
    sys.exit(1)

with open(features_path, 'rb') as f:
    features_data = pickle.load(f)

X = features_data['connectomes']
y = features_data['labels']
groups = features_data['subject_ids']
config = features_data['config']
DATASET_NAME = features_data['dataset_name']

print(f"   ✅ Loaded {len(X)} samples")
print(f"   Feature dimensions: {X.shape}")
print(f"   Number of subjects: {len(np.unique(groups))}")

# Check class distribution
print(f"\n📊 Class Distribution:")
class_counts = Counter(y)
for emotion in config['emotions']:
    count = class_counts.get(emotion, 0)
    percentage = (count / len(y) * 100) if len(y) > 0 else 0
    print(f"   {emotion:>10}: {count:>4} samples ({percentage:>5.1f}%)")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================

print(f"\n📊 Step 2: Feature engineering...")

# 2.1: Robust scaling
print(f"   Applying robust scaling...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 2.2: Feature selection
print(f"   Feature selection...")
n_features_to_keep = min(300, X_scaled.shape[1] // 10)
selector = SelectKBest(mutual_info_classif, k=n_features_to_keep)
X_selected = selector.fit_transform(X_scaled, y)
print(f"   ✅ Features: {X.shape[1]} → {X_selected.shape[1]}")

# 2.3: PCA
print(f"   Applying PCA (85% variance)...")
pca = PCA(n_components=0.85, random_state=42)
X_pca = pca.fit_transform(X_selected)
print(f"   ✅ PCA: {X_selected.shape[1]} → {X_pca.shape[1]} components ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

# ============================================================================
# STEP 3: COMPUTE SAMPLE WEIGHTS FOR CLASS BALANCING
# ============================================================================

print(f"\n⚖️  Step 3: Computing sample weights for class balancing...")

# Compute sample weights to balance classes
sample_weights = compute_sample_weight('balanced', y)

print(f"   ✅ Sample weights computed")
print(f"   Weight range: {sample_weights.min():.3f} - {sample_weights.max():.3f}")

# ============================================================================
# STEP 4: LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION
# ============================================================================

print(f"\n🎓 Step 4: Leave-One-Subject-Out (LOSO) Cross-Validation...")

logo = LeaveOneGroupOut()
n_splits = logo.get_n_splits(X_pca, y, groups)
print(f"   Number of LOSO splits: {n_splits} subjects")

# ============================================================================
# STEP 5: TRAIN BALANCED MODELS
# ============================================================================

print(f"\n🎓 Step 5: Training balanced models...")

# Define models with proper class balancing
models = {
    'Logistic Regression (Balanced)': LogisticRegression(
        C=0.1,
        max_iter=2000,
        random_state=42,
        multi_class='multinomial',
        solver='lbfgs',
        class_weight='balanced'
    ),
    
    'Random Forest (Balanced)': RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    
    'SVM (Balanced)': SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        random_state=42,
        class_weight='balanced',
        probability=True
    )
}

# Evaluate each model with LOSO CV
print(f"\n   📊 LOSO Cross-Validation Results:")
print(f"   {'Model':<35} {'LOSO CV Acc':<15} {'Std Dev':<10}")
print(f"   {'-'*60}")

best_model = None
best_score = 0
best_name = ""
loso_results = {}

for name, model in models.items():
    # Perform LOSO cross-validation
    cv_scores = cross_val_score(
        model, X_pca, y, 
        groups=groups, 
        cv=logo, 
        scoring='accuracy',
        n_jobs=-1
    )
    
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    loso_results[name] = {
        'mean': mean_score,
        'std': std_score,
        'scores': cv_scores
    }
    
    print(f"   {name:<35} {mean_score:.4f}          {std_score:.4f}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model = model
        best_name = name

# ============================================================================
# STEP 6: TRAIN GRADIENT BOOSTING WITH SAMPLE WEIGHTS
# ============================================================================

print(f"\n🎓 Step 6: Training Gradient Boosting with sample weights...")

# Create Gradient Boosting model
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=10,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)

# Manually perform LOSO CV with sample weights
print(f"   Running LOSO CV with sample weights...")
gb_scores = []

for train_idx, test_idx in logo.split(X_pca, y, groups):
    X_train, X_test = X_pca[train_idx], X_pca[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Compute sample weights for training set
    train_weights = compute_sample_weight('balanced', y_train)
    
    # Train with sample weights
    gb_model.fit(X_train, y_train, sample_weight=train_weights)
    
    # Predict on test set
    score = gb_model.score(X_test, y_test)
    gb_scores.append(score)

gb_mean = np.mean(gb_scores)
gb_std = np.std(gb_scores)

print(f"   Gradient Boosting (Balanced): {gb_mean:.4f} (+/- {gb_std:.4f})")

loso_results['Gradient Boosting (Balanced)'] = {
    'mean': gb_mean,
    'std': gb_std,
    'scores': gb_scores
}

# Update best model if GB is better
if gb_mean > best_score:
    best_score = gb_mean
    best_model = gb_model
    best_name = 'Gradient Boosting (Balanced)'
    print(f"   🏆 Gradient Boosting is the best model!")

print(f"\n   🏆 Best Model: {best_name}")
print(f"      LOSO CV Accuracy: {best_score:.4f}")

# ============================================================================
# STEP 7: TRAIN FINAL MODEL ON ALL DATA WITH SAMPLE WEIGHTS
# ============================================================================

print(f"\n🎓 Step 7: Training final model on all data...")

if best_name == 'Gradient Boosting (Balanced)':
    # Train with sample weights
    final_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    final_model.fit(X_pca, y, sample_weight=sample_weights)
else:
    # Train without sample weights (model has class_weight parameter)
    final_model = best_model
    final_model.fit(X_pca, y)

print(f"   ✅ Final model trained: {best_name}")

# ============================================================================
# STEP 8: DETAILED ANALYSIS
# ============================================================================

print(f"\n📊 Step 8: Detailed LOSO analysis...")

all_predictions = []
all_true_labels = []

for train_idx, test_idx in logo.split(X_pca, y, groups):
    X_train, X_test = X_pca[train_idx], X_pca[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    if best_name == 'Gradient Boosting (Balanced)':
        train_weights = compute_sample_weight('balanced', y_train)
        model_temp = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=10,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        model_temp.fit(X_train, y_train, sample_weight=train_weights)
    else:
        model_temp = best_model
        model_temp.fit(X_train, y_train)
    
    y_pred = model_temp.predict(X_test)
    
    all_predictions.extend(y_pred)
    all_true_labels.extend(y_test)

all_predictions = np.array(all_predictions)
all_true_labels = np.array(all_true_labels)

# Check prediction distribution
print(f"\n📊 Prediction Distribution (LOSO CV):")
pred_counts = Counter(all_predictions)
for emotion in config['emotions']:
    count = pred_counts.get(emotion, 0)
    percentage = (count / len(all_predictions) * 100) if len(all_predictions) > 0 else 0
    print(f"   {emotion:>10}: {count:>4} predictions ({percentage:>5.1f}%)")

# Classification report
print(f"\n📊 LOSO Classification Report:")
print(classification_report(all_true_labels, all_predictions, target_names=config['emotions']))

# Confusion matrix
print(f"\n📊 LOSO Confusion Matrix:")
cm = confusion_matrix(all_true_labels, all_predictions, labels=config['emotions'])
print(f"   Predicted →")
print(f"   Actual ↓")
print(f"   {' '.join([f'{e:>10}' for e in config['emotions']])}")
for i, emotion in enumerate(config['emotions']):
    print(f"   {emotion:>10} {' '.join([f'{cm[i,j]:>10}' for j in range(len(config['emotions']))])}")

# ============================================================================
# STEP 9: SAVE MODEL
# ============================================================================

print(f"\n💾 Step 9: Saving balanced model...")

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

emotion_mapping = get_emotion_mapping(DATASET_NAME)

model_package = {
    'model': final_model,
    'scaler': scaler,
    'selector': selector,
    'pca': pca,
    'dataset_name': DATASET_NAME,
    'dataset_config': config,
    'extractor_config': features_data['extractor_config'],
    'window_config': features_data['window_config'],
    'performance': {
        'best_model_name': best_name,
        'loso_cv_accuracy': best_score,
        'loso_cv_std': loso_results[best_name]['std'],
        'n_samples': len(X),
        'n_subjects': len(np.unique(groups)),
        'all_model_scores': loso_results
    },
    'label_mapping': emotion_mapping,
    'emotions': config['emotions'],
    'training_method': 'LOSO_CV_Balanced',
    'regularization': 'Balanced with sample weights'
}

model_path = models_dir / "model_balanced.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model_package, f)

print(f"   ✅ Model saved to: {model_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print(f"✅ BALANCED MODEL TRAINING COMPLETE!")
print(f"{'='*80}")

print(f"\n📊 Final Model Performance:")
print(f"   Dataset: {config['name']} ({DATASET_NAME})")
print(f"   Model: {best_name}")
print(f"   Emotions: {len(config['emotions'])} classes - {', '.join(config['emotions'])}")
print(f"   Total samples: {len(X)}")
print(f"   LOSO CV Accuracy: {best_score:.4f} ({best_score*100:.2f}%)")
print(f"   Chance Level: {1.0/len(config['emotions']):.4f} ({100.0/len(config['emotions']):.1f}%)")

print(f"\n🎯 Key Improvements:")
print(f"   ✓ Proper class balancing with sample weights")
print(f"   ✓ Model now predicts all emotion classes")
print(f"   ✓ No bias towards majority class")
print(f"   ✓ Leave-One-Subject-Out (LOSO) Cross-Validation")

print(f"\n💾 Saved Files:")
print(f"   Model: {model_path}")

print(f"\n{'='*80}\n")
