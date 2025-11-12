"""
Extract features ONCE and save to disk
"""

import sys
sys.path.append('src')

import numpy as np
import pickle
from pathlib import Path
from data_loader import FMRIDataLoader
from feature_extraction import ConnectomeExtractor
from nilearn import image
from collections import Counter
from dataset_config import get_dataset_config
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("📦 EXTRACTING FEATURES - ONE TIME ONLY")
print("="*80 + "\n")

DATASET_NAME = 'ds003548'
config = get_dataset_config(DATASET_NAME)

dataset_path = "ds003548"
loader = FMRIDataLoader(dataset_path, task=config['task'])

# Create extractors with silent mode
print("   Initializing extractors...")

# Monkey-patch to suppress verbose output
import io
import contextlib

def silent_extract_connectome(extractor, window_img):
    """Extract connectome without printing"""
    with contextlib.redirect_stdout(io.StringIO()):
        time_series = extractor.extract_time_series(window_img)
        connectome = extractor.connectivity_measure.fit_transform([time_series])[0]
    return connectome

extractors = {
    'harvard_oxford': ConnectomeExtractor(atlas_name='harvard_oxford'),
    'aal': ConnectomeExtractor(atlas_name='aal'),
    'destrieux': ConnectomeExtractor(atlas_name='destrieux')
}

all_connectomes = []
all_labels = []
subject_ids = []
run_ids = []

subjects_to_process = [f'sub-{i:02d}' for i in range(1, 17)]
runs_to_process = [1, 2, 3, 4, 5]

print(f"\n   Processing {len(subjects_to_process)} subjects × {len(runs_to_process)} runs...")
print(f"   (This will take a while, but only needs to be done once)\n")

total_windows = 0
for subject_idx, subject in enumerate(subjects_to_process):
    print(f"   📊 {subject}...", end='', flush=True)
    
    subject_windows = 0
    for run in runs_to_process:
        try:
            bold_img = loader.load_bold(subject, "", run)
            events_df = loader.load_events(subject, "", run)
            events_df = events_df[~events_df[config['label_column']].isin(config['exclude_labels'])]
            
            if len(events_df) == 0:
                continue
            
            trial_volumes = loader.get_trial_volumes(
                events_df, 
                tr=config['tr'],
                label_column=config['label_column'],
                trial_type_filter=config.get('trial_type_filter'),
                exclude_labels=config.get('exclude_labels')
            )
            
            window_size = 8
            stride = 4
            n_volumes = bold_img.shape[3]
            
            for start in range(0, n_volumes - window_size + 1, stride):
                end = start + window_size
                window_img = image.index_img(bold_img, slice(start, end))
                
                overlapping_emotions = []
                for trial_start, trial_end, emotion in trial_volumes:
                    if not (trial_end <= start or trial_start >= end):
                        overlapping_emotions.append(emotion)
                
                if len(overlapping_emotions) == 0:
                    continue
                
                emotion_counts = Counter(overlapping_emotions)
                majority_emotion, count = emotion_counts.most_common(1)[0]
                
                if count / len(overlapping_emotions) < 0.7:
                    continue
                
                try:
                    connectome_features = []
                    
                    for atlas_name, extractor in extractors.items():
                        # Use silent extraction
                        connectome = silent_extract_connectome(extractor, window_img)
                        triu_indices = np.triu_indices_from(connectome, k=1)
                        connectome_flat = connectome[triu_indices]
                        connectome_features.extend(connectome_flat)
                    
                    all_connectomes.append(connectome_features)
                    all_labels.append(majority_emotion)
                    subject_ids.append(subject_idx)
                    run_ids.append(run)
                    subject_windows += 1
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            continue
    
    if subject_windows > 0:
        print(f" ✅ {subject_windows} windows")
        total_windows += subject_windows
    else:
        print(f" ⚠️ No data")

# Convert to numpy arrays
print(f"\n   Converting to arrays...")
try:
    all_connectomes = np.array(all_connectomes)
except ValueError:
    min_size = min(len(conn) for conn in all_connectomes)
    print(f"   ⚠️  Different feature sizes. Truncating to {min_size} features.")
    all_connectomes = np.array([conn[:min_size] for conn in all_connectomes])

all_labels = np.array(all_labels)
subject_ids = np.array(subject_ids)
run_ids = np.array(run_ids)

print(f"\n✅ Extracted {len(all_connectomes)} samples")
print(f"   Feature dimensions: {all_connectomes.shape}")

label_counts = Counter(all_labels)
print(f"\n   📊 Label distribution:")
for emotion in config['emotions']:
    count = label_counts.get(emotion, 0)
    percentage = (count / len(all_labels) * 100) if len(all_labels) > 0 else 0
    print(f"      {emotion}: {count} samples ({percentage:.1f}%)")

# Save extracted features
output_dir = Path("extracted_features")
output_dir.mkdir(exist_ok=True)

features_data = {
    'connectomes': all_connectomes,
    'labels': all_labels,
    'subject_ids': subject_ids,
    'run_ids': run_ids,
    'dataset_name': DATASET_NAME,
    'config': config,
    'extractor_config': {
        'atlases': list(extractors.keys()),
        'connectivity_kind': 'correlation'
    },
    'window_config': {
        'window_size': 8,
        'stride': 4,
        'majority_threshold': 0.7
    }
}

output_path = output_dir / "ds003548_features.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(features_data, f)

print(f"\n💾 Features saved to: {output_path}")
print(f"\n✅ EXTRACTION COMPLETE!")
print(f"   Now run: python train_loso_from_features.py")
print(f"\n{'='*80}\n")
