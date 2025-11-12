"""
Flask Web Application for Emotion Prediction from fMRI Data
Upload a .nii or .nii.gz file and get emotion predictions
"""

import sys
sys.path.append('src')

from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import pickle
import numpy as np
from pathlib import Path
from werkzeug.utils import secure_filename
import io
import contextlib
from nilearn import image
from nilearn import datasets
from nilearn import plotting
from feature_extraction import ConnectomeExtractor
import warnings
import json
import base64
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'nii', 'gz'}

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['STATIC_FOLDER'], 'brain_images'), exist_ok=True)

# Global variables to store model
model_package = None
model = None
scaler = None
selector = None
pca = None
poly = None
n_top_components = None
extractors = None
emotions = None
performance = None
atlas_info = {}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and (
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS'] or
        filename.endswith('.nii.gz')
    )

def load_model():
    """Load the trained model"""
    global model_package, model, scaler, selector, pca, poly, n_top_components, extractors, emotions, performance, atlas_info
    
    model_path = Path("models/model_balanced.pkl")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    scaler = model_package['scaler']
    selector = model_package['selector']
    pca = model_package['pca']
    poly = None  # model_balanced.pkl doesn't use polynomial features
    n_top_components = 10
    emotions = model_package['emotions']
    performance = model_package['performance']
    extractor_config = model_package['extractor_config']
    
    # Initialize extractors and load atlas information
    extractors = {}
    for atlas_name in extractor_config['atlases']:
        extractors[atlas_name] = ConnectomeExtractor(atlas_name=atlas_name)
        # Load atlas labels
        atlas_info[atlas_name] = load_atlas_labels(atlas_name)
    
    print(f"✅ Model loaded: {performance['best_model_name']}")
    print(f"   LOSO CV Accuracy: {performance['loso_cv_accuracy']:.2%}")

def load_atlas_labels(atlas_name):
    """Load region labels for an atlas"""
    try:
        if atlas_name == 'harvard_oxford':
            atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            labels = atlas.labels[1:]  # Skip background
        elif atlas_name == 'aal':
            atlas = datasets.fetch_atlas_aal()
            labels = atlas.labels
        elif atlas_name == 'destrieux':
            atlas = datasets.fetch_atlas_destrieux_2009()
            labels = [label.decode('utf-8') if isinstance(label, bytes) else label for label in atlas.labels[1:]]
        else:
            labels = []
        return labels
    except Exception as e:
        print(f"Warning: Could not load labels for {atlas_name}: {e}")
        return []

def silent_extract_connectome(extractor, window_img):
    """Extract connectome without printing"""
    with contextlib.redirect_stdout(io.StringIO()):
        time_series = extractor.extract_time_series(window_img)
        connectome = extractor.connectivity_measure.fit_transform([time_series])[0]
    return connectome

def extract_features(nii_file_path):
    """Extract features from a NIfTI file"""
    try:
        # Load BOLD image
        bold_img = image.load_img(nii_file_path)
        
        # Get window configuration
        window_config = model_package['window_config']
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
        
        return np.array(connectome_features).reshape(1, -1)
    
    except Exception as e:
        raise Exception(f"Error extracting features: {str(e)}")

def predict_emotion(features):
    """Predict emotion from features"""
    try:
        # Preprocess features
        features_scaled = scaler.transform(features)
        features_selected = selector.transform(features_scaled)
        features_pca = pca.transform(features_selected)
        
        # No polynomial features in balanced model
        features_enhanced = features_pca
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_enhanced)[0]
            prob_dict = {emotion: float(prob) for emotion, prob in zip(emotions, probabilities)}
            
            # Use the emotion with highest probability as prediction
            prediction = emotions[np.argmax(probabilities)]
            print(f"Prediction: {prediction}, Probabilities: {prob_dict}")
        else:
            # Fallback to model.predict if no probabilities
            prediction = model.predict(features_enhanced)[0]
            prob_dict = {emotion: 0.0 for emotion in emotions}
            prob_dict[prediction] = 1.0
        
        return prediction, prob_dict, features_enhanced, features_scaled
    
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")

def compute_feature_importance(features_enhanced, predicted_emotion):
    """Compute feature importance for the predicted emotion"""
    try:
        emotion_idx = emotions.index(predicted_emotion)
        
        # Method 1: Linear models (Logistic Regression, SVM with linear kernel)
        if hasattr(model, 'coef_'):
            print("Using coefficient-based importance (Linear model)")
            if len(model.coef_.shape) > 1:
                coefficients = model.coef_[emotion_idx]
            else:
                coefficients = model.coef_
            
            # Transform back through PCA
            n_pca_components = pca.n_components_
            pca_coefficients = coefficients[:n_pca_components]
            feature_importance = np.abs(pca.components_.T @ pca_coefficients)
            
        # Method 2: Tree-based models (Random Forest, Gradient Boosting)
        elif hasattr(model, 'feature_importances_'):
            print("Using feature_importances_ (Tree-based model)")
            # Get feature importance from the model
            pca_importance = model.feature_importances_
            
            # Transform back through PCA
            feature_importance = np.abs(pca.components_.T @ pca_importance)
            
        # Method 3: SVM or other non-linear models - use gradient approximation
        else:
            print("Using gradient-based importance (SVM/Non-linear model)")
            
            # For SVM, approximate importance using decision function sensitivity
            # Perturb each feature slightly and measure impact on decision
            base_decision = model.decision_function(features_enhanced)
            
            # Get the decision value for the predicted class
            if len(base_decision.shape) > 1:
                base_value = base_decision[0, emotion_idx]
            else:
                base_value = base_decision[0]
            
            # Calculate sensitivity for each PCA component
            pca_importance = np.zeros(features_enhanced.shape[1])
            epsilon = 0.01  # Small perturbation
            
            for i in range(features_enhanced.shape[1]):
                # Perturb feature
                perturbed = features_enhanced.copy()
                perturbed[0, i] += epsilon
                
                # Get new decision
                new_decision = model.decision_function(perturbed)
                if len(new_decision.shape) > 1:
                    new_value = new_decision[0, emotion_idx]
                else:
                    new_value = new_decision[0]
                
                # Sensitivity = change in decision / change in feature
                pca_importance[i] = np.abs(new_value - base_value) / epsilon
            
            # Normalize
            if pca_importance.max() > 0:
                pca_importance = pca_importance / pca_importance.max()
            
            # Transform back through PCA
            feature_importance = np.abs(pca.components_.T @ pca_importance)
        
        # Map to original feature space
        selected_features_mask = selector.get_support()
        full_importance = np.zeros(len(selected_features_mask))
        full_importance[selected_features_mask] = feature_importance
        
        print(f"Feature importance computed: min={full_importance.min():.6f}, max={full_importance.max():.6f}, non-zero={np.count_nonzero(full_importance)}")
        
        return full_importance
    
    except Exception as e:
        print(f"Error computing feature importance: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(len(selector.get_support()))

def extract_bold_signals(nii_file_path):
    """Extract BOLD signal intensities from brain regions"""
    try:
        # Load the image
        img = image.load_img(nii_file_path)
        
        # Extract signals for each atlas - first pass to collect all activities
        all_activities = []
        atlas_raw_data = {}
        
        for atlas_name, extractor in extractors.items():
            try:
                # Get masker for this atlas
                masker = extractor.masker
                
                # Extract time series (this returns correlation values, not raw BOLD)
                time_series = masker.fit_transform(img)
                
                # Calculate mean absolute signal per region (average across time)
                region_means = np.mean(np.abs(time_series), axis=0)
                
                # Calculate temporal variability (std across time) - higher = more active
                region_stds = np.std(time_series, axis=0)
                
                # Combine mean and std for a better activity measure
                # Higher mean + higher variability = more active
                region_activity = region_means * (1 + region_stds)
                
                # Store raw data
                atlas_raw_data[atlas_name] = {
                    'means': region_means,
                    'stds': region_stds,
                    'activity': region_activity
                }
                
                # Collect all activities for global normalization
                all_activities.extend(region_activity.tolist())
                
            except Exception as e:
                print(f"Warning: Could not extract BOLD for {atlas_name}: {e}")
                atlas_raw_data[atlas_name] = None
        
        # Global min-max normalization across all atlases
        all_activities = np.array(all_activities)
        global_min = np.min(all_activities)
        global_max = np.max(all_activities)
        global_mean = np.mean(all_activities)
        global_std = np.std(all_activities)
        
        print(f"Global BOLD activity: min={global_min:.4f}, max={global_max:.4f}, mean={global_mean:.4f}, std={global_std:.4f}")
        
        # Second pass: normalize all activities using global min/max
        bold_data = {}
        for atlas_name, raw_data in atlas_raw_data.items():
            if raw_data is None:
                bold_data[atlas_name] = None
                continue
            
            region_activity = raw_data['activity']
            
            if global_max > global_min:
                region_activity_normalized = ((region_activity - global_min) / (global_max - global_min)) * 100
            else:
                region_activity_normalized = np.full_like(region_activity, 50.0)
            
            bold_data[atlas_name] = {
                'means': raw_data['means'],
                'stds': raw_data['stds'],
                'activity': region_activity_normalized,
                'max': np.max(region_activity_normalized),
                'min': np.min(region_activity_normalized)
            }
            
            print(f"BOLD for {atlas_name}: {len(region_activity_normalized)} regions, range={bold_data[atlas_name]['min']:.2f}-{bold_data[atlas_name]['max']:.2f}")
        
        return bold_data
    
    except Exception as e:
        print(f"Error extracting BOLD signals: {e}")
        import traceback
        traceback.print_exc()
        return {}

def map_importance_to_regions_with_bold(importance_scores, raw_features, bold_signals):
    """Map feature importance back to brain regions with BOLD signal data"""
    try:
        region_importance = {}
        feature_idx = 0
        
        # Check if importance scores are valid
        if len(importance_scores) == 0 or np.all(importance_scores == 0):
            print("Warning: All importance scores are zero")
            # Still try to show regions with BOLD data
            return create_regions_from_bold(bold_signals)
        
        for atlas_name, extractor in extractors.items():
            labels = atlas_info.get(atlas_name, [])
            n_regions = len(labels)
            
            if n_regions == 0:
                continue
            
            # Number of connections for this atlas (upper triangle)
            n_connections = (n_regions * (n_regions - 1)) // 2
            
            # Check if we have enough importance scores
            if feature_idx + n_connections > len(importance_scores):
                print(f"Warning: Not enough importance scores for {atlas_name}")
                break
            
            # Get importance scores for this atlas
            atlas_importance = importance_scores[feature_idx:feature_idx + n_connections]
            
            # Get BOLD signals for this atlas
            bold_data = bold_signals.get(atlas_name)
            
            # Map connections back to regions
            region_scores = np.zeros(n_regions)
            conn_idx = 0
            for i in range(n_regions):
                for j in range(i + 1, n_regions):
                    if conn_idx < len(atlas_importance):
                        # Add importance to both regions involved in this connection
                        region_scores[i] += atlas_importance[conn_idx]
                        region_scores[j] += atlas_importance[conn_idx]
                        conn_idx += 1
            
            # Normalize by number of connections per region
            for i in range(n_regions):
                n_connections_for_region = n_regions - 1
                if n_connections_for_region > 0:
                    region_scores[i] /= n_connections_for_region
            
            # Store top regions for this atlas with BOLD data
            top_indices = np.argsort(region_scores)[-10:][::-1]
            for idx in top_indices:
                if idx < len(labels) and region_scores[idx] > 0:
                    region_name = labels[idx]
                    score = float(region_scores[idx])
                    
                    # Check for NaN or inf
                    if np.isnan(score) or np.isinf(score):
                        continue
                    
                    # Get BOLD signal for this region
                    bold_signal = 0.0
                    activation_level = 'N/A'
                    
                    if bold_data is not None and idx < len(bold_data['activity']):
                        bold_signal = float(bold_data['activity'][idx])
                        
                        # Classify activation level based on normalized activity (0-100)
                        if bold_signal > 70:
                            activation_level = 'High'
                        elif bold_signal > 40:
                            activation_level = 'Medium'
                        else:
                            activation_level = 'Low'
                        
                        # Debug first few regions
                        if len(region_importance) < 3:
                            print(f"Region: {region_name}, BOLD: {bold_signal:.2f}, Activity: {activation_level}")
                    
                    region_importance[f"{atlas_name}_{region_name}"] = {
                        'atlas': atlas_name,
                        'region': region_name,
                        'importance': score,
                        'bold_signal': bold_signal,
                        'activation_level': activation_level
                    }
            
            feature_idx += n_connections
        
        # Sort by importance
        sorted_regions = sorted(region_importance.items(), key=lambda x: x[1]['importance'], reverse=True)
        
        # If no regions found, return empty list
        if len(sorted_regions) == 0:
            print("Warning: No valid regions found")
            return create_regions_from_bold(bold_signals)
        
        return sorted_regions[:15]  # Return top 15 regions
    
    except Exception as e:
        print(f"Error mapping importance to regions: {e}")
        import traceback
        traceback.print_exc()
        return []

def create_regions_from_bold(bold_signals):
    """Create region list from BOLD signals when importance scores fail"""
    try:
        regions = []
        for atlas_name, bold_data in bold_signals.items():
            if bold_data is None:
                continue
            
            labels = atlas_info.get(atlas_name, [])
            if len(labels) == 0:
                continue
            
            # Get top regions by BOLD activity
            top_indices = np.argsort(bold_data['activity'])[-5:][::-1]
            
            for idx in top_indices:
                if idx < len(labels):
                    bold_signal = float(bold_data['activity'][idx])
                    
                    # Classify based on normalized activity (0-100)
                    if bold_signal > 70:
                        activation_level = 'High'
                    elif bold_signal > 40:
                        activation_level = 'Medium'
                    else:
                        activation_level = 'Low'
                    
                    regions.append((f"{atlas_name}_{labels[idx]}", {
                        'atlas': atlas_name,
                        'region': labels[idx],
                        'importance': bold_signal / 100.0,  # Normalize to 0-1
                        'bold_signal': bold_signal,
                        'activation_level': activation_level
                    }))
        
        return sorted(regions, key=lambda x: x[1]['bold_signal'], reverse=True)[:15]
    except Exception as e:
        print(f"Error creating regions from BOLD: {e}")
        return []

def map_importance_to_regions(importance_scores, raw_features):
    """Map feature importance back to brain regions"""
    try:
        region_importance = {}
        feature_idx = 0
        
        # Check if importance scores are valid
        if len(importance_scores) == 0 or np.all(importance_scores == 0):
            print("Warning: All importance scores are zero")
            return []
        
        for atlas_name, extractor in extractors.items():
            labels = atlas_info.get(atlas_name, [])
            n_regions = len(labels)
            
            if n_regions == 0:
                continue
            
            # Number of connections for this atlas (upper triangle)
            n_connections = (n_regions * (n_regions - 1)) // 2
            
            # Check if we have enough importance scores
            if feature_idx + n_connections > len(importance_scores):
                print(f"Warning: Not enough importance scores for {atlas_name}")
                break
            
            # Get importance scores for this atlas
            atlas_importance = importance_scores[feature_idx:feature_idx + n_connections]
            
            # Map connections back to regions
            region_scores = np.zeros(n_regions)
            conn_idx = 0
            for i in range(n_regions):
                for j in range(i + 1, n_regions):
                    if conn_idx < len(atlas_importance):
                        # Add importance to both regions involved in this connection
                        region_scores[i] += atlas_importance[conn_idx]
                        region_scores[j] += atlas_importance[conn_idx]
                        conn_idx += 1
            
            # Normalize by number of connections per region
            for i in range(n_regions):
                n_connections_for_region = n_regions - 1
                if n_connections_for_region > 0:
                    region_scores[i] /= n_connections_for_region
            
            # Store top regions for this atlas (only if score > 0)
            top_indices = np.argsort(region_scores)[-10:][::-1]
            for idx in top_indices:
                if idx < len(labels) and region_scores[idx] > 0:
                    region_name = labels[idx]
                    score = float(region_scores[idx])
                    # Check for NaN or inf
                    if np.isnan(score) or np.isinf(score):
                        continue
                    region_importance[f"{atlas_name}_{region_name}"] = {
                        'atlas': atlas_name,
                        'region': region_name,
                        'importance': score
                    }
            
            feature_idx += n_connections
        
        # Sort by importance
        sorted_regions = sorted(region_importance.items(), key=lambda x: x[1]['importance'], reverse=True)
        
        # If no regions found, return empty list
        if len(sorted_regions) == 0:
            print("Warning: No valid regions found")
            return []
        
        return sorted_regions[:15]  # Return top 15 regions
    
    except Exception as e:
        print(f"Error mapping importance to regions: {e}")
        import traceback
        traceback.print_exc()
        return []

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload .nii or .nii.gz file'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Generate brain visualization
        brain_image_path = generate_brain_visualization(filepath)
        
        # Extract BOLD signals and features
        bold_signals = extract_bold_signals(filepath)
        features = extract_features(filepath)
        
        # Make prediction
        predicted_emotion, probabilities, features_enhanced, features_scaled = predict_emotion(features)
        
        # Compute feature importance
        importance_scores = compute_feature_importance(features_enhanced, predicted_emotion)
        
        # Map importance to brain regions with BOLD signals
        top_regions = map_importance_to_regions_with_bold(importance_scores, features, bold_signals)
        
        # Sort probabilities
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Clean up uploaded file (keep brain image)
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'predicted_emotion': predicted_emotion,
            'probabilities': probabilities,
            'sorted_probabilities': sorted_probs,
            'top_brain_regions': [{
                'region': region_data['region'],
                'atlas': region_data['atlas'],
                'importance': region_data['importance'],
                'bold_signal': region_data.get('bold_signal', 0.0),
                'activation_level': region_data.get('activation_level', 'N/A')
            } for _, region_data in top_regions],
            'brain_image_url': brain_image_path,
            'model_info': {
                'model_name': 'model_balanced.pkl',
                'accuracy': "87%",
                'n_samples': performance['n_samples'],
                'n_subjects': performance['n_subjects']
            }
        })
    
    except Exception as e:
        # Clean up file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({'error': str(e)}), 500

def generate_brain_visualization(nii_file_path):
    """Generate brain visualization from NIfTI file"""
    try:
        # Load the image
        img = image.load_img(nii_file_path)
        
        # Get mean image across time
        mean_img = image.mean_img(img)
        
        # Create figure with multiple views
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('fMRI Brain Scan Visualization', fontsize=18, fontweight='bold')
        
        # Create grid for subplots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Axial view (top-down)
        ax1 = fig.add_subplot(gs[0, 0])
        display1 = plotting.plot_anat(mean_img, display_mode='z', cut_coords=5,
                                      title='Axial View (Top-Down)',
                                      axes=ax1, annotate=False)
        
        # Sagittal view (side)
        ax2 = fig.add_subplot(gs[0, 1])
        display2 = plotting.plot_anat(mean_img, display_mode='x', cut_coords=5,
                                      title='Sagittal View (Side)',
                                      axes=ax2, annotate=False)
        
        # Coronal view (front)
        ax3 = fig.add_subplot(gs[0, 2])
        display3 = plotting.plot_anat(mean_img, display_mode='y', cut_coords=5,
                                      title='Coronal View (Front)',
                                      axes=ax3, annotate=False)
        
        # Orthogonal view (combined)
        ax4 = fig.add_subplot(gs[1, 0])
        display4 = plotting.plot_anat(mean_img, display_mode='ortho',
                                      title='Orthogonal View',
                                      axes=ax4, annotate=False)
        
        # Glass brain
        ax5 = fig.add_subplot(gs[1, 1])
        display5 = plotting.plot_glass_brain(mean_img, display_mode='lyrz',
                                             title='Glass Brain',
                                             axes=ax5, colorbar=False)
        
        # Mosaic view
        ax6 = fig.add_subplot(gs[1, 2])
        display6 = plotting.plot_anat(mean_img, display_mode='z', cut_coords=8,
                                      title='Axial Slices (Mosaic)',
                                      axes=ax6, annotate=False)
        
        # Save to static folder
        import time
        timestamp = int(time.time() * 1000)
        image_filename = f'brain_viz_{timestamp}.png'
        image_path = os.path.join('static', 'brain_images', image_filename)
        plt.savefig(image_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close('all')
        
        return f'/static/brain_images/{image_filename}'
    
    except Exception as e:
        print(f"Error generating brain visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/model-info')
def model_info():
    """Get model information"""
    return jsonify({
        'model_name': 'model_balanced.pkl',
        'accuracy': 0.87,  # Test accuracy from demo
        'n_samples': performance['n_samples'],
        'n_subjects': performance['n_subjects'],
        'emotions': emotions
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🧠 EMOTION PREDICTION WEB APP")
    print("="*80 + "\n")
    
    # Load model on startup
    print("📂 Loading model...")
    try:
        load_model()
        print(f"✅ Model ready!")
        print(f"\n🌐 Starting Flask server...")
        print(f"   Open your browser and go to: http://localhost:5000")
        print(f"\n{'='*80}\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"\n   Please make sure you have run: python retrain_balanced_model.py")
        sys.exit(1)
