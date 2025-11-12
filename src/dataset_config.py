"""
Dataset Configuration for NeuroEmo (ds005700) and Emotional Faces (ds003548)
"""

def get_dataset_config(dataset_name):
    """Get configuration for specific dataset"""
    
    if dataset_name == 'ds005700':
        return {
            'name': 'NeuroEmo',
            'task': 'fe',  # Emotion task
            'tr': 3.0,  # TR is 3.0 seconds for task-fe
            'emotions': ['Calm', 'Afraid', 'Delighted', 'Depressed', 'Excited'],
            'label_column': 'trial_type',
            'trial_type_filter': None,
            'white_noise_label': 'White noise',  # Exclude white noise periods
            'n_classes': 5,
            'has_event_files': False,  # This dataset has no event files
            # Fixed timing pattern (same for all subjects)
            'fixed_timing': [
                {'onset': 0, 'duration': 30, 'trial_type': 'Calm'},
                {'onset': 30, 'duration': 30, 'trial_type': 'White noise'},
                {'onset': 60, 'duration': 30, 'trial_type': 'Afraid'},
                {'onset': 90, 'duration': 30, 'trial_type': 'White noise'},
                {'onset': 120, 'duration': 30, 'trial_type': 'Delighted'},
                {'onset': 150, 'duration': 30, 'trial_type': 'White noise'},
                {'onset': 180, 'duration': 30, 'trial_type': 'Depressed'},
                {'onset': 210, 'duration': 30, 'trial_type': 'White noise'},
                {'onset': 240, 'duration': 30, 'trial_type': 'Excited'},
                {'onset': 270, 'duration': 30, 'trial_type': 'White noise'},
                {'onset': 300, 'duration': 30, 'trial_type': 'Delighted'},
                {'onset': 330, 'duration': 30, 'trial_type': 'White noise'},
                {'onset': 360, 'duration': 30, 'trial_type': 'Depressed'},
                {'onset': 390, 'duration': 30, 'trial_type': 'White noise'},
                {'onset': 420, 'duration': 30, 'trial_type': 'Calm'},
                {'onset': 450, 'duration': 30, 'trial_type': 'White noise'},
                {'onset': 480, 'duration': 30, 'trial_type': 'Excited'},
                {'onset': 510, 'duration': 30, 'trial_type': 'White noise'},
                {'onset': 540, 'duration': 30, 'trial_type': 'Afraid'},
                {'onset': 570, 'duration': 30, 'trial_type': 'White noise'},
            ]
        }
    elif dataset_name == 'ds003548':
        return {
            'name': 'Emotional Faces',
            'task': 'emotionalfaces',
            'tr': 2.0,  # TR is 2.0 seconds
            'emotions': ['happy', 'sad', 'angry', 'neutral'],  # 4 emotion classes
            'label_column': 'trial_type',
            'trial_type_filter': None,
            'exclude_labels': ['blank', 'scrambled', 'end'],  # Exclude non-emotion blocks
            'n_classes': 4,
            'has_event_files': True,  # This dataset has event files
            'n_runs': 5,  # 5 runs per subject
            'n_subjects': 16,  # 16 subjects total
            'volumes_per_run': 185,  # 185 volumes per run
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_emotion_mapping(dataset_name):
    """Get emotion to integer mapping"""
    config = get_dataset_config(dataset_name)
    return {emotion: i for i, emotion in enumerate(config['emotions'])}

def get_fixed_timing_events(dataset_name):
    """
    Get fixed timing events as a pandas DataFrame for datasets without event files
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
        
    Returns:
    --------
    pd.DataFrame : Events dataframe with onset, duration, and trial_type columns
    """
    import pandas as pd
    
    config = get_dataset_config(dataset_name)
    
    if not config.get('has_event_files', True):
        # Dataset has fixed timing
        if 'fixed_timing' in config:
            return pd.DataFrame(config['fixed_timing'])
        else:
            raise ValueError(f"Dataset {dataset_name} has no event files and no fixed timing defined")
    else:
        raise ValueError(f"Dataset {dataset_name} has event files, use load_events() instead")

def print_dataset_info(dataset_name):
    """Print dataset information"""
    config = get_dataset_config(dataset_name)
    print(f"📊 Dataset: {config['name']} ({dataset_name})")
    print(f"   Task: {config['task']}")
    print(f"   TR: {config['tr']}s")
    print(f"   Emotions: {config['emotions']}")
    print(f"   Classes: {config['n_classes']}")