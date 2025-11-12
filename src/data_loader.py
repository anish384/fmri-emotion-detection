"""
Data Loading and Preprocessing Module for fMRI Emotion Detection
Handles BIDS-formatted fMRI data and event files
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import nibabel as nib
from nilearn import image


class FMRIDataLoader:
    """
    Loads fMRI data and corresponding emotion labels from BIDS-formatted dataset
    """
    
    def __init__(self, dataset_path: str, task: str = "face"):
        """
        Initialize the data loader
        
        Parameters:
        -----------
        dataset_path : str
            Path to the BIDS dataset root directory
        task : str
            Task name (e.g., 'face' for emotion task)
        """
        self.dataset_path = Path(dataset_path)
        self.task = task
        self.subjects = []
        self.sessions = []
        
    def discover_data(self) -> Dict:
        """
        Discover all available subjects, sessions, and runs in the dataset
        
        Returns:
        --------
        dict : Dictionary containing data structure information
        """
        data_structure = {
            'subjects': [],
            'sessions': [],
            'runs': []
        }
        
        # Find all subject directories
        subject_dirs = sorted(self.dataset_path.glob('sub-*'))
        
        for sub_dir in subject_dirs:
            if not sub_dir.is_dir():
                continue
                
            subject_id = sub_dir.name
            data_structure['subjects'].append(subject_id)
            
            # Find sessions for this subject
            session_dirs = sorted(sub_dir.glob('ses-*'))
            
            for ses_dir in session_dirs:
                if not ses_dir.is_dir():
                    continue
                    
                session_id = ses_dir.name
                if session_id not in data_structure['sessions']:
                    data_structure['sessions'].append(session_id)
                
                # Find functional runs
                func_dir = ses_dir / 'func'
                if func_dir.exists():
                    bold_files = sorted(func_dir.glob(f'*task-{self.task}*bold.nii*'))
                    for bold_file in bold_files:
                        # Extract run number if present
                        if 'run-' in bold_file.name:
                            run_num = bold_file.name.split('run-')[1].split('_')[0]
                            if run_num not in data_structure['runs']:
                                data_structure['runs'].append(run_num)
        
        print(f"📊 Dataset Discovery:")
        print(f"   Subjects: {len(data_structure['subjects'])}")
        print(f"   Sessions: {data_structure['sessions']}")
        print(f"   Runs: {data_structure['runs']}")
        
        return data_structure
    
    def load_events(self, subject: str, session: str, run: str = None) -> pd.DataFrame:
        """
        Load event file (labels) for a specific subject/session/run
        
        Parameters:
        -----------
        subject : str
            Subject ID (e.g., 'sub-03')
        session : str
            Session ID (e.g., 'ses-1') or empty string for datasets without sessions
        run : str, optional
            Run number (e.g., '1', '2', etc.)
            
        Returns:
        --------
        pd.DataFrame : Events dataframe with timing and labels
        """
        # First, try to find event files at dataset root (for shared event files like ds003548)
        if run:
            root_pattern = f"task-{self.task}_run-{run}_events.tsv"
            root_event_files = list(self.dataset_path.glob(root_pattern))
            if root_event_files:
                events_df = pd.read_csv(root_event_files[0], sep='\t')
                return events_df
        
        # If not found at root, try subject-specific directories
        # Handle datasets with or without session structure
        if session:
            func_dir = self.dataset_path / subject / session / 'func'
        else:
            func_dir = self.dataset_path / subject / 'func'
        
        # Build pattern based on available information
        if session and run:
            pattern = f"{subject}_{session}_task-{self.task}_run-{run}_events.tsv"
        elif session:
            pattern = f"{subject}_{session}_task-{self.task}_events.tsv"
        elif run:
            pattern = f"{subject}_task-{self.task}_run-{run}_events.tsv"
        else:
            pattern = f"{subject}_task-{self.task}_events.tsv"
        
        event_files = list(func_dir.glob(pattern))
        
        if not event_files:
            raise FileNotFoundError(f"No event file found for {subject}/{session}/run-{run}")
        
        events_df = pd.read_csv(event_files[0], sep='\t')
        return events_df
    
    def load_bold(self, subject: str, session: str, run: str = None) -> nib.Nifti1Image:
        """
        Load BOLD fMRI image for a specific subject/session/run
        
        Parameters:
        -----------
        subject : str
            Subject ID (e.g., 'sub-03')
        session : str
            Session ID (e.g., 'ses-1') or empty string for datasets without sessions
        run : str, optional
            Run number (e.g., '1', '2', etc.)
            
        Returns:
        --------
        nibabel.Nifti1Image : 4D BOLD image
        """
        # Handle datasets with or without session structure
        if session:
            func_dir = self.dataset_path / subject / session / 'func'
        else:
            func_dir = self.dataset_path / subject / 'func'
        
        # Build pattern based on available information
        if session and run:
            pattern = f"{subject}_{session}_task-{self.task}*run-{run}_bold.nii*"
        elif session:
            pattern = f"{subject}_{session}_task-{self.task}_bold.nii*"
        elif run:
            pattern = f"{subject}_task-{self.task}*run-{run}_bold.nii*"
        else:
            pattern = f"{subject}_task-{self.task}_bold.nii*"
        
        bold_files = list(func_dir.glob(pattern))
        
        if not bold_files:
            raise FileNotFoundError(f"No BOLD file found for {subject}/{session}/run-{run}")
        
        print(f"📂 Loading: {bold_files[0].name}")
        bold_img = nib.load(str(bold_files[0]))
        
        return bold_img
    
    def extract_emotion_labels(self, events_df: pd.DataFrame, 
                               label_column: str = 'expression',
                               trial_type_filter: str = None,
                               exclude_labels: List[str] = None) -> List[str]:
        """
        Extract emotion labels from events dataframe
        
        Parameters:
        -----------
        events_df : pd.DataFrame
            Events dataframe
        label_column : str
            Column name containing emotion labels
        trial_type_filter : str, optional
            Filter for trial_type column (e.g., 'STIM')
        exclude_labels : list, optional
            Labels to exclude (e.g., ['White noise'])
            
        Returns:
        --------
        list : List of emotion labels
        """
        # Filter based on trial_type if specified
        if trial_type_filter and 'trial_type' in events_df.columns:
            stimulus_events = events_df[events_df['trial_type'].str.contains(trial_type_filter, na=False)]
        else:
            stimulus_events = events_df
        
        # Get labels from specified column
        if label_column not in stimulus_events.columns:
            raise ValueError(f"Column '{label_column}' not found in events file")
        
        emotions = stimulus_events[label_column].dropna().tolist()
        
        # Exclude specified labels (e.g., rest periods)
        if exclude_labels:
            emotions = [e for e in emotions if e not in exclude_labels]
        
        return emotions
    
    def get_trial_volumes(self, events_df: pd.DataFrame, tr: float = 0.7,
                         label_column: str = 'expression',
                         trial_type_filter: str = None,
                         exclude_labels: List[str] = None) -> List[Tuple[int, int, str]]:
        """
        Convert event onsets to volume indices
        
        Parameters:
        -----------
        events_df : pd.DataFrame
            Events dataframe with 'onset', 'duration', and emotion labels
        tr : float
            Repetition time (TR) in seconds
        label_column : str
            Column name containing emotion labels
        trial_type_filter : str, optional
            Filter for trial_type column (e.g., 'STIM')
        exclude_labels : list, optional
            Labels to exclude (e.g., ['White noise'])
            
        Returns:
        --------
        list : List of tuples (start_volume, end_volume, emotion_label)
        """
        # Filter based on trial_type if specified
        if trial_type_filter and 'trial_type' in events_df.columns:
            stimulus_events = events_df[events_df['trial_type'].str.contains(trial_type_filter, na=False)]
        else:
            stimulus_events = events_df
        
        trial_volumes = []
        for _, row in stimulus_events.iterrows():
            start_vol = int(row['onset'] / tr)
            end_vol = int((row['onset'] + row['duration']) / tr)
            
            # Get emotion from specified column or trial_type
            if label_column in row:
                emotion = row[label_column]
            elif 'trial_type' in row:
                emotion = row['trial_type']
            else:
                emotion = 'unknown'
            
            # Skip excluded labels (e.g., rest periods)
            if exclude_labels and emotion in exclude_labels:
                continue
            
            trial_volumes.append((start_vol, end_vol, emotion))
        
        return trial_volumes
    
    def load_all_runs(self, subject: str, session: str) -> List[Tuple[nib.Nifti1Image, pd.DataFrame]]:
        """
        Load all runs for a given subject and session
        
        Parameters:
        -----------
        subject : str
            Subject ID
        session : str
            Session ID
            
        Returns:
        --------
        list : List of tuples (bold_img, events_df) for each run
        """
        func_dir = self.dataset_path / subject / session / 'func'
        bold_files = sorted(func_dir.glob(f"*task-{self.task}*bold.nii*"))
        
        data_list = []
        for bold_file in bold_files:
            # Extract run number
            if 'run-' in bold_file.name:
                run = bold_file.name.split('run-')[1].split('_')[0]
            else:
                run = None
            
            try:
                bold_img = self.load_bold(subject, session, run)
                events_df = self.load_events(subject, session, run)
                data_list.append((bold_img, events_df, run))
            except FileNotFoundError as e:
                print(f"⚠️  Skipping: {e}")
                continue
        
        return data_list


class FMRIPreprocessor:
    """
    Basic preprocessing utilities for fMRI data
    """
    
    @staticmethod
    def smooth_image(img: nib.Nifti1Image, fwhm: float = 6.0) -> nib.Nifti1Image:
        """
        Apply Gaussian smoothing to fMRI image
        
        Parameters:
        -----------
        img : nibabel.Nifti1Image
            Input 4D fMRI image
        fwhm : float
            Full-width at half-maximum of Gaussian kernel in mm
            
        Returns:
        --------
        nibabel.Nifti1Image : Smoothed image
        """
        print(f"🔄 Smoothing with FWHM={fwhm}mm...")
        smoothed_img = image.smooth_img(img, fwhm=fwhm)
        return smoothed_img
    
    @staticmethod
    def standardize_image(img: nib.Nifti1Image) -> nib.Nifti1Image:
        """
        Standardize (z-score) the fMRI image
        
        Parameters:
        -----------
        img : nibabel.Nifti1Image
            Input 4D fMRI image
            
        Returns:
        --------
        nibabel.Nifti1Image : Standardized image
        """
        print("🔄 Standardizing signal...")
        data = img.get_fdata()
        
        # Standardize across time for each voxel
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        std[std == 0] = 1  # Avoid division by zero
        
        standardized_data = (data - mean) / std
        
        standardized_img = nib.Nifti1Image(standardized_data, img.affine, img.header)
        return standardized_img
    
    @staticmethod
    def extract_trial_volumes(img: nib.Nifti1Image, 
                             start_vol: int, 
                             end_vol: int) -> nib.Nifti1Image:
        """
        Extract specific volumes from 4D image
        
        Parameters:
        -----------
        img : nibabel.Nifti1Image
            Input 4D fMRI image
        start_vol : int
            Starting volume index
        end_vol : int
            Ending volume index
            
        Returns:
        --------
        nibabel.Nifti1Image : Extracted volumes
        """
        return image.index_img(img, slice(start_vol, end_vol))


if __name__ == "__main__":
    # Example usage
    dataset_path = r"c:\Users\Hp\Documents\emotionDetectionFmri\ds003477"
    
    loader = FMRIDataLoader(dataset_path, task="face")
    
    # Discover dataset structure
    data_info = loader.discover_data()
    
    # Load data for one subject
    subject = "sub-03"
    session = "ses-1"
    run = "5"
    
    # Load BOLD and events
    bold_img = loader.load_bold(subject, session, run)
    events_df = loader.load_events(subject, session, run)
    
    print(f"\n📊 BOLD Image Shape: {bold_img.shape}")
    print(f"📊 Events Shape: {events_df.shape}")
    
    # Extract emotion labels
    emotions = loader.extract_emotion_labels(events_df)
    print(f"\n🎭 Emotions found: {set(emotions)}")
    print(f"🎭 Total trials: {len(emotions)}")
