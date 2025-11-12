"""
Feature Extraction Module for fMRI Data
Creates functional connectivity matrices (connectomes) from fMRI scans
"""

import numpy as np
import nibabel as nib
from typing import List, Tuple, Optional
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
import matplotlib.pyplot as plt


class ConnectomeExtractor:
    """
    Extracts functional connectivity features (connectomes) from fMRI data
    """
    
    def __init__(self, atlas_name: str = 'harvard_oxford', 
                 connectivity_kind: str = 'correlation',
                 standardize: bool = True):
        """
        Initialize the connectome extractor
        
        Parameters:
        -----------
        atlas_name : str
            Name of brain atlas to use. Options:
            - 'harvard_oxford': Harvard-Oxford cortical atlas (48 regions)
            - 'aal': AAL atlas (116 regions)
            - 'destrieux': Destrieux atlas (148 regions)
            - 'schaefer': Schaefer atlas (100-1000 regions)
        connectivity_kind : str
            Type of connectivity measure: 'correlation', 'partial correlation', 'covariance'
        standardize : bool
            Whether to standardize time series before computing connectivity
        """
        self.atlas_name = atlas_name
        self.connectivity_kind = connectivity_kind
        self.standardize = standardize
        self.atlas = None
        self.masker = None
        self.connectivity_measure = None
        
        self._load_atlas()
        self._initialize_masker()
        self._initialize_connectivity_measure()
    
    def _load_atlas(self):
        """Load the specified brain atlas"""
        print(f"🧠 Loading {self.atlas_name} atlas...")
        
        if self.atlas_name == 'harvard_oxford':
            self.atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            self.atlas_img = self.atlas.maps
            self.atlas_labels = self.atlas.labels
            print(f"   ✓ Loaded Harvard-Oxford atlas with {len(self.atlas_labels)} regions")
            
        elif self.atlas_name == 'aal':
            self.atlas = datasets.fetch_atlas_aal()
            self.atlas_img = self.atlas.maps
            self.atlas_labels = self.atlas.labels
            print(f"   ✓ Loaded AAL atlas with {len(self.atlas_labels)} regions")
            
        elif self.atlas_name == 'destrieux':
            self.atlas = datasets.fetch_atlas_destrieux_2009()
            self.atlas_img = self.atlas.maps
            self.atlas_labels = self.atlas.labels
            print(f"   ✓ Loaded Destrieux atlas with {len(self.atlas_labels)} regions")
            
        elif self.atlas_name == 'schaefer':
            self.atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
            self.atlas_img = self.atlas.maps
            self.atlas_labels = self.atlas.labels
            print(f"   ✓ Loaded Schaefer atlas with {len(self.atlas_labels)} regions")
            
        else:
            raise ValueError(f"Unknown atlas: {self.atlas_name}")
    
    def _initialize_masker(self):
        """Initialize the masker for extracting time series"""
        print(f"🔧 Initializing masker...")
        
        self.masker = NiftiLabelsMasker(
            labels_img=self.atlas_img,
            standardize=self.standardize,
            memory='nilearn_cache',
            verbose=0
        )
        print(f"   ✓ Masker ready")
    
    def _initialize_connectivity_measure(self):
        """Initialize connectivity measure"""
        print(f"🔧 Initializing connectivity measure ({self.connectivity_kind})...")
        
        self.connectivity_measure = ConnectivityMeasure(
            kind=self.connectivity_kind,
            vectorize=False,  # Keep as 2D matrix
            discard_diagonal=False
        )
        print(f"   ✓ Connectivity measure ready")
    
    def extract_time_series(self, fmri_img: nib.Nifti1Image) -> np.ndarray:
        """
        Extract regional time series from fMRI image using the atlas
        
        Parameters:
        -----------
        fmri_img : nibabel.Nifti1Image
            4D fMRI image
            
        Returns:
        --------
        np.ndarray : Time series matrix (n_timepoints x n_regions)
        """
        print(f"📊 Extracting time series...")
        time_series = self.masker.fit_transform(fmri_img)
        print(f"   ✓ Time series shape: {time_series.shape}")
        return time_series
    
    def compute_connectome(self, time_series: np.ndarray) -> np.ndarray:
        """
        Compute functional connectivity matrix from time series
        
        Parameters:
        -----------
        time_series : np.ndarray
            Time series matrix (n_timepoints x n_regions)
            
        Returns:
        --------
        np.ndarray : Connectivity matrix (n_regions x n_regions)
        """
        print(f"🔗 Computing connectome...")
        connectome = self.connectivity_measure.fit_transform([time_series])[0]
        print(f"   ✓ Connectome shape: {connectome.shape}")
        return connectome
    
    def extract_connectome_from_image(self, fmri_img: nib.Nifti1Image) -> np.ndarray:
        """
        Complete pipeline: Extract connectome directly from fMRI image
        
        Parameters:
        -----------
        fmri_img : nibabel.Nifti1Image
            4D fMRI image
            
        Returns:
        --------
        np.ndarray : Connectivity matrix (n_regions x n_regions)
        """
        time_series = self.extract_time_series(fmri_img)
        connectome = self.compute_connectome(time_series)
        return connectome
    
    def extract_multiple_connectomes(self, 
                                    fmri_images: List[nib.Nifti1Image]) -> np.ndarray:
        """
        Extract connectomes from multiple fMRI images
        
        Parameters:
        -----------
        fmri_images : list
            List of 4D fMRI images
            
        Returns:
        --------
        np.ndarray : Array of connectivity matrices (n_samples x n_regions x n_regions)
        """
        connectomes = []
        
        for i, img in enumerate(fmri_images):
            print(f"\n🔄 Processing image {i+1}/{len(fmri_images)}...")
            connectome = self.extract_connectome_from_image(img)
            connectomes.append(connectome)
        
        return np.array(connectomes)
    
    def visualize_connectome(self, connectome: np.ndarray, 
                            title: str = "Functional Connectivity Matrix",
                            save_path: Optional[str] = None):
        """
        Visualize a connectivity matrix
        
        Parameters:
        -----------
        connectome : np.ndarray
            Connectivity matrix (n_regions x n_regions)
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the matrix
        im = ax.imshow(connectome, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Brain Region', fontsize=12)
        ax.set_ylabel('Brain Region', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def visualize_connectome_glass_brain(self, connectome: np.ndarray,
                                         threshold: float = 0.8,
                                         title: str = "Connectivity Glass Brain",
                                         save_path: Optional[str] = None):
        """
        Visualize connectivity on a glass brain
        
        Parameters:
        -----------
        connectome : np.ndarray
            Connectivity matrix
        threshold : float
            Threshold for displaying connections (0-1)
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
        """
        # Get coordinates of atlas regions
        coords = plotting.find_parcellation_cut_coords(labels_img=self.atlas_img)
        
        # Plot on glass brain
        fig = plotting.plot_connectome(
            connectome,
            coords,
            edge_threshold=f"{int(threshold*100)}%",
            title=title,
            display_mode='lyrz',
            colorbar=True
        )
        
        if save_path:
            fig.savefig(save_path, dpi=300)
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def get_n_features(self) -> int:
        """
        Get the number of features in flattened connectome
        
        Returns:
        --------
        int : Number of features (n_regions * n_regions)
        """
        n_regions = len(self.atlas_labels)
        return n_regions * n_regions


class TrialBasedExtractor:
    """
    Extracts connectomes for individual trials/events within a run
    """
    
    def __init__(self, connectome_extractor: ConnectomeExtractor):
        """
        Initialize trial-based extractor
        
        Parameters:
        -----------
        connectome_extractor : ConnectomeExtractor
            Initialized connectome extractor
        """
        self.extractor = connectome_extractor
    
    def extract_trial_connectomes(self, 
                                  fmri_img: nib.Nifti1Image,
                                  trial_volumes: List[Tuple[int, int, str]]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract connectomes for individual trials
        
        Parameters:
        -----------
        fmri_img : nibabel.Nifti1Image
            Full 4D fMRI image
        trial_volumes : list
            List of (start_vol, end_vol, emotion_label) tuples
            
        Returns:
        --------
        tuple : (connectomes array, emotion labels list)
        """
        from nilearn import image
        
        connectomes = []
        labels = []
        
        print(f"\n🎯 Extracting {len(trial_volumes)} trial connectomes...")
        
        for i, (start, end, emotion) in enumerate(trial_volumes):
            # Extract volumes for this trial
            trial_img = image.index_img(fmri_img, slice(start, end))
            
            # Skip if too few volumes
            if trial_img.shape[-1] < 3:
                print(f"   ⚠️  Trial {i+1}: Too few volumes ({trial_img.shape[-1]}), skipping")
                continue
            
            # Extract connectome
            try:
                connectome = self.extractor.extract_connectome_from_image(trial_img)
                connectomes.append(connectome)
                labels.append(emotion)
                print(f"   ✓ Trial {i+1}/{len(trial_volumes)}: {emotion}")
            except Exception as e:
                print(f"   ⚠️  Trial {i+1}: Error - {e}")
                continue
        
        return np.array(connectomes), labels


if __name__ == "__main__":
    # Example usage
    from data_loader import FMRIDataLoader
    
    # Load data
    dataset_path = r"c:\Users\Hp\Documents\emotionDetectionFmri\ds003477"
    loader = FMRIDataLoader(dataset_path, task="face")
    
    subject = "sub-03"
    session = "ses-1"
    run = "5"
    
    bold_img = loader.load_bold(subject, session, run)
    events_df = loader.load_events(subject, session, run)
    
    # Initialize extractor
    extractor = ConnectomeExtractor(atlas_name='harvard_oxford')
    
    # Extract connectome from full run
    print("\n" + "="*60)
    print("FULL RUN CONNECTOME")
    print("="*60)
    connectome = extractor.extract_connectome_from_image(bold_img)
    
    # Visualize
    extractor.visualize_connectome(connectome, title="Full Run Connectivity")
    
    # Extract trial-based connectomes
    print("\n" + "="*60)
    print("TRIAL-BASED CONNECTOMES")
    print("="*60)
    trial_volumes = loader.get_trial_volumes(events_df)
    
    trial_extractor = TrialBasedExtractor(extractor)
    trial_connectomes, trial_labels = trial_extractor.extract_trial_connectomes(
        bold_img, trial_volumes
    )
    
    print(f"\n✅ Extracted {len(trial_connectomes)} trial connectomes")
    print(f"   Shape: {trial_connectomes.shape}")
    print(f"   Labels: {set(trial_labels)}")
