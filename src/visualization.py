"""
Visualization Utilities for fMRI Emotion Detection
Advanced plotting functions for results and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path


class ResultsVisualizer:
    """
    Comprehensive visualization tools for fMRI ML results
    """
    
    def __init__(self, output_dir: str = "results/figures"):
        """
        Initialize visualizer
        
        Parameters:
        -----------
        output_dir : str
            Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_data_distribution(self, 
                               labels: List[str],
                               title: str = "Emotion Distribution",
                               save_name: Optional[str] = None):
        """
        Plot distribution of emotion labels
        
        Parameters:
        -----------
        labels : list
            List of emotion labels
        title : str
            Plot title
        save_name : str, optional
            Filename to save
        """
        from collections import Counter
        
        label_counts = Counter(labels)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = sns.color_palette("Set2", len(label_counts))
        bars = ax.bar(label_counts.keys(), label_counts.values(), color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Emotion', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_connectome_comparison(self,
                                   connectomes: np.ndarray,
                                   labels: List[str],
                                   n_samples: int = 3,
                                   save_name: Optional[str] = None):
        """
        Plot multiple connectomes side by side for comparison
        
        Parameters:
        -----------
        connectomes : np.ndarray
            Array of connectivity matrices
        labels : list
            Corresponding labels
        n_samples : int
            Number of samples per class to show
        save_name : str, optional
            Filename to save
        """
        unique_labels = list(set(labels))
        
        fig, axes = plt.subplots(len(unique_labels), n_samples, 
                                figsize=(4*n_samples, 4*len(unique_labels)))
        
        if len(unique_labels) == 1:
            axes = axes.reshape(1, -1)
        
        for i, label in enumerate(unique_labels):
            # Get indices for this label
            label_indices = [j for j, l in enumerate(labels) if l == label]
            sample_indices = label_indices[:n_samples]
            
            for j, idx in enumerate(sample_indices):
                ax = axes[i, j] if len(unique_labels) > 1 else axes[j]
                
                im = ax.imshow(connectomes[idx], cmap='RdBu_r', vmin=-1, vmax=1)
                ax.set_title(f"{label} - Sample {j+1}", fontsize=12, fontweight='bold')
                ax.axis('off')
        
        # Add colorbar
        fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label='Correlation')
        
        plt.suptitle('Connectivity Matrix Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self,
                             results: Dict,
                             metric: str = 'accuracy',
                             title: Optional[str] = None,
                             save_name: Optional[str] = None):
        """
        Compare multiple models on a given metric
        
        Parameters:
        -----------
        results : dict
            Dictionary of model results
        metric : str
            Metric to compare ('accuracy', 'f1_score', 'roc_auc')
        title : str, optional
            Plot title
        save_name : str, optional
            Filename to save
        """
        model_names = []
        scores = []
        model_types = []
        
        for model_name, model_results in results.items():
            if metric in model_results:
                model_names.append(model_name.replace('_', ' ').title())
                scores.append(model_results[metric])
                
                # Determine model type
                if 'cnn' in model_name.lower() or 'deep' in model_name.lower():
                    model_types.append('Deep Learning')
                else:
                    model_types.append('Classical ML')
        
        # Create dataframe
        df = pd.DataFrame({
            'Model': model_names,
            'Score': scores,
            'Type': model_types
        })
        
        # Sort by score
        df = df.sort_values('Score', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.barplot(data=df, x='Model', y='Score', hue='Type', ax=ax, palette='Set2')
        
        # Add value labels
        for i, (idx, row) in enumerate(df.iterrows()):
            ax.text(i, row['Score'] + 0.02, f"{row['Score']:.3f}",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        if title is None:
            title = f"Model Comparison - {metric.replace('_', ' ').title()}"
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Model', fontsize=14)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        ax.legend(title='Model Type', fontsize=12)
        plt.xticks(rotation=15, ha='right')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self,
                               feature_importance: np.ndarray,
                               n_regions: int,
                               top_k: int = 20,
                               title: str = "Top Important Connections",
                               save_name: Optional[str] = None):
        """
        Plot feature importance for classical ML models
        
        Parameters:
        -----------
        feature_importance : np.ndarray
            Feature importance scores (flattened)
        n_regions : int
            Number of brain regions
        top_k : int
            Number of top features to show
        title : str
            Plot title
        save_name : str, optional
            Filename to save
        """
        # Reshape to matrix
        importance_matrix = feature_importance.reshape(n_regions, n_regions)
        
        # Get top k connections
        flat_indices = np.argsort(np.abs(feature_importance))[-top_k:]
        row_indices = flat_indices // n_regions
        col_indices = flat_indices % n_regions
        
        # Create labels
        connection_labels = [f"R{r+1}-R{c+1}" for r, c in zip(row_indices, col_indices)]
        importance_values = feature_importance[flat_indices]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        colors = ['#e74c3c' if v < 0 else '#3498db' for v in importance_values]
        ax1.barh(range(top_k), importance_values, color=colors, alpha=0.7)
        ax1.set_yticks(range(top_k))
        ax1.set_yticklabels(connection_labels)
        ax1.set_xlabel('Importance', fontsize=12)
        ax1.set_title('Top Important Connections', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Heatmap
        im = ax2.imshow(importance_matrix, cmap='RdBu_r', aspect='auto')
        ax2.set_title('Feature Importance Matrix', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Brain Region', fontsize=12)
        ax2.set_ylabel('Brain Region', fontsize=12)
        plt.colorbar(im, ax=ax2, label='Importance')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curves(self,
                            train_scores: List[float],
                            val_scores: List[float],
                            metric_name: str = "Accuracy",
                            save_name: Optional[str] = None):
        """
        Plot learning curves for model training
        
        Parameters:
        -----------
        train_scores : list
            Training scores per epoch
        val_scores : list
            Validation scores per epoch
        metric_name : str
            Name of the metric
        save_name : str, optional
            Filename to save
        """
        epochs = range(1, len(train_scores) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, train_scores, 'o-', label='Training', linewidth=2, markersize=4)
        ax.plot(epochs, val_scores, 's-', label='Validation', linewidth=2, markersize=4)
        
        ax.set_title(f'Learning Curves - {metric_name}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel(metric_name, fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add best validation score
        best_val = max(val_scores)
        best_epoch = val_scores.index(best_val) + 1
        ax.axhline(y=best_val, color='r', linestyle='--', alpha=0.5)
        ax.text(len(epochs)*0.7, best_val + 0.02, 
               f'Best: {best_val:.4f} (Epoch {best_epoch})',
               fontsize=10, color='r')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()
    
    def create_results_dashboard(self,
                                results: Dict,
                                connectomes: np.ndarray,
                                labels: List[str],
                                save_name: str = "results_dashboard.png"):
        """
        Create a comprehensive dashboard with all key visualizations
        
        Parameters:
        -----------
        results : dict
            Dictionary of all model results
        connectomes : np.ndarray
            Connectivity matrices
        labels : list
            Emotion labels
        save_name : str
            Filename to save
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Data distribution
        ax1 = fig.add_subplot(gs[0, 0])
        from collections import Counter
        label_counts = Counter(labels)
        ax1.bar(label_counts.keys(), label_counts.values(), color=['#3498db', '#e74c3c'])
        ax1.set_title('Data Distribution', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Sample connectomes
        for i in range(2):
            ax = fig.add_subplot(gs[0, i+1])
            ax.imshow(connectomes[i], cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title(f'Sample {i+1}: {labels[i]}', fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # 3. Model comparison
        ax3 = fig.add_subplot(gs[1, :])
        model_names = []
        accuracies = []
        for model, res in results.items():
            if 'accuracy' in res:
                model_names.append(model.replace('_', ' ').title())
                accuracies.append(res['accuracy'])
        
        bars = ax3.bar(model_names, accuracies, color=sns.color_palette("Set2", len(model_names)))
        ax3.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_ylim([0, 1])
        ax3.grid(axis='y', alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
        
        # 4. Best model details (if available)
        if results:
            best_model = max(results, key=lambda k: results[k].get('accuracy', 0))
            
            ax4 = fig.add_subplot(gs[2, :])
            ax4.axis('off')
            
            summary_text = f"""
            📊 RESULTS SUMMARY
            
            Dataset: {len(connectomes)} samples
            Classes: {set(labels)}
            Best Model: {best_model.upper()}
            Best Accuracy: {results[best_model].get('accuracy', 0):.4f}
            
            All Models:
            """
            
            for model, res in results.items():
                if 'accuracy' in res:
                    summary_text += f"\n   • {model.upper()}: {res['accuracy']:.4f}"
            
            ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center')
        
        plt.suptitle('🧠 fMRI Emotion Detection - Results Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Dashboard saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("Visualization Utilities Demo")
    
    # Create dummy data
    n_samples = 100
    n_regions = 48
    
    connectomes = np.random.randn(n_samples, n_regions, n_regions)
    labels = ['neutral'] * 50 + ['smiling'] * 50
    
    results = {
        'svm': {'accuracy': 0.75, 'f1_score': 0.74},
        'random_forest': {'accuracy': 0.68, 'f1_score': 0.67},
        'cnn': {'accuracy': 0.78, 'f1_score': 0.77}
    }
    
    # Initialize visualizer
    viz = ResultsVisualizer(output_dir="results/figures")
    
    # Create visualizations
    viz.plot_data_distribution(labels)
    viz.plot_connectome_comparison(connectomes, labels, n_samples=3)
    viz.plot_model_comparison(results, metric='accuracy')
    viz.create_results_dashboard(results, connectomes, labels)
