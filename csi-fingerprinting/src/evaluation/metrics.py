import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from scipy.stats import entropy
from typing import Dict, Tuple

class SubjectMetrics:
    """Metrics for evaluating discovered subjects"""
    
    def __init__(self, num_subjects: int):
        self.num_subjects = num_subjects
    
    def compute(self,
                signatures: np.ndarray,
                subject_assignments: np.ndarray,
                activity_labels: np.ndarray,
                activity_preds: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive metrics for subject discovery
        
        Args:
            signatures: Subject signatures (N, D)
            subject_assignments: Discovered subject labels (N,)
            activity_labels: True activity labels (N,)
            activity_preds: Predicted activity labels (N,)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Clustering metrics
        if len(np.unique(subject_assignments)) > 1:
            metrics['silhouette_score'] = silhouette_score(signatures, subject_assignments)
            metrics['davies_bouldin'] = davies_bouldin_score(signatures, subject_assignments)
        else:
            metrics['silhouette_score'] = 0
            metrics['davies_bouldin'] = float('inf')
        
        # Subject consistency across activities
        metrics['subject_activity_independence'] = self._compute_independence(
            subject_assignments, activity_labels
        )
        
        # Temporal consistency (subjects should be consistent over time)
        metrics['temporal_consistency'] = self._compute_temporal_consistency(
            subject_assignments
        )
        
        # Subject distribution entropy (are all subjects used?)
        subject_dist = np.bincount(subject_assignments, minlength=self.num_subjects)
        subject_dist = subject_dist / subject_dist.sum()
        metrics['subject_entropy'] = entropy(subject_dist)
        metrics['subject_entropy_normalized'] = entropy(subject_dist) / np.log(self.num_subjects)
        
        # Activity recognition metrics
        metrics['activity_accuracy'] = accuracy_score(activity_labels, activity_preds)
        metrics['activity_f1'] = f1_score(activity_labels, activity_preds, average='macro')
        
        return metrics
    
    def _compute_independence(self, subjects, activities):
        """Measure independence between subjects and activities"""
        # Use normalized mutual information (lower is better for independence)
        nmi = normalized_mutual_info_score(subjects, activities)
        return 1 - nmi  # Convert so higher is better
    
    def _compute_temporal_consistency(self, subjects, window=10):
        """Measure how consistent subject assignments are over time"""
        consistency_scores = []
        
        for i in range(len(subjects) - window):
            window_subjects = subjects[i:i+window]
            # Count most common subject in window
            unique, counts = np.unique(window_subjects, return_counts=True)
            max_count = counts.max()
            consistency = max_count / window
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0