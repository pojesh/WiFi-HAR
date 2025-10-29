import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class TemporalContrastiveLoss(nn.Module):
    """
    Contrastive loss that uses temporal proximity as weak supervision
    for subject discovery
    """
    
    def __init__(self, temperature: float = 0.1, temporal_window: int = 10):
        super().__init__()
        self.temperature = temperature
        self.temporal_window = temporal_window
        
    def forward(self, 
                signatures: torch.Tensor, 
                indices: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute temporal contrastive loss
        
        Args:
            signatures: Subject signatures (B, D)
            indices: Sample indices in dataset (B,)
            mask: Optional mask for valid pairs
            
        Returns:
            Contrastive loss value
        """
        B = signatures.shape[0]
        device = signatures.device
        
        # Normalize signatures
        signatures = F.normalize(signatures, p=2, dim=1)
        
        # Compute all pairwise similarities
        sim_matrix = torch.matmul(signatures, signatures.T) / self.temperature
        
        # Create temporal proximity mask
        index_diff = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0))
        temporal_mask = (index_diff > 0) & (index_diff <= self.temporal_window)
        
        # Mask out self-similarities
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        temporal_mask = temporal_mask & ~self_mask
        
        if mask is not None:
            temporal_mask = temporal_mask & mask
        
        # Compute loss for each sample
        loss = 0
        valid_samples = 0
        
        for i in range(B):
            pos_mask_i = temporal_mask[i]
            neg_mask_i = ~temporal_mask[i] & ~self_mask[i]
            
            if pos_mask_i.sum() > 0 and neg_mask_i.sum() > 0:
                # Positive similarities (temporally close)
                pos_sim = sim_matrix[i][pos_mask_i]
                
                # Negative similarities (temporally distant)
                neg_sim = sim_matrix[i][neg_mask_i]
                
                # InfoNCE loss
                pos_exp = torch.exp(pos_sim)
                neg_exp = torch.exp(neg_sim).sum()
                
                sample_loss = -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp + 1e-8))
                loss += sample_loss
                valid_samples += 1
        
        if valid_samples > 0:
            loss = loss / valid_samples

        else:
            loss = torch.tensor(0., device=device, requires_grad=False)
        
        return loss

class SubjectConsistencyLoss(nn.Module):
    """
    Ensures discovered subjects are consistent across different activities
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self,
                subject_logits: torch.Tensor,
                activity_labels: torch.Tensor) -> torch.Tensor:
        """
        Encourage subject assignments to be independent of activity
        
        Args:
            subject_logits: Subject classification logits (B, K)
            activity_labels: Activity labels (B,)
            
        Returns:
            Consistency loss
        """
        # Group by activity
        unique_activities = torch.unique(activity_labels)
        
        if len(unique_activities) < 2:
            return torch.tensor(0.0, device=subject_logits.device)
        
        # Compute subject distributions per activity
        subject_probs = F.softmax(subject_logits, dim=1)
        
        activity_distributions = []
        for activity in unique_activities:
            mask = activity_labels == activity
            if mask.sum() > 0:
                activity_dist = subject_probs[mask].mean(dim=0)
                activity_distributions.append(activity_dist)
        
        if len(activity_distributions) < 2:
            return torch.tensor(0.0, device=subject_logits.device)
        
        # Compute KL divergence between activity distributions
        loss = 0
        count = 0
        for i in range(len(activity_distributions)):
            for j in range(i+1, len(activity_distributions)):
                kl = F.kl_div(
                    activity_distributions[i].log(),
                    activity_distributions[j],
                    reduction='sum'
                )
                loss += kl
                count += 1
        
        return loss / max(count, 1)