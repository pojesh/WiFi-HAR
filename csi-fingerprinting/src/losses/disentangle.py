import torch
import torch.nn as nn
import torch.nn.functional as F

class DisentanglementLoss(nn.Module):
    """
    Loss to disentangle activity and subject features
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        
    def forward(self,
                activity_features: torch.Tensor,
                subject_signatures: torch.Tensor,
                disentangle_scores: torch.Tensor) -> torch.Tensor:
        """
        Encourage orthogonality between activity and subject features
        
        Args:
            activity_features: Activity-specific features (B, D1)
            subject_signatures: Subject-specific signatures (B, D2)
            disentangle_scores: Predicted disentanglement (B, 2)
            
        Returns:
            Disentanglement loss
        """
        B = activity_features.shape[0]
        
        # Normalize features
        activity_norm = F.normalize(activity_features, p=2, dim=1)
        subject_norm = F.normalize(subject_signatures, p=2, dim=1)
        
        # Compute correlation matrix
        if activity_norm.shape[1] == subject_norm.shape[1]:
            correlation = torch.abs(torch.matmul(activity_norm, subject_norm.T))
            ortho_loss = correlation.mean()
        else:
            # Use projection if dimensions don't match
            proj_dim = min(activity_norm.shape[1], subject_norm.shape[1])
            activity_proj = activity_norm[:, :proj_dim]
            subject_proj = subject_norm[:, :proj_dim]
            correlation = torch.abs((activity_proj * subject_proj).sum(dim=1))
            ortho_loss = correlation.mean()
        
        # Disentanglement prediction loss
        target = torch.tensor([[1.0, 0.0]], device=disentangle_scores.device).repeat(B, 1)
        disentangle_loss = F.mse_loss(F.softmax(disentangle_scores, dim=1), target)
        
        return ortho_loss + self.beta * disentangle_loss