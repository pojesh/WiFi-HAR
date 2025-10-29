import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Optional, Dict

class TemporalConvBlock(nn.Module):
    """Temporal convolution block for CSI processing"""
    
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                     stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        return self.conv(x)

class CSISignatureNet(nn.Module):
    """
    Neural network for discovering hidden subjects in CSI data
    while maintaining activity recognition capability
    """
    
    def __init__(self, 
                 input_dim: int = 90,
                 seq_len: int = 250,
                 num_activities: int = 7,
                 latent_subjects: int = 5,
                 hidden_dim: int = 256,
                 signature_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_activities = num_activities
        self.latent_subjects = latent_subjects
        self.signature_dim = signature_dim
        
        # Shared CSI feature extractor
        self.feature_extractor = nn.Sequential(
            TemporalConvBlock(input_dim, 128, kernel_size=7, stride=2),
            TemporalConvBlock(128, 256, kernel_size=5, stride=2),
            TemporalConvBlock(256, 256, kernel_size=3, stride=1),
            nn.AdaptiveAvgPool1d(16)
        )
        
        feature_dim = 256 * 16
        
        # Activity recognition branch
        self.activity_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.activity_classifier = nn.Linear(hidden_dim, num_activities)
        
        # Subject signature branch
        self.signature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, signature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(signature_dim, signature_dim)
        )
        
        # Learnable subject prototypes
        self.subject_prototypes = nn.Parameter(
            torch.randn(latent_subjects, signature_dim)
        )
        nn.init.xavier_uniform_(self.subject_prototypes)
        
        # Attention mechanism for prototype matching
        self.prototype_attention = nn.MultiheadAttention(
            signature_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Disentanglement network
        self.disentangle_mlp = nn.Sequential(
            nn.Linear(signature_dim + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)  # [activity_relevant, subject_relevant]
        )
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from CSI data"""
        # x: (B, 250, 90)
        x = x.transpose(1, 2)  # (B, 90, 250)
        features = self.feature_extractor(x)  # (B, 256, 16)
        features = features.flatten(1)  # (B, 256*16)
        return features
    
    def forward(self, 
                x: torch.Tensor, 
                return_all: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: CSI data (B, 250, 90)
            return_all: Return all intermediate outputs
            
        Returns:
            Dictionary with model outputs
        """
        B = x.shape[0]
        
        # Extract shared features
        features = self.extract_features(x)
        
        # Activity branch
        activity_features = self.activity_encoder(features)
        activity_logits = self.activity_classifier(activity_features)
        
        # Subject signature branch  
        signature = self.signature_encoder(features)
        signature_norm = F.normalize(signature, p=2, dim=1)
        
        # Prototype matching with attention
        prototypes_norm = F.normalize(self.subject_prototypes, p=2, dim=1)
        
        # Compute similarity to prototypes
        subject_similarities = torch.matmul(signature_norm, prototypes_norm.T)
        subject_logits = subject_similarities / 0.1  # temperature scaling
        
        # Attention-based prototype aggregation
        signature_expanded = signature_norm.unsqueeze(1)  # (B, 1, D)
        prototypes_expanded = prototypes_norm.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)
        
        attended_prototypes, attention_weights = self.prototype_attention(
            signature_expanded, 
            prototypes_expanded, 
            prototypes_expanded
        )
        attended_prototypes = attended_prototypes.squeeze(1)  # (B, D)
        
        outputs = {
            'activity_logits': activity_logits,
            'activity_features': activity_features,
            'signature': signature_norm,
            'subject_logits': subject_logits,
            'attended_prototypes': attended_prototypes,
            'attention_weights': attention_weights.squeeze(1)
        }
        
        if return_all:
            # Disentanglement scores
            combined = torch.cat([signature, activity_features], dim=1)
            disentangle_scores = self.disentangle_mlp(combined)
            outputs['disentangle_scores'] = disentangle_scores
            outputs['prototypes'] = prototypes_norm
            
        return outputs
    
    def get_subject_assignment(self, x: torch.Tensor) -> torch.Tensor:
        """Get hard subject assignments"""
        outputs = self.forward(x)
        return torch.argmax(outputs['subject_logits'], dim=1)