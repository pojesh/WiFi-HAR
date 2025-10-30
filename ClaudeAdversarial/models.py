"""
Neural Network Models for WiFi CSI-based HAR
Includes base CNN-GRU model and adversarial domain adaptation model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import logging

from config import Config


logger = logging.getLogger(__name__)


class GradientReversalLayer(Function):
    """
    Gradient Reversal Layer for Domain Adversarial Training
    Forwards input as-is but reverses gradient during backpropagation
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GRL(nn.Module):
    """Gradient Reversal Layer Module"""
    
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)


class CNNGRUBase(nn.Module):
    """
    Base CNN-GRU model for WiFi CSI-based activity recognition
    Architecture:
    1. Conv2D layers to process spatial features (subcarriers × packets)
    2. Conv1D layers to process temporal features
    3. GRU for sequential modeling
    4. Fully connected classifier
    """
    
    def __init__(
        self,
        input_channels: int = Config.INPUT_CHANNELS,
        num_classes: int = Config.NUM_ACTIVITIES,
        conv2d_channels: list = None,
        conv1d_channels: list = None,
        gru_hidden_size: int = Config.GRU_HIDDEN_SIZE,
        gru_num_layers: int = Config.GRU_NUM_LAYERS,
        dropout: float = Config.DROPOUT_RATE
    ):
        super(CNNGRUBase, self).__init__()
        
        if conv2d_channels is None:
            conv2d_channels = Config.CONV2D_CHANNELS
        if conv1d_channels is None:
            conv1d_channels = Config.CONV1D_CHANNELS
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Conv2D processor for spatial features
        # Input: [batch, antennas, window_size, subcarriers, packets]
        self.conv2d_processor = nn.Sequential(
            nn.Conv2d(1, conv2d_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2d_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(conv2d_channels[0], conv2d_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2d_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate conv2d output size
        # Input: (114, 10) -> after 2 pooling: (28, 2)
        conv2d_h = Config.NUM_SUBCARRIERS // 4
        conv2d_w = Config.NUM_PACKETS // 4
        self.conv2d_output_size = conv2d_channels[1] * conv2d_h * conv2d_w
        
        # Conv1D input channels
        conv1d_input_channels = input_channels * self.conv2d_output_size
        
        # Conv1D encoder for temporal features
        self.conv1d_encoder = nn.Sequential(
            nn.Conv1d(conv1d_input_channels, conv1d_channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(conv1d_channels[0]),
            nn.ReLU(),
            
            nn.Conv1d(conv1d_channels[0], conv1d_channels[1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(conv1d_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # GRU for sequential modeling
        self.gru = nn.GRU(
            input_size=conv1d_channels[1],
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_size, num_classes)
        )
        
        logger.info(f"Initialized CNNGRUBase: Conv2D out={self.conv2d_output_size}, "
                   f"Conv1D in={conv1d_input_channels}, GRU hidden={gru_hidden_size}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, antennas, window_size, subcarriers, packets]
            
        Returns:
            Activity predictions [batch, num_classes]
        """
        batch_size, antennas, window_size, subcarriers, packets = x.shape
        
        # Reshape for Conv2D: [batch * antennas * window_size, 1, subcarriers, packets]
        x = x.view(batch_size * antennas * window_size, 1, subcarriers, packets)
        
        # Apply Conv2D
        x = self.conv2d_processor(x)
        
        # Reshape: [batch, antennas, window_size, conv2d_output_size]
        x = x.view(batch_size, antennas, window_size, -1)
        
        # Permute and reshape for Conv1D: [batch, antennas * conv2d_output_size, window_size]
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, -1, window_size)
        
        # Apply Conv1D
        x = self.conv1d_encoder(x)
        
        # Permute for GRU: [batch, seq_len, features]
        x = x.permute(0, 2, 1)
        
        # Apply GRU
        _, h = self.gru(x)
        
        # Use last hidden state
        features = h[-1]
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def extract_features(self, x):
        """
        Extract features before classification layer
        Useful for adversarial training
        
        Args:
            x: Input tensor
            
        Returns:
            Feature vector before classification
        """
        batch_size, antennas, window_size, subcarriers, packets = x.shape
        
        # Conv2D processing
        x = x.view(batch_size * antennas * window_size, 1, subcarriers, packets)
        x = self.conv2d_processor(x)
        x = x.view(batch_size, antennas, window_size, -1)
        
        # Conv1D processing
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, -1, window_size)
        x = self.conv1d_encoder(x)
        
        # GRU processing
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)
        features = h[-1]
        
        return features


class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator for Adversarial Domain Adaptation
    Tries to predict which subject the sample comes from
    """
    
    def __init__(
        self,
        input_size: int = Config.GRU_HIDDEN_SIZE,
        hidden_size: int = Config.DOMAIN_HIDDEN_SIZE,
        num_domains: int = Config.NUM_SUBJECTS,
        dropout: float = Config.DROPOUT_RATE
    ):
        super(DomainDiscriminator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, num_domains)
        )
        
        logger.info(f"Initialized DomainDiscriminator: input={input_size}, "
                   f"hidden={hidden_size}, domains={num_domains}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Feature vector [batch, feature_size]
            
        Returns:
            Domain predictions [batch, num_domains]
        """
        return self.network(x)


class AdversarialDANN(nn.Module):
    """
    Domain Adversarial Neural Network (DANN) for subject bias mitigation
    
    Architecture from research: 
    Feature extractor learns domain-invariant features by minimizing 
    distribution discrepancy between source and target domains using 
    adversarial training with gradient reversal layer
    
    """
    
    def __init__(
        self,
        input_channels: int = Config.INPUT_CHANNELS,
        num_activities: int = Config.NUM_ACTIVITIES,
        num_subjects: int = Config.NUM_SUBJECTS,
        gru_hidden_size: int = Config.GRU_HIDDEN_SIZE,
        domain_hidden_size: int = Config.DOMAIN_HIDDEN_SIZE,
        alpha: float = Config.GRL_ALPHA
    ):
        super(AdversarialDANN, self).__init__()
        
        # Feature extractor (shared)
        self.feature_extractor = CNNGRUBase(
            input_channels=input_channels,
            num_classes=num_activities  # Will be replaced with separate classifier
        )
        
        # Remove the original classifier
        self.feature_extractor.classifier = nn.Identity()
        
        # Activity classifier
        self.activity_classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(gru_hidden_size, num_activities)
        )
        
        # Gradient reversal layer
        self.grl = GRL(alpha=alpha)
        
        # Domain discriminator
        self.domain_discriminator = DomainDiscriminator(
            input_size=gru_hidden_size,
            hidden_size=domain_hidden_size,
            num_domains=num_subjects
        )
        
        logger.info("Initialized AdversarialDANN with GRL")
    
    def forward(self, x, alpha=None):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, antennas, window_size, subcarriers, packets]
            alpha: Alpha value for GRL (if None, uses default)
            
        Returns:
            Tuple of (activity_predictions, domain_predictions)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Activity classification
        activity_output = self.activity_classifier(features)
        
        # Domain classification with gradient reversal
        if alpha is not None:
            self.grl.alpha = alpha
        
        reversed_features = self.grl(features)
        domain_output = self.domain_discriminator(reversed_features)
        
        return activity_output, domain_output
    
    def set_alpha(self, alpha: float):
        """Set alpha value for gradient reversal"""
        self.grl.alpha = alpha
    
    def predict_activity(self, x):
        """Predict only activity (for inference)"""
        features = self.feature_extractor(x)
        return self.activity_classifier(features)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models
    print("Testing models...")
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(
        batch_size,
        Config.INPUT_CHANNELS,
        Config.WINDOW_SIZE,
        Config.NUM_SUBCARRIERS,
        Config.NUM_PACKETS
    )
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Test base model
    print("\n" + "="*80)
    print("Testing CNNGRUBase...")
    base_model = CNNGRUBase()
    output = base_model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {count_parameters(base_model):,}")
    
    # Test feature extraction
    features = base_model.extract_features(dummy_input)
    print(f"Features shape: {features.shape}")
    
    # Test adversarial model
    print("\n" + "="*80)
    print("Testing AdversarialDANN...")
    adv_model = AdversarialDANN()
    activity_output, domain_output = adv_model(dummy_input)
    print(f"Activity output shape: {activity_output.shape}")
    print(f"Domain output shape: {domain_output.shape}")
    print(f"Parameters: {count_parameters(adv_model):,}")
    
    print("\n" + "="*80)
    print("✓ All models working correctly!")