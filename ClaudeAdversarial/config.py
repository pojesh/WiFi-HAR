"""
Configuration file for WiFi CSI-based Human Activity Recognition
Centralized configuration management for reproducibility
"""

import os
from pathlib import Path

class Config:
    """Configuration class for the entire project"""
    
    # ==================== Path Configuration ====================
    # Raw data directory (adjust this to your MMFi dataset location)
    RAW_DATA_ROOT = Path('C:/Users/Pojesh/Documents/project1/working/mmfi/E01')
    
    # Processed data directory
    PROCESSED_DATA_ROOT = Path('data/processed_mmfi_window')
    
    # Results directory
    RESULTS_ROOT = Path('results')
    
    # Model checkpoint directory
    CHECKPOINT_ROOT = Path('checkpoints')
    
    # Logs directory
    LOGS_ROOT = Path('logs')
    
    # ==================== Dataset Configuration ====================
    # MMFi daily activities subset (14 activities)
    DAILY_ACTIVITIES = [
        'A02',  # Chest expansion (horizontal)
        'A03',  # Chest expansion (vertical)
        'A04',  # Twist (left)
        'A05',  # Twist (right)
        'A13',  # Raising hand (left)
        'A14',  # Raising hand (right)
        'A17',  # Waving hand (left)
        'A18',  # Waving hand (right)
        'A19',  # Picking up things
        'A20',  # Throwing (toward left side)
        'A21',  # Throwing (toward right side)
        'A22',  # Kicking (toward left side)
        'A23',  # Kicking (toward right side)
        'A27'   # Bowing
    ]
    
    # Subject IDs (10 subjects from E01)
    SUBJECTS = [f'S{str(i).zfill(2)}' for i in range(1, 11)]
    
    # Activity to index mapping
    ACTIVITY_TO_IDX = {act: idx for idx, act in enumerate(DAILY_ACTIVITIES)}
    
    # Subject to index mapping
    SUBJECT_TO_IDX = {subj: idx for idx, subj in enumerate(SUBJECTS)}
    
    # Number of classes
    NUM_ACTIVITIES = len(DAILY_ACTIVITIES)
    NUM_SUBJECTS = len(SUBJECTS)
    
    # ==================== Preprocessing Configuration ====================
    # Window size for temporal segmentation
    WINDOW_SIZE = 64 #100 
    
    # Step size for sliding window (50% overlap)
    STEP_SIZE = 32 #50 
    
    # CSI data dimensions (from MMFi dataset)
    NUM_ANTENNAS = 3
    NUM_SUBCARRIERS = 114
    NUM_PACKETS = 10  # Third dimension from MMFi CSI data
    
    # Normalization strategy: 'global' or 'per_window'
    NORMALIZATION = 'global'
    
    # Handle missing values
    FILL_NAN_VALUE = 0.0
    
    # ==================== Model Configuration ====================
    # CNN-GRU Base Model
    INPUT_CHANNELS = NUM_ANTENNAS
    CONV2D_CHANNELS = [16, 32]  # Progressive channel increase
    CONV1D_CHANNELS = [128, 256]
    GRU_HIDDEN_SIZE = 128
    GRU_NUM_LAYERS = 1
    
    # Dropout rate
    DROPOUT_RATE = 0.65
    
    # ==================== Training Configuration ====================
    # Basic training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4 #1e-5
    NUM_EPOCHS = 30
    
    # Validation split (for non-LOSO experiments)
    VAL_SPLIT = 0.20
    
    # Early stopping patience
    EARLY_STOPPING_PATIENCE = 10
    
    # Learning rate scheduler
    LR_SCHEDULER = 'ReduceLROnPlateau'
    LR_SCHEDULER_PATIENCE = 3
    LR_SCHEDULER_FACTOR = 0.5
    
    # ==================== Adversarial Training Configuration ====================
    # Domain discriminator configuration
    DOMAIN_HIDDEN_SIZE = 256
    
    # Adversarial loss weight (lambda parameter)
    ADVERSARIAL_LAMBDA = 0.05
    
    # Gradient reversal layer alpha (for DANN)
    GRL_ALPHA = 1.0
    
    # Progressive adversarial training
    # Gradually increase adversarial loss weight
    PROGRESSIVE_ADVERSARIAL = True
    ADVERSARIAL_WARMUP_EPOCHS = 5
    
    # ==================== LOSO-CV Configuration ====================
    # Number of folds for LOSO-CV (equal to number of subjects)
    LOSO_NUM_FOLDS = NUM_SUBJECTS
    
    # ==================== Evaluation Configuration ====================
    # Metrics to compute
    METRICS = ['accuracy', 'precision', 'recall', 'f1']
    
    # Save confusion matrix
    SAVE_CONFUSION_MATRIX = True
    
    # Save per-class metrics
    SAVE_PER_CLASS_METRICS = True
    
    # ==================== Visualization Configuration ====================
    # Figure size for plots
    FIGURE_SIZE = (12, 8)
    
    # DPI for saving figures
    FIGURE_DPI = 300
    
    # Color scheme
    COLOR_SCHEME = 'Blues'
    
    # ==================== Reproducibility Configuration ====================
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # CUDA configuration
    USE_CUDA = True
    CUDA_DEVICE = 0
    
    # Number of workers for data loading
    NUM_WORKERS = 0
    
    # Pin memory for faster data transfer
    PIN_MEMORY = True
    
    # ==================== Logging Configuration ====================
    # Logging level
    LOG_LEVEL = 'INFO'
    
    # Log to file
    LOG_TO_FILE = True
    
    # Tensorboard logging
    USE_TENSORBOARD = True
    
    # ==================== Experiment Configuration ====================
    # Experiment name (for organizing results)
    EXPERIMENT_NAME = 'wifi_har_adversarial'
    
    # Experiment mode: 'base' or 'adversarial'
    EXPERIMENT_MODE = 'adversarial'
    
    # Cross-validation mode: 'standard' or 'loso'
    CV_MODE = 'standard'
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.PROCESSED_DATA_ROOT,
            cls.RESULTS_ROOT,
            cls.CHECKPOINT_ROOT,
            cls.LOGS_ROOT,
            cls.RESULTS_ROOT / cls.EXPERIMENT_NAME,
            cls.CHECKPOINT_ROOT / cls.EXPERIMENT_NAME
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    @classmethod
    def get_device(cls):
        """Get the appropriate device for training"""
        import torch
        if cls.USE_CUDA and torch.cuda.is_available():
            device = torch.device(f'cuda:{cls.CUDA_DEVICE}')
        else:
            device = torch.device('cpu')
        return device
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 80)
        print("Configuration Settings")
        print("=" * 80)
        
        sections = {
            'Paths': ['RAW_DATA_ROOT', 'PROCESSED_DATA_ROOT', 'RESULTS_ROOT'],
            'Dataset': ['NUM_ACTIVITIES', 'NUM_SUBJECTS', 'WINDOW_SIZE', 'STEP_SIZE'],
            'Model': ['INPUT_CHANNELS', 'GRU_HIDDEN_SIZE', 'DROPOUT_RATE'],
            'Training': ['BATCH_SIZE', 'LEARNING_RATE', 'NUM_EPOCHS'],
            'Adversarial': ['ADVERSARIAL_LAMBDA', 'GRL_ALPHA', 'PROGRESSIVE_ADVERSARIAL'],
            'Experiment': ['EXPERIMENT_NAME', 'EXPERIMENT_MODE', 'CV_MODE']
        }
        
        for section, attrs in sections.items():
            print(f"\n{section}:")
            for attr in attrs:
                value = getattr(cls, attr)
                print(f"  {attr}: {value}")
        
        print("=" * 80)


# Create directories on import
Config.create_directories()