"""
PyTorch Dataset implementation for WiFi CSI-based HAR
Provides efficient data loading with proper transformations
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pathlib import Path
from typing import Tuple, List, Optional
import logging

from config import Config


logger = logging.getLogger(__name__)


class WiFiCSIDataset(Dataset):
    """
    PyTorch Dataset for WiFi CSI windowed data
    """
    
    def __init__(
        self,
        labels_csv: Path,
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset
        
        Args:
            labels_csv: Path to labels CSV file
            transform: Optional transform to apply to data
        """
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform
        
        # Create label encodings
        self.activity_to_idx = Config.ACTIVITY_TO_IDX
        self.subject_to_idx = Config.SUBJECT_TO_IDX
        
        logger.info(f"Loaded dataset with {len(self.labels_df)} samples")
    
    def __len__(self) -> int:
        return len(self.labels_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (data, activity_label, subject_label)
        """
        # Get file path and labels
        row = self.labels_df.iloc[idx]
        file_path = row['file']
        activity = row['activity']
        subject = row['subject']
        
        # Load data
        data = np.load(file_path)  # Shape: [window_size, antennas, subcarriers, packets]
        
        # Transpose to [antennas, window_size, subcarriers, packets]
        data = np.transpose(data, (1, 0, 2, 3))
        
        # Apply transform if provided
        if self.transform is not None:
            data = self.transform(data)
        
        # Convert to tensors
        data = torch.from_numpy(data).float()
        activity_label = torch.tensor(self.activity_to_idx[activity], dtype=torch.long)
        subject_label = torch.tensor(self.subject_to_idx[subject], dtype=torch.long)
        
        return data, activity_label, subject_label
    
    def get_subject_indices(self, subject: str) -> List[int]:
        """Get all indices for a specific subject"""
        return self.labels_df[self.labels_df['subject'] == subject].index.tolist()
    
    def get_subjects_indices(self, subjects: List[str]) -> List[int]:
        """Get all indices for multiple subjects"""
        return self.labels_df[self.labels_df['subject'].isin(subjects)].index.tolist()


class DataAugmentation:
    """
    Data augmentation for WiFi CSI data
    """
    
    @staticmethod
    def add_gaussian_noise(data: np.ndarray, std: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to data"""
        noise = np.random.normal(0, std, data.shape)
        return data + noise
    
    @staticmethod
    def time_shift(data: np.ndarray, max_shift: int = 10) -> np.ndarray:
        """Apply random time shift"""
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(data, shift, axis=1)
    
    @staticmethod
    def amplitude_scaling(data: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Apply random amplitude scaling"""
        scale = np.random.uniform(*scale_range)
        return data * scale
    
    @staticmethod
    def random_augment(data: np.ndarray, prob: float = 0.5) -> np.ndarray:
        """Apply random augmentations with probability"""
        if np.random.random() < prob:
            data = DataAugmentation.add_gaussian_noise(data)
        if np.random.random() < prob:
            data = DataAugmentation.time_shift(data)
        if np.random.random() < prob:
            data = DataAugmentation.amplitude_scaling(data)
        return data


def create_data_loaders(
    labels_csv: Path,
    batch_size: int = Config.BATCH_SIZE,
    val_split: float = Config.VAL_SPLIT,
    test_subjects: Optional[List[str]] = None,
    augment_train: bool = False,
    seed: int = Config.RANDOM_SEED
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        labels_csv: Path to labels CSV
        batch_size: Batch size
        val_split: Validation split ratio
        test_subjects: List of subjects for test set (for LOSO-CV)
        augment_train: Whether to apply augmentation to training data
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create transform for training data
    train_transform = DataAugmentation.random_augment if augment_train else None
    
    # Create dataset
    dataset = WiFiCSIDataset(labels_csv, transform=None)
    
    if test_subjects is not None:
        # Use subject-wise split (for LOSO-CV)
        test_indices = dataset.get_subjects_indices(test_subjects)
        train_val_subjects = [s for s in Config.SUBJECTS if s not in test_subjects]
        train_val_indices = dataset.get_subjects_indices(train_val_subjects)
        
        # Split train_val into train and val
        np.random.shuffle(train_val_indices)
        val_size = int(len(train_val_indices) * val_split)
        train_indices = train_val_indices[val_size:]
        val_indices = train_val_indices[:val_size]
        
        logger.info(f"Subject-wise split: Train subjects={train_val_subjects}, Test subjects={test_subjects}")
    else:
        # Random split
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        
        test_size = int(len(indices) * 0.2)
        val_size = int(len(indices) * val_split)
        
        test_indices = indices[:test_size]
        val_indices = indices[test_size:test_size+val_size]
        train_indices = indices[test_size+val_size:]
    
    logger.info(f"Data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_indices),
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader


def create_loso_data_loaders(
    labels_csv: Path,
    test_subject: str,
    batch_size: int = Config.BATCH_SIZE,
    val_split: float = Config.VAL_SPLIT,
    seed: int = Config.RANDOM_SEED
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for Leave-One-Subject-Out cross-validation
    
    Args:
        labels_csv: Path to labels CSV
        test_subject: Subject to use for testing
        batch_size: Batch size
        val_split: Validation split ratio from training subjects
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    return create_data_loaders(
        labels_csv=labels_csv,
        batch_size=batch_size,
        val_split=val_split,
        test_subjects=[test_subject],
        augment_train=False,
        seed=seed
    )


if __name__ == '__main__':
    # Test dataset loading
    labels_csv = Config.PROCESSED_DATA_ROOT / 'labels.csv'
    
    if labels_csv.exists():
        print("Testing dataset loading...")
        
        # Create dataset
        dataset = WiFiCSIDataset(labels_csv)
        print(f"Dataset size: {len(dataset)}")
        
        # Get a sample
        data, activity, subject = dataset[0]
        print(f"Sample shape: {data.shape}")
        print(f"Activity label: {activity}, Subject label: {subject}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(labels_csv)
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test LOSO split
        train_loader, val_loader, test_loader = create_loso_data_loaders(
            labels_csv, test_subject='S10'
        )
        print(f"\nLOSO (test=S10):")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
    else:
        print(f"Labels file not found: {labels_csv}")
        print("Run preprocessing first!")