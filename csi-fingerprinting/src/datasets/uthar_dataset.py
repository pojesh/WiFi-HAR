import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict

class UTHARDataset(Dataset):
    """UT-HAR Dataset loader with support for subject discovery"""
    
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',
                 transform: Optional[callable] = None,
                 return_index: bool = False,
                 augment: bool = False):
        
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.return_index = return_index
        self.augment = augment
        
        # Load data
        self.x, self.y = self._load_data()
        
        # Data statistics for normalization
        self.mean = np.mean(self.x, axis=(0, 1), keepdims=True)
        self.std = np.std(self.x, axis=(0, 1), keepdims=True) + 1e-6
        
        print(f"Loaded {split} set: {self.x.shape[0]} samples")
        print(f"Shape: {self.x.shape}, Labels: {np.unique(self.y)}")
        
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load UT-HAR data from binary numpy files with .csv extension"""
        
        # Map split names if needed
        if self.split == 'val':
            file_split = 'valid'  # Some versions use 'valid' instead of 'val'
        else:
            file_split = self.split
            
        # Try different possible file names
        possible_x_files = [
            self.data_path / f'x_{self.split}.csv',
            self.data_path / f'X_{self.split}.csv',
            self.data_path / f'x_{file_split}.csv',
            self.data_path / f'X_{file_split}.csv',
        ]
        
        possible_y_files = [
            self.data_path / f'y_{self.split}.csv',
            self.data_path / f'Y_{self.split}.csv',
            self.data_path / f'y_{file_split}.csv',
            self.data_path / f'Y_{file_split}.csv',
        ]
        
        # Find the actual files
        x_file = None
        y_file = None
        
        for f in possible_x_files:
            if f.exists():
                x_file = f
                break
                
        for f in possible_y_files:
            if f.exists():
                y_file = f
                break
        
        if x_file is None or y_file is None:
            raise FileNotFoundError(
                f"Could not find data files for split '{self.split}' in {self.data_path}\n"
                f"Looking for: {possible_x_files}\n"
                f"Available files: {list(self.data_path.glob('*.csv'))}"
            )
        
        print(f"Loading from: {x_file.name} and {y_file.name}")
        
        # Load the binary numpy files (despite .csv extension)
        x = np.load(str(x_file))
        y = np.load(str(y_file))
        
        # Reshape X to (N, 250, 90) and ensure correct types
        x = x.reshape(-1, 250, 90).astype(np.float32)
        y = y.astype(np.int64)
        
        # Ensure y is 1D
        if len(y.shape) > 1:
            y = y.flatten()
        
        # Verify shapes
        assert x.shape[1:] == (250, 90), f"Expected shape (N, 250, 90), got {x.shape}"
        assert len(y) == len(x), f"Mismatch between data and labels: {len(x)} vs {len(y)}"
        
        # Check for valid labels
        unique_labels = np.unique(y)
        print(f"Unique labels in {self.split}: {unique_labels}")
        
        return x, y
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize CSI data"""
        return (x - self.mean) / self.std
    
    def augment_csi(self, x: np.ndarray) -> np.ndarray:
        """Data augmentation for CSI"""
        if not self.augment or self.split != 'train':
            return x
        
        # Time shift
        if np.random.random() > 0.5:
            shift = np.random.randint(-10, 10)
            x = np.roll(x, shift, axis=0)
        
        # Amplitude scaling
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            x = x * scale
        
        # Add noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.01, x.shape).astype(np.float32)
            x = x + noise
            
        return x
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int) -> Dict:
        x = self.normalize(self.x[idx])  # Shape: (250, 90)
        y = self.y[idx]
        
        if self.augment:
            x = self.augment_csi(x)
        
        if self.transform:
            x = self.transform(x)
        
        # Ensure x is 2D (250, 90)
        x = torch.FloatTensor(x).squeeze()  # Remove any extra dimensions
        
        # Ensure correct shape
        if x.dim() == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        
        # Final check
        assert x.shape == (250, 90), f"Expected shape (250, 90), got {x.shape}"
        
        y = torch.tensor(y, dtype=torch.long)  # Scalar tensor
        
        sample = {
            'csi': x,  # Shape: (250, 90)
            'activity': y
        }
        
        if self.return_index:
            sample['index'] = idx
            sample['temporal_position'] = idx
            
        return sample

def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders"""
    
    # Set num_workers to 0 for Windows
    num_workers = 0 if config.get('num_workers', 0) > 0 else 0
    
    train_dataset = UTHARDataset(
        config['dataset']['root'],
        split='train',
        return_index=True,
        augment=True
    )
    
    val_dataset = UTHARDataset(
        config['dataset']['root'],
        split='val',
        return_index=True,
        augment=False
    )
    
    test_dataset = UTHARDataset(
        config['dataset']['root'],
        split='test',
        return_index=True,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader