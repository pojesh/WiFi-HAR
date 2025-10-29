#!/usr/bin/env python3
"""
Test script to verify UT-HAR data loading
"""

import numpy as np
from pathlib import Path
import sys
sys.path.append('.')

from src.datasets.uthar_dataset import UTHARDataset, create_dataloaders

def test_direct_load():
    """Test direct numpy loading"""
    data_path = Path('./data/UTD-HAR')
    
    print("Testing direct numpy load...")
    
    for split in ['train', 'val', 'test']:
        # Try different file names
        for prefix in ['x', 'X']:
            for split_name in [split, 'valid' if split == 'val' else split]:
                file_path = data_path / f'{prefix}_{split_name}.csv'
                if file_path.exists():
                    print(f"\nFound: {file_path}")
                    try:
                        data = np.load(str(file_path))
                        print(f"  Raw shape: {data.shape}")
                        
                        # Reshape
                        data = data.reshape(-1, 250, 90).astype(np.float32)
                        print(f"  Reshaped: {data.shape}")
                        print(f"  Min: {data.min():.3f}, Max: {data.max():.3f}")
                        
                    except Exception as e:
                        print(f"  Error: {e}")

def test_dataset_class():
    """Test the dataset class"""
    print("\n" + "="*50)
    print("Testing UTHARDataset class...")
    
    config = {
        'dataset': {
            'root': './data/UTD-HAR',
            'num_classes': 7
        },
        'training': {
            'batch_size': 16
        },
        'num_workers': 0
    }
    
    try:
        # Test each split
        for split in ['train', 'val', 'test']:
            print(f"\nTesting {split} split...")
            dataset = UTHARDataset(
                config['dataset']['root'],
                split=split,
                return_index=True,
                augment=False
            )
            
            # Test getting a sample
            sample = dataset[0]
            print(f"  Sample keys: {sample.keys()}")
            print(f"  CSI shape: {sample['csi'].shape}")
            print(f"  Activity: {sample['activity']}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

def test_dataloader():
    """Test the dataloader"""
    print("\n" + "="*50)
    print("Testing DataLoader...")
    
    config = {
        'dataset': {
            'root': './data/UTD-HAR',
            'num_classes': 7
        },
        'training': {
            'batch_size': 16
        },
        'num_workers': 0
    }
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config)
        
        # Test one batch
        for batch in train_loader:
            print(f"\nBatch keys: {batch.keys()}")
            print(f"CSI shape: {batch['csi'].shape}")
            print(f"Activity shape: {batch['activity'].shape}")
            print(f"Index shape: {batch['index'].shape}")
            print(f"Activities in batch: {batch['activity'].unique()}")
            break
            
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("UT-HAR Data Loading Test")
    print("="*50)
    
    test_direct_load()
    test_dataset_class()
    test_dataloader()
    
    print("\n" + "="*50)
    print("Test complete!")


    """
    PS C:\Users\Pojesh\Documents\GitHub\WiFi-HAR\csi-fingerprinting> python scripts/test_data.py
UT-HAR Data Loading Test
==================================================
Testing direct numpy load...

Found: data\UTD-HAR\x_train.csv
  Raw shape: (3977, 250, 90)
  Reshaped: (3977, 250, 90)
  Min: -10.666, Max: 30.519

Found: data\UTD-HAR\x_train.csv
  Raw shape: (3977, 250, 90)
  Reshaped: (3977, 250, 90)
  Min: -10.666, Max: 30.519

Found: data\UTD-HAR\X_train.csv
  Raw shape: (3977, 250, 90)
  Reshaped: (3977, 250, 90)
  Min: -10.666, Max: 30.519

Found: data\UTD-HAR\X_train.csv
  Raw shape: (3977, 250, 90)
  Reshaped: (3977, 250, 90)
  Min: -10.666, Max: 30.519

Found: data\UTD-HAR\x_val.csv
  Raw shape: (496, 250, 90)
  Reshaped: (496, 250, 90)
  Min: -9.956, Max: 30.539

Found: data\UTD-HAR\X_val.csv
  Raw shape: (496, 250, 90)
  Reshaped: (496, 250, 90)
  Min: -9.956, Max: 30.539

Found: data\UTD-HAR\x_test.csv
  Raw shape: (500, 250, 90)
  Reshaped: (500, 250, 90)
  Min: -9.901, Max: 30.539

Found: data\UTD-HAR\x_test.csv
  Raw shape: (500, 250, 90)
  Reshaped: (500, 250, 90)
  Min: -9.901, Max: 30.539

Found: data\UTD-HAR\X_test.csv
  Raw shape: (500, 250, 90)
  Reshaped: (500, 250, 90)
  Min: -9.901, Max: 30.539

Found: data\UTD-HAR\X_test.csv
  Raw shape: (500, 250, 90)
  Reshaped: (500, 250, 90)
  Min: -9.901, Max: 30.539

==================================================
Testing UTHARDataset class...

Testing train split...
Loading from: x_train.csv and y_train.csv
Unique labels in train: [0 1 2 3 4 5 6]
Loaded train set: 3977 samples
Shape: (3977, 250, 90), Labels: [0 1 2 3 4 5 6]
  Sample keys: dict_keys(['csi', 'activity', 'index', 'temporal_position'])
  CSI shape: torch.Size([1, 250, 90])
  Activity: 0

Testing val split...
Loading from: x_val.csv and y_val.csv
Unique labels in val: [0 1 2 3 4 5 6]
Loaded val set: 496 samples
Shape: (496, 250, 90), Labels: [0 1 2 3 4 5 6]
  Sample keys: dict_keys(['csi', 'activity', 'index', 'temporal_position'])
  CSI shape: torch.Size([1, 250, 90])
  Activity: 0

Testing test split...
Loading from: x_test.csv and y_test.csv
Unique labels in test: [0 1 2 3 4 5 6]
Loaded test set: 500 samples
Shape: (500, 250, 90), Labels: [0 1 2 3 4 5 6]
  Sample keys: dict_keys(['csi', 'activity', 'index', 'temporal_position'])
  CSI shape: torch.Size([1, 250, 90])
  Activity: 0

==================================================
Testing DataLoader...
Loading from: x_train.csv and y_train.csv
Unique labels in train: [0 1 2 3 4 5 6]
Loaded train set: 3977 samples
Shape: (3977, 250, 90), Labels: [0 1 2 3 4 5 6]
Loading from: x_val.csv and y_val.csv
Unique labels in val: [0 1 2 3 4 5 6]
Loaded val set: 496 samples
Shape: (496, 250, 90), Labels: [0 1 2 3 4 5 6]
Loading from: x_test.csv and y_test.csv
Unique labels in test: [0 1 2 3 4 5 6]
Loaded test set: 500 samples
Shape: (500, 250, 90), Labels: [0 1 2 3 4 5 6]

Batch keys: dict_keys(['csi', 'activity', 'index', 'temporal_position'])
CSI shape: torch.Size([16, 1, 250, 90])
Activity shape: torch.Size([16])
Index shape: torch.Size([16])
Activities in batch: tensor([0, 2, 3, 4, 5, 6])

==================================================
Test complete!

    """