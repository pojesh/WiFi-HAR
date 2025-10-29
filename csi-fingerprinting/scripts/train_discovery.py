#!/usr/bin/env python3
"""
Main training script for subject discovery in UT-HAR
"""

import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path

from src.datasets.uthar_dataset import create_dataloaders
from src.models.csi_signature_net import CSISignatureNet
from src.trainers.subject_discovery_trainer import SubjectDiscoveryTrainer

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle config inheritance
    if 'inherits' in config:
        parent_path = Path(config_path).parent / config['inherits']
        with open(parent_path, 'r') as f:
            parent_config = yaml.safe_load(f)
        # Merge configs (child overrides parent)
        for key, value in parent_config.items():
            if key not in config:
                config[key] = value
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Train subject discovery model on UT-HAR')
    parser.add_argument('--config', type=str, default='configs/subject_discovery.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config['seed'])
    
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Create model
    model = CSISignatureNet(
        input_dim=config['model']['input_dim'],
        seq_len=config['model']['seq_len'],
        num_activities=config['dataset']['num_classes'],
        latent_subjects=config['model']['latent_subjects'],
        hidden_dim=config['model']['hidden_dim'],
        signature_dim=config['model']['signature_dim'],
        dropout=config['model']['dropout']
    )
    
    # Create trainer
    trainer = SubjectDiscoveryTrainer(model, config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {trainer.epoch}")
    
    # Train model
    trainer.train(train_loader, val_loader, test_loader)
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")

if __name__ == '__main__':
    main()