"""
Training script for base CNN-GRU model
Standard training without adversarial learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import json
from datetime import datetime

from config import Config
from dataset import create_data_loaders
from models import CNNGRUBase, count_parameters


# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseTrainer:
    """Trainer for base CNN-GRU model"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        save_dir: Path,
        experiment_name: str = 'base_model'
    ):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        
        # Create directories
        self.checkpoint_dir = self.save_dir / 'checkpoints'
        self.results_dir = self.save_dir / 'results'
        self.logs_dir = self.save_dir / 'logs'
        
        for dir_path in [self.checkpoint_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard writer
        if Config.USE_TENSORBOARD:
            self.writer = SummaryWriter(self.logs_dir / experiment_name)
        else:
            self.writer = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        logger.info(f"Initialized trainer for {experiment_name}")
        logger.info(f"Model parameters: {count_parameters(model):,}")
    
    def train_epoch(
        self,
        train_loader,
        optimizer,
        criterion,
        epoch: int
    ):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (data, activity_labels, _) in enumerate(pbar):
            data = data.to(self.device)
            activity_labels = activity_labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, activity_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item() * data.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(activity_labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(
        self,
        val_loader,
        criterion
    ):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, activity_labels, _ in val_loader:
                data = data.to(self.device)
                activity_labels = activity_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = criterion(outputs, activity_labels)
                
                # Statistics
                total_loss += loss.item() * data.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(activity_labels.cpu().numpy())
        
        # Compute metrics
        val_loss = total_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        
        return val_loss, val_acc
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = Config.NUM_EPOCHS,
        learning_rate: float = Config.LEARNING_RATE,
        weight_decay: float = Config.WEIGHT_DECAY
    ):
        """Full training loop"""
        
        # Setup optimizer and criterion
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=Config.LR_SCHEDULER_FACTOR,
            patience=Config.LR_SCHEDULER_PATIENCE
        )
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, epoch
            )
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Log to tensorboard
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            # Print progress
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth')
                logger.info(f"✓ Saved best model (Val Acc: {val_acc:.4f})")
        
        logger.info(f"Training complete! Best Val Acc: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
    
    def evaluate(
        self,
        test_loader,
        load_best: bool = True
    ):
        """Evaluate model on test set"""
        
        if load_best:
            self.load_checkpoint('best_model.pth')
            logger.info("Loaded best model for evaluation")
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, activity_labels, _ in tqdm(test_loader, desc='Testing'):
                data = data.to(self.device)
                activity_labels = activity_labels.to(self.device)
                
                outputs = self.model(data)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(activity_labels.cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Classification report
        report = classification_report(
            all_labels,
            all_preds,
            target_names=Config.DAILY_ACTIVITIES,
            digits=4,
            zero_division=0
        )
        
        # Save results
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        # Print results
        logger.info("="*80)
        logger.info("Test Results")
        logger.info("="*80)
        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        logger.info("="*80)
        logger.info("\nClassification Report:")
        logger.info("\n" + report)
        
        # Save to file
        results_file = self.results_dir / f'{self.experiment_name}_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {results_file}")
        
        # Plot and save confusion matrix
        self.plot_confusion_matrix(cm, Config.DAILY_ACTIVITIES)
        
        # Plot training history
        self.plot_training_history()
        
        return results
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot and save confusion matrix"""
        plt.figure(figsize=Config.FIGURE_SIZE)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=Config.COLOR_SCHEME,
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_file = self.results_dir / f'{self.experiment_name}_confusion_matrix.png'
        plt.savefig(cm_file, dpi=Config.FIGURE_DPI)
        plt.close()
        logger.info(f"Confusion matrix saved to {cm_file}")
    
    def plot_training_history(self):
        """Plot and save training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        history_file = self.results_dir / f'{self.experiment_name}_training_history.png'
        plt.savefig(history_file, dpi=Config.FIGURE_DPI)
        plt.close()
        logger.info(f"Training history saved to {history_file}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'history': self.history
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch = checkpoint['best_epoch']
        self.history = checkpoint['history']


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train base CNN-GRU model')
    parser.add_argument('--labels_csv', type=str, default=str(Config.PROCESSED_DATA_ROOT / 'labels.csv'))
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    parser.add_argument('--experiment_name', type=str, default='base_model')
    
    args = parser.parse_args()
    
    # Print configuration
    Config.print_config()
    
    # Set device
    device = Config.get_device()
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # Create data loaders
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        labels_csv=Path(args.labels_csv),
        batch_size=args.batch_size
    )
    
    # Create model
    logger.info("Creating model...")
    model = CNNGRUBase()
    
    # Create trainer
    save_dir = Config.RESULTS_ROOT / args.experiment_name
    trainer = BaseTrainer(model, device, save_dir, args.experiment_name)
    
    # Train
    logger.info("Starting training...")
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # Evaluate
    logger.info("Evaluating on test set...")
    results = trainer.evaluate(test_loader)
    
    logger.info("✓ Training and evaluation complete!")


if __name__ == '__main__':
    main()