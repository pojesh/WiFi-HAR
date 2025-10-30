"""
Training script for adversarial domain adaptation model
Implements Domain Adversarial Neural Network (DANN) for subject bias mitigation
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

from config import Config
from dataset import create_data_loaders
from models import AdversarialDANN, count_parameters


# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdversarialTrainer:
    """Trainer for adversarial domain adaptation model"""
    
    def __init__(
        self,
        model: AdversarialDANN,
        device: torch.device,
        save_dir: Path,
        experiment_name: str = 'adversarial_model',
        adversarial_lambda: float = Config.ADVERSARIAL_LAMBDA
    ):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.adversarial_lambda = adversarial_lambda
        
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
            'train_activity_loss': [],
            'train_domain_loss': [],
            'train_total_loss': [],
            'train_activity_acc': [],
            'train_domain_acc': [],
            'val_activity_loss': [],
            'val_activity_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        logger.info(f"Initialized adversarial trainer for {experiment_name}")
        logger.info(f"Model parameters: {count_parameters(model):,}")
        logger.info(f"Adversarial lambda: {adversarial_lambda}")
    
    def compute_alpha(self, epoch: int, num_epochs: int) -> float:
        """
        Compute alpha for gradient reversal layer
        Gradually increase from 0 to 1 during training for stable learning
        """
        if Config.PROGRESSIVE_ADVERSARIAL:
            p = epoch / num_epochs
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            return alpha
        else:
            return Config.GRL_ALPHA
    
    def train_epoch(
        self,
        train_loader,
        optimizer,
        activity_criterion,
        domain_criterion,
        epoch: int,
        num_epochs: int
    ):
        """Train for one epoch with adversarial learning"""
        self.model.train()
        
        # Compute alpha for GRL
        alpha = self.compute_alpha(epoch, num_epochs)
        self.model.set_alpha(alpha)
        
        total_activity_loss = 0
        total_domain_loss = 0
        total_loss = 0
        
        all_activity_preds = []
        all_activity_labels = []
        all_domain_preds = []
        all_domain_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} (α={alpha:.3f})')
        for batch_idx, (data, activity_labels, subject_labels) in enumerate(pbar):
            data = data.to(self.device)
            activity_labels = activity_labels.to(self.device)
            subject_labels = subject_labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            activity_output, domain_output = self.model(data, alpha=alpha)
            
            # Compute losses
            activity_loss = activity_criterion(activity_output, activity_labels)
            domain_loss = domain_criterion(domain_output, subject_labels)
            
            # Combined loss with adversarial weight
            loss = activity_loss + self.adversarial_lambda * domain_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_activity_loss += activity_loss.item() * data.size(0)
            total_domain_loss += domain_loss.item() * data.size(0)
            total_loss += loss.item() * data.size(0)
            
            # Predictions
            activity_preds = torch.argmax(activity_output, dim=1)
            domain_preds = torch.argmax(domain_output, dim=1)
            
            all_activity_preds.extend(activity_preds.cpu().numpy())
            all_activity_labels.extend(activity_labels.cpu().numpy())
            all_domain_preds.extend(domain_preds.cpu().numpy())
            all_domain_labels.extend(subject_labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'act_loss': activity_loss.item(),
                'dom_loss': domain_loss.item()
            })
        
        # Compute metrics
        epoch_activity_loss = total_activity_loss / len(train_loader.dataset)
        epoch_domain_loss = total_domain_loss / len(train_loader.dataset)
        epoch_total_loss = total_loss / len(train_loader.dataset)
        epoch_activity_acc = accuracy_score(all_activity_labels, all_activity_preds)
        epoch_domain_acc = accuracy_score(all_domain_labels, all_domain_preds)
        
        return (epoch_activity_loss, epoch_domain_loss, epoch_total_loss,
                epoch_activity_acc, epoch_domain_acc)
    
    def validate(self, val_loader, criterion):
        """Validate model (only on activity classification)"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, activity_labels, _ in val_loader:
                data = data.to(self.device)
                activity_labels = activity_labels.to(self.device)
                
                # Forward pass (only need activity output)
                outputs = self.model.predict_activity(data)
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
        """Full training loop with adversarial learning"""
        
        # Setup optimizer and criteria
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        activity_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=Config.LR_SCHEDULER_FACTOR,
            patience=Config.LR_SCHEDULER_PATIENCE
        )
        
        logger.info(f"Starting adversarial training for {num_epochs} epochs")
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            # Train
            (train_activity_loss, train_domain_loss, train_total_loss,
             train_activity_acc, train_domain_acc) = self.train_epoch(
                train_loader, optimizer, activity_criterion, domain_criterion,
                epoch, num_epochs
            )
            
            # Validate
            val_activity_loss, val_activity_acc = self.validate(
                val_loader, activity_criterion
            )
            
            # Update scheduler
            scheduler.step(val_activity_acc)
            
            # Save history
            self.history['train_activity_loss'].append(train_activity_loss)
            self.history['train_domain_loss'].append(train_domain_loss)
            self.history['train_total_loss'].append(train_total_loss)
            self.history['train_activity_acc'].append(train_activity_acc)
            self.history['train_domain_acc'].append(train_domain_acc)
            self.history['val_activity_loss'].append(val_activity_loss)
            self.history['val_activity_acc'].append(val_activity_acc)
            
            # Log to tensorboard
            if self.writer is not None:
                self.writer.add_scalar('Loss/train_activity', train_activity_loss, epoch)
                self.writer.add_scalar('Loss/train_domain', train_domain_loss, epoch)
                self.writer.add_scalar('Loss/train_total', train_total_loss, epoch)
                self.writer.add_scalar('Loss/val_activity', val_activity_loss, epoch)
                self.writer.add_scalar('Accuracy/train_activity', train_activity_acc, epoch)
                self.writer.add_scalar('Accuracy/train_domain', train_domain_acc, epoch)
                self.writer.add_scalar('Accuracy/val_activity', val_activity_acc, epoch)
            
            # Print progress
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Act Loss: {train_activity_loss:.4f}, Dom Loss: {train_domain_loss:.4f} | "
                f"Act Acc: {train_activity_acc:.4f}, Dom Acc: {train_domain_acc:.4f} | "
                f"Val Loss: {val_activity_loss:.4f}, Val Acc: {val_activity_acc:.4f}"
            )
            
            # Save best model
            if val_activity_acc > self.best_val_acc:
                self.best_val_acc = val_activity_acc
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth')
                logger.info(f"✓ Saved best model (Val Acc: {val_activity_acc:.4f})")
        
        logger.info(f"Training complete! Best Val Acc: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
    
    def evaluate(self, test_loader, load_best: bool = True):
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
                
                outputs = self.model.predict_activity(data)
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
        logger.info("Test Results (Adversarial Model)")
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
        
        # Plot confusion matrix
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
        plt.title('Confusion Matrix (Adversarial Model)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_file = self.results_dir / f'{self.experiment_name}_confusion_matrix.png'
        plt.savefig(cm_file, dpi=Config.FIGURE_DPI)
        plt.close()
        logger.info(f"Confusion matrix saved to {cm_file}")
    
    def plot_training_history(self):
        """Plot and save training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Activity loss
        axes[0, 0].plot(self.history['train_activity_loss'], label='Train')
        axes[0, 0].plot(self.history['val_activity_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Activity Classification Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Domain loss
        axes[0, 1].plot(self.history['train_domain_loss'], label='Domain Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Domain Discrimination Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Activity accuracy
        axes[1, 0].plot(self.history['train_activity_acc'], label='Train')
        axes[1, 0].plot(self.history['val_activity_acc'], label='Val')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Activity Classification Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Domain accuracy
        axes[1, 1].plot(self.history['train_domain_acc'], label='Domain Acc')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Domain Discrimination Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
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
    
    parser = argparse.ArgumentParser(description='Train adversarial domain adaptation model')
    parser.add_argument('--labels_csv', type=str, default=str(Config.PROCESSED_DATA_ROOT / 'labels.csv'))
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    parser.add_argument('--adversarial_lambda', type=float, default=Config.ADVERSARIAL_LAMBDA)
    parser.add_argument('--experiment_name', type=str, default='adversarial_model')
    
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
    logger.info("Creating adversarial model...")
    model = AdversarialDANN()
    
    # Create trainer
    save_dir = Config.RESULTS_ROOT / args.experiment_name
    trainer = AdversarialTrainer(
        model, device, save_dir, args.experiment_name,
        adversarial_lambda=args.adversarial_lambda
    )
    
    # Train
    logger.info("Starting adversarial training...")
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # Evaluate
    logger.info("Evaluating on test set...")
    results = trainer.evaluate(test_loader)
    
    logger.info("✓ Adversarial training and evaluation complete!")


if __name__ == '__main__':
    main()