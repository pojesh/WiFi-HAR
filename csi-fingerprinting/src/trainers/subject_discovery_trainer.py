import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np

from src.losses.contrastive import TemporalContrastiveLoss, SubjectConsistencyLoss
from src.losses.disentangle import DisentanglementLoss
from src.evaluation.metrics import SubjectMetrics

class SubjectDiscoveryTrainer:
    """Trainer for subject discovery model"""
    
    def __init__(self,
                 model: nn.Module,
                 config: dict,
                 save_dir: str = './results'):
        
        self.model = model
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Losses
        self.activity_loss = nn.CrossEntropyLoss()
        self.temporal_contrastive = TemporalContrastiveLoss(
            temperature=config['model']['temperature'],
            temporal_window=config['subject_discovery']['temporal_window']
        )
        self.consistency_loss = SubjectConsistencyLoss()
        self.disentangle_loss = DisentanglementLoss()
        
        # Loss weights
        self.weights = config['losses']
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler with warmup
        self.warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config['training']['scheduler']['warmup_epochs']
        )
        
        self.main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'] - config['training']['scheduler']['warmup_epochs']
        )
        
        # Logging
        self.save_dir = Path(save_dir) / config['experiment']['name']
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.save_dir / 'tensorboard')
        self.metrics = SubjectMetrics(num_subjects=config['model']['latent_subjects'])
        
        self.best_val_acc = 0
        self.epoch = 0
        
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_losses = {
            'total': 0, 'activity': 0, 'temporal': 0,
            'consistency': 0, 'disentangle': 0, 'entropy': 0
        }
        
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch}')
        
        for batch in pbar:
            csi = batch['csi'].to(self.device)
            activity_labels = batch['activity'].to(self.device)
            indices = batch['index'].to(self.device)
            
            # Forward pass
            outputs = self.model(csi, return_all=True)
            
            # Activity loss (supervised)
            activity_loss = self.activity_loss(
                outputs['activity_logits'], 
                activity_labels
            )
            
            # Temporal contrastive loss (self-supervised)
            temporal_loss = self.temporal_contrastive(
                outputs['signature'],
                indices
            )
            
            # Consistency loss
            consistency_loss = self.consistency_loss(
                outputs['subject_logits'],
                activity_labels
            )
            
            # Disentanglement loss
            disentangle_loss = self.disentangle_loss(
                outputs['activity_features'],
                outputs['signature'],
                outputs['disentangle_scores']
            )
            
            # Entropy regularization for subject assignments
            subject_probs = torch.softmax(outputs['subject_logits'], dim=1)
            entropy = -torch.sum(subject_probs * torch.log(subject_probs + 1e-8), dim=1).mean()
            entropy_loss = -entropy  # Maximize entropy
            
            # Total loss
            total_loss = (
                self.weights['activity_weight'] * activity_loss +
                self.weights['subject_weight'] * temporal_loss +
                self.weights['entropy_weight'] * entropy_loss +
                self.weights['disentangle_weight'] * disentangle_loss +
                0.1 * consistency_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            total_losses['total'] += total_loss.item()
            total_losses['activity'] += activity_loss.item()
            total_losses['temporal'] += temporal_loss.item()
            total_losses['consistency'] += consistency_loss.item()
            total_losses['disentangle'] += disentangle_loss.item()
            total_losses['entropy'] += entropy_loss.item()
            
            _, predicted = torch.max(outputs['activity_logits'], 1)
            correct += (predicted == activity_labels).sum().item()
            total += activity_labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        # Average losses
        num_batches = len(train_loader)
        for key in total_losses:
            total_losses[key] /= num_batches
        
        total_losses['accuracy'] = 100. * correct / total
        
        return total_losses
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        all_signatures = []
        all_subject_assignments = []
        all_activity_labels = []
        all_activity_preds = []
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                csi = batch['csi'].to(self.device)
                activity_labels = batch['activity'].to(self.device)
                
                outputs = self.model(csi)
                
                # Activity loss
                loss = self.activity_loss(outputs['activity_logits'], activity_labels)
                total_loss += loss.item()
                
                # Activity predictions
                _, predicted = torch.max(outputs['activity_logits'], 1)
                correct += (predicted == activity_labels).sum().item()
                total += activity_labels.size(0)
                
                # Collect outputs for analysis
                all_signatures.append(outputs['signature'].cpu())
                all_subject_assignments.append(
                    torch.argmax(outputs['subject_logits'], dim=1).cpu()
                )
                all_activity_labels.append(activity_labels.cpu())
                all_activity_preds.append(predicted.cpu())
        
        # Concatenate all outputs
        all_signatures = torch.cat(all_signatures, dim=0)
        all_subject_assignments = torch.cat(all_subject_assignments, dim=0)
        all_activity_labels = torch.cat(all_activity_labels, dim=0)
        all_activity_preds = torch.cat(all_activity_preds, dim=0)
        
        # Compute metrics
        metrics = self.metrics.compute(
            all_signatures.numpy(),
            all_subject_assignments.numpy(),
            all_activity_labels.numpy(),
            all_activity_preds.numpy()
        )
        
        metrics['loss'] = total_loss / len(val_loader)
        metrics['accuracy'] = 100. * correct / total
        
        return metrics
    
    def train(self, train_loader, val_loader, test_loader=None):
        """Main training loop"""
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            if epoch < self.config['training']['scheduler']['warmup_epochs']:
                self.warmup_scheduler.step()
            else:
                self.main_scheduler.step()
            
            # Logging
            self._log_metrics(train_metrics, val_metrics)
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint('best_model.pt', val_metrics)
            
            # Regular checkpointing
            if (epoch + 1) % self.config['logging']['save_frequency'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', val_metrics)
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"Train Loss: {train_metrics['total']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"Subject Metrics - Silhouette: {val_metrics['silhouette_score']:.4f}, "
                  f"Davies-Bouldin: {val_metrics['davies_bouldin']:.4f}")
        
        # Final test evaluation
        if test_loader:
            test_metrics = self.validate(test_loader)
            print(f"\nFinal Test Accuracy: {test_metrics['accuracy']:.2f}%")
            self.save_results(test_metrics)
    
    def _log_metrics(self, train_metrics, val_metrics):
        """Log metrics to tensorboard"""
        
        # Training metrics
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, self.epoch)
        
        # Validation metrics
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Val/{key}', value, self.epoch)
    
    def save_checkpoint(self, filename, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def save_results(self, metrics):
        """Save final results"""
        import json
        with open(self.save_dir / 'results.json', 'w') as f:
            json.dump(metrics, f, indent=4, default=str)