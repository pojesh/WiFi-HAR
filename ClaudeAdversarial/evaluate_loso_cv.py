"""
Leave-One-Subject-Out Cross-Validation for WiFi CSI-based HAR
Evaluates both base and adversarial models for subject generalization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import json
from datetime import datetime

from config import Config
from dataset import create_loso_data_loaders
from models import CNNGRUBase, AdversarialDANN, count_parameters


# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LOSOCrossValidator:
    """
    Leave-One-Subject-Out Cross-Validation evaluator
    Tests subject generalization capability
    """
    
    def __init__(
        self,
        model_type: str = 'base',  # 'base' or 'adversarial'
        device: torch.device = None,
        save_dir: Path = None
    ):
        self.model_type = model_type
        self.device = device if device is not None else Config.get_device()
        self.save_dir = save_dir if save_dir is not None else Config.RESULTS_ROOT / f'loso_{model_type}'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.fold_results = []
        self.all_predictions = []
        self.all_labels = []
        
        logger.info(f"Initialized LOSO-CV for {model_type} model")
        logger.info(f"Testing on {len(Config.SUBJECTS)} subjects")
    
    def create_model(self):
        """Create model instance"""
        if self.model_type == 'base':
            model = CNNGRUBase()
        elif self.model_type == 'adversarial':
            model = AdversarialDANN()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def train_fold(
        self,
        model,
        train_loader,
        val_loader,
        fold_idx: int,
        test_subject: str,
        num_epochs: int = Config.NUM_EPOCHS,
        learning_rate: float = Config.LEARNING_RATE
    ):
        """Train model for one fold"""
        
        # Setup optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Setup criteria
        activity_criterion = nn.CrossEntropyLoss()
        
        if self.model_type == 'adversarial':
            domain_criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=Config.LR_SCHEDULER_FACTOR,
            patience=Config.LR_SCHEDULER_PATIENCE,
            verbose=False
        )
        
        best_val_acc = 0.0
        best_model_state = None
        
        logger.info(f"Training fold {fold_idx+1} (test subject: {test_subject})")
        
        for epoch in range(1, num_epochs + 1):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for data, activity_labels, subject_labels in train_loader:
                data = data.to(self.device)
                activity_labels = activity_labels.to(self.device)
                
                optimizer.zero_grad()
                
                if self.model_type == 'base':
                    # Base model training
                    outputs = model(data)
                    loss = activity_criterion(outputs, activity_labels)
                else:
                    # Adversarial model training
                    subject_labels = subject_labels.to(self.device)
                    activity_outputs, domain_outputs = model(data)
                    
                    activity_loss = activity_criterion(activity_outputs, activity_labels)
                    domain_loss = domain_criterion(domain_outputs, subject_labels)
                    loss = activity_loss + Config.ADVERSARIAL_LAMBDA * domain_loss
                    outputs = activity_outputs
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * data.size(0)
                preds = torch.argmax(outputs, dim=1)
                train_correct += (preds == activity_labels).sum().item()
                train_total += activity_labels.size(0)
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, activity_labels, _ in val_loader:
                    data = data.to(self.device)
                    activity_labels = activity_labels.to(self.device)
                    
                    if self.model_type == 'base':
                        outputs = model(data)
                    else:
                        outputs = model.predict_activity(data)
                    
                    loss = activity_criterion(outputs, activity_labels)
                    
                    val_loss += loss.item() * data.size(0)
                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == activity_labels).sum().item()
                    val_total += activity_labels.size(0)
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            # Update scheduler
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if epoch % 10 == 0:
                logger.info(
                    f"  Epoch {epoch}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
        
        # Load best model
        model.load_state_dict(best_model_state)
        logger.info(f"  Best validation accuracy: {best_val_acc:.4f}")
        
        return model, best_val_acc
    
    def test_fold(self, model, test_loader, test_subject: str):
        """Test model on held-out subject"""
        
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, activity_labels, _ in test_loader:
                data = data.to(self.device)
                activity_labels = activity_labels.to(self.device)
                
                if self.model_type == 'base':
                    outputs = model(data)
                else:
                    outputs = model.predict_activity(data)
                
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(activity_labels.cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        logger.info(
            f"  Test Results (Subject {test_subject}): "
            f"Acc={accuracy:.4f}, Precision={precision:.4f}, "
            f"Recall={recall:.4f}, F1={f1:.4f}"
        )
        
        return {
            'test_subject': test_subject,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def run(
        self,
        labels_csv: Path,
        num_epochs: int = Config.NUM_EPOCHS,
        batch_size: int = Config.BATCH_SIZE
    ):
        """Run complete LOSO cross-validation"""
        
        logger.info("="*80)
        logger.info(f"Starting LOSO Cross-Validation ({self.model_type} model)")
        logger.info("="*80)
        
        # Set random seed
        torch.manual_seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)
        
        # Iterate over each subject as test set
        for fold_idx, test_subject in enumerate(Config.SUBJECTS):
            logger.info(f"\n{'='*80}")
            logger.info(f"Fold {fold_idx+1}/{len(Config.SUBJECTS)}: Testing on {test_subject}")
            logger.info(f"{'='*80}")
            
            # Create data loaders
            train_loader, val_loader, test_loader = create_loso_data_loaders(
                labels_csv=labels_csv,
                test_subject=test_subject,
                batch_size=batch_size
            )
            
            # Create fresh model for this fold
            model = self.create_model()
            
            # Train
            model, best_val_acc = self.train_fold(
                model, train_loader, val_loader,
                fold_idx, test_subject, num_epochs
            )
            
            # Test
            fold_result = self.test_fold(model, test_loader, test_subject)
            fold_result['best_val_acc'] = best_val_acc
            self.fold_results.append(fold_result)
            
            # Store predictions
            self.all_predictions.extend(fold_result['predictions'])
            self.all_labels.extend(fold_result['labels'])
        
        # Compute overall metrics
        self.compute_overall_metrics()
        
        # Save results
        self.save_results()
        
        # Plot results
        self.plot_results()
        
        logger.info("\n" + "="*80)
        logger.info("LOSO Cross-Validation Complete!")
        logger.info("="*80)
    
    def compute_overall_metrics(self):
        """Compute overall metrics across all folds"""
        
        # Average metrics across folds
        avg_accuracy = np.mean([r['accuracy'] for r in self.fold_results])
        avg_precision = np.mean([r['precision'] for r in self.fold_results])
        avg_recall = np.mean([r['recall'] for r in self.fold_results])
        avg_f1 = np.mean([r['f1'] for r in self.fold_results])
        
        std_accuracy = np.std([r['accuracy'] for r in self.fold_results])
        std_precision = np.std([r['precision'] for r in self.fold_results])
        std_recall = np.std([r['recall'] for r in self.fold_results])
        std_f1 = np.std([r['f1'] for r in self.fold_results])
        
        # Overall metrics using all predictions
        overall_accuracy = accuracy_score(self.all_labels, self.all_predictions)
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            self.all_labels, self.all_predictions, average='weighted'
        )
        
        self.overall_metrics = {
            'average': {
                'accuracy': float(avg_accuracy),
                'precision': float(avg_precision),
                'recall': float(avg_recall),
                'f1': float(avg_f1)
            },
            'std': {
                'accuracy': float(std_accuracy),
                'precision': float(std_precision),
                'recall': float(std_recall),
                'f1': float(std_f1)
            },
            'overall': {
                'accuracy': float(overall_accuracy),
                'precision': float(overall_precision),
                'recall': float(overall_recall),
                'f1': float(overall_f1)
            }
        }
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("LOSO Cross-Validation Summary")
        logger.info("="*80)
        logger.info(f"Average Accuracy:  {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        logger.info(f"Average Precision: {avg_precision:.4f} ± {std_precision:.4f}")
        logger.info(f"Average Recall:    {avg_recall:.4f} ± {std_recall:.4f}")
        logger.info(f"Average F1-Score:  {avg_f1:.4f} ± {std_f1:.4f}")
        logger.info(f"\nOverall Accuracy:  {overall_accuracy:.4f}")
        logger.info(f"Overall Precision: {overall_precision:.4f}")
        logger.info(f"Overall Recall:    {overall_recall:.4f}")
        logger.info(f"Overall F1-Score:  {overall_f1:.4f}")
        logger.info("="*80)
    
    def save_results(self):
        """Save LOSO-CV results to file"""
        
        results = {
            'model_type': self.model_type,
            'num_folds': len(Config.SUBJECTS),
            'fold_results': self.fold_results,
            'overall_metrics': self.overall_metrics,
            'config': {
                'num_epochs': Config.NUM_EPOCHS,
                'batch_size': Config.BATCH_SIZE,
                'learning_rate': Config.LEARNING_RATE
            }
        }
        
        # Save as JSON
        results_file = self.save_dir / f'loso_cv_results_{self.model_type}.json'
        with open(results_file, 'w') as f:
            # Don't save predictions/labels (too large)
            results_to_save = results.copy()
            for fold in results_to_save['fold_results']:
                fold.pop('predictions', None)
                fold.pop('labels', None)
            json.dump(results_to_save, f, indent=4)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save per-fold results as CSV
        df = pd.DataFrame(self.fold_results)
        df = df.drop(['predictions', 'labels'], axis=1, errors='ignore')
        csv_file = self.save_dir / f'loso_cv_per_fold_{self.model_type}.csv'
        df.to_csv(csv_file, index=False)
        logger.info(f"Per-fold results saved to {csv_file}")
    
    def plot_results(self):
        """Plot LOSO-CV results"""
        
        # Extract data
        subjects = [r['test_subject'] for r in self.fold_results]
        accuracies = [r['accuracy'] for r in self.fold_results]
        f1_scores = [r['f1'] for r in self.fold_results]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Per-subject accuracy
        axes[0, 0].bar(range(len(subjects)), accuracies, color='skyblue', edgecolor='black')
        axes[0, 0].axhline(y=np.mean(accuracies), color='r', linestyle='--', label=f'Mean: {np.mean(accuracies):.3f}')
        axes[0, 0].set_xlabel('Test Subject')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title(f'Per-Subject Accuracy ({self.model_type.capitalize()} Model)')
        axes[0, 0].set_xticks(range(len(subjects)))
        axes[0, 0].set_xticklabels(subjects, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Per-subject F1-score
        axes[0, 1].bar(range(len(subjects)), f1_scores, color='lightgreen', edgecolor='black')
        axes[0, 1].axhline(y=np.mean(f1_scores), color='r', linestyle='--', label=f'Mean: {np.mean(f1_scores):.3f}')
        axes[0, 1].set_xlabel('Test Subject')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_title(f'Per-Subject F1-Score ({self.model_type.capitalize()} Model)')
        axes[0, 1].set_xticks(range(len(subjects)))
        axes[0, 1].set_xticklabels(subjects, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Metrics comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            self.overall_metrics['average']['accuracy'],
            self.overall_metrics['average']['precision'],
            self.overall_metrics['average']['recall'],
            self.overall_metrics['average']['f1']
        ]
        stds = [
            self.overall_metrics['std']['accuracy'],
            self.overall_metrics['std']['precision'],
            self.overall_metrics['std']['recall'],
            self.overall_metrics['std']['f1']
        ]
        
        axes[1, 0].bar(metrics, values, yerr=stds, capsize=5, color='coral', edgecolor='black')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title(f'Average Metrics Across All Folds ({self.model_type.capitalize()})')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Box plot of accuracies
        axes[1, 1].boxplot([accuracies], labels=['Accuracy'])
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title(f'Accuracy Distribution Across Subjects ({self.model_type.capitalize()})')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.save_dir / f'loso_cv_results_{self.model_type}.png'
        plt.savefig(plot_file, dpi=Config.FIGURE_DPI)
        plt.close()
        logger.info(f"Results plot saved to {plot_file}")


def compare_models(labels_csv: Path, num_epochs: int = Config.NUM_EPOCHS):
    """Compare base and adversarial models using LOSO-CV"""
    
    logger.info("\n" + "="*80)
    logger.info("Comparing Base and Adversarial Models with LOSO-CV")
    logger.info("="*80 + "\n")
    
    device = Config.get_device()
    
    # Run LOSO-CV for base model
    logger.info("="*80)
    logger.info("Part 1: Evaluating Base Model")
    logger.info("="*80)
    base_validator = LOSOCrossValidator('base', device)
    base_validator.run(labels_csv, num_epochs)
    
    # Run LOSO-CV for adversarial model
    logger.info("\n" + "="*80)
    logger.info("Part 2: Evaluating Adversarial Model")
    logger.info("="*80)
    adv_validator = LOSOCrossValidator('adversarial', device)
    adv_validator.run(labels_csv, num_epochs)
    
    # Compare results
    logger.info("\n" + "="*80)
    logger.info("Comparison Summary")
    logger.info("="*80)
    
    base_acc = base_validator.overall_metrics['average']['accuracy']
    base_f1 = base_validator.overall_metrics['average']['f1']
    adv_acc = adv_validator.overall_metrics['average']['accuracy']
    adv_f1 = adv_validator.overall_metrics['average']['f1']
    
    logger.info(f"Base Model      - Accuracy: {base_acc:.4f}, F1: {base_f1:.4f}")
    logger.info(f"Adversarial     - Accuracy: {adv_acc:.4f}, F1: {adv_f1:.4f}")
    logger.info(f"Improvement     - Accuracy: {(adv_acc-base_acc):.4f} ({(adv_acc-base_acc)/base_acc*100:.2f}%)")
    logger.info(f"                  F1:       {(adv_f1-base_f1):.4f} ({(adv_f1-base_f1)/base_f1*100:.2f}%)")
    logger.info("="*80)
    
    # Create comparison plot
    create_comparison_plot(base_validator, adv_validator)


def create_comparison_plot(base_validator, adv_validator):
    """Create comparison plot between base and adversarial models"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Per-subject accuracy comparison
    subjects = [r['test_subject'] for r in base_validator.fold_results]
    base_acc = [r['accuracy'] for r in base_validator.fold_results]
    adv_acc = [r['accuracy'] for r in adv_validator.fold_results]
    
    x = np.arange(len(subjects))
    width = 0.35
    
    axes[0].bar(x - width/2, base_acc, width, label='Base Model', color='skyblue', edgecolor='black')
    axes[0].bar(x + width/2, adv_acc, width, label='Adversarial Model', color='coral', edgecolor='black')
    axes[0].set_xlabel('Test Subject')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Per-Subject Accuracy: Base vs Adversarial')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(subjects, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Overall metrics comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    base_values = [
        base_validator.overall_metrics['average'][m.lower().replace('-', '_')] 
        for m in metrics
    ]
    adv_values = [
        adv_validator.overall_metrics['average'][m.lower().replace('-', '_')] 
        for m in metrics
    ]
    
    x = np.arange(len(metrics))
    axes[1].bar(x - width/2, base_values, width, label='Base Model', color='skyblue', edgecolor='black')
    axes[1].bar(x + width/2, adv_values, width, label='Adversarial Model', color='coral', edgecolor='black')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Average Metrics: Base vs Adversarial')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = Config.RESULTS_ROOT / 'loso_cv_comparison.png'
    plt.savefig(plot_file, dpi=Config.FIGURE_DPI)
    plt.close()
    logger.info(f"Comparison plot saved to {plot_file}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LOSO Cross-Validation')
    parser.add_argument('--labels_csv', type=str, default=str(Config.PROCESSED_DATA_ROOT / 'labels.csv'))
    parser.add_argument('--model_type', type=str, choices=['base', 'adversarial', 'both'], default='both')
    parser.add_argument('--epochs', type=int, default=30)  # Fewer epochs for CV
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    
    args = parser.parse_args()
    
    labels_csv = Path(args.labels_csv)
    
    if not labels_csv.exists():
        logger.error(f"Labels file not found: {labels_csv}")
        logger.error("Please run preprocessing first!")
        return
    
    if args.model_type == 'both':
        compare_models(labels_csv, args.epochs)
    else:
        device = Config.get_device()
        validator = LOSOCrossValidator(args.model_type, device)
        validator.run(labels_csv, args.epochs, args.batch_size)


if __name__ == '__main__':
    main()