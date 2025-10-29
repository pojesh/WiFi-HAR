#!/usr/bin/env python3
"""
Analyze discovered subjects and create visualizations
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from pathlib import Path

from src.datasets.uthar_dataset import create_dataloaders
from src.models.csi_signature_net import CSISignatureNet

# import pyyaml
import yaml

def visualize_subjects(model, dataloader, save_dir):
    """Create comprehensive visualizations of discovered subjects"""
    
    model.eval()
    
    all_signatures = []
    all_subjects = []
    all_activities = []
    all_indices = []
    
    with torch.no_grad():
        for batch in dataloader:
            csi = batch['csi'].to(model.device)
            outputs = model(csi, return_all=True)
            
            signatures = outputs['signature'].cpu().numpy()
            subjects = torch.argmax(outputs['subject_logits'], dim=1).cpu().numpy()
            activities = batch['activity'].numpy()
            indices = batch['index'].numpy()
            
            all_signatures.append(signatures)
            all_subjects.append(subjects)
            all_activities.append(activities)
            all_indices.append(indices)
    
    # Concatenate all
    signatures = np.vstack(all_signatures)
    subjects = np.hstack(all_subjects)
    activities = np.hstack(all_activities)
    indices = np.hstack(all_indices)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. UMAP visualization
    print("Creating UMAP visualization...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(signatures)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Color by discovered subjects
    scatter1 = axes[0].scatter(embedding[:, 0], embedding[:, 1], 
                               c=subjects, cmap='tab10', s=10, alpha=0.6)
    axes[0].set_title('UMAP - Discovered Subjects')
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Color by activities
    scatter2 = axes[1].scatter(embedding[:, 0], embedding[:, 1],
                               c=activities, cmap='tab20', s=10, alpha=0.6)
    axes[1].set_title('UMAP - Activities')
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'umap_visualization.png', dpi=300)
    plt.show()
    
    # 2. Temporal consistency plot
    print("Creating temporal consistency plot...")
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Sort by index (temporal order)
    sorted_idx = np.argsort(indices)
    subjects_sorted = subjects[sorted_idx]
    
    # Plot subject assignments over time
    ax.scatter(range(len(subjects_sorted)), subjects_sorted, 
               c=subjects_sorted, cmap='tab10', s=1)
    ax.set_xlabel('Temporal Index')
    ax.set_ylabel('Discovered Subject')
    ax.set_title('Subject Assignments Over Time')
    plt.savefig(save_dir / 'temporal_consistency.png', dpi=300)
    plt.show()
    
    # 3. Subject-Activity distribution
    print("Creating subject-activity distribution...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create confusion matrix style plot
    subject_activity_matrix = np.zeros((len(np.unique(subjects)), 
                                        len(np.unique(activities))))
    
    for s, a in zip(subjects, activities):
        subject_activity_matrix[s, a] += 1
    
    # Normalize by row (subject)
    subject_activity_matrix = subject_activity_matrix / subject_activity_matrix.sum(axis=1, keepdims=True)
    
    sns.heatmap(subject_activity_matrix, annot=True, fmt='.2f', 
                cmap='YlOrRd', cbar_kws={'label': 'Proportion'})
    ax.set_xlabel('Activity')
    ax.set_ylabel('Discovered Subject')
    ax.set_title('Subject-Activity Distribution\n(Should be uniform if subjects are activity-independent)')
    plt.tight_layout()
    plt.savefig(save_dir / 'subject_activity_distribution.png', dpi=300)
    plt.show()
    
    # 4. Subject prototype analysis
    print("Analyzing subject prototypes...")
    prototypes = model.subject_prototypes.detach().cpu().numpy()
    
    # Compute prototype similarities
    prototype_sim = np.corrcoef(prototypes)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(prototype_sim, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, cbar_kws={'label': 'Cosine Similarity'})
    ax.set_xlabel('Prototype')
    ax.set_ylabel('Prototype')
    ax.set_title('Subject Prototype Similarities')
    plt.tight_layout()
    plt.savefig(save_dir / 'prototype_similarities.png', dpi=300)
    plt.show()
    
    # Save numerical results
    results = {
        'num_discovered_subjects': len(np.unique(subjects)),
        'subject_distribution': np.bincount(subjects).tolist(),
        'subject_entropy': float(entropy(np.bincount(subjects) / len(subjects))),
        'temporal_consistency_score': compute_temporal_consistency(subjects),
        'subject_activity_independence': compute_independence(subjects, activities)
    }
    
    import json
    with open(save_dir / 'subject_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nAnalysis complete! Results saved to {save_dir}")
    return results

def compute_temporal_consistency(subjects, window=10):
    """Compute temporal consistency of subject assignments"""
    consistencies = []
    for i in range(len(subjects) - window):
        window_subjects = subjects[i:i+window]
        unique, counts = np.unique(window_subjects, return_counts=True)
        consistency = counts.max() / window
        consistencies.append(consistency)
    return float(np.mean(consistencies))

def compute_independence(subjects, activities):
    """Compute independence between subjects and activities"""
    from sklearn.metrics import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(subjects, activities)
    return float(1 - nmi)

def entropy(p):
    """Compute entropy of probability distribution"""
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def main():
    parser = argparse.ArgumentParser(description='Analyze discovered subjects')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/subject_discovery.yaml',
                       help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, default='./results/subject_analysis',
                       help='Directory to save analysis results')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataloaders
    _, val_loader, test_loader = create_dataloaders(config)
    
    # Load model
    model = CSISignatureNet(
        input_dim=config['model']['input_dim'],
        seq_len=config['model']['seq_len'],
        num_activities=config['dataset']['num_classes'],
        latent_subjects=config['model']['latent_subjects'],
        hidden_dim=config['model']['hidden_dim'],
        signature_dim=config['model']['signature_dim']
    )
    
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.device = device
    
    # Analyze test set
    results = visualize_subjects(model, test_loader, args.save_dir)
    
    print("\nSubject Discovery Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    main()