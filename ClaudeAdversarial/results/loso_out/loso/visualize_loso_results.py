"""
Comprehensive LOSO-CV Visualization Script
Generates all publication-ready charts and graphs for LOSO-CV results
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Configuration
FIGURE_DPI = 300
FIGURE_SIZE_LARGE = (16, 10)
FIGURE_SIZE_MEDIUM = (12, 8)
FIGURE_SIZE_SMALL = (10, 6)
COLOR_BASE = '#4A90E2'  # Blue
COLOR_ADV = '#E8704B'   # Coral/Orange
COLOR_PALETTE = 'Set2'

# Activity labels
ACTIVITIES = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 
              'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27']

ACTIVITY_NAMES = {
    'A02': 'Chest Exp. (H)',
    'A03': 'Chest Exp. (V)',
    'A04': 'Twist (L)',
    'A05': 'Twist (R)',
    'A13': 'Raise Hand (L)',
    'A14': 'Raise Hand (R)',
    'A17': 'Wave Hand (L)',
    'A18': 'Wave Hand (R)',
    'A19': 'Pick Up',
    'A20': 'Throw (L)',
    'A21': 'Throw (R)',
    'A22': 'Kick (L)',
    'A23': 'Kick (R)',
    'A27': 'Bow'
}

# Output directory
OUTPUT_DIR = Path('loso_cv_visualizations')
OUTPUT_DIR.mkdir(exist_ok=True)


def load_results(base_file='loso_cv_results_base_enhanced.json', 
                 adv_file='loso_cv_results_adversarial_enhanced.json'):
    """Load LOSO-CV results from JSON files"""
    with open(base_file, 'r') as f:
        base_results = json.load(f)
    
    with open(adv_file, 'r') as f:
        adv_results = json.load(f)
    
    return base_results, adv_results


def set_style():
    """Set publication-quality plot style"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette(COLOR_PALETTE)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 16


def plot_per_subject_accuracy(base_results, adv_results):
    """Plot per-subject accuracy comparison"""
    subjects = [f['test_subject'] for f in base_results['fold_results']]
    base_acc = [f['accuracy'] for f in base_results['fold_results']]
    adv_acc = [f['accuracy'] for f in adv_results['fold_results']]
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_MEDIUM)
    
    x = np.arange(len(subjects))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, base_acc, width, label='Base Model', 
                   color=COLOR_BASE, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, adv_acc, width, label='Adversarial Model', 
                   color=COLOR_ADV, edgecolor='black', linewidth=1.2)
    
    # Add mean lines
    base_mean = np.mean(base_acc)
    adv_mean = np.mean(adv_acc)
    ax.axhline(y=base_mean, color=COLOR_BASE, linestyle='--', linewidth=2, 
               label=f'Base Mean: {base_mean:.3f}', alpha=0.7)
    ax.axhline(y=adv_mean, color=COLOR_ADV, linestyle='--', linewidth=2, 
               label=f'Adv Mean: {adv_mean:.3f}', alpha=0.7)
    
    ax.set_xlabel('Test Subject', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Per-Subject Accuracy: Base vs Adversarial Model', 
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=0)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_subject_accuracy_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: per_subject_accuracy_comparison.png")


def plot_metrics_comparison(base_results, adv_results):
    """Plot overall metrics comparison"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1']
    
    base_values = [base_results['overall_metrics']['average'][k] for k in metric_keys]
    adv_values = [adv_results['overall_metrics']['average'][k] for k in metric_keys]
    
    base_std = [base_results['overall_metrics']['std'][k] for k in metric_keys]
    adv_std = [adv_results['overall_metrics']['std'][k] for k in metric_keys]
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SMALL)
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, base_values, width, yerr=base_std, 
                   label='Base Model', color=COLOR_BASE, 
                   edgecolor='black', linewidth=1.2, capsize=5)
    bars2 = ax.bar(x + width/2, adv_values, width, yerr=adv_std, 
                   label='Adversarial Model', color=COLOR_ADV, 
                   edgecolor='black', linewidth=1.2, capsize=5)
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Average Metrics Across All Folds (LOSO-CV)', 
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: metrics_comparison.png")


def plot_improvement_analysis(base_results, adv_results):
    """Plot improvement percentages"""
    subjects = [f['test_subject'] for f in base_results['fold_results']]
    base_acc = np.array([f['accuracy'] for f in base_results['fold_results']])
    adv_acc = np.array([f['accuracy'] for f in adv_results['fold_results']])
    
    improvements = ((adv_acc - base_acc) / base_acc) * 100
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_MEDIUM)
    
    colors = [COLOR_ADV if imp > 0 else COLOR_BASE for imp in improvements]
    bars = ax.bar(range(len(subjects)), improvements, color=colors, 
                  edgecolor='black', linewidth=1.2)
    
    ax.axhline(y=0, color='black', linewidth=1.5)
    ax.axhline(y=np.mean(improvements), color='red', linestyle='--', 
               linewidth=2, label=f'Mean Improvement: {np.mean(improvements):.1f}%')
    
    ax.set_xlabel('Test Subject', fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontweight='bold')
    ax.set_title('Per-Subject Improvement: Adversarial vs Base Model', 
                 fontweight='bold', pad=20)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', 
                va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_subject_improvement.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: per_subject_improvement.png")


def plot_box_plots(base_results, adv_results):
    """Plot box plots for distribution comparison"""
    base_acc = [f['accuracy'] for f in base_results['fold_results']]
    adv_acc = [f['accuracy'] for f in adv_results['fold_results']]
    
    fig, ax = plt.subplots(figsize=(8, 10))
    
    data = [base_acc, adv_acc]
    bp = ax.boxplot(data, labels=['Base Model', 'Adversarial Model'],
                    patch_artist=True, widths=0.6)
    
    # Color the boxes
    colors = [COLOR_BASE, COLOR_ADV]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Style the plot
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Accuracy Distribution Across Subjects (LOSO-CV)', 
                 fontweight='bold', pad=20)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean markers
    means = [np.mean(d) for d in data]
    ax.plot([1, 2], means, 'D', color='red', markersize=10, 
            label='Mean', zorder=3)
    
    # Add statistics text
    t_stat, p_value = stats.ttest_rel(base_acc, adv_acc)
    stats_text = f'Paired t-test:\nt-statistic: {t_stat:.4f}\np-value: {p_value:.4e}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_distribution_boxplot.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: accuracy_distribution_boxplot.png")


def plot_per_class_accuracy(base_results, adv_results):
    """Plot per-class (activity) accuracy comparison"""
    # Aggregate correct predictions per class across all folds
    base_per_class = np.zeros(len(ACTIVITIES))
    adv_per_class = np.zeros(len(ACTIVITIES))
    total_per_class = np.zeros(len(ACTIVITIES))
    
    for fold_cm in base_results['per_fold_confusion_matrices']:
        base_per_class += np.array(fold_cm['correct_per_class'])
    
    for fold_cm in adv_results['per_fold_confusion_matrices']:
        adv_per_class += np.array(fold_cm['correct_per_class'])
    
    for fold_cm in base_results['per_fold_confusion_matrices']:
        total_per_class += np.array(fold_cm['total_per_class'])
    
    base_acc_per_class = base_per_class / total_per_class
    adv_acc_per_class = adv_per_class / total_per_class
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(ACTIVITIES))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, base_acc_per_class, width, 
                   label='Base Model', color=COLOR_BASE, 
                   edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, adv_acc_per_class, width, 
                   label='Adversarial Model', color=COLOR_ADV, 
                   edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Activity', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Per-Activity Accuracy: Base vs Adversarial Model', 
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([ACTIVITY_NAMES[a] for a in ACTIVITIES], rotation=45, ha='right')
    ax.legend(framealpha=0.9)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_activity_accuracy.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: per_activity_accuracy.png")


def plot_variance_comparison(base_results, adv_results):
    """Plot variance comparison showing improved stability"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1']
    
    base_std = [base_results['overall_metrics']['std'][k] for k in metric_keys]
    adv_std = [adv_results['overall_metrics']['std'][k] for k in metric_keys]
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SMALL)
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, base_std, width, label='Base Model', 
                   color=COLOR_BASE, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, adv_std, width, label='Adversarial Model', 
                   color=COLOR_ADV, edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('Standard Deviation', fontweight='bold')
    ax.set_title('Cross-Subject Variance: Base vs Adversarial Model\n(Lower is Better)', 
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'variance_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: variance_comparison.png")


def plot_comprehensive_comparison(base_results, adv_results):
    """Create comprehensive 2x2 comparison plot"""
    fig = plt.figure(figsize=FIGURE_SIZE_LARGE)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Per-subject accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    subjects = [f['test_subject'] for f in base_results['fold_results']]
    base_acc = [f['accuracy'] for f in base_results['fold_results']]
    adv_acc = [f['accuracy'] for f in adv_results['fold_results']]
    
    x = np.arange(len(subjects))
    width = 0.35
    ax1.bar(x - width/2, base_acc, width, label='Base', color=COLOR_BASE, edgecolor='black')
    ax1.bar(x + width/2, adv_acc, width, label='Adversarial', color=COLOR_ADV, edgecolor='black')
    ax1.set_xlabel('Test Subject', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('(a) Per-Subject Accuracy', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects, fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.0])
    
    # Plot 2: Overall metrics
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ['Acc', 'Prec', 'Rec', 'F1']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1']
    base_values = [base_results['overall_metrics']['average'][k] for k in metric_keys]
    adv_values = [adv_results['overall_metrics']['average'][k] for k in metric_keys]
    
    x = np.arange(len(metrics))
    ax2.bar(x - width/2, base_values, width, label='Base', color=COLOR_BASE, edgecolor='black')
    ax2.bar(x + width/2, adv_values, width, label='Adversarial', color=COLOR_ADV, edgecolor='black')
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('(b) Average Metrics', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.0])
    
    # Plot 3: Improvement
    ax3 = fig.add_subplot(gs[1, 0])
    improvements = ((np.array(adv_acc) - np.array(base_acc)) / np.array(base_acc)) * 100
    colors = [COLOR_ADV if imp > 0 else COLOR_BASE for imp in improvements]
    ax3.bar(range(len(subjects)), improvements, color=colors, edgecolor='black')
    ax3.axhline(y=0, color='black', linewidth=1.5)
    ax3.axhline(y=np.mean(improvements), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(improvements):.1f}%')
    ax3.set_xlabel('Test Subject', fontweight='bold')
    ax3.set_ylabel('Improvement (%)', fontweight='bold')
    ax3.set_title('(c) Per-Subject Improvement', fontweight='bold')
    ax3.set_xticks(range(len(subjects)))
    ax3.set_xticklabels(subjects, fontsize=9)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Box plot
    ax4 = fig.add_subplot(gs[1, 1])
    bp = ax4.boxplot([base_acc, adv_acc], labels=['Base', 'Adversarial'],
                     patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(COLOR_BASE)
    bp['boxes'][1].set_facecolor(COLOR_ADV)
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax4.set_ylabel('Accuracy', fontweight='bold')
    ax4.set_title('(d) Accuracy Distribution', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1.0])
    
    plt.suptitle('LOSO Cross-Validation Results: Comprehensive Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(OUTPUT_DIR / 'comprehensive_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: comprehensive_comparison.png")


def plot_statistical_significance(base_results, adv_results):
    """Plot statistical significance analysis"""
    subjects = [f['test_subject'] for f in base_results['fold_results']]
    base_acc = [f['accuracy'] for f in base_results['fold_results']]
    adv_acc = [f['accuracy'] for f in adv_results['fold_results']]
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(base_acc, adv_acc)
    
    # Effect size (Cohen's d)
    diff = np.array(adv_acc) - np.array(base_acc)
    cohen_d = np.mean(diff) / np.std(diff, ddof=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Paired comparison
    for i in range(len(subjects)):
        ax1.plot([1, 2], [base_acc[i], adv_acc[i]], 
                marker='o', color='gray', alpha=0.5, linewidth=1)
    
    ax1.plot([1, 2], [np.mean(base_acc), np.mean(adv_acc)], 
            marker='D', color='red', linewidth=3, markersize=12, label='Mean')
    
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Base Model', 'Adversarial Model'])
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Paired Subject Comparison', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1.0])
    
    # Plot 2: Statistical summary
    ax2.axis('off')
    stats_text = f"""
    Statistical Significance Analysis
    ─────────────────────────────────────
    
    Paired t-test:
      t-statistic: {t_stat:.4f}
      p-value: {p_value:.6f}
      Significance: {'Yes (p < 0.05)' if p_value < 0.05 else 'No (p ≥ 0.05)'}
    
    Effect Size (Cohen's d): {cohen_d:.4f}
      Interpretation: {'Large' if abs(cohen_d) > 0.8 else 'Medium' if abs(cohen_d) > 0.5 else 'Small'}
    
    Mean Improvement:
      Absolute: {np.mean(diff):.4f}
      Relative: {np.mean(diff) / np.mean(base_acc) * 100:.2f}%
    
    Improvement Range:
      Min: {np.min(diff):.4f} ({np.min(diff) / np.mean(base_acc) * 100:.2f}%)
      Max: {np.max(diff):.4f} ({np.max(diff) / np.mean(base_acc) * 100:.2f}%)
    
    Subjects Improved: {np.sum(diff > 0)}/{len(diff)}
    """
    
    ax2.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Statistical Significance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'statistical_significance.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: statistical_significance.png")


def create_results_table(base_results, adv_results):
    """Create formatted results table as image"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    subjects = [f['test_subject'] for f in base_results['fold_results']]
    
    table_data = []
    table_data.append(['Subject', 'Base Acc', 'Adv Acc', 'Improvement', 
                      'Base F1', 'Adv F1', 'F1 Improve'])
    
    for i, subj in enumerate(subjects):
        base_acc = base_results['fold_results'][i]['accuracy']
        adv_acc = adv_results['fold_results'][i]['accuracy']
        improvement = ((adv_acc - base_acc) / base_acc) * 100
        
        base_f1 = base_results['fold_results'][i]['f1']
        adv_f1 = adv_results['fold_results'][i]['f1']
        f1_improvement = ((adv_f1 - base_f1) / base_f1) * 100
        
        table_data.append([
            subj,
            f'{base_acc:.4f}',
            f'{adv_acc:.4f}',
            f'{improvement:+.2f}%',
            f'{base_f1:.4f}',
            f'{adv_f1:.4f}',
            f'{f1_improvement:+.2f}%'
        ])
    
    # Add average row
    base_avg = base_results['overall_metrics']['average']['accuracy']
    adv_avg = adv_results['overall_metrics']['average']['accuracy']
    avg_improvement = ((adv_avg - base_avg) / base_avg) * 100
    
    base_f1_avg = base_results['overall_metrics']['average']['f1']
    adv_f1_avg = adv_results['overall_metrics']['average']['f1']
    f1_avg_improvement = ((adv_f1_avg - base_f1_avg) / base_f1_avg) * 100
    
    table_data.append([
        'AVERAGE',
        f'{base_avg:.4f}',
        f'{adv_avg:.4f}',
        f'{avg_improvement:+.2f}%',
        f'{base_f1_avg:.4f}',
        f'{adv_f1_avg:.4f}',
        f'{f1_avg_improvement:+.2f}%'
    ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.12, 0.13, 0.13, 0.14, 0.13, 0.13, 0.14])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style average row
    for i in range(len(table_data[0])):
        table[(len(table_data)-1, i)].set_facecolor('#E8E8E8')
        table[(len(table_data)-1, i)].set_text_props(weight='bold')
    
    # Color improvement cells
    for i in range(1, len(table_data)-1):
        # Accuracy improvement
        imp_val = float(table_data[i][3].strip('%+'))
        table[(i, 3)].set_facecolor