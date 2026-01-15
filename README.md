# WiFi CSI-based Human Activity Recognition with Adversarial Learning

**Complete implementation of WiFi CSI-based HAR with subject bias mitigation using adversarial domain adaptation**

## 📋 Project Overview

This project implements a robust Human Activity Recognition (HAR) system using WiFi Channel State Information (CSI) from the MMFi dataset. It features:

- **Base CNN-GRU Model**: Deep learning architecture for activity classification
- **Adversarial Domain Adaptation**: Subject bias mitigation using DANN (Domain Adversarial Neural Network)
- **Leave-One-Subject-Out Cross-Validation**: Rigorous evaluation of subject generalization
- **Comprehensive Evaluation**: Detailed metrics, visualizations, and statistical analysis

### Key Features

✅ **Production-Ready Code**: Fully tested, documented, and error-handled  
✅ **Modular Architecture**: Easy to extend and customize  
✅ **Reproducible Results**: Seed-based reproducibility  
✅ **Comprehensive Logging**: Tensorboard integration and detailed logs  
✅ **Publication-Quality Visualizations**: High-resolution plots and confusion matrices

---

## 📊 Dataset

**MMFi Dataset Subset**:
- **10 subjects** (S01-S10) from Environment E01
- **14 daily activities** (chest expansion, hand raising, waving, picking, throwing, kicking, bowing)
- **WiFi CSI data**: 3 antennas × 114 subcarriers × 10 packets
- **Windowing**: 256 frames with 50% overlap (128 frame step)

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- MMFi dataset downloaded

### Setup

```bash
# Clone repository (or extract files)
cd WiFi-HAR-Adversarial

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tqdm>=4.62.0
tensorboard>=2.8.0
```

---

## 📂 Project Structure

```
WiFi-HAR-Adversarial/
│
├── config.py                    # Configuration management
├── preprocess_data.py          # Data preprocessing pipeline
├── dataset.py                  # PyTorch dataset implementation
├── models.py                   # Neural network architectures
├── train_base.py               # Base model training
├── train_adversarial.py        # Adversarial model training
├── evaluate_loso_cv.py         # LOSO cross-validation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/                       # Data directory
│   └── processed_mmfi_window/  # Processed windowed data
│       ├── *.npy               # Individual windows
│       └── labels.csv          # Labels file
│
├── results/                    # Results directory
│   ├── base_model/            # Base model results
│   ├── adversarial_model/     # Adversarial model results
│   └── loso_*/                # LOSO-CV results
│
├── checkpoints/               # Model checkpoints
└── logs/                      # Training logs
```

---

## 🚀 Quick Start

### 1. Configure Paths

Edit `config.py` to set your data path:

```python
# Line ~21 in config.py
RAW_DATA_ROOT = Path('C:/Users/YourName/path/to/mmfi/E01')
```

### 2. Preprocess Data

```bash
python preprocess_data.py
```

**Output**: Windowed `.npy` files and `labels.csv` in `data/processed_mmfi_window/`

### 3. Train Base Model

```bash
python train_base.py --epochs 50
```

**Output**: Trained model, confusion matrix, training history in `results/base_model/`

### 4. Train Adversarial Model

```bash
python train_adversarial.py --epochs 50 --adversarial_lambda 0.1
```

**Output**: Adversarial model, enhanced confusion matrix in `results/adversarial_model/`

### 5. Run LOSO Cross-Validation

```bash
python evaluate_loso_cv.py --model_type both --epochs 30
```

**Output**: Complete LOSO-CV results comparing base and adversarial models

---

## 📖 Detailed Usage

### Configuration

All parameters are centralized in `config.py`:

```python
# Key parameters
WINDOW_SIZE = 256           # Window size for temporal segmentation
STEP_SIZE = 128            # Step size (50% overlap)
BATCH_SIZE = 32            # Training batch size
LEARNING_RATE = 1e-3       # Learning rate
NUM_EPOCHS = 50            # Training epochs
ADVERSARIAL_LAMBDA = 0.1   # Adversarial loss weight
```

### Preprocessing Options

```bash
# Custom preprocessing
python preprocess_data.py \
    --raw_root /path/to/mmfi/E01 \
    --output_root /path/to/output \
    --window_size 256 \
    --step_size 128
```

### Training Options

#### Base Model
```bash
python train_base.py \
    --labels_csv data/processed_mmfi_window/labels.csv \
    --epochs 30 \
    --batch_size 16 \
    --experiment_name my_base_model
```

#### Adversarial Model
```bash
python train_adversarial.py \
    --labels_csv data/processed_mmfi_window/labels.csv \
    --epochs 30 \
    --batch_size 16 \
    --adversarial_lambda 0.05 \
    --experiment_name my_adv_model
```

### LOSO Cross-Validation

```bash
# Evaluate both models
python evaluate_loso_cv.py --model_type both --epochs 30

# Evaluate only base model
python evaluate_loso_cv.py --model_type base --epochs 30

# Evaluate only adversarial model
python evaluate_loso_cv.py --model_type adversarial --epochs 30
```

---

## 🔬 Model Architecture

### Base CNN-GRU Model

```
Input: [batch, 3, 256, 114, 10]
    ↓
Conv2D Processor (spatial features)
    ├─ Conv2D(1→16, 3×3) + BN + ReLU + MaxPool
    └─ Conv2D(16→32, 3×3) + BN + ReLU + MaxPool
    ↓
Conv1D Encoder (temporal features)
    ├─ Conv1D(input→128, k=7, s=2)
    └─ Conv1D(128→256, k=5, s=2)
    ↓
GRU (sequential modeling)
    └─ GRU(256→128, 1 layer)
    ↓
Classifier
    └─ Dropout(0.5) + Linear(128→14)
```

### Adversarial DANN Model

```
                    Input
                      ↓
            Feature Extractor
         (Shared CNN-GRU)
                      ↓
         ┌────────────┴────────────┐
         ↓                         ↓
  Activity Classifier    Gradient Reversal Layer
   (CrossEntropyLoss)              ↓
                           Domain Discriminator
                           (CrossEntropyLoss)
```

**Loss Function**:
```
Total Loss = Activity Loss + λ × Domain Loss
```

- **λ (lambda)**: Adversarial weight (default: 0.1)
- **Gradient Reversal**: Forces feature extractor to learn domain-invariant features

---

## 📈 Results

### Expected Performance (not real values)

Based on research literature, you should achieve:

**Base Model (Standard Split)**:
- Accuracy: ~85-90%
- F1-Score: ~83-88%

**Base Model (LOSO-CV)**:
- Average Accuracy: ~65-75%
- Significant variance across subjects

**Adversarial Model (LOSO-CV)**:
- Average Accuracy: ~75-80%
- **~10-15% improvement** over base model
- Lower variance across subjects

### Output Files

After training and evaluation, you'll have:

```
results/
├── base_model/
│   ├── best_model.pth
│   ├── base_model_results.json
│   ├── base_model_confusion_matrix.png
│   └── base_model_training_history.png
│
├── adversarial_model/
│   ├── best_model.pth
│   ├── adversarial_model_results.json
│   ├── adversarial_model_confusion_matrix.png
│   └── adversarial_model_training_history.png
│
└── loso_base/ & loso_adversarial/
    ├── loso_cv_results_*.json
    ├── loso_cv_per_fold_*.csv
    └── loso_cv_results_*.png
```

---

## 🎯 Research Publication Tips

### For Your Paper

**1. Problem Statement**:
- Subject-specific biases in WiFi CSI-based HAR
- Poor generalization to unseen subjects
- Need for subject-independent recognition

**2. Methodology**:
- CNN-GRU baseline for temporal CSI processing
- Domain Adversarial Neural Network (DANN) for bias mitigation
- Gradient Reversal Layer for learning domain-invariant features
- Leave-One-Subject-Out Cross-Validation for rigorous evaluation

**3. Key Results to Report**:
- Base model LOSO-CV accuracy
- Adversarial model LOSO-CV accuracy
- Improvement percentage
- Per-subject performance variance
- Confusion matrices for both models
- Statistical significance tests (t-test on fold results)

**4. Visualizations**:
- Training curves (loss and accuracy)
- Confusion matrices
- Per-subject accuracy comparison
- Domain discrimination accuracy (shows adversarial learning)

**5. Ablation Studies** (Optional):
- Effect of adversarial lambda (0.01, 0.1, 0.5, 1.0)
- Effect of window size (128, 256, 512)
- Effect of model architecture components

### Citation Template

```bibtex
@article{yourname2024wifi,
  title={Subject-Independent WiFi CSI-based Human Activity Recognition Using Adversarial Domain Adaptation},
  author={Your Name and Co-authors},
  journal={Target Journal/Conference},
  year={2024}
}
```

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train_base.py --batch_size 16
```

**2. Data Not Found**
```bash
# Check paths in config.py
# Verify MMFi dataset structure
ls C:/Users/Pojesh/Documents/project1/working/mmfi/E01
```

**3. Low Accuracy**
- Check data preprocessing (NaN handling)
- Verify label mappings
- Increase training epochs
- Adjust learning rate

**4. LOSO-CV Takes Too Long**
```bash
# Reduce epochs for LOSO-CV (30 instead of 50)
python evaluate_loso_cv.py --epochs 30
```

---

## 🔧 Advanced Usage

### Custom Model Architecture

Edit `models.py` to modify:
- CNN layers and channels
- GRU hidden size
- Domain discriminator architecture

### Custom Data Augmentation

Edit `dataset.py` to add augmentations:
- Time warping
- Frequency masking
- Amplitude perturbation

### Hyperparameter Tuning

Create a grid search script:
```python
for lr in [1e-4, 1e-3, 1e-2]:
    for lambda_val in [0.01, 0.1, 1.0]:
        # Train with different hyperparameters
```

---

## 📚 References

### Papers
1. MMFi Dataset: Yang et al., "MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset", NeurIPS 2023
2. SenseFi Benchmark: Yang et al., "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing", Patterns 2023
3. Domain Adaptation: Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation", ICML 2015

### Datasets
- [MMFi Dataset](https://github.com/ybhbingo/MMFi_dataset)
- [SenseFi Benchmark](https://github.com/Marsrocky/Awesome-WiFi-CSI-Sensing)

---

## 📝 License

This project is for academic and research purposes. Please cite appropriately if you use this code in your research.

---

## 👥 Contact

For questions or issues:
- Create an issue in the repository
- Email: [your email]

---

## ✅ Checklist for Publication

- [ ] Run preprocessing on full dataset
- [ ] Train base model (50 epochs)
- [ ] Train adversarial model (50 epochs)
- [ ] Run LOSO-CV for both models (30 epochs per fold)
- [ ] Generate all visualizations
- [ ] Compute statistical significance (t-test)
- [ ] Write methodology section
- [ ] Write results section with tables and figures
- [ ] Prepare supplementary materials
- [ ] Review and proofread

---

## 🎉 Acknowledgments

This implementation is based on:
- MMFi Dataset by Yang et al.
- SenseFi Benchmark by Yang et al.
- PyTorch Deep Learning Framework
