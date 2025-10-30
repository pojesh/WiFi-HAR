"""
Complete Pipeline Execution Script
Runs the entire WiFi CSI-based HAR workflow from preprocessing to evaluation
"""

import argparse
import logging
import sys
from pathlib import Path
import subprocess

from config import Config


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str):
    """Execute a command and handle errors"""
    logger.info("="*80)
    logger.info(description)
    logger.info("="*80)
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with error: {e}")
        return False
    except FileNotFoundError:
        logger.error(f"✗ Python script not found. Make sure you're in the correct directory.")
        return False


def check_prerequisites():
    """Check if all necessary files and dependencies exist"""
    logger.info("Checking prerequisites...")
    
    # Check Python files
    required_files = [
        'config.py',
        'preprocess_data.py',
        'dataset.py',
        'models.py',
        'train_base.py',
        'train_adversarial.py',
        'evaluate_loso_cv.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    # Check if raw data exists
    if not Config.RAW_DATA_ROOT.exists():
        logger.warning(f"Raw data directory not found: {Config.RAW_DATA_ROOT}")
        logger.warning("Please update the path in config.py")
        return False
    
    # Check Python dependencies
    try:
        import torch
        import numpy
        import pandas
        import sklearn
        import matplotlib
        import seaborn
        import scipy
        import tqdm
        logger.info("✓ All dependencies are installed")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please run: pip install -r requirements.txt")
        return False
    
    logger.info("✓ All prerequisites satisfied")
    return True


def run_preprocessing(force: bool = False):
    """Run data preprocessing"""
    labels_csv = Config.PROCESSED_DATA_ROOT / 'labels.csv'
    
    if labels_csv.exists() and not force:
        logger.info("Preprocessed data already exists. Use --force to reprocess.")
        return True
    
    cmd = ['python', 'preprocess_data.py']
    return run_command(cmd, "Data Preprocessing")


def run_base_training(epochs: int, batch_size: int, experiment_name: str):
    """Run base model training"""
    cmd = [
        'python', 'train_base.py',
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--experiment_name', experiment_name
    ]
    return run_command(cmd, "Base Model Training")


def run_adversarial_training(epochs: int, batch_size: int, lambda_val: float, experiment_name: str):
    """Run adversarial model training"""
    cmd = [
        'python', 'train_adversarial.py',
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--adversarial_lambda', str(lambda_val),
        '--experiment_name', experiment_name
    ]
    return run_command(cmd, "Adversarial Model Training")


def run_loso_evaluation(model_type: str, epochs: int):
    """Run LOSO cross-validation"""
    cmd = [
        'python', 'evaluate_loso_cv.py',
        '--model_type', model_type,
        '--epochs', str(epochs)
    ]
    return run_command(cmd, f"LOSO Cross-Validation ({model_type})")


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(
        description='Complete WiFi CSI-based HAR Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline (quick mode)
  python run_pipeline.py --mode quick
  
  # Run complete pipeline (full mode)
  python run_pipeline.py --mode full
  
  # Run only preprocessing
  python run_pipeline.py --steps preprocessing
  
  # Run preprocessing and base training
  python run_pipeline.py --steps preprocessing base_training
  
  # Run LOSO-CV only
  python run_pipeline.py --steps loso_cv --loso_model both
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['quick', 'full'],
        default='quick',
        help='Quick (30 epochs) or full (50 epochs) mode'
    )
    
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['preprocessing', 'base_training', 'adversarial_training', 'loso_cv', 'all'],
        default=['all'],
        help='Pipeline steps to execute'
    )
    
    parser.add_argument(
        '--force_preprocess',
        action='store_true',
        help='Force reprocess data even if it exists'
    )
    
    parser.add_argument(
        '--base_epochs',
        type=int,
        default=None,
        help='Epochs for base model training'
    )
    
    parser.add_argument(
        '--adv_epochs',
        type=int,
        default=None,
        help='Epochs for adversarial model training'
    )
    
    parser.add_argument(
        '--loso_epochs',
        type=int,
        default=None,
        help='Epochs per fold for LOSO-CV'
    )
    
    parser.add_argument(
        '--loso_model',
        type=str,
        choices=['base', 'adversarial', 'both'],
        default='both',
        help='Model type for LOSO-CV'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=Config.BATCH_SIZE,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--adversarial_lambda',
        type=float,
        default=Config.ADVERSARIAL_LAMBDA,
        help='Adversarial loss weight'
    )
    
    args = parser.parse_args()
    
    # Set epochs based on mode
    if args.mode == 'quick':
        base_epochs = args.base_epochs or 30
        adv_epochs = args.adv_epochs or 30
        loso_epochs = args.loso_epochs or 20
    else:  # full
        base_epochs = args.base_epochs or 50
        adv_epochs = args.adv_epochs or 50
        loso_epochs = args.loso_epochs or 30
    
    # Determine steps
    steps = args.steps
    if 'all' in steps:
        steps = ['preprocessing', 'base_training', 'adversarial_training', 'loso_cv']
    
    # Print configuration
    logger.info("="*80)
    logger.info("WiFi CSI-based HAR Pipeline")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Steps: {', '.join(steps)}")
    logger.info(f"Base epochs: {base_epochs}")
    logger.info(f"Adversarial epochs: {adv_epochs}")
    logger.info(f"LOSO epochs: {loso_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Adversarial lambda: {args.adversarial_lambda}")
    logger.info("="*80)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed. Please fix the issues and try again.")
        sys.exit(1)
    
    # Execute pipeline steps
    success = True
    
    if 'preprocessing' in steps:
        if not run_preprocessing(args.force_preprocess):
            logger.error("Preprocessing failed. Stopping pipeline.")
            sys.exit(1)
    
    if 'base_training' in steps:
        if not run_base_training(base_epochs, args.batch_size, 'base_model'):
            logger.error("Base model training failed.")
            success = False
    
    if 'adversarial_training' in steps:
        if not run_adversarial_training(
            adv_epochs, args.batch_size, args.adversarial_lambda, 'adversarial_model'
        ):
            logger.error("Adversarial model training failed.")
            success = False
    
    if 'loso_cv' in steps:
        if not run_loso_evaluation(args.loso_model, loso_epochs):
            logger.error("LOSO cross-validation failed.")
            success = False
    
    # Summary
    logger.info("\n" + "="*80)
    if success:
        logger.info("✓ Pipeline completed successfully!")
        logger.info("="*80)
        logger.info("\nResults are available in:")
        logger.info(f"  - {Config.RESULTS_ROOT}")
        logger.info(f"  - {Config.CHECKPOINT_ROOT}")
        logger.info(f"  - {Config.LOGS_ROOT}")
    else:
        logger.error("✗ Pipeline completed with errors.")
        logger.error("Please check the logs above for details.")
        logger.info("="*80)
        sys.exit(1)


if __name__ == '__main__':
    main()