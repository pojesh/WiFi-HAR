"""
Environment Verification Script
Checks if the environment is properly set up for running the WiFi HAR pipeline
"""

import sys
from pathlib import Path
import logging


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(text)
    print("="*80)


def check_python_version():
    """Check Python version"""
    print_header("Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Python version: {version_str}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8 or higher is required!")
        return False
    
    logger.info("✓ Python version is compatible")
    return True


def check_dependencies():
    """Check if all required dependencies are installed"""
    print_header("Checking Dependencies")
    
    dependencies = {
        'torch': 'PyTorch (Deep Learning)',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy (Numerical Computing)',
        'scipy': 'SciPy (Scientific Computing)',
        'pandas': 'Pandas (Data Analysis)',
        'matplotlib': 'Matplotlib (Plotting)',
        'seaborn': 'Seaborn (Statistical Visualization)',
        'sklearn': 'Scikit-learn (Machine Learning)',
        'tqdm': 'TQDM (Progress Bars)',
        'tensorboard': 'TensorBoard (Logging)'
    }
    
    missing = []
    installed = []
    
    for module, description in dependencies.items():
        try:
            if module == 'sklearn':
                __import__('sklearn')
            else:
                __import__(module)
            print(f"✓ {description:40s} - Installed")
            installed.append(module)
        except ImportError:
            print(f"✗ {description:40s} - Missing")
            missing.append(module)
    
    if missing:
        logger.error(f"\nMissing dependencies: {', '.join(missing)}")
        logger.error("Please run: pip install -r requirements.txt")
        return False
    
    logger.info(f"\n✓ All {len(installed)} dependencies are installed")
    return True


def check_pytorch_cuda():
    """Check PyTorch CUDA availability"""
    print_header("PyTorch CUDA Status")
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            logger.info("✓ CUDA is available (GPU acceleration enabled)")
        else:
            logger.warning("⚠ CUDA not available (will use CPU - training will be slower)")
            logger.warning("  For GPU support, install PyTorch with CUDA:")
            logger.warning("  Visit: https://pytorch.org/get-started/locally/")
        
        return True
    except Exception as e:
        logger.error(f"Error checking PyTorch: {e}")
        return False


def check_project_files():
    """Check if all required project files exist"""
    print_header("Checking Project Files")
    
    required_files = {
        'config.py': 'Configuration file',
        'preprocess_data.py': 'Data preprocessing script',
        'dataset.py': 'PyTorch dataset implementation',
        'models.py': 'Neural network models',
        'train_base.py': 'Base model training script',
        'train_adversarial.py': 'Adversarial training script',
        'evaluate_loso_cv.py': 'LOSO-CV evaluation script',
        'requirements.txt': 'Dependencies list',
        'README.md': 'Documentation'
    }
    
    missing = []
    found = []
    
    for file, description in required_files.items():
        if Path(file).exists():
            print(f"✓ {description:40s} - Found")
            found.append(file)
        else:
            print(f"✗ {description:40s} - Missing")
            missing.append(file)
    
    if missing:
        logger.error(f"\nMissing files: {', '.join(missing)}")
        logger.error("Please ensure all project files are in the current directory")
        return False
    
    logger.info(f"\n✓ All {len(found)} project files are present")
    return True


def check_data_directory():
    """Check data directory setup"""
    print_header("Checking Data Directory")
    
    try:
        from config import Config
        
        print(f"Raw data root: {Config.RAW_DATA_ROOT}")
        print(f"Processed data root: {Config.PROCESSED_DATA_ROOT}")
        
        if Config.RAW_DATA_ROOT.exists():
            # Check for expected structure
            sample_files = list(Config.RAW_DATA_ROOT.glob('S*/A*/wifi-csi/frame*.mat'))
            if sample_files:
                logger.info(f"✓ Raw data directory found with {len(sample_files)} sample files")
            else:
                logger.warning("⚠ Raw data directory exists but no .mat files found")
                logger.warning("  Please verify the MMFi dataset structure")
        else:
            logger.warning("⚠ Raw data directory not found")
            logger.warning(f"  Expected location: {Config.RAW_DATA_ROOT}")
            logger.warning("  Please download the MMFi dataset and update config.py")
        
        # Check if processed data exists
        labels_csv = Config.PROCESSED_DATA_ROOT / 'labels.csv'
        if labels_csv.exists():
            logger.info("✓ Preprocessed data found (can skip preprocessing step)")
        else:
            logger.info("ℹ No preprocessed data found (will need to run preprocessing)")
        
        return True
    except Exception as e:
        logger.error(f"Error checking data directory: {e}")
        return False


def test_model_creation():
    """Test if models can be created"""
    print_header("Testing Model Creation")
    
    try:
        import torch
        from models import CNNGRUBase, AdversarialDANN
        
        # Test base model
        print("Creating base model...")
        base_model = CNNGRUBase()
        base_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print(f"✓ Base model created ({base_params:,} parameters)")
        
        # Test adversarial model
        print("Creating adversarial model...")
        adv_model = AdversarialDANN()
        adv_params = sum(p.numel() for p in adv_model.parameters() if p.requires_grad)
        print(f"✓ Adversarial model created ({adv_params:,} parameters)")
        
        # Test forward pass
        print("Testing forward pass...")
        dummy_input = torch.randn(2, 3, 256, 114, 10)
        
        with torch.no_grad():
            base_output = base_model(dummy_input)
            adv_activity, adv_domain = adv_model(dummy_input)
        
        print(f"✓ Base model output shape: {base_output.shape}")
        print(f"✓ Adversarial model outputs: {adv_activity.shape}, {adv_domain.shape}")
        
        logger.info("\n✓ Models can be created and used successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Error testing models: {e}")
        logger.error("Please check the models.py file for errors")
        return False


def test_data_loading():
    """Test if data can be loaded (if preprocessed data exists)"""
    print_header("Testing Data Loading")
    
    try:
        from config import Config
        from dataset import WiFiCSIDataset
        
        labels_csv = Config.PROCESSED_DATA_ROOT / 'labels.csv'
        
        if not labels_csv.exists():
            logger.info("ℹ No preprocessed data found (skipping data loading test)")
            logger.info("  Run preprocessing first: python preprocess_data.py")
            return True
        
        print("Loading dataset...")
        dataset = WiFiCSIDataset(labels_csv)
        print(f"✓ Dataset loaded with {len(dataset)} samples")
        
        print("Testing data sample...")
        data, activity, subject = dataset[0]
        print(f"✓ Sample shape: {data.shape}")
        print(f"✓ Activity label: {activity.item()}, Subject label: {subject.item()}")
        
        logger.info("\n✓ Data loading works correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Error testing data loading: {e}")
        logger.error("Please check if preprocessing completed successfully")
        return False


def print_summary(results):
    """Print test summary"""
    print_header("Test Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    print("\nTest Results:")
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8s} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*80)
        print("🎉 Environment is ready! You can now run the pipeline.")
        print("="*80)
        print("\nQuick start:")
        print("  1. Update data path in config.py (if needed)")
        print("  2. Run preprocessing: python preprocess_data.py")
        print("  3. Run pipeline: python run_pipeline.py --mode quick")
        print("\nFor detailed instructions, see README.md")
    else:
        print("\n" + "="*80)
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("="*80)
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Update data path in config.py")
        print("  - Ensure all project files are present")


def main():
    """Run all tests"""
    print("="*80)
    print("WiFi CSI-based HAR - Environment Verification")
    print("="*80)
    print("\nThis script will verify that your environment is properly set up.")
    print("It will check:")
    print("  - Python version")
    print("  - Required dependencies")
    print("  - PyTorch CUDA availability")
    print("  - Project files")
    print("  - Data directory")
    print("  - Model creation")
    print("  - Data loading (if preprocessed data exists)")
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'PyTorch CUDA': check_pytorch_cuda(),
        'Project Files': check_project_files(),
        'Data Directory': check_data_directory(),
        'Model Creation': test_model_creation(),
        'Data Loading': test_data_loading()
    }
    
    print_summary(results)
    
    # Return exit code
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()