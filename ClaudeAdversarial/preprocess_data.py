"""
Data Preprocessing Script for WiFi CSI-based HAR
Handles raw MMFi dataset and creates windowed samples with proper normalization
"""

import os
import numpy as np
import scipy.io as scio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

from config import Config


# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor for MMFi WiFi CSI data
    Converts raw .mat files to windowed numpy arrays with proper normalization
    """
    
    def __init__(
        self,
        raw_root: Path = Config.RAW_DATA_ROOT,
        output_root: Path = Config.PROCESSED_DATA_ROOT,
        window_size: int = Config.WINDOW_SIZE,
        step_size: int = Config.STEP_SIZE
    ):
        self.raw_root = Path(raw_root)
        self.output_root = Path(output_root)
        self.window_size = window_size
        self.step_size = step_size
        
        # Create output directory
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Statistics for global normalization
        self.global_min = None
        self.global_max = None
        
        logger.info(f"Initialized preprocessor with window_size={window_size}, step_size={step_size}")
    
    def load_csi_frame(self, mat_file: Path) -> np.ndarray:
        """
        Load a single CSI frame from .mat file
        
        Args:
            mat_file: Path to .mat file
            
        Returns:
            CSI amplitude data as numpy array
        """
        try:
            mat_data = scio.loadmat(str(mat_file))
            csi_amp = mat_data['CSIamp']
            
            # Handle infinity values
            csi_amp[np.isinf(csi_amp)] = np.nan
            
            return csi_amp
        except Exception as e:
            logger.error(f"Error loading {mat_file}: {e}")
            return None
    
    def handle_nan_values(self, csi_data: np.ndarray) -> np.ndarray:
        """
        Handle NaN values in CSI data using intelligent imputation
        
        Args:
            csi_data: CSI amplitude data with potential NaN values
            
        Returns:
            CSI data with NaN values handled
        """
        # Process each subcarrier column independently
        for i in range(csi_data.shape[-1]):
            temp_col = csi_data[:, :, i]
            nan_count = np.count_nonzero(np.isnan(temp_col))
            
            if nan_count > 0:
                # Use column mean for imputation
                temp_not_nan = temp_col[~np.isnan(temp_col)]
                if len(temp_not_nan) > 0:
                    mean_val = temp_not_nan.mean()
                    temp_col[np.isnan(temp_col)] = mean_val
                else:
                    # If entire column is NaN, use zero
                    temp_col[np.isnan(temp_col)] = Config.FILL_NAN_VALUE
                
                csi_data[:, :, i] = temp_col
        
        # Final check: replace any remaining NaN with zero
        csi_data = np.nan_to_num(csi_data, nan=Config.FILL_NAN_VALUE)
        
        return csi_data
    
    def normalize_data(self, csi_data: np.ndarray, global_norm: bool = True) -> np.ndarray:
        """
        Normalize CSI data using min-max normalization
        
        Args:
            csi_data: CSI amplitude data
            global_norm: If True, use global statistics; else per-sequence
            
        Returns:
            Normalized CSI data
        """
        if global_norm and self.global_min is not None:
            # Use pre-computed global statistics
            min_val = self.global_min
            max_val = self.global_max
        else:
            # Compute local statistics
            min_val = np.min(csi_data)
            max_val = np.max(csi_data)
        
        # Avoid division by zero
        if (max_val - min_val) < 1e-8:
            logger.warning("Near-zero range detected, returning zeros")
            return np.zeros_like(csi_data)
        
        normalized = (csi_data - min_val) / (max_val - min_val)
        return normalized.astype(np.float32)
    
    def create_windows(
        self,
        csi_sequence: np.ndarray,
        subject: str,
        activity: str
    ) -> List[Tuple[np.ndarray, str, str, int]]:
        """
        Create sliding windows from CSI sequence
        
        Args:
            csi_sequence: Full CSI sequence [frames, antennas, subcarriers, packets]
            subject: Subject ID
            activity: Activity label
            
        Returns:
            List of (window, subject, activity, start_idx) tuples
        """
        num_frames = csi_sequence.shape[0]
        windows = []
        
        for start in range(0, num_frames - self.window_size + 1, self.step_size):
            window = csi_sequence[start:start + self.window_size]
            windows.append((window, subject, activity, start))
        
        return windows
    
    def process_subject_activity(
        self,
        subject: str,
        activity: str
    ) -> List[Tuple[np.ndarray, str, str, int]]:
        """
        Process all CSI frames for a given subject-activity pair
        
        Args:
            subject: Subject ID (e.g., 'S01')
            activity: Activity label (e.g., 'A02')
            
        Returns:
            List of processed windows
        """
        # Construct path to CSI data
        csi_dir = self.raw_root / subject / activity / 'wifi-csi'
        
        if not csi_dir.exists():
            logger.warning(f"Directory not found: {csi_dir}")
            return []
        
        # Get all frame files
        frame_files = sorted(csi_dir.glob('frame*.mat'))
        
        if len(frame_files) == 0:
            logger.warning(f"No frame files found in {csi_dir}")
            return []
        
        # Load all frames
        all_frames = []
        for frame_file in frame_files:
            csi_frame = self.load_csi_frame(frame_file)
            if csi_frame is not None:
                # Add time dimension: [antennas, subcarriers, packets] -> [1, antennas, subcarriers, packets]
                all_frames.append(csi_frame[np.newaxis, ...])
        
        if len(all_frames) == 0:
            logger.warning(f"No valid frames loaded for {subject}-{activity}")
            return []
        
        # Concatenate along time axis: [frames, antennas, subcarriers, packets]
        csi_sequence = np.concatenate(all_frames, axis=0)
        
        # Handle NaN values
        csi_sequence = self.handle_nan_values(csi_sequence)
        
        # Create windows
        windows = self.create_windows(csi_sequence, subject, activity)
        
        logger.info(f"Processed {subject}-{activity}: {len(all_frames)} frames -> {len(windows)} windows")
        
        return windows
    
    def compute_global_statistics(self, all_windows: List[Tuple[np.ndarray, str, str, int]]):
        """
        Compute global min and max for normalization
        
        Args:
            all_windows: List of all window data
        """
        logger.info("Computing global statistics for normalization...")
        
        all_values = []
        for window, _, _, _ in tqdm(all_windows, desc="Computing statistics"):
            all_values.append(window.flatten())
        
        all_values = np.concatenate(all_values)
        
        self.global_min = np.min(all_values)
        self.global_max = np.max(all_values)
        
        logger.info(f"Global statistics: min={self.global_min:.6f}, max={self.global_max:.6f}")
    
    def save_windows(
        self,
        windows: List[Tuple[np.ndarray, str, str, int]],
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Save windows to disk and create labels CSV
        
        Args:
            windows: List of (window, subject, activity, start) tuples
            normalize: Whether to normalize data
            
        Returns:
            DataFrame with file paths and labels
        """
        records = []
        
        for window, subject, activity, start_idx in tqdm(windows, desc="Saving windows"):
            # Normalize if requested
            if normalize:
                window = self.normalize_data(window, global_norm=(Config.NORMALIZATION == 'global'))
            
            # Create filename
            filename = f"{subject}_{activity}_{start_idx:06d}.npy"
            filepath = self.output_root / filename
            
            # Save window
            np.save(filepath, window)
            
            # Record metadata
            records.append({
                'file': str(filepath),
                'subject': subject,
                'activity': activity,
                'start_idx': start_idx
            })
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Save labels CSV
        labels_csv = self.output_root / 'labels.csv'
        df.to_csv(labels_csv, index=False)
        
        logger.info(f"Saved {len(records)} windows and labels to {labels_csv}")
        
        return df
    
    def process_all(self) -> pd.DataFrame:
        """
        Process entire dataset
        
        Returns:
            DataFrame with all processed windows
        """
        logger.info("Starting preprocessing pipeline...")
        logger.info(f"Processing {len(Config.SUBJECTS)} subjects and {len(Config.DAILY_ACTIVITIES)} activities")
        
        # Collect all windows
        all_windows = []
        
        for subject in Config.SUBJECTS:
            for activity in Config.DAILY_ACTIVITIES:
                windows = self.process_subject_activity(subject, activity)
                all_windows.extend(windows)
        
        if len(all_windows) == 0:
            logger.error("No windows created! Check your data path.")
            return pd.DataFrame()
        
        logger.info(f"Total windows created: {len(all_windows)}")
        
        # Compute global statistics if using global normalization
        if Config.NORMALIZATION == 'global':
            self.compute_global_statistics(all_windows)
        
        # Save windows
        df = self.save_windows(all_windows, normalize=True)
        
        # Print summary statistics
        self.print_summary(df)
        
        logger.info("Preprocessing complete!")
        
        return df
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics of processed data"""
        print("\n" + "="*80)
        print("Dataset Summary")
        print("="*80)
        print(f"Total windows: {len(df)}")
        print(f"\nWindows per subject:")
        print(df['subject'].value_counts().sort_index())
        print(f"\nWindows per activity:")
        print(df['activity'].value_counts().sort_index())
        print("="*80 + "\n")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess MMFi CSI data')
    parser.add_argument(
        '--raw_root',
        type=str,
        default=str(Config.RAW_DATA_ROOT),
        help='Root directory containing raw MMFi .mat files'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default=str(Config.PROCESSED_DATA_ROOT),
        help='Output directory for processed .npy files'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=Config.WINDOW_SIZE,
        help='Window size for segmentation'
    )
    parser.add_argument(
        '--step_size',
        type=int,
        default=Config.STEP_SIZE,
        help='Step size for sliding window'
    )
    
    args = parser.parse_args()
    
    # Update config if arguments provided
    if args.raw_root != str(Config.RAW_DATA_ROOT):
        Config.RAW_DATA_ROOT = Path(args.raw_root)
    if args.output_root != str(Config.PROCESSED_DATA_ROOT):
        Config.PROCESSED_DATA_ROOT = Path(args.output_root)
    
    # Create preprocessor
    preprocessor = DataPreprocessor(
        raw_root=Config.RAW_DATA_ROOT,
        output_root=Config.PROCESSED_DATA_ROOT,
        window_size=args.window_size,
        step_size=args.step_size
    )
    
    # Process all data
    df = preprocessor.process_all()
    
    if len(df) > 0:
        logger.info("✓ Preprocessing successful!")
    else:
        logger.error("✗ Preprocessing failed!")


if __name__ == '__main__':
    main()