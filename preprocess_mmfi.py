import os
import glob
import numpy as np
import scipy.io as scio
from tqdm import tqdm
import pickle
import json

class MMFiCSIPreprocessor:
    """
    Preprocess and store MMFi E01 WiFi CSI data using official MMFi methodology
    """
    def __init__(self, dataset_root, environment='E01'):
        """
        Args:
            dataset_root: Root directory of MMFi dataset
            environment: Environment to process (default: 'E01')
        """
        self.dataset_root = dataset_root
        self.environment = environment
        self.env_path = os.path.join(dataset_root, environment)
        
        # Activity mapping
        self.activity_names = {
            'A01': 'Stretching and relaxing',
            'A02': 'Chest expansion (horizontal)',
            'A03': 'Chest expansion (vertical)',
            'A04': 'Twist (left)',
            'A05': 'Twist (right)',
            'A06': 'Mark time',
            'A07': 'Limb extension (left)',
            'A08': 'Limb extension (right)',
            'A09': 'Lunge (toward left-front)',
            'A10': 'Lunge (toward right-front)',
            'A11': 'Limb extension (both)',
            'A12': 'Squat',
            'A13': 'Raising hand (left)',
            'A14': 'Raising hand (right)',
            'A15': 'Lunge (toward left side)',
            'A16': 'Lunge (toward right side)',
            'A17': 'Waving hand (left)',
            'A18': 'Waving hand (right)',
            'A19': 'Picking up things',
            'A20': 'Throwing (toward left side)',
            'A21': 'Throwing (toward right side)',
            'A22': 'Kicking (toward left side)',
            'A23': 'Kicking (toward right side)',
            'A24': 'Body extension (left)',
            'A25': 'Body extension (right)',
            'A26': 'Jumping up',
            'A27': 'Bowing'
        }
        
    def preprocess_csi_frame(self, csi_mat_path):
        """
        Preprocess single CSI frame using official MMFi method
        
        Args:
            csi_mat_path: Path to frame*.mat file
            
        Returns:
            Preprocessed CSI data as numpy array
        """
        # Load CSI amplitude from .mat file
        data = scio.loadmat(csi_mat_path)['CSIamp']
        
        # Replace inf values with nan
        data[np.isinf(data)] = np.nan
        
        # Handle nan values for each antenna (10 antennas as per MMFi)
        for i in range(10):  # 10 antennas in MMFi WiFi CSI
            temp_col = data[:, :, i]
            nan_num = np.count_nonzero(temp_col != temp_col)
            
            if nan_num != 0:
                # Get non-nan values
                temp_not_nan_col = temp_col[temp_col == temp_col]
                # Replace nan with mean of non-nan values
                temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
        
        # Min-max normalization to [0, 1]
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        return np.array(data)
    
    def process_activity_sequence(self, activity_dir):
        """
        Process all frames in an activity sequence
        
        Args:
            activity_dir: Path to activity directory containing wifi-csi folder
            
        Returns:
            List of preprocessed frames, or None if no frames found
        """
        wifi_csi_dir = os.path.join(activity_dir, 'wifi-csi')
        
        if not os.path.exists(wifi_csi_dir):
            return None
        
        # Get all frame*.mat files, sorted
        frame_files = sorted(glob.glob(os.path.join(wifi_csi_dir, "frame*.mat")))
        
        if len(frame_files) == 0:
            return None
        
        # Process each frame
        frames = []
        for frame_file in frame_files:
            try:
                frame_data = self.preprocess_csi_frame(frame_file)
                frames.append(frame_data)
            except Exception as e:
                print(f"Error processing {frame_file}: {e}")
                continue
        
        if len(frames) == 0:
            return None
            
        return np.array(frames)
    
    def process_and_save_dataset(self, output_dir='processed_mmfi_e01'):
        """
        Process entire E01 dataset and save to disk
        
        Args:
            output_dir: Directory to save processed data
            
        Returns:
            Dictionary with dataset statistics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Subjects in E01
        subjects = [f'S{i:02d}' for i in range(1, 11)]  # S01 to S10
        # Activities
        activities = [f'A{i:02d}' for i in range(1, 28)]  # A01 to A27
        
        # Storage for all processed data
        all_data = []
        all_labels = []
        all_metadata = []
        
        # Statistics
        stats = {
            'total_sequences': 0,
            'total_frames': 0,
            'subjects': {},
            'activities': {},
            'frame_shapes': []
        }
        
        print(f"Processing {self.environment} dataset...")
        print(f"Subjects: {len(subjects)}, Activities: {len(activities)}")
        
        # Process each subject
        for subject in tqdm(subjects, desc='Processing subjects'):
            subject_path = os.path.join(self.env_path, subject)
            
            if not os.path.exists(subject_path):
                print(f"Warning: {subject} not found")
                continue
            
            stats['subjects'][subject] = {'sequences': 0, 'frames': 0}
            
            # Process each activity
            for activity in tqdm(activities, desc=f'{subject} activities', leave=False):
                activity_path = os.path.join(subject_path, activity)
                
                if not os.path.exists(activity_path):
                    continue
                
                # Process activity sequence
                sequence_data = self.process_activity_sequence(activity_path)
                
                if sequence_data is None:
                    continue
                
                # Store data
                all_data.append(sequence_data)
                
                # Store label (activity index: A01->0, A02->1, ..., A27->26)
                activity_label = int(activity[1:]) - 1
                all_labels.append(activity_label)
                
                # Store metadata
                metadata = {
                    'subject': subject,
                    'subject_id': int(subject[1:]) - 1,  # S01->0, S02->1, ...
                    'activity': activity,
                    'activity_id': activity_label,
                    'activity_name': self.activity_names[activity],
                    'num_frames': len(sequence_data),
                    'frame_shape': sequence_data.shape[1:]  # (height, width, channels)
                }
                all_metadata.append(metadata)
                
                # Update statistics
                stats['total_sequences'] += 1
                stats['total_frames'] += len(sequence_data)
                stats['subjects'][subject]['sequences'] += 1
                stats['subjects'][subject]['frames'] += len(sequence_data)
                
                if activity not in stats['activities']:
                    stats['activities'][activity] = {'sequences': 0, 'frames': 0}
                stats['activities'][activity]['sequences'] += 1
                stats['activities'][activity]['frames'] += len(sequence_data)
                
                if len(stats['frame_shapes']) == 0:
                    stats['frame_shapes'].append(sequence_data.shape[1:])
        
        print(f"\n✓ Processed {stats['total_sequences']} sequences")
        print(f"✓ Total frames: {stats['total_frames']}")
        print(f"✓ Frame shape: {stats['frame_shapes'][0] if stats['frame_shapes'] else 'N/A'}")
        
        # Save processed data
        print("\nSaving processed data...")
        
        # Save as numpy arrays
        np.save(os.path.join(output_dir, 'data.npy'), np.array(all_data, dtype=object))
        np.save(os.path.join(output_dir, 'labels.npy'), np.array(all_labels))
        
        # Save metadata
        with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(all_metadata, f)
        
        # Save statistics
        with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
            # Convert to serializable format
            serializable_stats = {
                'total_sequences': stats['total_sequences'],
                'total_frames': stats['total_frames'],
                'frame_shape': stats['frame_shapes'][0] if stats['frame_shapes'] else None,
                'subjects': stats['subjects'],
                'activities': stats['activities']
            }
            json.dump(serializable_stats, f, indent=2)
        
        print(f"✓ Data saved to {output_dir}/")
        print(f"  - data.npy: {stats['total_sequences']} sequences")
        print(f"  - labels.npy: {stats['total_sequences']} labels")
        print(f"  - metadata.pkl: Detailed metadata")
        print(f"  - statistics.json: Dataset statistics")
        
        return stats

# Usage example for preprocessing
if __name__ == '__main__':
    # Initialize preprocessor
    preprocessor = MMFiCSIPreprocessor(
        dataset_root='C:\\Users\\Pojesh\\Documents\\project1\\working\\mmfi',  #path to mmfi - mmfi/e01/s01/a01/wifi-csi/frames mat
        environment='E01'
    )
    
    # Process and save
    stats = preprocessor.process_and_save_dataset(
        output_dir='processed_mmfi_e01'
    )

'''
OUTPUT
Processing E01 dataset...
Subjects: 10, Activities: 27
Processing subjects: 100%|██████████| 10/10 [12:36<00:00, 75.66s/it]

✓ Processed 270 sequences
✓ Total frames: 80190
✓ Frame shape: (3, 114, 10)

Saving processed data...
✓ Data saved to processed_mmfi_e01/
  - data.npy: 270 sequences
  - labels.npy: 270 labels
  - metadata.pkl: Detailed metadata
  - statistics.json: Dataset statistics
'''