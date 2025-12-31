"""
config.py
Configuration module: Centralized management of all hyperparameters and paths
"""

import os
import torch


class Config:
    """Configuration class: Centralized management of all hyperparameters and paths"""
  
    # ==================== Path Configuration ====================
    # Project root directory (automatically obtained from current file location)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  
    # Data paths
    # Shear data
    DATA_PATH = os.path.join(BASE_DIR, 'data_shear.npy')
    # Tensile data
    # DATA_PATH = os.path.join(BASE_DIR, 'data_axial.npy')
  
    # Checkpoint directory
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'model.pth')
    SCALER_X_PATH = os.path.join(CHECKPOINT_DIR, 'scaler_X.joblib')
    SCALER_Y_PATH = os.path.join(CHECKPOINT_DIR, 'scaler_y.joblib')
  
    # Output directory
    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
    RESULTS_PATH = os.path.join(OUTPUT_DIR, 'training_results.xlsx')
  
    # ==================== Data Parameters ====================
    INPUT_FEATURES = ['Strain', 'Up', 'Down', 'Step', 'Angle']
    TARGET_FEATURE = 'Stress'
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
  
    # ==================== Model Parameters ====================
    INPUT_DIM = 5
    HIDDEN_DIMS = [32, 64, 32]
    OUTPUT_DIM = 1
  
    # ==================== Training Parameters ====================
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 5000
    PRINT_INTERVAL = 100
  
    # ==================== Device Configuration ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = torch.device('cpu')  # Uncomment to force CPU usage
  
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        print(f"Directories created/confirmed: {cls.CHECKPOINT_DIR}, {cls.OUTPUT_DIR}")
  
    @classmethod
    def print_config(cls):
        """Print current configuration information"""
        print("\n" + "=" * 60)
        print("Current Configuration Information")
        print("=" * 60)
        print(f"Data path: {cls.DATA_PATH}")
        print(f"Model save path: {cls.MODEL_PATH}")
        print(f"Device: {cls.DEVICE}")
        print(f"Hidden layer structure: {cls.HIDDEN_DIMS}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Training epochs: {cls.EPOCHS}")
        print("=" * 60 + "\n")