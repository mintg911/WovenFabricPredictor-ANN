"""
utils.py
Utility module: contains general utility functions
"""

import os
import random
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def set_seed(seed=42):
    """
    Set random seed to ensure experiment reproducibility
  
    Args:
        seed: random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def print_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    Memory: {memory_total:.2f} GB")
    else:
        print("CUDA not available, using CPU")


def calculate_metrics(y_true, y_pred):
    """
    Calculate all evaluation metrics
  
    Args:
        y_true: true values
        y_pred: predicted values
  
    Returns:
        dict: dictionary containing all metrics
    """
    # Flatten arrays
    if hasattr(y_true, 'values'):
        y_true = y_true.values.flatten()
    else:
        y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
  
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
  
    # Calculate MAPE, avoid division by zero
    mask = np.abs(y_true) > 1e-8
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf
  
    return {
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }


def print_metrics(metrics, title="Model Evaluation Metrics"):
    """
    Print evaluation metrics
  
    Args:
        metrics: metrics dictionary
        title: title
    """
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(f"R² Score:  {metrics['r2']:.6f}")
    print(f"MSE:       {metrics['mse']:.6f}")
    print(f"RMSE:      {metrics['rmse']:.6f}")
    print(f"MAE:       {metrics['mae']:.6f}")
    print(f"MAPE:      {metrics['mape']:.2f}%")
    print("=" * 50)


def print_final_metrics(y_test, y_pred):
    """
    Print final evaluation metrics (compatible with original interface)
  
    Args:
        y_test: true values
        y_pred: predicted values
    """
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "Final Model Evaluation Metrics")


def count_parameters(model):
    """
    Count model parameters
  
    Args:
        model: PyTorch model
  
    Returns:
        total: total parameter count
        trainable: trainable parameter count
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  
    print(f"Model parameter statistics:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
  
    return total, trainable


def ensure_dir(path):
    """
    Ensure directory exists, create if not exists
  
    Args:
        path: directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")


class EarlyStopping:
    """
    Early stopping mechanism: stop training when validation metrics no longer improve
    """
  
    def __init__(self, patience=10, min_delta=0, mode='max', verbose=True):
        """
        Initialize early stopping mechanism
      
        Args:
            patience: number of epochs to tolerate
            min_delta: minimum improvement amount
            mode: 'max' means higher metric is better, 'min' means lower is better
            verbose: whether to print information
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
  
    def __call__(self, score):
        """
        Check if early stopping should be triggered
      
        Args:
            score: current metric value
      
        Returns:
            bool: whether early stopping should be triggered
        """
        if self.best_score is None:
            self.best_score = score
            return False
      
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
      
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
      
        return self.early_stop


class AverageMeter:
    """
    Compute and store average and current value
    """
  
    def __init__(self, name='Metric'):
        self.name = name
        self.reset()
  
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
  
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
  
    def __str__(self):
        return f'{self.name}: {self.avg:.6f}'