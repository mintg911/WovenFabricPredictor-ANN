"""
predict.py
Simplified prediction script: Load trained model and make predictions
Function: Input parameters to plot prediction curves and compare with original data
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import warnings

# Suppress libpng warning
warnings.filterwarnings("ignore", message="libpng warning: iCCP: cHRM chunk does not match sRGB")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from config import Config
from model import load_model
from data_processor import DataProcessor
from utils import set_seed, print_gpu_info

class Predictor:
    """Predictor class: Load model and make predictions"""
    
    def __init__(self, config=None):
        """
        Initialize predictor
        
        Args:
            config: Configuration object, defaults to Config()
        """
        if config is None:
            config = Config()
        
        self.config = config
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.raw_data = None  # Store original data for comparison
    
    def load(self, model_path=None, scaler_X_path=None, scaler_y_path=None):
        """
        Load model and scalers
        
        Args:
            model_path: Model file path
            scaler_X_path: X scaler file path
            scaler_y_path: y scaler file path
        """
        print("\n" + "=" * 50)
        print("Loading prediction model...")
        print("=" * 50)
        
        # Load model
        if model_path is None:
            model_path = self.config.MODEL_PATH
        
        self.model = load_model(self.config, model_path)
        self.model.eval()
        
        # Load scalers
        self.processor = DataProcessor(self.config)
        self.processor.load_scalers(scaler_X_path, scaler_y_path)
        
        # Load original data
        # self.raw_data = pd.read_excel(self.config.DATA_PATH)
        self.raw_data = self.processor.load_data(self.config.DATA_PATH)
        
        self.is_loaded = True
        print("Model and scalers loaded successfully!")
    
    def predict_stress_strain_curve(self, up, down, step, angle, 
                                     strain_min=0.0, strain_max=0.1, n_points=100):
        """
        Predict complete stress-strain curve
        
        Args:
            up: Up parameter
            down: Down parameter
            step: Step parameter
            angle: Angle parameter
            strain_min: Minimum strain value
            strain_max: Maximum strain value
            n_points: Number of sampling points
        
        Returns:
            strain_values: Strain values array
            stress_values: Stress values array
        """
        if not self.is_loaded:
            raise RuntimeError("Please call load() method to load model first!")
        
        # Generate strain sequence
        strain_values = np.linspace(strain_min, strain_max, n_points)
        
        # Construct input data
        input_data = pd.DataFrame({
            'Strain': strain_values,
            'Up': [up] * n_points,
            'Down': [down] * n_points,
            'Step': [step] * n_points,
            'Angle': [angle] * n_points
        })
        
        # Preprocess
        X_tensor = self.processor.preprocess_for_prediction(input_data)
        X_tensor = X_tensor.to(self.config.DEVICE)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            y_pred_norm = self.model(X_tensor)
            y_pred = self.processor.inverse_transform_prediction(y_pred_norm)
        
        return strain_values, y_pred.flatten()
    
    def get_raw_data_for_params(self, up, down, step, angle):
        """
        Get original data matching the parameters
        
        Args:
            up: Up parameter
            down: Down parameter
            step: Step parameter
            angle: Angle parameter
        
        Returns:
            strain_values: Strain values array
            stress_values: Stress values array
        """
        if self.raw_data is None:
            return None, None
        
        # Filter data matching parameters
        mask = (
            (self.raw_data['Up'] == up) &
            (self.raw_data['Down'] == down) &
            (self.raw_data['Step'] == step) &
            (self.raw_data['Angle'] == angle)
        )
        
        filtered_data = self.raw_data[mask]
        
        if len(filtered_data) == 0:
            return None, None
        
        # Sort by strain
        filtered_data = filtered_data.sort_values('Strain')
        
        return filtered_data['Strain'].values, filtered_data['Stress'].values


def set_plot_style():
    """Set plot style"""
    # Set font
    plt.rcParams["font.family"] = ["Times New Roman"]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 12


def plot_comparison(predictor, up, down, step, angle):
    """
    Plot comparison between predicted curve and original data
    
    Args:
        predictor: Predictor object
        up: Up parameter
        down: Down parameter
        step: Step parameter
        angle: Angle parameter
    """
    print(f"\nPlotting comparison - Up={up}, Down={down}, Step={step}, Angle={angle}")
    
    # Set strain range based on angle
    if angle == 45:
        strain_min, strain_max = 0.0, 0.1
        curve_type = "Shear stress-strain"
    elif angle == 0:
        strain_min, strain_max = 0.0, 0.006
        curve_type = "Axial tensile stress-strain"
    else:
        strain_min, strain_max = 0.0, 0.1
        curve_type = "Stress-strain"
    
    print(f"  Curve type: {curve_type}")
    print(f"  Strain range: {strain_min} ~ {strain_max}")
    
    # Predict curve
    pred_strain, pred_stress = predictor.predict_stress_strain_curve(
        up=up, down=down, step=step, angle=angle,
        strain_min=strain_min, strain_max=strain_max, n_points=100
    )
    
    # Get original data
    raw_strain, raw_stress = predictor.get_raw_data_for_params(up, down, step, angle)
    
    # Plot
    set_plot_style()
    plt.figure(figsize=(10, 6))
    
    # Plot predicted curve
    plt.plot(pred_strain, pred_stress, 'b-', linewidth=2, label='Predicted curve')
    
    # Plot original data (if exists)
    if raw_strain is not None and raw_stress is not None:
        plt.scatter(raw_strain, raw_stress, c='r', s=30, alpha=0.7, label='Original data', zorder=5)
        print(f"  Found {len(raw_strain)} original data points")
    else:
        print("  No matching original data found")
    
    plt.xlabel('Strain', fontsize=14)
    plt.ylabel('Stress (MPa)', fontsize=14)
    plt.title(f'{curve_type} curve comparison\nUp={up}, Down={down}, Step={step}, Angle={angle}°', 
              fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show plot
    plt.show()
    
    return pred_strain, pred_stress, raw_strain, raw_stress


def interactive_prediction(predictor):
    """
    Interactive prediction mode
    Users input parameters to plot prediction curves and compare with original data
    """
    print("\n" + "=" * 60)
    print("Interactive Prediction Mode")
    print("=" * 60)
    print("Input parameters to plot prediction curves and compare with original data")
    print("Instructions:")
    print("  Angle=45° for shear stress-strain, strain range 0~0.1")
    print("  Angle=0°  for axial tensile stress-strain, strain range 0~0.006")
    print("Enter 'q' to quit")
    print("-" * 60)
    
    while True:
        try:
            print("\nPlease enter parameters Up, Down, Step and Angle:")
            
            # Get user input
            user_input = input("Up parameter: ").strip()
            if user_input.lower() == 'q':
                print("Exiting prediction mode")
                break
            
            up = float(user_input)
            down = float(input("Down parameter: "))
            step = float(input("Step parameter: "))
            angle = float(input("Angle parameter (0 or 45): "))
            
            # Validate angle
            if angle not in [0, 45]:
                print("Warning: Angle should be 0 or 45, using default range 0~0.1")
            
            # Plot comparison
            plot_comparison(predictor, up, down, step, angle)
            
            # Ask if continue
            continue_input = input("\nContinue? (y/n): ").strip().lower()
            if continue_input == 'n':
                print("Exiting prediction mode")
                break
                
        except ValueError as e:
            print(f"Input error: {e}")
            print("Please enter valid numbers!")
        except KeyboardInterrupt:
            print("\nExiting prediction mode")
            break
        except Exception as e:
            print(f"Error occurred: {e}")


def main():
    """Main function: Prediction script entry point"""
    
    print("\n" + "=" * 60)
    print("Stress-Strain Curve Prediction - Simplified Mode")
    print("=" * 60)
    print("Function: Input parameters to plot prediction curves and compare with original data")
    print("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Set random seed
    set_seed(config.RANDOM_SEED)
    
    # Print GPU info
    print_gpu_info()
    
    # Check if model file exists
    if not os.path.exists(config.MODEL_PATH):
        print(f"\nError: Model file not found: {config.MODEL_PATH}")
        print("Please run train.py to train the model first!")
        return
    
    if not os.path.exists(config.SCALER_X_PATH) or not os.path.exists(config.SCALER_Y_PATH):
        print(f"\nError: Scaler files not found!")
        print("Please run train.py to train the model first!")
        return
    
    # Create predictor and load model
    predictor = Predictor(config)
    predictor.load()
    
    # Interactive prediction
    print("\n" + "=" * 60)
    user_input = input("Enter interactive prediction mode? (y/n): ").strip().lower()
    if user_input == 'y':
        interactive_prediction(predictor)
    
    print("\n" + "=" * 60)
    print("Prediction completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
