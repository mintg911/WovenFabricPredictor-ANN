"""
train.py
Training script: main entry point for model training
"""

import sys
import os

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from config import Config
from model import NeuralNetwork, create_model
from data_processor import DataProcessor
from trainer import Trainer
from visualizer import Visualizer
from utils import set_seed, print_gpu_info, print_final_metrics, count_parameters


def main():
    """Main function: program entry point"""
  
    # ==================== 1. Initialize configuration ====================
    print("\n" + "=" * 60)
    print("Stress-Strain Curve Prediction Neural Network - Training Mode")
    print("=" * 60)
  
    config = Config()
    config.create_dirs()  # Create necessary directories
    config.print_config()  # Print configuration information
  
    # Set random seed (for reproducibility)
    set_seed(config.RANDOM_SEED)
  
    # Print GPU information
    print_gpu_info()
  
    # ==================== 2. Data processing ====================
    print("\n" + "-" * 40)
    print("Step 1: Data loading and preprocessing")
    print("-" * 40)
  
    # Initialize data processor
    processor = DataProcessor(config)
  
    # Load data
    data = processor.load_data()
  
    # Split training and test sets
    train_data, test_data = processor.split_data(data)
  
    # Extract features and labels
    X_train, y_train, X_test, y_test = processor.prepare_features(train_data, test_data)
  
    # Data normalization
    X_train_norm, y_train_norm, X_test_norm, y_test_norm = processor.normalize_data(
        X_train, y_train, X_test, y_test
    )
  
    # Save scalers (for use during prediction)
    processor.save_scalers()
  
    # Convert to tensors
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = processor.to_tensors(
        X_train_norm, y_train_norm, X_test_norm, y_test_norm
    )
  
    # Create data loader
    train_loader = processor.create_dataloader(X_train_tensor, y_train_tensor, config.BATCH_SIZE)
  
    print("Data preprocessing completed!")
  
    # ==================== 3. Create model ====================
    print("\n" + "-" * 40)
    print("Step 2: Create model")
    print("-" * 40)
  
    model = create_model(config)
    print(f"\nModel structure:\n{model}")
  
    # Count parameters
    count_parameters(model)
  
    # ==================== 4. Train model ====================
    print("\n" + "-" * 40)
    print("Step 3: Model training")
    print("-" * 40)
  
    # Initialize trainer
    trainer = Trainer(model, config, processor.scaler_y)
  
    # Start training
    y_pred = trainer.train(train_loader, X_test_tensor, y_test)
  
    # Save training results
    trainer.save_results()
  
    # Ensure final model is saved
    trainer.save_model()
    print(f"Final model saved to: {config.MODEL_PATH}")
  
    # ==================== 5. Evaluate results ====================
    print("\n" + "-" * 40)
    print("Step 4: Model evaluation")
    print("-" * 40)
  
    # Print final metrics
    print_final_metrics(y_test, y_pred)
  
    # ==================== 6. Visualization ====================
    print("\n" + "-" * 40)
    print("Step 5: Result visualization")
    print("-" * 40)
  
    # Initialize visualizer
    visualizer = Visualizer(output_dir=config.OUTPUT_DIR)
  
    # Plot training loss curve
    visualizer.plot_training_loss(trainer.train_losses, show=False)
  
    # Plot R² score curve
    visualizer.plot_r2_curve(trainer.train_r2_scores, show=False)
  
    # Plot scatter plot
    visualizer.plot_scatter(y_test, y_pred, show=False)
  
    # Plot residual distribution
    visualizer.plot_residuals(y_test, y_pred, show=False)
  
    # Plot comprehensive training metrics
    visualizer.plot_all_training_metrics(trainer, show=False)
  
    # Plot stress-strain curves by parameter combinations
    visualizer.plot_stress_strain_by_parameters(
        data, model, processor.scaler_X, processor.scaler_y, config.DEVICE, 
        n_groups=3, show=False
    )
  
    # Parameter sensitivity analysis
    visualizer.plot_parameter_analysis(
        data, model, processor.scaler_X, processor.scaler_y, config.DEVICE, 
        show=False
    )
  
    print(f"\nAll images saved to: {config.OUTPUT_DIR}")
  
    # ==================== 7. Training completed ====================
    print("\n" + "=" * 60)
    print("Training process fully completed!")
    print("=" * 60)
    print(f"Model file: {config.MODEL_PATH}")
    print(f"Scalers: {config.SCALER_X_PATH}, {config.SCALER_Y_PATH}")
    print(f"Training results: {config.RESULTS_PATH}")
    print(f"Visualization images: {config.OUTPUT_DIR}")
    print("=" * 60)
  
    return model, processor, trainer


if __name__ == '__main__':
    main()