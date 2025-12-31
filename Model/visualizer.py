"""
visualizer.py
Visualization module: responsible for plotting and result display
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


class Visualizer:
    """Visualization class: responsible for plotting and result display"""
  
    def __init__(self, output_dir=None):
        """
        Initialize visualizer
      
        Args:
            output_dir: image output directory
        """
        self.output_dir = output_dir if output_dir else '.'
      
        # Set Chinese font
        plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
  
    def _save_figure(self, filename):
        """Helper method to save image"""
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Image saved: {filepath}")
  
    def plot_training_loss(self, train_losses, show=True):
        """
        Plot training loss curve
      
        Args:
            train_losses: training loss list
            show: whether to display image
        """
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, 'b-', linewidth=1.5)
        plt.title('Training Loss Curve', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_figure('training_loss.png')
        if show:
            plt.show()
        plt.close()
  
    def plot_r2_curve(self, r2_scores, show=True):
        """
        Plot R² score change curve
      
        Args:
            r2_scores: R² score list
            show: whether to display image
        """
        plt.figure(figsize=(10, 5))
        plt.plot(r2_scores, 'g-', linewidth=1.5)
        plt.title('R² Score Change Curve', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_figure('r2_curve.png')
        if show:
            plt.show()
        plt.close()
  
    def plot_scatter(self, y_test, y_pred, show=True):
        """
        Plot predicted values vs true values scatter plot
      
        Args:
            y_test: true values
            y_pred: predicted values
            show: whether to display image
        """
        plt.figure(figsize=(8, 8))
      
        # Flatten arrays
        if hasattr(y_test, 'values'):
            y_test_flat = y_test.values.flatten()
        else:
            y_test_flat = np.array(y_test).flatten()
        y_pred_flat = np.array(y_pred).flatten()
      
        plt.scatter(y_test_flat, y_pred_flat, alpha=0.5, edgecolors='k', linewidth=0.5)
      
        # Plot ideal line y=x
        min_val = min(y_test_flat.min(), y_pred_flat.min())
        max_val = max(y_test_flat.max(), y_pred_flat.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                 linewidth=2, label='Ideal Line (y=x)')
      
        # Calculate and display metrics
        r2 = r2_score(y_test_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
      
        plt.title(f'Predicted vs Actual Values\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=14)
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        self._save_figure('scatter_plot.png')
        if show:
            plt.show()
        plt.close()
  
    def plot_residuals(self, y_test, y_pred, show=True):
        """
        Plot residual distribution
      
        Args:
            y_test: true values
            y_pred: predicted values
            show: whether to display image
        """
        # Flatten arrays
        if hasattr(y_test, 'values'):
            y_test_flat = y_test.values.flatten()
        else:
            y_test_flat = np.array(y_test).flatten()
        y_pred_flat = np.array(y_pred).flatten()
      
        residuals = y_test_flat - y_pred_flat
      
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
      
        # Residual scatter plot
        axes[0].scatter(y_pred_flat, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values', fontsize=12)
        axes[0].set_ylabel('Residuals', fontsize=12)
        axes[0].set_title('Residual Scatter Plot', fontsize=14)
        axes[0].grid(True, alpha=0.3)
      
        # Residual histogram
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residuals', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'Residual Distribution (Mean={residuals.mean():.4f}, Std={residuals.std():.4f})', 
                          fontsize=14)
        axes[1].grid(True, alpha=0.3)
      
        plt.tight_layout()
        self._save_figure('residuals.png')
        if show:
            plt.show()
        plt.close()
  
    def plot_stress_strain_by_parameters(self, data, model, scaler_X, scaler_y, 
                                          device, n_groups=3, show=True):
        """
        Plot stress-strain curves by parameter combinations
      
        Args:
            data: raw data
            model: trained model
            scaler_X: X scaler
            scaler_y: y scaler
            device: computing device
            n_groups: number of parameter combinations
            show: whether to display image
        """
        # Get different parameter combinations
        param_columns = ['Up', 'Down', 'Step', 'Angle']
        unique_params = data[param_columns].drop_duplicates()
      
        # Select first n_groups parameter combinations
        selected_params = unique_params.head(n_groups)
      
        fig, axes = plt.subplots(1, n_groups, figsize=(6*n_groups, 5))
        if n_groups == 1:
            axes = [axes]
      
        for i, (idx, params) in enumerate(selected_params.iterrows()):
            # Filter all data with this parameter combination
            mask = (data['Up'] == params['Up']) & \
                   (data['Down'] == params['Down']) & \
                   (data['Step'] == params['Step']) & \
                   (data['Angle'] == params['Angle'])
          
            group_data = data[mask].sort_values('Strain')
          
            if len(group_data) == 0:
                continue
          
            # Extract strain and stress
            strain_actual = group_data['Strain'].values
            stress_actual = group_data['Stress'].values
          
            # Prepare prediction data
            X_pred = group_data[['Strain', 'Up', 'Down', 'Step', 'Angle']].copy()
            X_pred_norm = scaler_X.transform(X_pred)
            X_pred_tensor = torch.tensor(X_pred_norm, dtype=torch.float32).to(device)
          
            # Predict
            model.eval()
            with torch.no_grad():
                stress_pred_norm = model(X_pred_tensor).cpu().numpy()
                stress_pred = scaler_y.inverse_transform(stress_pred_norm).flatten()
          
            # Plot
            ax = axes[i]
            ax.plot(strain_actual, stress_actual, 'b-o', label='Actual Data', 
                    markersize=4, linewidth=1.5)
            ax.plot(strain_actual, stress_pred, 'r--', label='Prediction', linewidth=2)
          
            # Add parameter information
            param_info = f"Up={params['Up']}\nDown={params['Down']}\n" \
                         f"Step={params['Step']}\nAngle={params['Angle']}"
            ax.text(0.05, 0.95, param_info, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
          
            ax.set_xlabel('Strain', fontsize=12)
            ax.set_ylabel('Stress (Stress/MPa)', fontsize=12)
            ax.set_title(f'Parameter Combination {i+1}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
      
        plt.tight_layout()
        self._save_figure('stress_strain_by_parameters.png')
        if show:
            plt.show()
        plt.close()
  
    def plot_parameter_analysis(self, data, model, scaler_X, scaler_y, device, show=True):
        """
        Parameter sensitivity analysis: fix other parameters, change single parameter to observe stress change
      
        Args:
            data: raw data
            model: trained model
            scaler_X: X scaler
            scaler_y: y scaler
            device: computing device
            show: whether to display image
        """
        # Select a base parameter combination
        base_params = data[['Up', 'Down', 'Step', 'Angle']].iloc[0].to_dict()
      
        # Generate strain sequence for prediction
        strain_range = np.linspace(data['Strain'].min(), data['Strain'].max(), 50)
      
        # Analyze influence of each parameter
        param_names = ['Up', 'Down', 'Step', 'Angle']
        param_ranges = {
            'Up': sorted(data['Up'].unique()),
            'Down': sorted(data['Down'].unique()),
            'Step': sorted(data['Step'].unique()),
            'Angle': sorted(data['Angle'].unique())
        }
      
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
      
        for i, param_name in enumerate(param_names):
            ax = axes[i]
          
            # Get different values for this parameter
            param_values = param_ranges[param_name]
          
            # Plot curve for each parameter value
            colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
          
            for j, param_value in enumerate(param_values):
                # Construct input data
                X_input = pd.DataFrame({
                    'Strain': strain_range,
                    'Up': [base_params['Up'] if param_name != 'Up' else param_value] * len(strain_range),
                    'Down': [base_params['Down'] if param_name != 'Down' else param_value] * len(strain_range),
                    'Step': [base_params['Step'] if param_name != 'Step' else param_value] * len(strain_range),
                    'Angle': [base_params['Angle'] if param_name != 'Angle' else param_value] * len(strain_range)
                })
              
                # Predict
                X_input_norm = scaler_X.transform(X_input)
                X_input_tensor = torch.tensor(X_input_norm, dtype=torch.float32).to(device)
              
                model.eval()
                with torch.no_grad():
                    stress_pred_norm = model(X_input_tensor).cpu().numpy()
                    stress_pred = scaler_y.inverse_transform(stress_pred_norm).flatten()
              
                ax.plot(strain_range, stress_pred, color=colors[j], 
                        label=f'{param_name}={param_value}', linewidth=2)
          
            ax.set_xlabel('Strain', fontsize=12)
            ax.set_ylabel('Stress (Stress/MPa)', fontsize=12)
            ax.set_title(f'{param_name} Parameter Sensitivity Analysis', fontsize=14)
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
      
        plt.tight_layout()
        self._save_figure('parameter_sensitivity_analysis.png')
        if show:
            plt.show()
        plt.close()
  
    def plot_all_training_metrics(self, trainer, show=True):
        """
        Plot comprehensive graph of all training metrics
      
        Args:
            trainer: trainer object
            show: whether to display image
        """
        history = trainer.get_training_history()
      
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
      
        # Loss curve
        axes[0, 0].plot(history['losses'], 'b-', linewidth=1.5)
        axes[0, 0].set_title('Training Loss Curve', fontsize=14)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
      
        # R² curve
        axes[0, 1].plot(history['r2_scores'], 'g-', linewidth=1.5)
        axes[0, 1].axhline(y=history['best_r2'], color='r', linestyle='--', 
                           label=f'Best R²={history["best_r2"]:.4f}')
        axes[0, 1].set_title('R² Score Change Curve', fontsize=14)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('R² Score', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
      
        # MSE curve
        axes[1, 0].plot(history['mse_scores'], 'r-', linewidth=1.5)
        axes[1, 0].set_title('MSE Change Curve', fontsize=14)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('MSE', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
      
        # MAE curve
        axes[1, 1].plot(history['mae_scores'], 'm-', linewidth=1.5)
        axes[1, 1].set_title('MAE Change Curve', fontsize=14)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('MAE', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
      
        plt.tight_layout()
        self._save_figure('all_training_metrics.png')
        if show:
            plt.show()
        plt.close()