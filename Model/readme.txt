WovenFabricPredictor-NN/
│
├── config.py              # Configuration parameters
├── model.py               # Neural network model definition
├── data_processor.py      # Data processing module
├── trainer.py             # Training module
├── visualizer.py          # Visualization module
├── utils.py               # Utility functions
├── train.py               # Training script (main entry)
├── predict.py             # Prediction script (load best model)
│
├── data_axial.npy         # Data file of axial tensile loading
├── data_shear.npy        # Data file of off-axial loading
├── checkpoints/           # Model saving directory
│   ├── model.pth
│   ├── scaler_X.joblib
│   └── scaler_y.joblib
└── outputs/               # Output directory
    ├── training_results.xlsx
    └── *.png              # Image files
