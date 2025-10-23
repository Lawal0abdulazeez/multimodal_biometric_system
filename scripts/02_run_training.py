# scripts/02_run_training.py
import numpy as np
from pathlib import Path
import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress user warnings for cleaner output

from src.models.ecgwo_svm import ECGWO_SVM

# --- Configuration ---
PROCESSED_DATA_DIR = Path("./data/processed")
MODELS_DIR = Path("./models")

# Ensure the models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_model():
    """
    Loads the preprocessed data, trains the ECGWO-SVM model, and saves it.
    """
    print("--- Starting Model Training ---")

    # 1. Load Data
    print("Loading preprocessed data...")
    try:
        X_train = np.load(PROCESSED_DATA_DIR / 'X_train.npy')
        y_train = np.load(PROCESSED_DATA_DIR / 'y_train.npy')
    except FileNotFoundError:
        print("Error: Preprocessed data not found. Please run '01_run_preprocessing.py' first.")
        return

    print(f"Training data loaded. Shape: X_train={X_train.shape}, y_train={y_train.shape}")



    # === ADD THIS DIAGNOSTIC CODE HERE ===
    unique_classes, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique_classes, counts))
    print("\n--- Class Distribution in y_train ---")
    print(class_distribution)
    min_class_count = min(counts)
    print(f"The smallest class has {min_class_count} samples.")
    if min_class_count < 3:
        print("WARNING: The smallest class has fewer than 3 samples, which will cause the cross-validation to fail.\n")
    # =======================================



    # 2. Initialize Model
    # These parameters can be tuned. For a first run, they are reasonable.
    # To speed up, reduce num_wolves or max_iter values.
    # To improve performance, increase them.
    print("Initializing ECGWO-SVM model...")
    ecgwo_svm_model = ECGWO_SVM(
        num_wolves=20,      # Reduced for faster initial run
        max_iter_feat=50,  # Reduced for faster initial run
        max_iter_param=50, # Reduced for faster initial run
        alpha=0.99
    )


    # 3. Train Model
    print("Starting the training process (this may take a significant amount of time)...")
    start_time = time.time()
    
    ecgwo_svm_model.fit(X_train, y_train)
    
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"\n--- Training Finished ---")
    print(f"Total training time: {training_duration / 60:.2f} minutes")

    # 4. Save Model
    model_save_path = MODELS_DIR / 'ecgwo_svm_model.joblib'
    ecgwo_svm_model.save_model(model_save_path)
    
    print(f"\nModel successfully saved to {model_save_path}")

if __name__ == "__main__":
    train_model()