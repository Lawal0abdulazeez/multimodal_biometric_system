# scripts/04_run_comparison.py
import numpy as np
from pathlib import Path
import logging
import pandas as pd
import time
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress user warnings for cleaner

from src.models.gwo_models import GWO_Classifier, CGWO_Classifier, GWO_SVM
from src.models.ecgwo_svm import ECGWO_SVM
from src.evaluation.metrics import evaluate_performance

# --- Configuration ---
PROCESSED_DATA_DIR = Path("./data/processed")
REPORTS_DIR = Path("./reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Optimizer parameters (reduced for faster testing)
NUM_WOLVES = 10
MAX_ITER_FEAT = 20
MAX_ITER_PARAM = 20
ALPHA = 0.99

# Create timestamp for report files
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = REPORTS_DIR / f"comparison_report_{timestamp}.txt"

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a model, returning all metrics and training time.
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"Training and Evaluating: {model_name}")
    logging.info(f"{'='*60}")
    
    # Train the model and measure time
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    logging.info(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Evaluate the model
    results = evaluate_performance(model, X_test, y_test, threshold=0.5, training_time=training_time)
    
    return results

def format_results_table(all_results):
    """
    Format results into the requested table format.
    """
    logging.info("\n" + "="*80)
    logging.info("COMPARISON TABLE")
    logging.info("="*80)
    
    # Create DataFrame with the exact columns requested
    df = pd.DataFrame({
        'Tech': list(all_results.keys()),
        'TP': [all_results[tech]['tp'] for tech in all_results],
        'FN': [all_results[tech]['fn'] for tech in all_results],
        'TN': [all_results[tech]['tn'] for tech in all_results],
        'FP': [all_results[tech]['fp'] for tech in all_results],
        'FAR': [all_results[tech]['far'] for tech in all_results],
        'FRR': [all_results[tech]['frr'] for tech in all_results],
        'ERR': [all_results[tech]['err'] for tech in all_results],
        'SPEC (%)': [all_results[tech]['specificity'] * 100 for tech in all_results],
        'SEN (%)': [all_results[tech]['sensitivity'] * 100 for tech in all_results],
        'PRECISION (%)': [all_results[tech]['precision'] * 100 for tech in all_results],
        'ACCURACY (%)': [all_results[tech]['accuracy'] * 100 for tech in all_results],
        'TRAINING TIME(s)': [all_results[tech]['training_time'] for tech in all_results],
        'RECALL': [all_results[tech]['recall'] for tech in all_results],
        'Threshold value': [all_results[tech]['threshold'] for tech in all_results]
    })
    
    # Save to CSV
    csv_path = REPORTS_DIR / f"comparison_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"\nResults saved to: {csv_path}")
    
    # Display the table
    logging.info("\n" + df.to_string(index=False))
    
    return df

def run_comparison():
    """
    Main comparison function that trains and evaluates all four techniques.
    """
    logging.info("="*80)
    logging.info("MULTIMODAL BIOMETRIC SYSTEM - TECHNIQUE COMPARISON")
    logging.info("="*80)
    
    # Load data
    logging.info("\nLoading preprocessed data...")
    try:
        X_train = np.load(PROCESSED_DATA_DIR / 'X_train.npy')
        y_train = np.load(PROCESSED_DATA_DIR / 'y_train.npy')
        X_test = np.load(PROCESSED_DATA_DIR / 'X_test.npy')
        y_test = np.load(PROCESSED_DATA_DIR / 'y_test.npy')
        logging.info(f"Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")
    except FileNotFoundError:
        logging.error("Error: Preprocessed data not found. Please run preprocessing first.")
        return
    
    # Dictionary to store all results
    all_results = {}
    
    # 1. GWO (with Random Forest)
    logging.info("\n" + "="*80)
    logging.info("TECHNIQUE 1: GWO (Grey Wolf Optimizer with Random Forest)")
    logging.info("="*80)
    gwo_model = GWO_Classifier(
        num_wolves=NUM_WOLVES,
        max_iter=MAX_ITER_FEAT,
        alpha=ALPHA
    )
    all_results['GWO'] = train_and_evaluate_model(
        gwo_model, 'GWO', X_train, y_train, X_test, y_test
    )
    
    # 2. CGWO (Chaotic GWO with Random Forest)
    logging.info("\n" + "="*80)
    logging.info("TECHNIQUE 2: CGWO (Chaotic Grey Wolf Optimizer with Random Forest)")
    logging.info("="*80)
    cgwo_model = CGWO_Classifier(
        num_wolves=NUM_WOLVES,
        max_iter=MAX_ITER_FEAT,
        alpha=ALPHA
    )
    all_results['CGWO'] = train_and_evaluate_model(
        cgwo_model, 'CGWO', X_train, y_train, X_test, y_test
    )
    
    # 3. GWO-SVM
    logging.info("\n" + "="*80)
    logging.info("TECHNIQUE 3: GWO-SVM (Grey Wolf Optimizer with SVM)")
    logging.info("="*80)
    gwo_svm_model = GWO_SVM(
        num_wolves=NUM_WOLVES,
        max_iter_feat=MAX_ITER_FEAT,
        max_iter_param=MAX_ITER_PARAM,
        alpha=ALPHA
    )
    all_results['GWO-SVM'] = train_and_evaluate_model(
        gwo_svm_model, 'GWO-SVM', X_train, y_train, X_test, y_test
    )
    
    # 4. CGWO-SVM (Enhanced Chaotic GWO with SVM) - This is your ECGWO-SVM
    logging.info("\n" + "="*80)
    logging.info("TECHNIQUE 4: CGWO-SVM (Chaotic Grey Wolf Optimizer with SVM)")
    logging.info("="*80)
    cgwo_svm_model = ECGWO_SVM(
        num_wolves=NUM_WOLVES,
        max_iter_feat=MAX_ITER_FEAT,
        max_iter_param=MAX_ITER_PARAM,
        alpha=ALPHA
    )
    all_results['CGWO-SVM'] = train_and_evaluate_model(
        cgwo_svm_model, 'CGWO-SVM', X_train, y_train, X_test, y_test
    )
    
    # Format and display results
    results_df = format_results_table(all_results)
    
    logging.info("\n" + "="*80)
    logging.info("COMPARISON COMPLETE")
    logging.info(f"Full report saved to: {log_file}")
    logging.info("="*80)
    
    return results_df

if __name__ == "__main__":
    try:
        run_comparison()
    except Exception as e:
        logging.error("An unexpected error occurred during comparison.", exc_info=True)
