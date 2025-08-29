# scripts/03_run_evaluation.py
import numpy as np
from pathlib import Path
import logging # <-- Import logging
import json    # <-- Import json
from datetime import datetime # <-- To timestamp our reports

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from src.models.ecgwo_svm import ECGWO_SVM
from src.evaluation.metrics import evaluate_performance

# --- Configuration ---
PROCESSED_DATA_DIR = Path("./data/processed")
MODELS_DIR = Path("./models")
REPORTS_DIR = Path("./reports") # <-- Define reports directory
REPORTS_DIR.mkdir(parents=True, exist_ok=True) # Ensure it exists
CV_FOLDS = 3

# --- Setup Logging ---
# Create a timestamp for the report files
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = REPORTS_DIR / f"evaluation_report_{timestamp}.txt"

# Configure logger to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler() # This will print to console
    ]
)

def run_evaluation():
    """
    Main evaluation function with integrated logging and result saving.
    """
    logging.info("--- Starting Model Evaluation ---")
    
    # This dictionary will store all our final results for the JSON report
    final_results = {}

    # 1. Load Data
    logging.info("Loading test data...")
    # ... (loading data remains the same) ...
    try:
        X_test = np.load(PROCESSED_DATA_DIR / 'X_test.npy')
        y_test = np.load(PROCESSED_DATA_DIR / 'y_test.npy')
        # We need training data for the baselines
        X_train = np.load(PROCESSED_DATA_DIR / 'X_train.npy')
        y_train = np.load(PROCESSED_DATA_DIR / 'y_train.npy')
    except FileNotFoundError:
        logging.info("Error: Preprocessed data not found. Please run preprocessing and training first.")
        return


    # --- DIAGNOSTIC ---
    train_labels = set(np.unique(y_train))
    test_labels = set(np.unique(y_test))
    
    logging.info(f"\n--- DIAGNOSTIC INFO ---")
    logging.info(f"Number of unique labels in Training set: {len(train_labels)}")
    logging.info(f"Number of unique labels in Testing set: {len(test_labels)}")
    
    labels_in_test_but_not_train = test_labels - train_labels
    if labels_in_test_but_not_train:
        logging.info(f"WARNING: Found {len(labels_in_test_but_not_train)} labels in the test set that are NOT in the training set.")
        logging.info(f"Example missing labels: {list(labels_in_test_but_not_train)[:5]}")
    else:
        logging.info("OK: All labels in the test set are present in the training set.")
    logging.info(f"--- END DIAGNOSTIC ---\n")
    # --- END DIAGNOSTIC ---
    
    # 2. Load Our Trained Model
    model_path = MODELS_DIR / 'ecgwo_svm_model.joblib'
    if not model_path.exists():
        logging.info(f"Error: Trained model not found at {model_path}. Please run '02_run_training.py' first.")
        return
        
    our_model = ECGWO_SVM.load_model(model_path)
    
    # 3. Evaluate Our Model
    logging.info("\n" + "="*50)
    logging.info("        PERFORMANCE OF OUR PROPOSED ECGWO-SVM MODEL")
    logging.info("="*50)
    our_model_results = evaluate_performance(our_model, X_test, y_test)
    
    # Store results and model parameters
    final_results['ecgwo_svm'] = our_model_results
    final_results['ecgwo_svm']['model_params'] = {
        'num_selected_features': int(np.sum(our_model.best_feature_mask)),
        'total_features': len(our_model.best_feature_mask),
        'best_c': our_model.best_C,
        'best_gamma': our_model.best_gamma
    }
    
    # ... (rest of the script) ...
    # Now, replace all `print()` statements with `logging.info()` and adapt the logic
    
    logging.info("\n" + "="*50)
    logging.info("        PERFORMANCE OF BASELINE MODELS")
    logging.info("="*50)

    feature_mask = our_model.best_feature_mask
    X_train_reduced = X_train[:, feature_mask]
    X_test_reduced = X_test[:, feature_mask]

    # --- Baseline 1: Standard SVM ---
    logging.info("\n--- Training Baseline 1: SVM with GridSearchCV ---")
    
    # Robustness Check
    unique_train_labels, counts_train = np.unique(y_train, return_counts=True)
    min_class_size = int(np.min(counts_train))
    
    # --- MODIFIED LOGIC ---
    if min_class_size < 2:
        logging.warning(f"The smallest class in the training set has only {min_class_size} member(s).")
        logging.warning("Skipping GridSearchCV and training a default SVC instead.")
    
        baseline_svm = SVC(probability=True, C=1.0, gamma='scale', random_state=42)
        baseline_svm.fit(X_train_reduced, y_train)
    
        logging.info("Evaluating default SVC...")
        svm_results = evaluate_performance(baseline_svm, X_test_reduced, y_test)
        final_results['svm_default'] = svm_results
    else:
        dynamic_cv_folds = CV_FOLDS
        if min_class_size < CV_FOLDS:
            logging.warning(f"Smallest class has {min_class_size} members, reducing CV folds to {min_class_size}.")
            dynamic_cv_folds = min_class_size
        
        param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]} # Reduced for speed
        grid_svm = GridSearchCV(SVC(probability=True, random_state=42), param_grid, refit=True, verbose=0, cv=dynamic_cv_folds)
        grid_svm.fit(X_train_reduced, y_train)
        logging.info(f"Best parameters for standard SVM: {grid_svm.best_params_}")
        svm_results = evaluate_performance(grid_svm.best_estimator_, X_test_reduced, y_test)
        final_results['svm_gridsearch'] = svm_results
        final_results['svm_gridsearch']['best_params'] = grid_svm.best_params_

    # --- Baseline 2: Random Forest Classifier ---
    logging.info("\n--- Training Baseline 2: Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_reduced, y_train)
    rf_results = evaluate_performance(rf_model, X_test_reduced, y_test)
    final_results['random_forest'] = rf_results

    # --- Save the final results to a JSON file ---
    json_report_path = REPORTS_DIR / f"evaluation_results_{timestamp}.json"
    with open(json_report_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    
    logging.info("\n" + "="*50)
    logging.info("Evaluation complete.")
    logging.info(f"Full text report saved to: {log_file}")
    logging.info(f"Structured JSON results saved to: {json_report_path}")

# This part needs to be updated to use the logger
if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        logging.error("An unexpected error occurred during evaluation.", exc_info=True)