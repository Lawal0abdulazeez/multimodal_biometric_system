# src/evaluation/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import label_binarize
import warnings # <-- Import the warnings module

def calculate_far_frr(y_true, y_scores, threshold):
    """
    Calculates False Acceptance Rate (FAR) and False Rejection Rate (FRR).
    This is for a one-vs-rest scenario in a multi-class problem.
    """
    # Binarize the labels for one-vs-rest calculation
    classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)
    
    n_classes = len(classes)
    far_list, frr_list = [], []

    for i in range(n_classes):
        # Genuine scores are when the true class is `i`
        genuine_scores = y_scores[y_true_bin[:, i] == 1, i]
        # Impostor scores are when the true class is NOT `i`
        impostor_scores = y_scores[y_true_bin[:, i] == 0, i]

        # False Rejections: Genuine scores that fall below the threshold
        false_rejections = np.sum(genuine_scores < threshold)
        if len(genuine_scores) == 0:
            frr = 0.0
        else:
            frr = false_rejections / len(genuine_scores)

        # False Acceptances: Impostor scores that exceed the threshold
        false_acceptances = np.sum(impostor_scores >= threshold)
        if len(impostor_scores) == 0:
            far = 0.0
        else:
            far = false_acceptances / len(impostor_scores)
            
        far_list.append(far)
        frr_list.append(frr)
    
    # Return the average FAR and FRR across all classes
    return np.mean(far_list), np.mean(frr_list)

def calculate_eer(y_true, y_scores):
    """
    Calculates the Equal Error Rate (EER).
    EER is the point where FAR equals FRR.
    """
    thresholds = np.linspace(0.0, 1.0, 500) # Test 500 different thresholds
    min_diff = np.inf
    eer = 1.0
    eer_threshold = 0.5

    for t in thresholds:
        far, frr = calculate_far_frr(y_true, y_scores, t)
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2 # EER is the average of FAR and FRR at this point
            eer_threshold = t
    
    return eer, eer_threshold

def evaluate_performance(model, X_test, y_test):
    """
    Calculates a comprehensive set of performance metrics for the model.
    """
    # --- NEW: Suppress specific UserWarnings from sklearn ---
    # We are intentionally ignoring the warning about "y could represent a regression problem"
    # because we know our data structure and have handled it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        print("\n--- Evaluating Model Performance ---")
        
        # 1. Standard Metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision (Weighted): {precision:.4f}")
        
        # 2. Biometric-specific Metrics
        try:
            y_scores = model.predict_proba(X_test)
        except Exception as e:
            print(f"Could not get probability scores, skipping FAR/FRR/EER calculation. Error: {e}")
            return {} # Return empty dict on failure

        far_at_50, frr_at_50 = calculate_far_frr(y_test, y_scores, threshold=0.5)
        print(f"FAR (at threshold=0.5): {far_at_50 * 100:.2f}%")
        print(f"FRR (at threshold=0.5): {frr_at_50 * 100:.2f}%")
        
        print("Calculating EER...")
        eer, eer_threshold = calculate_eer(y_test, y_scores)
        print(f"Equal Error Rate (EER): {eer * 100:.2f}% (at threshold={eer_threshold:.3f})")

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "far_at_50": far_at_50,
            "frr_at_50": frr_at_50,
            "eer": eer
        }
        return results