# src/evaluation/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
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

def calculate_confusion_matrix_metrics(y_true, y_pred):
    """
    Calculates TP, FN, TN, FP from confusion matrix for multi-class problems.
    Returns averaged values across all classes using one-vs-rest approach.
    """
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    tp_list, fn_list, tn_list, fp_list = [], [], [], []
    
    for cls in classes:
        # Convert to binary classification: current class vs rest
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)
        
        # Calculate TP, FN, TN, FP for this class
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        
        tp_list.append(tp)
        fn_list.append(fn)
        tn_list.append(tn)
        fp_list.append(fp)
    
    # Return average values across all classes
    return np.mean(tp_list), np.mean(fn_list), np.mean(tn_list), np.mean(fp_list)

def evaluate_performance(model, X_test, y_test, threshold=0.5, training_time=None):
    """
    Calculates a comprehensive set of performance metrics for the model.
    
    Args:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: True test labels
        threshold: Threshold for FAR/FRR calculation (default 0.5)
        training_time: Training time in seconds (optional)
    
    Returns:
        dict: Dictionary containing all performance metrics
    """
    # --- NEW: Suppress specific UserWarnings from sklearn ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        print("\n--- Evaluating Model Performance ---")
        
        # 1. Get predictions
        y_pred = model.predict(X_test)
        
        # 2. Calculate confusion matrix metrics
        tp, fn, tn, fp = calculate_confusion_matrix_metrics(y_test, y_pred)
        
        # 3. Calculate standard metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # 4. Calculate Specificity and Sensitivity
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Sensitivity = TP / (TP + FN) - same as recall
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        print(f"TP: {tp:.2f}, FN: {fn:.2f}, TN: {tn:.2f}, FP: {fp:.2f}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision (Weighted): {precision:.4f}")
        print(f"Recall/Sensitivity: {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        
        # 5. Biometric-specific Metrics (FAR, FRR, EER)
        try:
            y_scores = model.predict_proba(X_test)
            
            far, frr = calculate_far_frr(y_test, y_scores, threshold=threshold)
            print(f"FAR (at threshold={threshold}): {far * 100:.2f}%")
            print(f"FRR (at threshold={threshold}): {frr * 100:.2f}%")
            
            print("Calculating EER...")
            eer, eer_threshold = calculate_eer(y_test, y_scores)
            print(f"Equal Error Rate (EER): {eer * 100:.2f}% (at threshold={eer_threshold:.3f})")
        except Exception as e:
            print(f"Could not get probability scores, skipping FAR/FRR/EER calculation. Error: {e}")
            far, frr, eer, eer_threshold = 0.0, 0.0, 0.0, threshold

        # 6. Compile all results
        results = {
            "tp": float(tp),
            "fn": float(fn),
            "tn": float(tn),
            "fp": float(fp),
            "far": far,
            "frr": frr,
            "err": eer,  # ERR is same as EER
            "specificity": specificity,
            "sensitivity": sensitivity,
            "precision": precision,
            "accuracy": accuracy,
            "recall": recall,
            "threshold": eer_threshold,
            "training_time": training_time if training_time else 0.0
        }
        
        return results
