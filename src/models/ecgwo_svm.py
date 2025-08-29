# src/models/ecgwo_svm.py
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import joblib
from collections import Counter

from src.optimizers.ecgwo import EnhancedChaoticGWO

class ECGWO_SVM:
    """
    An SVM classifier optimized by an Enhanced Chaotic Grey Wolf Optimizer.
    The optimization is performed in two stages:
    1. Feature Selection: ECGWO finds the optimal subset of features.
    2. Hyperparameter Tuning: ECGWO finds the optimal C and gamma for the SVM.
    """
    def __init__(self, num_wolves=10, max_iter_feat=20, max_iter_param=20, alpha=0.99):
        """
        Initializes the ECGWO-SVM model.

        Args:
            num_wolves (int): Number of wolves for the optimizer.
            max_iter_feat (int): Max iterations for feature selection.
            max_iter_param (int): Max iterations for hyperparameter tuning.
            alpha (float): Weighting factor for the feature selection fitness function.
                           Balances classification error and feature reduction ratio.
        """
        self.num_wolves = num_wolves
        self.max_iter_feat = max_iter_feat
        self.max_iter_param = max_iter_param
        self.alpha = alpha

        # These will be determined during the .fit() process
        self.best_feature_mask = None
        self.best_C = None
        self.best_gamma = None
        self.final_svm = None

    def _fitness_feature_selection(self, wolf_position, X, y):
        """
        Fitness function for feature selection.
        A wolf's position is a binary vector [0, 1, 0, 1, ...].
        Fitness = alpha * ClassificationError + (1 - alpha) * (FeatureRatio)
        """
        # A threshold of 0.5 is used to convert the continuous wolf position to binary
        feature_mask = wolf_position > 0.5
        
        num_selected_features = np.sum(feature_mask)
        
        # If no features are selected, return the worst possible fitness
        if num_selected_features == 0:
            return np.inf

        # Select the features from the dataset
        X_subset = X[:, feature_mask]
        
        # Use a default SVM for evaluating this feature subset
        temp_svm = SVC(C=1.0, gamma='scale') # Default parameters

        # === ADD THIS ROBUST LOGIC ===
        min_class_count = min(Counter(y).values())
        # The number of folds cannot be greater than the number of samples in the smallest class.
        # It must also be at least 2.
        n_splits = max(2, min_class_count) 
        # =============================

        # Use 2-fold cross-validation for a robust error estimate
        accuracies = cross_val_score(temp_svm, X_subset, y, cv=n_splits)
        classification_error = 1.0 - np.mean(accuracies)
        
        feature_ratio = num_selected_features / X.shape[1]
        
        # The objective is to minimize this fitness value
        fitness = self.alpha * classification_error + (1 - self.alpha) * feature_ratio
        
        return fitness

    def _fitness_hyperparameter_tuning(self, wolf_position, X, y):
        """
        Fitness function for SVM hyperparameter tuning.
        A wolf's position is a continuous vector [C, gamma].
        Fitness = ClassificationError
        """
        # Unpack the parameters from the wolf's position
        C = wolf_position[0]
        gamma = wolf_position[1]
        
        # Create an SVM with these parameters
        temp_svm = SVC(C=C, gamma=gamma)
        

        # === ADD THIS ROBUST LOGIC ===
        min_class_count = min(Counter(y).values())
        # The number of folds cannot be greater than the number of samples in the smallest class.
        # It must also be at least 2.
        n_splits = max(2, min_class_count) 
        # =============================

        # Use n_splits for cross-validation
        accuracies = cross_val_score(temp_svm, X, y, cv=n_splits)
        classification_error = 1.0 - np.mean(accuracies)
        
        return classification_error

    def fit(self, X, y):
        """
        Fits the model by performing feature selection, hyperparameter tuning,
        and finally training the SVM.
        """
        print("--- Stage 1: Feature Selection using ECGWO ---")
        
        num_features = X.shape[1]
        # Define the fitness function with the data baked in using a lambda
        fs_fitness_func = lambda pos: self._fitness_feature_selection(pos, X, y)
        
        # Initialize and run the optimizer for feature selection
        # Bounds are [0, 1] for the binary-like encoding
        ecgwo_fs = EnhancedChaoticGWO(
            fitness_function=fs_fitness_func,
            dim=num_features,
            num_wolves=self.num_wolves,
            max_iter=self.max_iter_feat,
            lower_bound=0,
            upper_bound=1
        )
        _, best_fs_position = ecgwo_fs.optimize()
        
        # Store the best feature mask
        self.best_feature_mask = best_fs_position > 0.5
        X_reduced = X[:, self.best_feature_mask]
        
        num_selected = np.sum(self.best_feature_mask)
        print(f"Feature selection complete. Selected {num_selected}/{num_features} features.")

        print("\n--- Stage 2: Hyperparameter Tuning using ECGWO ---")
        
        # Define the fitness function for hyperparameter tuning
        hp_fitness_func = lambda pos: self._fitness_hyperparameter_tuning(pos, X_reduced, y)
        
        # Set bounds for C and gamma. These are typical ranges.
        # C is the regularization parameter, gamma is the kernel coefficient.
        param_lower_bounds = [0.1, 0.001]  # Lower bounds for [C, gamma]
        param_upper_bounds = [100, 1]      # Upper bounds for [C, gamma]
        
        ecgwo_hp = EnhancedChaoticGWO(
            fitness_function=hp_fitness_func,
            dim=2, # We are optimizing 2 parameters: C and gamma
            num_wolves=self.num_wolves,
            max_iter=self.max_iter_param,
            lower_bound=param_lower_bounds,
            upper_bound=param_upper_bounds
        )
        _, best_hp_position = ecgwo_hp.optimize()
        
        self.best_C = best_hp_position[0]
        self.best_gamma = best_hp_position[1]
        print(f"Hyperparameter tuning complete. Best C={self.best_C:.4f}, Best gamma={self.best_gamma:.4f}")

        print("\n--- Stage 3: Training Final SVM Model ---")
        
        # Train the final SVM on the reduced dataset with the best hyperparameters
        self.final_svm = SVC(C=self.best_C, gamma=self.best_gamma, probability=True) # probability=True for score-level metrics
        self.final_svm.fit(X_reduced, y)
        
        print("Model training complete.")
        return self

    def predict(self, X):
        """
        Makes predictions on new data.
        """
        if self.final_svm is None:
            raise RuntimeError("The model has not been fitted yet. Call .fit() first.")
        
        # Apply the same feature mask to the new data
        X_reduced = X[:, self.best_feature_mask]
        
        return self.final_svm.predict(X_reduced)

    def predict_proba(self, X):
        """
        Returns probability estimates for each class.
        Useful for calculating ROC curves and EER.
        """
        if self.final_svm is None:
            raise RuntimeError("The model has not been fitted yet. Call .fit() first.")
            
        X_reduced = X[:, self.best_feature_mask]
        
        return self.final_svm.predict_proba(X_reduced)
        
    def save_model(self, filepath):
        """Saves the entire ECGWO_SVM object to a file."""
        print(f"Saving model to {filepath}")
        joblib.dump(self, filepath)

    @staticmethod
    def load_model(filepath):
        """Loads an ECGWO_SVM object from a file."""
        print(f"Loading model from {filepath}")
        return joblib.load(filepath)