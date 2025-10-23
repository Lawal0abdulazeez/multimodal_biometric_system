# src/models/gwo_models.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import joblib
from collections import Counter

from src.optimizers.gwo import GreyWolfOptimizer
from src.optimizers.ecgwo import EnhancedChaoticGWO

class GWO_Classifier:
    """
    A Random Forest classifier optimized by Grey Wolf Optimizer for feature selection.
    """
    def __init__(self, num_wolves=10, max_iter=20, alpha=0.99):
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.alpha = alpha
        self.best_feature_mask = None
        self.final_classifier = None

    def _fitness_feature_selection(self, wolf_position, X, y):
        """Fitness function for feature selection using GWO."""
        feature_mask = wolf_position > 0.5
        num_selected_features = np.sum(feature_mask)
        
        if num_selected_features == 0:
            return np.inf

        X_subset = X[:, feature_mask]
        temp_clf = RandomForestClassifier(n_estimators=50, random_state=42)

        min_class_count = min(Counter(y).values())
        n_splits = max(2, min_class_count)

        accuracies = cross_val_score(temp_clf, X_subset, y, cv=n_splits)
        classification_error = 1.0 - np.mean(accuracies)
        feature_ratio = num_selected_features / X.shape[1]
        
        fitness = self.alpha * classification_error + (1 - self.alpha) * feature_ratio
        return fitness

    def fit(self, X, y):
        """Fits the model using GWO for feature selection."""
        print("--- GWO Feature Selection ---")
        
        num_features = X.shape[1]
        fs_fitness_func = lambda pos: self._fitness_feature_selection(pos, X, y)
        
        gwo_fs = GreyWolfOptimizer(
            fitness_function=fs_fitness_func,
            dim=num_features,
            num_wolves=self.num_wolves,
            max_iter=self.max_iter,
            lower_bound=0,
            upper_bound=1
        )
        _, best_fs_position = gwo_fs.optimize()
        
        self.best_feature_mask = best_fs_position > 0.5
        X_reduced = X[:, self.best_feature_mask]
        
        num_selected = np.sum(self.best_feature_mask)
        print(f"GWO selected {num_selected}/{num_features} features.")

        print("Training Random Forest classifier...")
        self.final_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.final_classifier.fit(X_reduced, y)
        
        print("GWO model training complete.")
        return self

    def predict(self, X):
        if self.final_classifier is None:
            raise RuntimeError("The model has not been fitted yet.")
        X_reduced = X[:, self.best_feature_mask]
        return self.final_classifier.predict(X_reduced)

    def predict_proba(self, X):
        if self.final_classifier is None:
            raise RuntimeError("The model has not been fitted yet.")
        X_reduced = X[:, self.best_feature_mask]
        return self.final_classifier.predict_proba(X_reduced)


class GWO_SVM:
    """
    An SVM classifier optimized by GWO for feature selection and hyperparameter tuning.
    """
    def __init__(self, num_wolves=10, max_iter_feat=20, max_iter_param=20, alpha=0.99):
        self.num_wolves = num_wolves
        self.max_iter_feat = max_iter_feat
        self.max_iter_param = max_iter_param
        self.alpha = alpha
        self.best_feature_mask = None
        self.best_C = None
        self.best_gamma = None
        self.final_svm = None

    def _fitness_feature_selection(self, wolf_position, X, y):
        """Fitness function for feature selection."""
        feature_mask = wolf_position > 0.5
        num_selected_features = np.sum(feature_mask)
        
        if num_selected_features == 0:
            return np.inf

        X_subset = X[:, feature_mask]
        temp_svm = SVC(C=1.0, gamma='scale')

        min_class_count = min(Counter(y).values())
        n_splits = max(2, min_class_count)

        accuracies = cross_val_score(temp_svm, X_subset, y, cv=n_splits)
        classification_error = 1.0 - np.mean(accuracies)
        feature_ratio = num_selected_features / X.shape[1]
        
        fitness = self.alpha * classification_error + (1 - self.alpha) * feature_ratio
        return fitness

    def _fitness_hyperparameter_tuning(self, wolf_position, X, y):
        """Fitness function for SVM hyperparameter tuning."""
        C = wolf_position[0]
        gamma = wolf_position[1]
        
        temp_svm = SVC(C=C, gamma=gamma)

        min_class_count = min(Counter(y).values())
        n_splits = max(2, min_class_count)

        accuracies = cross_val_score(temp_svm, X, y, cv=n_splits)
        classification_error = 1.0 - np.mean(accuracies)
        
        return classification_error

    def fit(self, X, y):
        """Fits the model using GWO for feature selection and hyperparameter tuning."""
        print("--- Stage 1: GWO Feature Selection ---")
        
        num_features = X.shape[1]
        fs_fitness_func = lambda pos: self._fitness_feature_selection(pos, X, y)
        
        gwo_fs = GreyWolfOptimizer(
            fitness_function=fs_fitness_func,
            dim=num_features,
            num_wolves=self.num_wolves,
            max_iter=self.max_iter_feat,
            lower_bound=0,
            upper_bound=1
        )
        _, best_fs_position = gwo_fs.optimize()
        
        self.best_feature_mask = best_fs_position > 0.5
        X_reduced = X[:, self.best_feature_mask]
        
        num_selected = np.sum(self.best_feature_mask)
        print(f"Feature selection complete. Selected {num_selected}/{num_features} features.")

        print("\n--- Stage 2: GWO Hyperparameter Tuning ---")
        
        hp_fitness_func = lambda pos: self._fitness_hyperparameter_tuning(pos, X_reduced, y)
        
        param_lower_bounds = [0.1, 0.001]
        param_upper_bounds = [100, 1]
        
        gwo_hp = GreyWolfOptimizer(
            fitness_function=hp_fitness_func,
            dim=2,
            num_wolves=self.num_wolves,
            max_iter=self.max_iter_param,
            lower_bound=param_lower_bounds,
            upper_bound=param_upper_bounds
        )
        _, best_hp_position = gwo_hp.optimize()
        
        self.best_C = best_hp_position[0]
        self.best_gamma = best_hp_position[1]
        print(f"Hyperparameter tuning complete. Best C={self.best_C:.4f}, Best gamma={self.best_gamma:.4f}")

        print("\n--- Stage 3: Training Final SVM Model ---")
        
        self.final_svm = SVC(C=self.best_C, gamma=self.best_gamma, probability=True)
        self.final_svm.fit(X_reduced, y)
        
        print("GWO-SVM model training complete.")
        return self

    def predict(self, X):
        if self.final_svm is None:
            raise RuntimeError("The model has not been fitted yet.")
        X_reduced = X[:, self.best_feature_mask]
        return self.final_svm.predict(X_reduced)

    def predict_proba(self, X):
        if self.final_svm is None:
            raise RuntimeError("The model has not been fitted yet.")
        X_reduced = X[:, self.best_feature_mask]
        return self.final_svm.predict_proba(X_reduced)


class CGWO_Classifier:
    """
    A Random Forest classifier optimized by Chaotic GWO for feature selection.
    """
    def __init__(self, num_wolves=10, max_iter=20, alpha=0.99):
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.alpha = alpha
        self.best_feature_mask = None
        self.final_classifier = None

    def _fitness_feature_selection(self, wolf_position, X, y):
        """Fitness function for feature selection using CGWO."""
        feature_mask = wolf_position > 0.5
        num_selected_features = np.sum(feature_mask)
        
        if num_selected_features == 0:
            return np.inf

        X_subset = X[:, feature_mask]
        temp_clf = RandomForestClassifier(n_estimators=50, random_state=42)

        min_class_count = min(Counter(y).values())
        n_splits = max(2, min_class_count)

        accuracies = cross_val_score(temp_clf, X_subset, y, cv=n_splits)
        classification_error = 1.0 - np.mean(accuracies)
        feature_ratio = num_selected_features / X.shape[1]
        
        fitness = self.alpha * classification_error + (1 - self.alpha) * feature_ratio
        return fitness

    def fit(self, X, y):
        """Fits the model using CGWO for feature selection."""
        print("--- CGWO Feature Selection ---")
        
        num_features = X.shape[1]
        fs_fitness_func = lambda pos: self._fitness_feature_selection(pos, X, y)
        
        cgwo_fs = EnhancedChaoticGWO(
            fitness_function=fs_fitness_func,
            dim=num_features,
            num_wolves=self.num_wolves,
            max_iter=self.max_iter,
            lower_bound=0,
            upper_bound=1
        )
        _, best_fs_position = cgwo_fs.optimize()
        
        self.best_feature_mask = best_fs_position > 0.5
        X_reduced = X[:, self.best_feature_mask]
        
        num_selected = np.sum(self.best_feature_mask)
        print(f"CGWO selected {num_selected}/{num_features} features.")

        print("Training Random Forest classifier...")
        self.final_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.final_classifier.fit(X_reduced, y)
        
        print("CGWO model training complete.")
        return self

    def predict(self, X):
        if self.final_classifier is None:
            raise RuntimeError("The model has not been fitted yet.")
        X_reduced = X[:, self.best_feature_mask]
        return self.final_classifier.predict(X_reduced)

    def predict_proba(self, X):
        if self.final_classifier is None:
            raise RuntimeError("The model has not been fitted yet.")
        X_reduced = X[:, self.best_feature_mask]
        return self.final_classifier.predict_proba(X_reduced)
