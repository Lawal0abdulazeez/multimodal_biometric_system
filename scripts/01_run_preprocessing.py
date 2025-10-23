# scripts/01_run_preprocessing.py
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder # <-- Import LabelEncoder
import joblib # <-- Import joblib to save the encoder
from sklearn.model_selection import train_test_split # <-- Import train_test_split

from src.data_processing.face_processor import extract_face_features
from src.data_processing.iris_processor import extract_iris_features
from src.data_processing.fingerprint_processor import extract_fingerprint_features

MANIFEST_DIR = Path("./data")
OUTPUT_DIR = Path("./data/processed")


# (The process_manifest function is now simplified to process ONE file)
def process_master_manifest(manifest_path):
    manifest = pd.read_csv(manifest_path)
    all_features = []
    all_labels = [] # Will store subject_id
    for index, row in tqdm(manifest.iterrows(), total=len(manifest), desc=f"Processing {manifest_path.name}"):
        subject_id = row['subject_id']
        face_features = extract_face_features(row['face'])
        iris_features = extract_iris_features(row['iris'])
        fingerprint_features = extract_fingerprint_features(row['fingerprint'])
        if face_features is None or iris_features is None or fingerprint_features is None:
            print(f"Skipping a sample for subject {subject_id} due to a feature extraction error.")
            continue
        fused_features = np.concatenate([face_features, iris_features, fingerprint_features])
        all_features.append(fused_features)
        all_labels.append(subject_id)
    return np.array(all_features), np.array(all_labels)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("--- Processing Master Manifest ---")
    X_all, y_all_str = process_master_manifest(MANIFEST_DIR / 'master_manifest.csv')

    print("\n--- Encoding Labels ---")
    le = LabelEncoder()
    y_all = le.fit_transform(y_all_str) # Fit and transform on all labels at once
    print(f"Labels encoded. Found {len(le.classes_)} unique classes.")
    
    print("\n--- Performing Stratified Train-Test Split ---")
    # Split the data into 60% train and 40% test
    # Handle classes with only one sample: scikit-learn's stratify requires at least 2
    # samples per class. We'll place singleton classes into the training set and
    # perform a stratified split on the remaining data. If stratification is still
    # not possible (very small remaining set), fall back to a regular split.
    test_size = 0.40

    # Identify singleton classes (in the encoded labels)
    unique, counts = np.unique(y_all, return_counts=True)
    singleton_classes = unique[counts == 1]

    if len(singleton_classes) > 0:
        # Indices of samples that belong to singleton classes
        singleton_mask = np.isin(y_all, singleton_classes)
        non_singleton_mask = ~singleton_mask

        # Put singletons into training set
        X_singletons = X_all[singleton_mask]
        y_singletons = y_all[singleton_mask]

        # Data eligible for stratified split
        X_rest = X_all[non_singleton_mask]
        y_rest = y_all[non_singleton_mask]

        # If there is enough data to stratify, do so; otherwise fallback
        try:
            X_rest_train, X_rest_test, y_rest_train, y_rest_test = train_test_split(
                X_rest, y_rest, test_size=test_size, random_state=42, stratify=y_rest
            )
            # Combine rest-train with singletons to form final training set
            X_train = np.vstack([X_rest_train, X_singletons]) if len(X_rest_train) > 0 else X_singletons
            y_train = np.concatenate([y_rest_train, y_singletons]) if len(y_rest_train) > 0 else y_singletons
            X_test = X_rest_test
            y_test = y_rest_test
            print(f"Stratified split performed on non-singleton classes. Singletons ({len(X_singletons)}) moved to training set.")
        except ValueError:
            # Fallback: do a non-stratified split on the whole dataset
            print("Stratified split failed on non-singleton classes; falling back to non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=test_size, random_state=42, stratify=None
            )
    else:
        # No singletons -> safe to stratify on all labels
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=test_size, random_state=42, stratify=y_all
        )

    print(f"Data split complete. Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    print("\n--- Scaling Features ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Use the SAME scaler fitted on train data

    # Save everything
    np.save(OUTPUT_DIR / 'X_train.npy', X_train_scaled)
    np.save(OUTPUT_DIR / 'y_train.npy', y_train)
    np.save(OUTPUT_DIR / 'X_test.npy', X_test_scaled)
    np.save(OUTPUT_DIR / 'y_test.npy', y_test)
    joblib.dump(le, OUTPUT_DIR / 'label_encoder.joblib')
    
    print("\nPreprocessing complete. Processed files saved to:", OUTPUT_DIR)