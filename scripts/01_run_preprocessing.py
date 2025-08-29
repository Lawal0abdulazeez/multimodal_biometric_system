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
    # Split the data into 80% train and 20% test
    # stratify=y_all ensures that the class distribution is the same in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.40, random_state=42, stratify=y_all
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