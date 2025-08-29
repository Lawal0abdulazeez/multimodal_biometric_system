# scripts/00_create_manifest.py
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import collections

# --- Configuration ---
# !! IMPORTANT !! Ensure this is the correct ABSOLUTE path to your data.
# Use forward slashes.
BASE_DATA_PATH = Path("./Face Iris fingerprint Data") 
# BASE_DATA_PATH = Path("C:/Users/DELL/Desktop/multimodal_biometric_system/Face Iris fingerprint Data") 
OUTPUT_DIR = Path("./data")
# Define the mapping from our desired modality name to the actual folder names
# This makes the code cleaner and easier to adapt if folder names change.
DATA_MAP = {
    'train': {
        'face': BASE_DATA_PATH / 'TrainData' / 'facedatatrain',
        'iris': BASE_DATA_PATH / 'TrainData' / 'irisdatatrain',
        'iris_L': BASE_DATA_PATH / 'TrainData' / 'Lirisdatatrain', 
        'finger_L': BASE_DATA_PATH / 'TrainData' / 'Lfingerdatatrain',
        # 'finger_R': BASE_DATA_PATH / 'TrainData' / 'Rfingerdatatrain', # Assuming there's a right finger
    },
    'test': {
        'face': BASE_DATA_PATH / 'TestData' / 'facedatatest',
        'iris': BASE_DATA_PATH / 'TestData' / 'irisdatatest',
        'iris_L': BASE_DATA_PATH / 'TestData' / 'Lirisdatatest',
        'finger_L': BASE_DATA_PATH / 'TestData' / 'Lfingerdatatest',
        # 'finger_R': BASE_DATA_PATH / 'TestData' / 'Rfingerdatatest',
    }
}

def extract_sample_id(filename):
    """
    Extracts a unique sample ID from a filename.
    
    Assumption: Filenames are like '1_1.tif', '1_2.jpg', '180_1.bmp', etc.
    The unique ID for the sample is the part before the extension, e.g., '1_1'.
    """
    # Path(filename).stem gets the filename without the extension
    return Path(filename).stem

def create_manifest(data_split):
    """
    Scans the data directories for a given split (train/test) and creates a manifest.
    """
    print(f"--- Creating manifest for '{data_split}' data ---")
    
    path_map = DATA_MAP[data_split]
    
    # 1. Scan all directories and collect file paths, indexed by the unique sample ID.
    # We use a defaultdict to make it easier to add new keys.
    sample_data = collections.defaultdict(dict)

    for modality, folder_path in path_map.items():
        if not folder_path.exists():
            print(f"Warning: Directory not found, skipping: {folder_path}")
            continue

        print(f"Scanning {modality} in {folder_path}...")
        for filepath in tqdm(folder_path.glob('*.*')): # Scans for any file extension
            sample_id = extract_sample_id(filepath.name)
            
            if sample_id:
                # Store the path for this sample and modality
                sample_data[sample_id][modality] = str(filepath)

    # 2. Convert the dictionary of dictionaries to a list of records
    manifest_records = []
    for sample_id, modalities in sample_data.items():
        record = {'sample_id': sample_id}
        # Add the subject_id as a separate column for easier labeling
        record['subject_id'] = sample_id.split('_')[0] 
        record.update(modalities)
        manifest_records.append(record)

    if not manifest_records:
        print(f"FATAL: No files found for the '{data_split}' split. Check your BASE_DATA_PATH and folder names in DATA_MAP.")
        return pd.DataFrame() # Return an empty DataFrame

    # 3. Convert the list of records to a DataFrame
    manifest_df = pd.DataFrame(manifest_records)
    
    # 4. Clean up the manifest: drop rows where any modality is missing.
    # For a multimodal system, we need all three primary traits.
    # We will choose to use the Left iris and Left fingerprint for this project.
    # If a sample is missing any of these, we can't use it.
    required_columns = ['face', 'iris_L', 'finger_L']
    
    # Check if required columns exist before trying to drop NAs
    existing_required_columns = [col for col in required_columns if col in manifest_df.columns]
    if not existing_required_columns:
        print(f"FATAL: The DataFrame for '{data_split}' does not contain any of the required columns: {required_columns}.")
        return pd.DataFrame()
        
    manifest_df = manifest_df.dropna(subset=existing_required_columns, how='any')

    # 5. Select and rename the final columns for consistency
    # We will also keep the 'sample_id' and 'subject_id' for reference
    final_columns_map = {
        'sample_id': 'sample_id',
        'subject_id': 'subject_id',
        'face': 'face',
        'iris_L': 'iris',
        'finger_L': 'fingerprint'
    }
    
    # Filter to only keep columns that actually exist in the DataFrame
    final_df = manifest_df[list(final_columns_map.keys())]
    final_df = final_df.rename(columns=final_columns_map)

    if final_df.empty:
        print(f"Warning: Manifest for '{data_split}' is empty after cleaning! Check if each sample has a face, left iris, AND left fingerprint.")
    else:
        print(f"Found {len(final_df)} complete multimodal samples for '{data_split}'.")

    return final_df

    pass

if __name__ == "__main__":
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create manifests for both splits
    train_manifest = create_manifest('train')
    test_manifest = create_manifest('test')
    
    # --- NEW: Combine into a single master manifest ---
    if train_manifest.empty or test_manifest.empty:
        print("Could not create master manifest because one of the splits was empty.")
    else:
        master_manifest = pd.concat([train_manifest, test_manifest], ignore_index=True)
        master_manifest.to_csv(OUTPUT_DIR / 'master_manifest.csv', index=False)
        print("\nMaster manifest created successfully:")
        print(f" - {OUTPUT_DIR / 'master_manifest.csv'}")
        print(f"Total complete samples found: {len(master_manifest)}")
        print("\nSample of master manifest:")
        print(master_manifest.head())