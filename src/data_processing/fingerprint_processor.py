# src/data_processing/fingerprint_processor.py
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_fingerprint_features(image_path, target_size=(128, 128)):
    """
    Extracts features from a fingerprint image using LBP.
    This captures the texture of the ridges. A more advanced system would use minutiae.
    
    1. Reads image in grayscale.
    2. Enhances the image using histogram equalization.
    3. Computes the LBP histogram as the feature vector.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None
        
        # 1. Resize and enhance
        resized_fingerprint = cv2.resize(img, target_size)
        enhanced_fingerprint = cv2.equalizeHist(resized_fingerprint)
        
        # 2. LBP Feature Extraction (same as face)
        lbp = local_binary_pattern(enhanced_fingerprint, P=8, R=1, method='uniform')
        
        # 3. Create Histogram of LBP
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        return hist
        
    except Exception as e:
        print(f"Error processing fingerprint {image_path}: {e}")
        return None