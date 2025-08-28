# src/data_processing/face_processor.py
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# It's good practice to have a pre-trained model for face detection.
# Download this file and place it in a known location, or ensure your OpenCV installation has it.
# URL: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
# For simplicity, let's place it in the project root for now.
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def extract_face_features(image_path, target_size=(128, 128)):
    """
    Extracts features from a face image using LBP.
    1. Detects the face.
    2. Crops and converts to grayscale.
    3. Normalizes the image.
    4. Computes the LBP histogram as the feature vector.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Face Detection
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
            # If no face is detected, fallback to using the whole image
            # This is a basic fallback; more advanced methods could be used
            face_roi = gray
        else:
            # Use the largest detected face
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face_roi = gray[y:y+h, x:x+w]

        # 2. Resize and Normalize
        resized_face = cv2.resize(face_roi, target_size)
        # Histogram equalization improves contrast
        normalized_face = cv2.equalizeHist(resized_face)

        # 3. LBP Feature Extraction
        # Parameters: P=8 points, R=1 radius
        lbp = local_binary_pattern(normalized_face, P=8, R=1, method='uniform')
        
        # 4. Create Histogram of LBP
        # The number of bins for 'uniform' LBP is P + 2
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        return hist

    except Exception as e:
        print(f"Error processing face {image_path}: {e}")
        return None