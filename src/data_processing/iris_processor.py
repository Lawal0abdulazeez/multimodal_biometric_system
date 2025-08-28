# src/data_processing/iris_processor.py
import cv2
import numpy as np

def extract_iris_features(image_path, target_size=(256, 64)):
    """
    Extracts features from an iris image.
    This is a simplified version and does not perform true segmentation and normalization.
    A full implementation is a research project in itself.
    
    1. Reads image in grayscale.
    2. Applies a Log-Gabor filter to extract texture information.
    3. Computes the mean and standard deviation of the filter response as a feature vector.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None
        
        # --- A real system would have complex segmentation/normalization here ---
        # For this project, we assume the input images are roughly centered and cropped.
        # We will resize to a standard "unrolled" iris size.
        normalized_iris = cv2.resize(img, target_size)
        
        # 1. Create a Log-Gabor filter
        # These parameters are typical but can be tuned.
        gabor_kernel = cv2.getGaborKernel(
            ksize=(31, 31),  # Kernel size
            sigma=4.0,       # Standard deviation of the Gaussian envelope
            theta=np.pi/2,   # Orientation of the filter (vertical)
            lambd=10.0,      # Wavelength of the sinusoidal factor
            gamma=0.5,       # Spatial aspect ratio
            psi=0,           # Phase offset
            ktype=cv2.CV_32F
        )
        
        # 2. Filter the image
        filtered_img = cv2.filter2D(normalized_iris, cv2.CV_8UC3, gabor_kernel)

        # 3. Create feature vector from filter response
        # Using simple statistics (mean and std) as a feature vector
        mean_val = np.mean(filtered_img)
        std_val = np.std(filtered_img)

        # To make a longer, more descriptive vector, we can compute stats over blocks
        features = []
        block_size = 16
        for r in range(0, filtered_img.shape[0], block_size):
            for c in range(0, filtered_img.shape[1], block_size):
                block = filtered_img[r:r+block_size, c:c+block_size]
                features.append(np.mean(block))
                features.append(np.std(block))

        return np.array(features)

    except Exception as e:
        print(f"Error processing iris {image_path}: {e}")
        return None