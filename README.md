# Enhanced Chaotic Grey Wolf Optimization based SVM for Multimodal Biometric Security

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)

This repository contains the complete source code and implementation for the research project: **"An Enhanced Chaotic Grey Wolf Optimization based Support Vector Machine for a Multimodal Biometric Border Security System."**

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [📂 Project Structure](#-project-structure)
- [🔬 Methodology](#-methodology)
  - [1. Data Preprocessing & Feature Extraction](#1-data-preprocessing--feature-extraction)
  - [2. Feature-Level Fusion](#2-feature-level-fusion)
  - [3. The ECGWO-SVM Model](#3-the-ecgwo-svm-model)
- [🚀 Setup and Installation](#-setup-and-installation)
- [▶️ How to Run the Pipeline](#️-how-to-run-the-pipeline)
- [📊 Interpreting the Results](#-interpreting-the-results)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

## 🌟 Project Overview

This project implements a novel machine learning pipeline for a high-security biometric identification system. The system leverages three biometric modalities—**Face, Iris, and Fingerprint**—to ensure robust and reliable user authentication.

The core innovation lies in the application of an **Enhanced Chaotic Grey Wolf Optimizer (ECGWO)**. This advanced metaheuristic algorithm is uniquely employed in a two-stage process to simultaneously:
1.  **Select the most discriminative features** from the fused multimodal data, reducing dimensionality and improving model efficiency.
2.  **Tune the hyperparameters** of a Support Vector Machine (SVM) classifier, maximizing its predictive accuracy.

The entire pipeline is designed to be robust, reproducible, and demonstrates superior performance compared to standard machine learning baselines.

## ✨ Key Features

- **Multimodal Fusion:** Combines features from Face, Iris, and Fingerprint biometrics for enhanced security and accuracy.
- **Advanced Optimization:** Implements a standard Grey Wolf Optimizer (GWO) and an Enhanced Chaotic GWO using a Tent Map for superior global search capabilities.
- **Dual-Stage Optimization:** A novel framework where ECGWO is used for both feature selection and SVM hyperparameter tuning within a single, unified model (`ECGWO_SVM`).
- **End-to-End Pipeline:** Fully scripted workflow from raw image processing to final performance evaluation.
- **Robust Data Handling:** The pipeline correctly handles complex data structures, performs stratified train-test splits, and filters data to prevent common errors in cross-validation.
- **Comprehensive Benchmarking:** The proposed model is rigorously evaluated against standard `SVM with GridSearchCV` and `Random Forest` baselines.
- **Structured Reporting:** Automatically generates a timestamped text log and a structured JSON file with key performance metrics (Accuracy, Precision, FAR, FRR, EER) for easy analysis and reporting.

## 📂 Project Structure

The repository is organized into a modular and understandable structure:

```
multimodal_biometric_system/
│
├── data/
│   ├── raw/                # (Placeholder) Raw biometric images would go here.
│   └── processed/          # Processed feature vectors and labels (.npy files).
│
├── models/                 # Saved, trained model objects (.joblib files).
│
├── notebooks/              # Jupyter notebooks for exploration and result visualization.
│
├── reports/
│   ├── figures/            # Generated plots and graphs.
│   └── *.json, *.txt       # Timestamped JSON results and text logs from evaluations.
│
├── scripts/
│   ├── 00_create_manifest.py   # Scans data folders and creates a master manifest.
│   ├── 01_run_preprocessing.py # Extracts features, fuses them, and splits the data.
│   ├── 02_run_training.py      # Trains the main ECGWO-SVM model and saves it.
│   └── 03_run_evaluation.py    # Evaluates all models and generates reports.
│
├── src/
│   ├── data_processing/    # Modules for face, iris, and fingerprint feature extraction.
│   ├── evaluation/         # Module for calculating performance metrics (FAR, FRR, EER).
│   ├── models/             # The core ECGWO_SVM wrapper class.
│   └── optimizers/         # Implementations of GWO and ECGWO algorithms.
│
├── haarcascade_frontalface_default.xml # OpenCV model for face detection.
├── requirements.txt        # Python package dependencies.
└── README.md               # This file.
```

## 🔬 Methodology

The project follows a systematic, multi-stage pipeline:

### 1. Data Preprocessing & Feature Extraction
Raw biometric images are processed to extract meaningful numerical feature vectors.
- **Face:** Faces are detected using Haar Cascades, cropped, normalized, and features are extracted using the **Local Binary Patterns (LBP)** texture descriptor.
- **Iris:** Iris images are normalized and their rich texture is captured using **Log-Gabor filters**. The statistics of the filter responses form the feature vector.
- **Fingerprint:** Fingerprint images are enhanced, and their ridge patterns are converted into a feature vector using an LBP-based method (**FingerCode**).

### 2. Feature-Level Fusion
The extracted feature vectors from the three modalities for each sample are concatenated into a single, comprehensive feature vector. This fused vector is then scaled using `StandardScaler` to ensure all features have a consistent range.

### 3. The ECGWO-SVM Model
This is the core of the project, operating in two sequential stages:

#### Stage 1: Feature Selection
The ECGWO algorithm is tasked with finding an optimal binary mask to select a subset of features.
- A "wolf's position" represents a potential feature subset.
- The **fitness function** is designed to reward solutions that achieve low classification error while using the fewest features possible:
  `Fitness = α * (Classification Error) + (1 - α) * (Feature Selection Ratio)`

#### Stage 2: Hyperparameter Tuning
Using the reduced feature set selected in Stage 1, the ECGWO algorithm is run a second time to find the best hyperparameters for the SVM.
- A "wolf's position" represents a pair of `[C, gamma]` values.
- The **fitness function** is simply the cross-validated classification error, which the optimizer seeks to minimize.

Finally, a definitive SVM is trained on the entire feature-selected training set using the optimized `C` and `gamma` values.

## 🚀 Setup and Installation

Follow these steps to set up the project environment.

**1. Clone the Repository:**
```bash
git clone https://github.com/Lawal0abdulazeez/multimodal_biometric_system
cd multimodal_biometric_system
```

**2. Create and Activate the Conda Environment:**
This project uses Python 3.9.
```bash
conda create --name biometric_env python=3.9 -y
conda activate biometric_env
```

**3. Install Dependencies:**
All required packages are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

**4. Download Face Detection Model:**
The face detection module requires a pre-trained Haar Cascade model. If it's not already in the root directory, download it from the [OpenCV GitHub repository](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml) and place `haarcascade_frontalface_default.xml` in the project root.

**5. Place Data:**
Place your raw data folder (e.g., `Face Iris fingerprint Data`) in the project root or update the `BASE_DATA_PATH` variable in `scripts/00_create_manifest.py` to point to its location.

## ▶️ How to Run the Pipeline

Execute the scripts in the following order from the project's root directory.

**1. Create the Data Manifest:**
This script scans your data folders and creates a master CSV file mapping all biometric samples.
```bash
python scripts/00_create_manifest.py
```

**2. Preprocess Data and Create Splits:**
This script reads the manifest, extracts features for all samples, filters the data for robustness, and performs a stratified train-test split.
```bash
python scripts/01_run_preprocessing.py
```

**3. Train the ECGWO-SVM Model:**
This is the most time-consuming step. It runs the two-stage optimization and saves the final trained model to the `/models` directory.
> **Note:** For a quick test, you can reduce `num_wolves` and `max_iter` values inside the script. For the final, high-performance model, use the recommended higher values (this may take several hours).
```bash
python scripts/02_run_training.py
```

**4. Evaluate Models and Generate Reports:**
This script loads the trained model, evaluates it on the test set, and runs the baseline models for comparison. It generates the final report files.
```bash
python scripts/03_run_evaluation.py
```

## 📊 Interpreting the Results

After running the evaluation script, you will find two timestamped files in the `/reports` directory:

1.  **`evaluation_report_[timestamp].txt`**: A detailed log file containing all the console output from the evaluation run.
2.  **`evaluation_results_[timestamp].json`**: A structured JSON file containing the key performance metrics for each model. This file is ideal for programmatic analysis and generating plots.

**Sample JSON Output:**
```json
{
    "ecgwo_svm": {
        "accuracy": 0.9850,
        "eer": 0.021,
        "model_params": {
            "num_selected_features": 65,
            "total_features": 148,
            "best_c": 95.12,
            "best_gamma": 0.015
        }
    },
    "svm_gridsearch": {
        "accuracy": 0.9620,
        "eer": 0.045,
        "best_params": {
            "C": 10,
            "gamma": 0.01
        }
    },
    "random_forest": {
        "accuracy": 0.9580,
        "eer": 0.052
    }
}
```
You can use a script like the one in `notebooks/03_visualize_results.ipynb` to parse this JSON and create comparison charts for your final paper or presentation.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page or open a new one.

## 📜 License

This project is licensed under the Apache License - see the `LICENSE` file for details.