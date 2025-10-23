# Multimodal Biometric System - Comparison Guide

This guide explains how to run the comprehensive comparison of different optimization techniques.

## Techniques Compared

The system compares four different techniques:

1. **GWO** - Grey Wolf Optimizer with Random Forest classifier
2. **CGWO** - Chaotic Grey Wolf Optimizer with Random Forest classifier
3. **GWO-SVM** - Grey Wolf Optimizer with SVM (feature selection + hyperparameter tuning)
4. **CGWO-SVM** - Chaotic Grey Wolf Optimizer with SVM (your proposed ECGWO-SVM method)

## Output Metrics

The comparison generates a comprehensive table with the following metrics:

- **TP** - True Positives
- **FN** - False Negatives
- **TN** - True Negatives
- **FP** - False Positives
- **FAR** - False Acceptance Rate
- **FRR** - False Rejection Rate
- **ERR** - Equal Error Rate
- **SPEC (%)** - Specificity (percentage)
- **SEN (%)** - Sensitivity (percentage)
- **PRECISION (%)** - Precision (percentage)
- **ACCURACY (%)** - Accuracy (percentage)
- **TRAINING TIME(s)** - Training time in seconds
- **RECALL** - Recall score
- **Threshold value** - Optimal threshold value for EER

## How to Run

### Step 1: Ensure Preprocessing is Complete

Before running the comparison, make sure you have preprocessed data:

```bash
python scripts/01_run_preprocessing.py
```

### Step 2: Run the Comparison

Execute the comparison script:

```bash
python scripts/04_run_comparison.py
```

### Step 3: View Results

The script generates two output files in the `reports/` directory:

1. **CSV File** - `comparison_results_YYYY-MM-DD_HH-MM-SS.csv`
   - Contains the comparison table in CSV format
   - Can be opened in Excel or any spreadsheet application

2. **Text Log** - `comparison_report_YYYY-MM-DD_HH-MM-SS.txt`
   - Contains detailed training logs for each technique
   - Includes the final comparison table

## Customizing Parameters

You can adjust the optimization parameters in `scripts/04_run_comparison.py`:

```python
# Optimizer parameters
NUM_WOLVES = 10          # Number of search agents
MAX_ITER_FEAT = 20       # Iterations for feature selection
MAX_ITER_PARAM = 20      # Iterations for hyperparameter tuning
ALPHA = 0.99             # Balance between accuracy and feature count
```

**Note**: Increasing these values will improve results but significantly increase training time.

## Expected Training Time

With default parameters (NUM_WOLVES=10, MAX_ITER=20):
- GWO: ~5-10 minutes
- CGWO: ~5-10 minutes
- GWO-SVM: ~10-20 minutes
- CGWO-SVM: ~10-20 minutes

**Total**: ~30-60 minutes depending on dataset size and hardware

## Example Output

```
Tech      TP     FN     TN     FP    FAR    FRR    ERR   SPEC(%)  SEN(%)  PRECISION(%)  ACCURACY(%)  TRAINING TIME(s)  RECALL  Threshold
GWO       45.2   4.8    44.5   5.5   0.110  0.096  0.103  89.00    90.40   89.15         89.70        324.56           0.904   0.487
CGWO      46.1   3.9    45.2   4.8   0.096  0.078  0.087  90.40    92.20   90.56         91.30        342.12           0.922   0.492
GWO-SVM   47.3   2.7    46.8   3.2   0.064  0.054  0.059  93.60    94.60   93.67         94.10        512.34           0.946   0.501
CGWO-SVM  48.5   1.5    47.9   2.1   0.042  0.030  0.036  95.80    96.70   95.85         96.40        545.78           0.967   0.508
```

## Troubleshooting

### Error: "Preprocessed data not found"
- Run the preprocessing script first: `python scripts/01_run_preprocessing.py`

### Out of Memory Error
- Reduce NUM_WOLVES or MAX_ITER values
- Process fewer samples from your dataset

### Very Long Training Time
- The chaotic optimization with SVM takes the longest
- Consider reducing MAX_ITER_FEAT and MAX_ITER_PARAM for initial testing
- Use the default values for final comparison

## Notes

- All models are trained from scratch during comparison
- Results are automatically saved in CSV format
- The comparison uses the same train/test split for fair comparison
- Threshold values are automatically optimized for Equal Error Rate (EER)
