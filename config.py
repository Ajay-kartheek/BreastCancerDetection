# --- Configuration Constants ---

# File Path
FILE_PATH = "breast-cancer/data/breast-cancer.csv" # Adjust if your data folder is elsewhere relative to main.py

# Target Variable Definition
TARGET_COLUMN = 'class'
POSITIVE_CLASS_NAME = 'recurrence-events'
POSITIVE_CLASS_LABEL = 1 # Numerical label for the positive class

# Cost Definition
COST_FN = 15  # Cost of a False Negative
COST_FP = 1   # Cost of a False Positive

# Data Splitting Parameters
RANDOM_SEED = 42
FINAL_VALIDATION_SIZE = 0.2  # 20% for the final hold-out validation set
STREAM_PROPORTION_OF_REMAINDER = 0.7 # Stream is 70% of (Total - Final Validation)

# Adaptive Loop Parameters
BATCH_SIZE = 30
SLIDING_WINDOW_SIZE_BATCHES = 5
REOPTIMIZATION_INTERVAL_BATCHES = 3

# Model Parameters (can be expanded)
LOGISTIC_REGRESSION_SOLVER = 'liblinear'
LOGISTIC_REGRESSION_CLASS_WEIGHT = 'balanced'

# Plotting Parameters (optional)
PLOT_FIG_SIZE_MAIN = (14, 16)
PLOT_FIG_SIZE_COST_CURVE = (10, 6)