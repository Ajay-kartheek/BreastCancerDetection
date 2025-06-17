import pandas as pd
import numpy as np


FILE_PATH = "breast-cancer/data/breast-cancer.csv"

print("--- Task 1: Load Data & Initial Inspection ---")
print(f"Attempting to load data from: {FILE_PATH}")


try:
    df_original = pd.read_csv(FILE_PATH, na_values="?") # Tell pandas that '?' means missing
    print("Data loaded successfully!")
except FileNotFoundError:
    print(f"ERROR: File '{FILE_PATH}' not found in the current directory.")
    print("Please ensure the CSV file is in the same folder as the script, or update FILE_PATH.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV file: {e}")
    exit() 

print("\nFirst 5 rows of the loaded data (df_original):")
print(df_original.head())

print("\nInformation about the DataFrame (data types, non-null counts):")
df_original.info()

# The image shows column names. If your CSV doesn't have them, you'd add them here:
# column_names = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps',
#                 'deg-malig', 'breast', 'breast-quad', 'irradiat', 'class']
# df_original = pd.read_csv(FILE_PATH, names=column_names, na_values="?")
# print("\nFirst 5 rows (after potentially adding headers):")
# print(df_original.head())
# df_original.info() # Check info again if headers were added

print("\nDescriptive statistics (for all columns, including categorical):")
print(df_original.describe(include='all'))

print("\nUnique values in each column (helps understand categorical data and identify issues):")
for col in df_original.columns:
    unique_vals = df_original[col].unique()
    print(f"Column '{col}': {df_original[col].nunique()} unique values. Examples: {unique_vals[:min(5, len(unique_vals))]}") # Show up to 5 examples

print("\nValue counts for the 'class' column (our target):")
if 'class' in df_original.columns:
    print(df_original['class'].value_counts(dropna=False)) # dropna=False to see counts of NaN too
else:
    print("ERROR: 'class' column not found. Please check your CSV file's column names.")

print("\n--- End of Task 1 ---")




print("\n\n--- Task 2: Define Target/Features, Handle Missing, Encode ---")

# Make a copy to work on, to keep df_original unchanged
df_processed = df_original.copy()

# 1. Define Target (y) and Features (X)
TARGET_COLUMN = 'class'
# For now, we consider 'recurrence-events' as the positive class
POSITIVE_CLASS_NAME = 'recurrence-events'
# All other class values will be the negative class

if TARGET_COLUMN not in df_processed.columns:
    print(f"ERROR: Target column '{TARGET_COLUMN}' not found. Exiting.")
    exit()

y = df_processed[TARGET_COLUMN]
X = df_processed.drop(columns=[TARGET_COLUMN])

print(f"Target variable ('y') head:\n{y.head()}")
print(f"\nFeatures ('X') head:\n{X.head()}")

# 2. Convert Target Variable to Numerical (0 and 1)
# Let's map POSITIVE_CLASS_NAME to 1, and everything else to 0
y = y.apply(lambda x: 1 if x == POSITIVE_CLASS_NAME else 0)
print(f"\nNumerical target variable ('y') head (1 means '{POSITIVE_CLASS_NAME}'):\n{y.head()}")
print(f"Value counts for numerical y:\n{y.value_counts()}")
POSITIVE_CLASS_LABEL = 1 # Define this for consistency later

# 3. Handle Missing Values
# We identified missing values in 'falsede-caps' and 'breast-quad'.
# For categorical features, a common strategy is to fill with the mode (most frequent value).
missing_val_cols = ['falsede-caps', 'breast-quad']
for col in missing_val_cols:
    if col in X.columns:
        mode = X[col].mode()[0] # .mode() can return multiple if counts are tied, so take first
        X[col].fillna(mode, inplace=True)
        print(f"Missing values in '{col}' filled with mode: '{mode}'")
    else:
        print(f"Warning: Column '{col}' expected to have missing values was not found in X.")

print("\nInfo after handling missing values in X:")
X.info() # Should show no more NaNs in these columns

# 4. Encode Categorical Features
# We will use One-Hot Encoding for most categorical features.
# For features like 'age', 'tumor-size', 'inv-falsedes' which have an inherent order,
# Ordinal Encoding might be considered, but One-Hot is safer and simpler to start.
# The 'deg-malig' is already numerical (int64) but represents categories (1, 2, 3).
# 'irradiat' is already bool, pandas get_dummies handles bools fine.

# Identify categorical columns to encode (all 'object' type columns, and 'deg-malig' as it's categorical int)
# 'irradiat' is bool but get_dummies will convert it to 0/1 which is fine.
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
# 'deg-malig' is int but is categorical. If we one-hot encode it, ensure it's treated as such.
# For simplicity, let's convert 'deg-malig' to object first if we want to one-hot encode it like others.
# Or, we can leave it as is if its numerical nature (1, 2, 3) is meaningful as ordinal.
# Given it's "degree of malignancy", it IS ordinal.
# However, one-hot encoding is more general and makes fewer assumptions. Let's try one-hot for all non-numeric for now.

print(f"\nCategorical columns to be one-hot encoded: {categorical_cols}")

X_encoded = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols, drop_first=False)
# drop_first=True can be used to avoid multicollinearity, but for interpretation False is sometimes clearer.
# And some models (like tree-based) are not affected much by it. Start with False.

print("\nFeatures after One-Hot Encoding ('X_encoded') head:")
print(X_encoded.head())
print("\nShape of X_encoded:", X_encoded.shape)
print("\nInfo for X_encoded:")
X_encoded.info() # All columns should be numerical now

print("\n--- End of Task 2 ---")




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler # We'll need this for scaling features

print("\n\n--- Task 3: Split Data and Train Initial Model ---")

# 1. Split the Data into Training and Testing Sets
# X_encoded and y are our fully preprocessed features and target from Task 2.
# We'll set aside a portion of the data for testing (e.g., 30%).
# random_state ensures reproducibility of the split.
# stratify=y is important for imbalanced datasets to keep class proportions similar in train/test.

# Before splitting for the adaptive part, let's hold out a "final validation set"
# This set will simulate data from a "different hospital branch" or truly unseen future data.
# The rest will be split into initial_train and a "stream" for adaptive learning.

# Define the size for the final validation set (e.g., 20% of the total data)
FINAL_VALIDATION_SIZE = 0.2 # 20% for truly final hold-out
# The remaining 80% will be used for initial training and the simulated stream
INITIAL_TRAIN_PLUS_STREAM_SIZE = 1.0 - FINAL_VALIDATION_SIZE

# First, split off the final validation set
X_train_stream_temp, X_final_validation, y_train_stream_temp, y_final_validation = train_test_split(
    X_encoded, y,
    test_size=FINAL_VALIDATION_SIZE,
    random_state=42, # Use a consistent random state
    stratify=y      # Stratify based on the target variable y
)
print(f"Shape of X_final_validation: {X_final_validation.shape}, y_final_validation: {y_final_validation.shape}")
print(f"Class distribution in y_final_validation:\n{y_final_validation.value_counts(normalize=True)}")


# Now, from the remaining data (X_train_stream_temp), split into initial training and stream
# Let's say initial training is 30% OF THE REMAINING DATA (X_train_stream_temp)
# So, if X_train_stream_temp is 80% of total, and initial train is 30% of that,
# initial train size relative to total = 0.8 * 0.3 = 0.24 (or 24% of total)
# stream size relative to total = 0.8 * 0.7 = 0.56 (or 56% of total)

# Calculate the proportion for test_size for the second split
# If initial_train_size_of_remainder is 0.3, then stream_size_of_remainder is 0.7
STREAM_PROPORTION_OF_REMAINDER = 0.7

X_initial_train, X_stream, y_initial_train, y_stream = train_test_split(
    X_train_stream_temp, y_train_stream_temp,
    test_size=STREAM_PROPORTION_OF_REMAINDER, # This will be our "stream"
    random_state=42,    # Use the same random state for consistency if desired, or different
    stratify=y_train_stream_temp # Stratify this split as well
)

print(f"\nShape of X_initial_train: {X_initial_train.shape}, y_initial_train: {y_initial_train.shape}")
print(f"Class distribution in y_initial_train:\n{y_initial_train.value_counts(normalize=True)}")
print(f"\nShape of X_stream: {X_stream.shape}, y_stream: {y_stream.shape}") # This is for adaptive learning
print(f"Class distribution in y_stream:\n{y_stream.value_counts(normalize=True)}")


# 2. Feature Scaling
# Although one-hot encoded features are 0/1, the 'deg-malig' column is not.
# Logistic Regression can benefit from feature scaling.
# We fit the scaler ONLY on the training data and transform both train and stream/validation data.
scaler = StandardScaler()

# Fit on X_initial_train and transform it
X_initial_train_scaled = scaler.fit_transform(X_initial_train)

# Transform X_stream and X_final_validation using the SAME fitted scaler
X_stream_scaled = scaler.transform(X_stream)
X_final_validation_scaled = scaler.transform(X_final_validation)

print("\nFeature scaling applied.")
print(f"Example of scaled 'deg-malig' in training data (first 5): {X_initial_train_scaled[:5, X_encoded.columns.get_loc('deg-malig')]}")


# 3. Train an Initial Logistic Regression Model
# We'll use class_weight='balanced' to help with the imbalanced nature of the 'class' variable during model training.
initial_model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)

print("\nTraining the initial Logistic Regression model...")
initial_model.fit(X_initial_train_scaled, y_initial_train)
print("Initial model trained successfully!")

print("\n--- End of Task 3 ---")
 


from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt # For plotting later
import numpy as np # For numerical operations like linspace

print("\n\n--- Task 4: Initial Model Evaluation & Understanding Thresholds ---")

# --- PART 1: Make Predictions & Evaluate with Default Threshold (0.5) ---
# We'll use the X_initial_train_scaled for this initial check to see how it performs on data it saw.
# In a typical non-adaptive setup, you'd use an X_test set here.
# For our adaptive setup, we'll eventually evaluate on stream batches.
# For now, let's see performance on the training data itself as a first check.

print("\nEvaluating initial model on the TRAINING data (X_initial_train_scaled)...")

# Get predicted probabilities for the positive class (class 1: 'recurrence-events')
# initial_model.predict_proba(X_data) returns an array with two columns:
#   - Column 0: probability of class 0
#   - Column 1: probability of class 1
# We are interested in the probability of class 1.
y_initial_train_proba_positive = initial_model.predict_proba(X_initial_train_scaled)[:, 1]

# Get class predictions using the default threshold of 0.5
# initial_model.predict() implicitly uses a 0.5 threshold on the probabilities.
y_initial_train_pred_default_0_5 = initial_model.predict(X_initial_train_scaled)


# --- Calculate Metrics with Default 0.5 Threshold ---
print(f"\nMetrics for initial_model on TRAINING data with default 0.5 threshold:")

# Confusion Matrix:
# TN FP
# FN TP
# labels=[0, POSITIVE_CLASS_LABEL] ensures the order for TN, FP, FN, TP
# POSITIVE_CLASS_LABEL was defined in Task 2 as 1
cm_default_0_5 = confusion_matrix(y_initial_train, y_initial_train_pred_default_0_5, labels=[0, POSITIVE_CLASS_LABEL])
tn_default, fp_default, fn_default, tp_default = cm_default_0_5.ravel() # Flattens the 2x2 matrix into a 1D array

print(f"Confusion Matrix (TN, FP, FN, TP): ({tn_default}, {fp_default}, {fn_default}, {tp_default})")
# True Negatives (TN): Correctly predicted 'false-recurrence'
# False Positives (FP): Incorrectly predicted 'recurrence' (Type I error, false alarm)
# False Negatives (FN): Incorrectly predicted 'false-recurrence' (Type II error, MISSED recurrence - bad!)
# True Positives (TP): Correctly predicted 'recurrence'

precision_default_0_5 = precision_score(y_initial_train, y_initial_train_pred_default_0_5, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)
# Precision: TP / (TP + FP)
# Of all patients the model PREDICTED as 'recurrence', how many actually were?
print(f"Precision: {precision_default_0_5:.4f}")

recall_default_0_5 = recall_score(y_initial_train, y_initial_train_pred_default_0_5, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)
# Recall (Sensitivity): TP / (TP + FN)
# Of all patients who ACTUALLY had 'recurrence', how many did the model find? THIS IS CRITICAL.
print(f"Recall (Sensitivity): {recall_default_0_5:.4f}")

f1_default_0_5 = f1_score(y_initial_train, y_initial_train_pred_default_0_5, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)
# F1-score: 2 * (Precision * Recall) / (Precision + Recall)
# Harmonic mean of precision and recall, good for imbalanced classes.
print(f"F1-score: {f1_default_0_5:.4f}")


# --- PART 2: Explore Probabilities ---
print(f"\nExample predicted probabilities for class 1 (first 10 samples of training data):")
print(y_initial_train_proba_positive[:10])
# These are the raw scores the model produces before a threshold is applied.
# A threshold (like 0.5) converts these into 0 or 1 predictions.


# --- PART 3: Introduce Cost-Based Evaluation ---
COST_FN = 15  # Cost of a False Negative (missing a recurrence event)
COST_FP = 1   # Cost of a False Positive (false alarm for recurrence)

# Calculate cost for the default 0.5 threshold predictions on training data
total_cost_default_0_5 = (fn_default * COST_FN) + (fp_default * COST_FP)
print(f"\nCost-based evaluation (FN_cost={COST_FN}, FP_cost={COST_FP}) with 0.5 threshold:")
print(f"Total Cost on training data: ({fn_default} * {COST_FN}) + ({fp_default} * {COST_FP}) = {total_cost_default_0_5}")


# --- PART 4: Functions to Find an Optimal Static Threshold ---
# These functions will take true labels and predicted probabilities for the positive class.

def find_optimal_threshold_cost_based(y_true, y_proba_positive, cost_fn, cost_fp, positive_label=1):
    """
    Finds the threshold that minimizes the total cost.
    y_true: array-like of true binary labels (0 or 1)
    y_proba_positive: array-like of probabilities for the positive class
    cost_fn: cost of a false negative
    cost_fp: cost of a false positive
    positive_label: the label representing the positive class (e.g., 1)
    """
    thresholds = np.linspace(0.01, 0.99, 100) # Test 100 different thresholds
    min_total_cost = float('inf')
    best_threshold = 0.5
    cost_curve_data = [] # To store (threshold, cost) for plotting

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        print("Warning: Not enough data or only one class in y_true for threshold optimization.")
        return 0.5, float('inf'), []

    for t in thresholds:
        y_pred_temp = (y_proba_positive >= t).astype(int)
        # Ensure confusion matrix labels are [negative, positive] where negative is 0 and positive is positive_label
        # This assumes your negative class is 0.
        cm_temp = confusion_matrix(y_true, y_pred_temp, labels=[0, positive_label])
        tn_temp, fp_temp, fn_temp, tp_temp = cm_temp.ravel()
        current_total_cost = (fn_temp * cost_fn) + (fp_temp * cost_fp)
        cost_curve_data.append((t, current_total_cost))

        if current_total_cost < min_total_cost:
            min_total_cost = current_total_cost
            best_threshold = t
        # Tie-breaking: if costs are equal, prefer lower thresholds to slightly favor recall (more FNs are bad)
        # This is a small heuristic, could be more sophisticated.
        elif current_total_cost == min_total_cost:
            if t < best_threshold:
                 best_threshold = t


    return best_threshold, min_total_cost, cost_curve_data


def find_optimal_threshold_f1(y_true, y_proba_positive, positive_label=1):
    """
    Finds the threshold that maximizes the F1-score for the positive class.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    max_f1 = 0.0
    best_threshold = 0.5

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        print("Warning: Not enough data or only one class in y_true for F1 threshold optimization.")
        return 0.5, 0.0

    for t in thresholds:
        y_pred_temp = (y_proba_positive >= t).astype(int)
        current_f1 = f1_score(y_true, y_pred_temp, pos_label=positive_label, zero_division=0)
        if current_f1 > max_f1:
            max_f1 = current_f1
            best_threshold = t
        # Tie-breaking for F1: if F1s are equal, prefer lower thresholds to slightly favor recall
        elif current_f1 == max_f1:
            if t < best_threshold:
                best_threshold = t

    return best_threshold, max_f1

# Now, let's use these functions on our y_initial_train_proba_positive (probabilities on training data)
print(f"\nFinding optimal STATIC thresholds based on INITIAL TRAINING data probabilities:")

optimal_t_cost, min_train_cost, train_cost_curve = find_optimal_threshold_cost_based(
    y_initial_train, y_initial_train_proba_positive, COST_FN, COST_FP, positive_label=POSITIVE_CLASS_LABEL
)
print(f"Optimal threshold (cost-based) on training data: {optimal_t_cost:.4f} with min cost: {min_train_cost:.2f}")

optimal_t_f1, max_train_f1 = find_optimal_threshold_f1(
    y_initial_train, y_initial_train_proba_positive, positive_label=POSITIVE_CLASS_LABEL
)
print(f"Optimal threshold (F1-max) on training data: {optimal_t_f1:.4f} with max F1: {max_train_f1:.4f}")

# --- Optional: Visualize the Cost Curve for Cost-Based Threshold ---
if train_cost_curve:
    thresholds_plot, costs_plot = zip(*train_cost_curve)
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_plot, costs_plot, marker='.', linestyle='-')
    plt.scatter([optimal_t_cost], [min_train_cost], color='red', s=100, zorder=5,
                label=f'Optimal: T={optimal_t_cost:.2f}, Cost={min_train_cost:.2f}')
    plt.title(f'Threshold vs. Cost Trade-off on Training Data (FN_cost={COST_FN}, FP_cost={COST_FP})')
    plt.xlabel('Threshold')
    plt.ylabel('Total Cost')
    plt.legend()
    plt.grid(True)
    plt.show() # This will display the plot

print("\n--- End of Task 4 ---")



# ... (All your code from Task 1, Task 2, Task 3, Task 4 ends here) ...
# For example, the last line of Task 4 might be:
# print("\n--- End of Task 4 ---")
# Or it might be the plt.show() if you included the plot.

# ==============================================================================
# START OF COMBINED TASK 5 & TASK 6 SECTION
# This section replaces your separate Task 5 and Task 6 code blocks.
# ==============================================================================

import math # For math.ceil (should already be imported if used in Task 5)
# Ensure numpy is imported if not already at the top of your script
# import numpy as np # (if not already imported)
# Ensure pandas is imported if not already at the top (for pd.concat)
# import pandas as pd # (if not already imported)
# Ensure metrics are imported (should be from Task 4)
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

print("\n\n--- Task 5 & 6: Adaptive Loop with Batch Prediction, Performance Tracking, & Re-optimization ---")

# --- Configuration for the Adaptive Loop ---
BATCH_SIZE = 30
SLIDING_WINDOW_SIZE_BATCHES = 5
REOPTIMIZATION_INTERVAL_BATCHES = 3
# COST_FN and COST_FP should be available from Task 4
# POSITIVE_CLASS_LABEL should be available from Task 2

# --- Initializing/Resetting for Adaptive Simulation ---
# (This is the initialization block we discussed, placed correctly before the loop)
print("\nInitializing/Resetting for Adaptive Simulation...")

# 1. Establish Initial Adaptive Threshold
current_adaptive_threshold = optimal_t_cost # This was calculated in Task 4
print(f"Initial adaptive threshold set to: {current_adaptive_threshold:.4f}")

# Reset/Initialize sliding window lists
sliding_window_X_batches = []
sliding_window_y_batches = []
print("Sliding window lists initialized.")

# Reset/Initialize the history dictionary
history_adaptive = {
    'batch_num': [],
    'current_threshold_used': [],       # Threshold used for predictions ON THIS BATCH
    'threshold_updated_after_batch': [],# The threshold value AFTER any re-optimization in this step (for next batch)
    'num_samples_in_batch': [],
    'num_malignant_in_batch': [],
    'batch_precision': [],
    'batch_recall': [],
    'batch_f1': [],
    'batch_cost': [],
    'window_size_records_at_update': [], # Records in window when re-opt was ATTEMPTED
    'threshold_actually_changed': []     # Boolean: True if threshold value was modified in this step
}
print("history_adaptive dictionary initialized.")
# --- End of Initialization ---


# 2. Simulate Data Stream Processing Setup
num_total_stream_samples = len(X_stream_scaled) # X_stream_scaled from Task 3
num_batches_in_stream = math.ceil(num_total_stream_samples / BATCH_SIZE)

print(f"\nStarting adaptive simulation loop...")
print(f"Total stream samples: {num_total_stream_samples}")
print(f"Batch size: {BATCH_SIZE} samples")
print(f"Number of batches to process: {num_batches_in_stream}")
print(f"Sliding window will hold data from up to {SLIDING_WINDOW_SIZE_BATCHES} batches.")
print(f"Threshold re-optimization will be attempted every {REOPTIMIZATION_INTERVAL_BATCHES} batches if window is full.")


# --- Main Adaptive Simulation Loop ---
for i in range(num_batches_in_stream):
    batch_num_display = i + 1
    # print(f"\n--- Processing Batch {batch_num_display}/{num_batches_in_stream} ---") # Verbose, can be commented out

    start_idx = i * BATCH_SIZE
    end_idx = min((i + 1) * BATCH_SIZE, num_total_stream_samples)

    X_current_batch_scaled = X_stream_scaled[start_idx:end_idx]
    y_current_batch = y_stream.iloc[start_idx:end_idx] # y_stream from Task 3

    if len(X_current_batch_scaled) == 0:
        continue

    # --- Sliding Window Logic ---
    sliding_window_X_batches.append(X_current_batch_scaled)
    sliding_window_y_batches.append(y_current_batch)
    if len(sliding_window_X_batches) > SLIDING_WINDOW_SIZE_BATCHES:
        sliding_window_X_batches.pop(0)
        sliding_window_y_batches.pop(0)

    # --- Batch Prediction, Performance, and History Logging ---
    # Store the threshold that will be USED for THIS batch's predictions
    threshold_used_for_this_batch = current_adaptive_threshold
    history_adaptive['current_threshold_used'].append(threshold_used_for_this_batch)

    # 1. Predict on Current Batch
    y_batch_proba_positive = initial_model.predict_proba(X_current_batch_scaled)[:, 1] # initial_model from Task 3
    y_batch_pred_adaptive = (y_batch_proba_positive >= threshold_used_for_this_batch).astype(int)

    # 2. Calculate Batch Performance
    precision_batch, recall_batch, f1_batch, cost_batch = np.nan, np.nan, np.nan, np.nan # Default if not calculable
    if len(y_current_batch) > 0 and len(np.unique(y_current_batch)) > 0 :
        precision_batch = precision_score(y_current_batch, y_batch_pred_adaptive, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)
        recall_batch = recall_score(y_current_batch, y_batch_pred_adaptive, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)
        f1_batch = f1_score(y_current_batch, y_batch_pred_adaptive, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)
        cm_batch = confusion_matrix(y_current_batch, y_batch_pred_adaptive, labels=[0, POSITIVE_CLASS_LABEL])
        if cm_batch.size == 4:
            _, fp_b, fn_b, _ = cm_batch.ravel()
        else:
            fp_b = np.sum((y_current_batch == 0) & (y_batch_pred_adaptive == POSITIVE_CLASS_LABEL))
            fn_b = np.sum((y_current_batch == POSITIVE_CLASS_LABEL) & (y_batch_pred_adaptive == 0))
        cost_batch = (fn_b * COST_FN) + (fp_b * COST_FP)

    # 3. Store basic batch info and performance
    history_adaptive['batch_num'].append(batch_num_display)
    history_adaptive['num_samples_in_batch'].append(len(X_current_batch_scaled))
    history_adaptive['num_malignant_in_batch'].append(y_current_batch.sum())
    history_adaptive['batch_precision'].append(precision_batch)
    history_adaptive['batch_recall'].append(recall_batch)
    history_adaptive['batch_f1'].append(f1_batch)
    history_adaptive['batch_cost'].append(cost_batch)

    # --- Periodic Threshold Re-optimization ---
    threshold_actually_changed_this_iter = False
    logged_window_size_this_iter = np.nan
    window_is_full_enough = (len(sliding_window_X_batches) == SLIDING_WINDOW_SIZE_BATCHES)

    if (batch_num_display % REOPTIMIZATION_INTERVAL_BATCHES == 0) and window_is_full_enough:
        # print(f"  Batch {batch_num_display}: Attempting re-optimization...") # Verbose
        X_data_in_window = np.vstack(sliding_window_X_batches)
        y_data_in_window = pd.concat(sliding_window_y_batches)
        logged_window_size_this_iter = len(X_data_in_window)

        if len(y_data_in_window) > 0 and len(np.unique(y_data_in_window)) > 1:
            y_window_proba_positive = initial_model.predict_proba(X_data_in_window)[:, 1]
            new_optimal_threshold, new_min_cost, _ = find_optimal_threshold_cost_based(
                y_data_in_window, y_window_proba_positive, COST_FN, COST_FP, positive_label=POSITIVE_CLASS_LABEL
            )
            if new_optimal_threshold != current_adaptive_threshold: # Compare with the one used at start of this iter
                print(f"  Batch {batch_num_display}: Threshold Re-optimized! Old: {current_adaptive_threshold:.4f}, New: {new_optimal_threshold:.4f} (Window Cost: {new_min_cost:.2f})")
                current_adaptive_threshold = new_optimal_threshold # UPDATE for next batches
                threshold_actually_changed_this_iter = True
            else:
                print(f"  Batch {batch_num_display}: Threshold {current_adaptive_threshold:.4f} remains optimal (Window Cost: {new_min_cost:.2f}).")
        else:
            print(f"  Batch {batch_num_display}: Window data insufficient for re-optimization. Threshold maintained.")
    
    history_adaptive['window_size_records_at_update'].append(logged_window_size_this_iter)
    history_adaptive['threshold_actually_changed'].append(threshold_actually_changed_this_iter)
    # Log the threshold that will be used for the NEXT batch (or this one if it changed mid-step, though less common)
    history_adaptive['threshold_updated_after_batch'].append(current_adaptive_threshold)


    # Print a summary for the current batch
    if batch_num_display % 1 == 0 or batch_num_display == num_batches_in_stream :
        print(f"Batch {batch_num_display}: Samples={len(X_current_batch_scaled)}, Malignant={y_current_batch.sum()}, "
              f"Thresh_Used={threshold_used_for_this_batch:.4f}, Recall={recall_batch:.2f}, Cost={cost_batch:.2f}, "
              f"Next_Thresh={current_adaptive_threshold:.4f}")

# --- End of Main Adaptive Simulation Loop ---

print(f"\nCompleted adaptive simulation of {num_batches_in_stream} batches.")
print("\nSnapshot of history_adaptive (last 10 entries if available, else all):")
history_length = len(history_adaptive['batch_num'])
display_count = min(10, history_length)

for key in ['batch_num', 'current_threshold_used', 'threshold_updated_after_batch', 'batch_recall', 'batch_cost', 'window_size_records_at_update', 'threshold_actually_changed']:
    if key in history_adaptive and history_adaptive[key]: # Check if key exists and list is not empty
        print(f"  {key}: {history_adaptive[key][-display_count:]}") # Show last 'display_count' entries

print("\n--- End of Task 5 & 6 ---")

# ==============================================================================
# END OF COMBINED TASK 5 & TASK 6 SECTION
# ==============================================================================

import matplotlib.pyplot as plt # Should already be imported if used in Task 4
import numpy as np # Should already be imported

print("\n\n--- Task 7: Visualization of Performance Metrics & Threshold Evolution ---")

# Ensure history_adaptive has data before trying to plot
if not history_adaptive['batch_num']:
    print("No history to plot. Skipping Task 7.")
else:
    # Convert history dictionary to a pandas DataFrame for easier plotting (optional but can be handy)
    # For now, we'll plot directly from the dictionary lists.

    # Extract data from history for plotting
    batches = history_adaptive['batch_num']
    thresholds_used = history_adaptive['current_threshold_used']
    thresholds_after_update = history_adaptive['threshold_updated_after_batch'] # Shows effect of update
    precisions = history_adaptive['batch_precision']
    recalls = history_adaptive['batch_recall']
    f1_scores = history_adaptive['batch_f1']
    costs = history_adaptive['batch_cost']
    threshold_changed_flags = history_adaptive['threshold_actually_changed']

    # Create figure and axes for subplots
    # We'll have 3 main plots: Thresholds, Performance Metrics (P, R, F1), and Cost
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle('Adaptive Thresholding System Performance Over Batches', fontsize=16)

    # --- Plot 1: Threshold Evolution ---
    axs[0].plot(batches, thresholds_used, marker='o', linestyle='-', color='blue', label='Threshold Used for Batch Predictions')
    # To show when updates happened more clearly, we can plot the 'threshold_updated_after_batch'
    # This line represents the threshold value that is set *after* the current batch's processing and re-opt.
    axs[0].step(batches, thresholds_after_update, marker='.', where='post', linestyle='--', color='green', label='Threshold Value After Update (for next batch)')

    # Add vertical lines where the threshold actually changed
    for i, changed_flag in enumerate(threshold_changed_flags):
        if changed_flag: # If True
            axs[0].axvline(x=batches[i], color='red', linestyle=':', linewidth=1, label='Threshold Changed' if i == threshold_changed_flags.index(True) else "")

    axs[0].set_ylabel('Threshold Value')
    axs[0].set_title('Threshold Evolution')
    axs[0].legend(loc='best')
    axs[0].grid(True)

    # --- Plot 2: Performance Metrics (Precision, Recall, F1-score) ---
    axs[1].plot(batches, precisions, marker='.', linestyle='-', color='dodgerblue', label='Precision')
    axs[1].plot(batches, recalls, marker='.', linestyle='-', color='orangered', label='Recall (Sensitivity)')
    axs[1].plot(batches, f1_scores, marker='.', linestyle='-', color='forestgreen', label='F1-score')
    axs[1].set_ylabel('Score (0.0 to 1.0)')
    axs[1].set_title('Batch Performance Metrics')
    axs[1].legend(loc='best')
    axs[1].grid(True)
    axs[1].set_ylim(-0.05, 1.05) # Set y-axis limits for scores

    # --- Plot 3: Cost per Batch ---
    axs[2].plot(batches, costs, marker='x', linestyle='-', color='purple', label='Total Cost per Batch')
    axs[2].set_xlabel('Batch Number')
    axs[2].set_ylabel(f'Cost (FN={COST_FN}, FP={COST_FP})')
    axs[2].set_title('Cost per Batch')
    axs[2].legend(loc='best')
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show() # Display the combined plot

    # --- Optional: Re-display the Threshold vs. Cost Trade-off from Task 4 (Initial Training Data) ---
    # This helps remember why the initial threshold was chosen.
    # The `train_cost_curve` variable should be available from Task 4.
    if 'train_cost_curve' in globals() and train_cost_curve:
        print("\nRe-displaying initial Threshold vs. Cost Trade-off (from Task 4 on training data):")
        thresholds_task4, costs_task4 = zip(*train_cost_curve)
        optimal_t_cost_task4 = optimal_t_cost # from Task 4
        min_train_cost_task4 = min_train_cost # from Task 4

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds_task4, costs_task4, marker='.', linestyle='-')
        plt.scatter([optimal_t_cost_task4], [min_train_cost_task4], color='red', s=100, zorder=5,
                    label=f'Initial Optimal: T={optimal_t_cost_task4:.2f}, Cost={min_train_cost_task4:.2f}')
        plt.title(f'Initial Threshold vs. Cost (Training Data - Task 4) (FN_cost={COST_FN}, FP_cost={COST_FP})')
        plt.xlabel('Threshold')
        plt.ylabel('Total Cost')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("\n`train_cost_curve` not found or empty, skipping re-display of initial cost curve.")

print("\n--- End of Task 7 ---")



print("\n\n--- Task 8: Final Model Validation & Interpretation of Results ---")

# --- 1. Final Validation on Hold-out Set ---
# We use:
# - initial_model (trained once in Task 3)
# - X_final_validation_scaled, y_final_validation (from Task 3, completely unseen since)
# - current_adaptive_threshold (the VERY LAST value it had at the end of the Task 6 loop)

if 'X_final_validation_scaled' in globals() and len(X_final_validation_scaled) > 0:
    print("\nPerforming final validation on the completely unseen hold-out set...")
    print(f"Using final adapted threshold: {current_adaptive_threshold:.4f}") # This is the last value from the loop

    # Get probabilities on the validation set
    y_final_val_proba_positive = initial_model.predict_proba(X_final_validation_scaled)[:, 1]

    # Apply the final adapted threshold
    y_final_val_pred = (y_final_val_proba_positive >= current_adaptive_threshold).astype(int)

    # Calculate metrics for the validation set
    precision_final_val = precision_score(y_final_validation, y_final_val_pred, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)
    recall_final_val = recall_score(y_final_validation, y_final_val_pred, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)
    f1_final_val = f1_score(y_final_validation, y_final_val_pred, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)

    cm_final_val = confusion_matrix(y_final_validation, y_final_val_pred, labels=[0, POSITIVE_CLASS_LABEL])
    if cm_final_val.size == 4:
        tn_fv, fp_fv, fn_fv, tp_fv = cm_final_val.ravel()
    else: # Manual calculation for robustness
        tp_fv = np.sum((y_final_validation == POSITIVE_CLASS_LABEL) & (y_final_val_pred == POSITIVE_CLASS_LABEL))
        fp_fv = np.sum((y_final_validation == 0) & (y_final_val_pred == POSITIVE_CLASS_LABEL))
        fn_fv = np.sum((y_final_validation == POSITIVE_CLASS_LABEL) & (y_final_val_pred == 0))
        tn_fv = np.sum((y_final_validation == 0) & (y_final_val_pred == 0)) # Added for completeness

    cost_final_val = (fn_fv * COST_FN) + (fp_fv * COST_FP)

    print(f"\nFinal Validation Set Performance (Threshold: {current_adaptive_threshold:.4f}):")
    print(f"  Confusion Matrix (TN, FP, FN, TP): ({tn_fv}, {fp_fv}, {fn_fv}, {tp_fv})")
    print(f"  Precision: {precision_final_val:.4f}")
    print(f"  Recall (Sensitivity): {recall_final_val:.4f}")
    print(f"  F1-score: {f1_final_val:.4f}")
    print(f"  Total Cost: {cost_final_val:.2f}")

else:
    print("\nFinal validation set not available or empty. Skipping final validation.")


# --- 2. Interpretation of Simulation Results & Report Summary (Textual Output) ---
print("\n\n--- Interpretation of Results & Report Summary ---")
print("Objective: Develop an adaptive classification threshold optimization system.")

if 'history_adaptive' in globals() and history_adaptive['batch_num']:
    print("\n1. Threshold Adjustments & Adaptation:")
    num_threshold_changes = sum(history_adaptive['threshold_actually_changed'])
    print(f"  - The classification threshold was dynamically adjusted {num_threshold_changes} time(s) during the simulation.")
    
    adjustment_points = [history_adaptive['batch_num'][i] for i, changed in enumerate(history_adaptive['threshold_actually_changed']) if changed]
    if adjustment_points:
        print(f"  - Threshold value changes occurred after processing batches: {adjustment_points}")
        for point in adjustment_points:
            idx = history_adaptive['batch_num'].index(point)
            old_t = history_adaptive['current_threshold_used'][idx] # Threshold used for batch 'point'
            new_t = history_adaptive['threshold_updated_after_batch'][idx] # Threshold after update for batch 'point'
            print(f"    - After batch {point}: Threshold changed from approx {old_t:.4f} to {new_t:.4f}")
    else:
        print(f"  - The threshold remained static at {current_adaptive_threshold:.4f} after initial setting (or was never updated).")

    print(f"  - Initial threshold was {history_adaptive['current_threshold_used'][0]:.4f} (cost-optimized on training data).")
    print(f"  - Re-optimization was attempted every {REOPTIMIZATION_INTERVAL_BATCHES} batches using data from a sliding window of up to {SLIDING_WINDOW_SIZE_BATCHES} batches.")
    # You can analyze the plots from Task 7 visually:
    #   - Did recall improve or stabilize after threshold changes?
    #   - Did cost per batch decrease or stabilize?
    #   - If you had a drift simulation mechanism: Did the system adapt when drift was introduced?
    #     (Our current setup doesn't explicitly simulate drift, but real data streams inherently have it)
    print("  - (Visually inspect plots from Task 7 to see how performance metrics like Recall and Cost per Batch responded to threshold changes).")


    print("\n2. Handling Imbalance & Cost Sensitivity:")
    print(f"  - The dataset has an imbalance (Positive class (1) proportion in y_stream: {y_stream.mean():.2f}).")
    print(f"  - Cost-based thresholding (FN_cost={COST_FN}, FP_cost={COST_FP}) was used, strongly penalizing False Negatives.")
    print(f"  - This approach aims to maximize recall (sensitivity) to detect 'recurrence-events' without excessively increasing overall cost.")
    print(f"  - (Observe the 'Recall (Sensitivity)' line in the performance plot from Task 7; it should generally be high or improve with adaptation if FNs were initially high).")

    print("\n3. Final Hold-out Validation:")
    if 'X_final_validation_scaled' in globals() and len(X_final_validation_scaled) > 0:
        print(f"  - The system (initial_model + final adapted threshold of {current_adaptive_threshold:.4f}) was tested on a completely unseen hold-out set.")
        print(f"  - Performance on this set (Recall: {recall_final_val:.4f}, Cost: {cost_final_val:.2f}) indicates generalization ability.")
        print(f"  - Compare these final validation metrics to the average performance observed during the stream simulation.")
    else:
        print("  - Final hold-out validation was not performed (no validation set).")

else:
    print("No simulation history found to interpret.")

print("\n4. Recommendations for Long-Term Deployment & Improvements:")
print(f"  - **Continuous Monitoring:** Implement robust, automated monitoring of not just model metrics but also data drift on input features (e.g., using statistical tests like KS-test, Population Stability Index).")
print(f"  - **Sophisticated Drift Detection:** Instead of purely periodic re-optimization, integrate dedicated drift detection algorithms (e.g., DDM, ADWIN, Page-Hinkley) to trigger threshold (and potentially model) re-evaluation more dynamically when significant drift is confirmed.")
print(f"  - **Model Retraining/Updating:** This system focuses on threshold adaptation. For significant or prolonged concept drift, the underlying `initial_model` itself may need retraining or updating. Plan for periodic model retraining or retraining triggered by severe drift or performance degradation.")
print(f"  - **Feedback Loop with Clinicians:** Regularly review cases where model predictions (especially FNs and FPs) differ from clinical outcomes to understand model weaknesses, changing data characteristics, and refine cost assignments.")
print(f"  - **Cost Function Refinement:** The costs (FN, FP) might evolve or vary by sub-population. Allow for mechanisms to update these costs based on new clinical guidelines or economic analyses, which would then influence threshold selection.")
print(f"  - **Explore Different Base Models:** While Logistic Regression is a good start, other models (e.g., Gradient Boosting, Random Forests, Neural Networks) might offer better baseline performance, which could then be further enhanced by adaptive thresholding.")
print(f"  - **Advanced Adaptive Strategies:** Explore more complex adaptive methods, such as those that adapt model parameters online or use ensemble methods that can weigh models based on recent performance.")
print(f"  - **A/B Testing in Deployment:** Before fully rolling out a new adaptive strategy or a significantly changed threshold logic, consider A/B testing or shadow deployment in a controlled manner to validate its real-world impact.")
print(f"  - **Demographic Factor Analysis (as per original problem):** To evaluate how demographic changes impact threshold selection, one could: ")
print(f"    a) Track statistics (mean, std, distribution) of key demographic-related features (e.g., 'age' bins, 'tumor-size' bins if available as raw) within the sliding window.")
print(f"    b) Correlate significant changes in these feature statistics with points where the optimal threshold shifted. If the threshold consistently changes when, for example, the average tumor size in the window shifts, it indicates adaptation to that demographic trend.")
print(f"    c) This requires having interpretable features related to demographics to monitor. Our one-hot encoded features make direct mean/std tracking less intuitive than on original binned/numerical features.")

print("\n--- End of Task 8 ---")