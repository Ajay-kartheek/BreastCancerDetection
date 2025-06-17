import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def split_data_for_simulation(X_processed, y_processed, final_val_size, stream_prop_remainder, random_seed_val):
    print("\n--- Data Splitting for Simulation ---")
    X_train_stream_temp, X_final_val, y_train_stream_temp, y_final_val = train_test_split(
        X_processed, y_processed, test_size=final_val_size, random_state=random_seed_val, stratify=y_processed
    )
    X_init_train, X_str, y_init_train, y_str = train_test_split(
        X_train_stream_temp, y_train_stream_temp, test_size=stream_prop_remainder, random_state=random_seed_val, stratify=y_train_stream_temp
    )
    print(f"Data split: Initial Train ({X_init_train.shape[0]}), Stream ({X_str.shape[0]}), Final Validation ({X_final_val.shape[0]}) samples.")
    return X_init_train, X_str, X_final_val, y_init_train, y_str, y_final_val

def scale_features(X_init_train, X_str, X_final_val):
    print("\n--- Feature Scaling ---")
    scaler_obj = StandardScaler()
    X_init_train_scaled = scaler_obj.fit_transform(X_init_train)
    X_str_scaled = scaler_obj.transform(X_str)
    X_final_val_scaled = scaler_obj.transform(X_final_val) if X_final_val.shape[0] > 0 else X_final_val.copy()
    print("Features scaled.")
    return X_init_train_scaled, X_str_scaled, X_final_val_scaled, scaler_obj

def train_initial_model(X_train_scaled, y_train, solver, class_weight, random_seed_val):
    print("\n--- Initial Model Training ---")
    model = LogisticRegression(solver=solver, class_weight=class_weight, random_state=random_seed_val)
    model.fit(X_train_scaled, y_train)
    print(f"Initial Logistic Regression model trained (solver={solver}, class_weight={class_weight}).")
    return model

def find_optimal_threshold_cost_based(y_true, y_proba_positive, cost_fn, cost_fp, positive_label):
    thresholds = np.linspace(0.01, 0.99, 100)
    min_total_cost = float('inf')
    best_threshold = 0.5
    cost_curve_data = []
    if len(y_true) == 0 or len(np.unique(y_true)) < 2: return 0.5, float('inf'), []
    for t in thresholds:
        y_pred_temp = (y_proba_positive >= t).astype(int)
        cm_temp = confusion_matrix(y_true, y_pred_temp, labels=[0, positive_label])
        if cm_temp.size != 4: continue
        _, fp_temp, fn_temp, _ = cm_temp.ravel()
        current_total_cost = (fn_temp * cost_fn) + (fp_temp * cost_fp)
        cost_curve_data.append((t, current_total_cost))
        if current_total_cost < min_total_cost:
            min_total_cost = current_total_cost
            best_threshold = t
        elif current_total_cost == min_total_cost and t < best_threshold:
            best_threshold = t
    return best_threshold, min_total_cost, cost_curve_data



# The f1 optimization is validated only for comparison purposes.
# It is not used in the cost-based adaptive loop.
def find_optimal_threshold_f1(y_true, y_proba_positive, positive_label):
    thresholds = np.linspace(0.01, 0.99, 100)
    max_f1 = 0.0
    best_threshold = 0.5
    if len(y_true) == 0 or len(np.unique(y_true)) < 2: return 0.5, 0.0
    for t in thresholds:
        y_pred_temp = (y_proba_positive >= t).astype(int)
        current_f1 = f1_score(y_true, y_pred_temp, pos_label=positive_label, zero_division=0)
        if current_f1 > max_f1:
            max_f1 = current_f1
            best_threshold = t
        elif current_f1 == max_f1 and t < best_threshold:
            best_threshold = t
    return best_threshold, max_f1



def get_initial_optimal_thresholds(model, X_train_scaled, y_train, cost_fn_param, cost_fp_param, positive_label_param):
    print("\n--- Finding Initial Optimal Thresholds (on Training Data) ---")
    y_train_proba_positive = model.predict_proba(X_train_scaled)[:, 1]
    
    y_train_pred_default = model.predict(X_train_scaled)
    recall_default = recall_score(y_train, y_train_pred_default, pos_label=positive_label_param, zero_division=0)
    cm_default = confusion_matrix(y_train, y_train_pred_default, labels=[0, positive_label_param])
    cost_default = "N/A" 
    if cm_default.size == 4:
        _, fp_d, fn_d, _ = cm_default.ravel()
        cost_default = (fn_d * cost_fn_param) + (fp_d * cost_fp_param)
    print(f"Baseline on Training (0.5 thresh): Recall={recall_default:.4f}, Cost={cost_default}")

    opt_t_cost, min_t_cost, cost_curve = find_optimal_threshold_cost_based(
        y_train, y_train_proba_positive, cost_fn_param, cost_fp_param, positive_label=positive_label_param
    )
    print(f"Optimal threshold (cost-based) on training data: {opt_t_cost:.4f} with min cost: {min_t_cost:.2f}")
    opt_t_f1, max_t_f1 = find_optimal_threshold_f1(
        y_train, y_train_proba_positive, positive_label=positive_label_param
    )
    print(f"Optimal threshold (F1-max) on training data: {opt_t_f1:.4f} with max F1: {max_t_f1:.4f}")
    return opt_t_cost, min_t_cost, cost_curve, opt_t_f1, max_t_f1