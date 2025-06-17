import numpy as np
import pandas as pd
import math
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from model_utils import find_optimal_threshold_cost_based 

def run_adaptive_simulation(model_obj, X_str_scaled, y_str_series,
                            initial_thresh, batch_size_val, window_size_batches_val, reopt_interval_val,
                            cost_fn_val, cost_fp_val, pos_label_val):
    print("\n--- Running Adaptive Simulation Loop ---")
    
    current_thresh = initial_thresh
    sliding_X_batches = []
    sliding_y_batches = []
    
    history = {
        'batch_num': [], 'current_threshold_used': [], 'threshold_updated_after_batch': [],
        'num_samples_in_batch': [], 'num_malignant_in_batch': [],
        'batch_precision': [], 'batch_recall': [], 'batch_f1': [], 'batch_cost': [],
        'window_size_records_at_update': [], 'threshold_actually_changed': []
    }

    num_total_samples = len(X_str_scaled)
    num_stream_batches = math.ceil(num_total_samples / batch_size_val)
    print(f"Processing {num_stream_batches} batches from stream data...")

    for i in range(num_stream_batches):
        batch_n_display = i + 1
        start = i * batch_size_val
        end = min((i + 1) * batch_size_val, num_total_samples)
        X_batch = X_str_scaled[start:end]
        y_batch = y_str_series.iloc[start:end]

        if len(X_batch) == 0: continue

        sliding_X_batches.append(X_batch)
        sliding_y_batches.append(y_batch)
        if len(sliding_X_batches) > window_size_batches_val:
            sliding_X_batches.pop(0)
            sliding_y_batches.pop(0)

        thresh_used_this_batch = current_thresh
        history['current_threshold_used'].append(thresh_used_this_batch)

        y_batch_proba = model_obj.predict_proba(X_batch)[:, 1]
        y_batch_pred = (y_batch_proba >= thresh_used_this_batch).astype(int)

        prec, rec, f1, cost = np.nan, np.nan, np.nan, np.nan
        if len(y_batch) > 0 and len(np.unique(y_batch)) > 0:
            prec = precision_score(y_batch, y_batch_pred, pos_label=pos_label_val, zero_division=0)
            rec = recall_score(y_batch, y_batch_pred, pos_label=pos_label_val, zero_division=0)
            f1 = f1_score(y_batch, y_batch_pred, pos_label=pos_label_val, zero_division=0)
            cm = confusion_matrix(y_batch, y_batch_pred, labels=[0, pos_label_val])
            if cm.size == 4: _, fp, fn, _ = cm.ravel()
            else:
                fp = np.sum((y_batch == 0) & (y_batch_pred == pos_label_val))
                fn = np.sum((y_batch == pos_label_val) & (y_batch_pred == 0))
            cost = (fn * cost_fn_val) + (fp * cost_fp_val)

        history['batch_num'].append(batch_n_display)
        history['num_samples_in_batch'].append(len(X_batch))
        history['num_malignant_in_batch'].append(y_batch.sum())
        history['batch_precision'].append(prec)
        history['batch_recall'].append(rec)
        history['batch_f1'].append(f1)
        history['batch_cost'].append(cost)

        thresh_changed_iter = False
        win_size_iter = np.nan
        window_full = (len(sliding_X_batches) == window_size_batches_val)

        if (batch_n_display % reopt_interval_val == 0) and window_full:
            X_win = np.vstack(sliding_X_batches)
            y_win = pd.concat(sliding_y_batches)
            win_size_iter = len(X_win)
            if len(y_win) > 0 and len(np.unique(y_win)) > 1:
                y_win_proba = model_obj.predict_proba(X_win)[:, 1]
                new_t, new_c, _ = find_optimal_threshold_cost_based( # This is now imported
                    y_win, y_win_proba, cost_fn_val, cost_fp_val, pos_label_val
                )
                if new_t != current_thresh:
                    print(f"  Batch {batch_n_display}: Threshold Re-optimized! Old: {current_thresh:.4f}, New: {new_t:.4f} (Win Cost: {new_c:.2f})")
                    current_thresh = new_t
                    thresh_changed_iter = True
                else:
                     print(f"  Batch {batch_n_display}: Threshold {current_thresh:.4f} remains optimal (Win Cost: {new_c:.2f}).")
            else:
                print(f"  Batch {batch_n_display}: Window data insufficient for re-optimization. Threshold maintained.")
        
        history['window_size_records_at_update'].append(win_size_iter)
        history['threshold_actually_changed'].append(thresh_changed_iter)
        history['threshold_updated_after_batch'].append(current_thresh)
        
        if batch_n_display % 10 == 0 or batch_n_display == num_stream_batches:
             print(f"Processed Batch {batch_n_display}/{num_stream_batches}. Next Thresh: {current_thresh:.4f}")
    return history, current_thresh