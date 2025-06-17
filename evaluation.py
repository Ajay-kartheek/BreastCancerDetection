import numpy as np
import pandas as pd # For pd.Series.mean() in report
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# def plot_simulation_results(history_dict, initial_cost_curve, opt_t_initial, min_cost_initial,
#                             cost_fn_val, cost_fp_val, plot_fig_size_main, plot_fig_size_cost_curve, title_prefix=""):
#     print("\n--- Visualizing Simulation Results ---")
#     if not history_dict['batch_num']:
#         print("No history to plot.")
#         return

#     batches = history_dict['batch_num']
#     thresh_used = history_dict['current_threshold_used']
#     thresh_after_update = history_dict['threshold_updated_after_batch']
#     precisions = history_dict['batch_precision']
#     recalls = history_dict['batch_recall']
#     f1s = history_dict['batch_f1']
#     costs = history_dict['batch_cost']
#     thresh_changed = history_dict['threshold_actually_changed']

#     full_title = f'{title_prefix} Adaptive Thresholding System Performance'.strip()
#     fig, axs = plt.subplots(3, 1, figsize=plot_fig_size_main, sharex=True)
#     fig.suptitle(full_title, fontsize=16)

#     axs[0].plot(batches, thresh_used, marker='o', ms=4, linestyle='-', label='Threshold Used for Batch')
#     axs[0].step(batches, thresh_after_update, where='post', linestyle='--', color='green', label='Threshold After Update')
#     changed_indices = [b for i, (b, flag) in enumerate(zip(batches, thresh_changed)) if flag]
#     if changed_indices:
#         unique_label_added = False
#         for b_idx, b_val in enumerate(batches):
#             if thresh_changed[b_idx]:
#                 axs[0].axvline(x=b_val, color='red', linestyle=':', linewidth=1, 
#                                label='Threshold Changed Point' if not unique_label_added else "")
#                 unique_label_added = True
#     axs[0].set_ylabel('Threshold Value'); axs[0].set_title('Threshold Evolution'); axs[0].legend(loc='best'); axs[0].grid(True)

#     axs[1].plot(batches, precisions, marker='.', label='Precision')
#     axs[1].plot(batches, recalls, marker='.', label='Recall (Sensitivity)')
#     axs[1].plot(batches, f1s, marker='.', label='F1-score')
#     axs[1].set_ylabel('Score'); axs[1].set_title('Batch Performance Metrics'); axs[1].legend(loc='best'); axs[1].grid(True); axs[1].set_ylim(-0.05, 1.05)

#     axs[2].plot(batches, costs, marker='x', label='Cost per Batch')
#     axs[2].set_xlabel('Batch Number'); axs[2].set_ylabel(f'Cost (FN={cost_fn_val}, FP={cost_fp_val})'); axs[2].set_title('Cost per Batch'); axs[2].legend(loc='best'); axs[2].grid(True)

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()

#     if initial_cost_curve:
#         plt.figure(figsize=plot_fig_size_cost_curve)
#         t_plot, c_plot = zip(*initial_cost_curve)
#         plt.plot(t_plot, c_plot, marker='.')
#         plt.scatter([opt_t_initial], [min_cost_initial], color='red', s=100, zorder=5, label=f'Initial Optimal: T={opt_t_initial:.2f}, Cost={min_cost_initial:.2f}')
#         plt.title(f'{title_prefix} Initial Threshold vs. Cost (Training Data)'.strip())
#         plt.xlabel('Threshold'); plt.ylabel('Total Cost'); plt.legend(); plt.grid(True)
#         plt.show()

def perform_final_validation(model, X_val_s, y_val, final_thresh_val, cost_fn_val, cost_fp_val, pos_label_val):
    print("\n--- Final Validation on Hold-out Set ---")
    if X_val_s.shape[0] == 0:
        print("Final validation set is empty. Skipping.")
        return None 
    print(f"Using final adapted threshold: {final_thresh_val:.4f}")
    y_val_proba = model.predict_proba(X_val_s)[:, 1]
    y_val_pred = (y_val_proba >= final_thresh_val).astype(int)

    metrics = {'threshold': final_thresh_val}
    metrics['precision'] = precision_score(y_val, y_val_pred, pos_label=pos_label_val, zero_division=0)
    metrics['recall'] = recall_score(y_val, y_val_pred, pos_label=pos_label_val, zero_division=0)
    metrics['f1_score'] = f1_score(y_val, y_val_pred, pos_label=pos_label_val, zero_division=0)
    cm_fv = confusion_matrix(y_val, y_val_pred, labels=[0, pos_label_val])
    
    if cm_fv.size == 4:
        tn, fp, fn, tp = cm_fv.ravel()
        metrics['cm_string'] = f"({tn}, {fp}, {fn}, {tp})"
        metrics['cost'] = (fn * cost_fn_val) + (fp * cost_fp_val)
        print(f"  CM (TN,FP,FN,TP): {metrics['cm_string']}")
    else:
        fp = np.sum((y_val == 0) & (y_val_pred == pos_label_val))
        fn = np.sum((y_val == pos_label_val) & (y_val_pred == 0))
        metrics['cm_string'] = f"N/A (FPs:{fp}, FNs:{fn})"
        metrics['cost'] = (fn * cost_fn_val) + (fp * cost_fp_val)
        print(f"  CM: {metrics['cm_string']}")
        
    print(f"  P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}, Cost: {metrics['cost']:.2f}")
    return metrics

def generate_report_summary(history, final_val_res, init_thresh_static, reopt_int_val, win_size_b_val, 
                            stream_y_data, cost_fn_val, cost_fp_val, positive_class_label_val):
    print("\n\n--- Interpretation of Results & Report Summary ---")
    print("Objective: Develop an adaptive classification threshold optimization system.")

    if not (history and history.get('batch_num')):
        print("No simulation history found for detailed report.")
        return

    print("\n1. Threshold Adjustments & Adaptation Dynamics:")
    num_changes = sum(history['threshold_actually_changed'])
    init_thresh_used = history['current_threshold_used'][0] if history['current_threshold_used'] else init_thresh_static
    print(f"  - Threshold dynamically adjusted {num_changes} time(s) over {len(history['batch_num'])} batches.")
    print(f"  - Initial stream threshold: {init_thresh_used:.4f} (cost-optimized on training data).")
    # ... (rest of your detailed report prints from original Task 8) ...
    # This is just a snippet for brevity
    print("\n2. Handling Imbalance & Cost Sensitivity:")
    prop_positive_stream = pd.Series(stream_y_data).mean() if len(stream_y_data) > 0 else "N/A"
    print(f"  - Stream data positive class proportion: {prop_positive_stream:.2f}")
    print(f"  - Cost-based thresholding (FN={cost_fn_val}, FP={cost_fp_val}) drove optimization.")

    print("\n3. Final Hold-out Validation Performance:")
    if final_val_res and isinstance(final_val_res, dict):
        print(f"  - System (final threshold: {final_val_res.get('threshold', 'N/A'):.4f}) on hold-out: "
              f"Recall={final_val_res.get('recall', float('nan')):.4f}, "
              f"Cost={final_val_res.get('cost', float('nan')):.2f}")
    else:
        print("  - Final hold-out validation not performed or results unavailable.")
    
    print("\n4. Key Challenges Addressed & Observations:") # Your existing detailed points
    print(f"  - Threshold Drift Detection & Adaptation: Implemented via periodic re-optimization on a sliding window...")
    
    print("\n5. Recommendations for Long-Term Deployment & Improvements:") # Your existing detailed points
    print(f"  - **Continuous Monitoring:** ...")