# --- Main Script to Run the Adaptive Thresholding Simulation ---

# Import configurations
import config as cfg # Alias for brevity

# Import functions from custom modules
from data_processing import load_and_inspect_data, preprocess_data
from model_utils import (split_data_for_simulation, scale_features, train_initial_model,
                         get_initial_optimal_thresholds) # Removed unused F1 threshold finders for main flow
from simulation import run_adaptive_simulation
from evaluation import perform_final_validation

# Standard library imports (if any specific to main, like time)
import time

if __name__ == "__main__":
    print("Starting Adaptive Thresholding Simulation...")
    start_time = time.time()

    # --- Task 1 & 2: Load and Preprocess Data ---
    df_raw = load_and_inspect_data(cfg.FILE_PATH, cfg.TARGET_COLUMN)
    X_processed_df, y_processed_series = preprocess_data(
        df_raw, cfg.TARGET_COLUMN, cfg.POSITIVE_CLASS_NAME, cfg.POSITIVE_CLASS_LABEL
    )

    # --- Task 3 (Part 1): Data Splitting ---
    X_initial_train_df, X_stream_df, X_final_validation_df, \
    y_initial_train_series, y_stream_series, y_final_validation_series = split_data_for_simulation(
        X_processed_df, y_processed_series, 
        cfg.FINAL_VALIDATION_SIZE, cfg.STREAM_PROPORTION_OF_REMAINDER, cfg.RANDOM_SEED
    )

    # --- Task 3 (Part 2): Feature Scaling ---
    X_initial_train_scl, X_stream_scl, X_final_validation_scl, data_scaler = scale_features(
        X_initial_train_df, X_stream_df, X_final_validation_df
    )

    # --- Task 3 (Part 3): Model Training ---
    main_model = train_initial_model(
        X_initial_train_scl, y_initial_train_series, 
        cfg.LOGISTIC_REGRESSION_SOLVER, cfg.LOGISTIC_REGRESSION_CLASS_WEIGHT, cfg.RANDOM_SEED
    )

    # --- Task 4: Get Initial Optimal Thresholds ---
    initial_optimal_t_cost, initial_min_train_cost, training_cost_curve_data, \
    initial_optimal_t_f1, initial_max_train_f1 = get_initial_optimal_thresholds( # Pass costs from cfg
        main_model, X_initial_train_scl, y_initial_train_series,
        cfg.COST_FN, cfg.COST_FP, cfg.POSITIVE_CLASS_LABEL
    )

    # --- Tasks 5 & 6: Run Adaptive Simulation ---
    simulation_history, final_adapted_threshold = run_adaptive_simulation(
        main_model, X_stream_scl, y_stream_series,
        initial_thresh=initial_optimal_t_cost, # Use the cost-optimized one
        batch_size_val=cfg.BATCH_SIZE,
        window_size_batches_val=cfg.SLIDING_WINDOW_SIZE_BATCHES,
        reopt_interval_val=cfg.REOPTIMIZATION_INTERVAL_BATCHES,
        cost_fn_val=cfg.COST_FN,
        cost_fp_val=cfg.COST_FP,
        pos_label_val=cfg.POSITIVE_CLASS_LABEL
    )

    # --- Task 7: Visualization ---
    # plot_simulation_results(
    #     simulation_history, training_cost_curve_data, 
    #     initial_optimal_t_cost, initial_min_train_cost,
    #     cfg.COST_FN, cfg.COST_FP,
    #     cfg.PLOT_FIG_SIZE_MAIN, cfg.PLOT_FIG_SIZE_COST_CURVE,
    #     title_prefix="Main Simulation" # Example prefix
    # )

    # --- Task 8: Final Validation & Report ---
    final_validation_results = perform_final_validation(
        main_model, X_final_validation_scl, y_final_validation_series,
        final_thresh_val=final_adapted_threshold, # Use the one from simulation
        cost_fn_val=cfg.COST_FN,
        cost_fp_val=cfg.COST_FP,
        pos_label_val=cfg.POSITIVE_CLASS_LABEL
    )

    # generate_report_summary(
    #     simulation_history, final_validation_results, 
    #     initial_thresh_static=initial_optimal_t_cost,
    #     reopt_int_val=cfg.REOPTIMIZATION_INTERVAL_BATCHES, 
    #     win_size_b_val=cfg.SLIDING_WINDOW_SIZE_BATCHES, 
    #     stream_y_data=y_stream_series, # Pass the actual stream y data
    #     cost_fn_val=cfg.COST_FN, 
    #     cost_fp_val=cfg.COST_FP,
    #     positive_class_label_val=cfg.POSITIVE_CLASS_LABEL
    # )

    end_time = time.time()
    print(f"\n\nSCRIPT EXECUTION COMPLETED. Total time: {end_time - start_time:.2f} seconds.")