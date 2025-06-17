import pandas as pd


def load_and_inspect_data(file_path, target_column_name):
    print("--- Data Loading & Inspection ---")
    print(f"Attempting to load data from: {file_path}")
    try:
        df = pd.read_csv(file_path, na_values="?")
        print("Data loaded successfully!")
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the CSV file: {e}")
        exit()

    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataFrame Info:")
    df.info()
    print("\nDescriptive statistics:")
    print(df.describe(include='all'))
    print("\nValue counts for target column:")
    if target_column_name in df.columns:
        print(df[target_column_name].value_counts(dropna=False))
    else:
        print(f"ERROR: Target column '{target_column_name}' not found in loaded data.")
        exit()
    return df

def preprocess_data(df_original, target_col_name, positive_class_name_val, positive_label_val):
    print("\n--- Data Preprocessing ---")
    df = df_original.copy()

    y_series = df[target_col_name]
    X_df = df.drop(columns=[target_col_name])

    y_numerical = y_series.apply(lambda x: positive_label_val if x == positive_class_name_val else 0)
    print(f"Target column '{target_col_name}' converted to numerical (0/1).")

    missing_val_cols = ['falsede-caps', 'breast-quad'] # Specific to this dataset
    for col in missing_val_cols:
        if col in X_df.columns and X_df[col].isnull().any():
            mode = X_df[col].mode()[0]
            X_df[col].fillna(mode, inplace=True)
            print(f"Missing values in '{col}' (in X) filled with mode: '{mode}'")

    categorical_cols_to_encode = X_df.select_dtypes(include=['object', 'bool']).columns.tolist()
    X_encoded_df = pd.get_dummies(X_df, columns=categorical_cols_to_encode, prefix=categorical_cols_to_encode, drop_first=False)
    print(f"Categorical features one-hot encoded. New X shape: {X_encoded_df.shape}")
    return X_encoded_df, y_numerical