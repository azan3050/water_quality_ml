# processing.py

import pandas as pd
import numpy as np

# --- Define Constants ---
# These are the lists of columns you discovered in your EDA
# We use the *renamed* column names

SKEWED_COLS = [
    'conductivity_mhocmmin', 'conductivity_mhocmmax', 'bod_mglmin', 'bod_mglmax',
    'nitrate_n_mglmin', 'nitrate_n_mglmax', 'fecal_coliform_mpn100mlmin', 
    'fecal_coliform_mpn100mlmax', 'total_coliform_mpn100mlmin', 'total_coliform_mpn100mlmax'
]

FEATURE_PAIRS = {
    'temperature': ('temperature_cmin', 'temperature_cmax'),
    'dissolved_oxygen': ('dissolved_oxygenmglmin', 'dissolved_oxygenmglmax'),
    'ph': ('phmin', 'phmax'),
    'log_fecal_coliform': ('log_fecal_coliform_mpn100mlmin', 'log_fecal_coliform_mpn100mlmax'),
    'log_total_coliform': ('log_total_coliform_mpn100mlmin', 'log_total_coliform_mpn100mlmax'),
    'log_nitrate': ('log_nitrate_n_mglmin', 'log_nitrate_n_mglmax'),
    'log_conductivity': ('log_conductivity_mhocmmin', 'log_conductivity_mhocmmax'),
    'log_bod': ('log_bod_mglmin', 'log_bod_mglmax')
}

COLS_TO_DROP = ['log_total_coliform_range', 'log_total_coliform_avg']

# --- Individual Helper Functions ---

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and standardizes column names."""
    proc_df = df.copy()
    proc_df.columns = (
        proc_df.columns.str.replace(r'\s+', '_', regex=True)
                        .str.replace(r'[^\w]', '', regex=True)
                        .str.lower()
    )
    return proc_df

def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Converts all columns (except state_name) to numeric, coercing errors."""
    proc_df = df.copy()
    numeric_cols = [col for col in proc_df.columns if col != 'state_name']
    for col in numeric_cols:
        proc_df[col] = pd.to_numeric(proc_df[col], errors='coerce')
    return proc_df

def apply_log_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """Applies log1p to skewed columns."""
    proc_df = df.copy()
    for col in SKEWED_COLS:
        if col in proc_df.columns:
            proc_df[f'log_{col}'] = np.log1p(proc_df[col])
    return proc_df

def apply_imputation(df: pd.DataFrame, medians_dict: dict) -> pd.DataFrame:
    """Fills NaNs using the saved medians from the artifact."""
    proc_df = df.copy()
    for col_name, median_val in medians_dict.items():
        if col_name in proc_df.columns:
            proc_df[col_name] = proc_df[col_name].fillna(median_val)
        
        # Also fill original columns (for features that weren't logged)
        original_col = col_name.replace('log_', '')
        if original_col in proc_df.columns and pd.api.types.is_numeric_dtype(proc_df[original_col]):
             proc_df[original_col] = proc_df[original_col].fillna(median_val)
    return proc_df

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Creates _avg and _range features."""
    proc_df = df.copy()
    for prefix, (min_col, max_col) in FEATURE_PAIRS.items():
        min_col_to_use = f'log_{min_col}' if f'log_{min_col}' in proc_df.columns else min_col
        max_col_to_use = f'log_{max_col}' if f'log_{max_col}' in proc_df.columns else max_col
        
        if min_col_to_use in proc_df.columns and max_col_to_use in proc_df.columns:
            proc_df[f'{prefix}_avg'] = (proc_df[min_col_to_use] + proc_df[max_col_to_use]) / 2.0
            proc_df[f'{prefix}_range'] = (proc_df[max_col_to_use] - proc_df[min_col_to_use]).abs()
    return proc_df

def apply_clipping(df: pd.DataFrame, quantiles_dict: dict) -> pd.DataFrame:
    """Clips features based on saved quantiles."""
    proc_df = df.copy()
    for col_name, (lower, upper) in quantiles_dict.items():
        if col_name in proc_df.columns:
            proc_df[col_name] = proc_df[col_name].clip(lower, upper)
    return proc_df

def drop_correlated_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drops columns identified as highly correlated."""
    existing_cols_to_drop = [col for col in COLS_TO_DROP if col in df.columns]
    proc_df = df.drop(columns=existing_cols_to_drop)
    return proc_df

# --- Main Orchestrator Function ---

def process_raw_data(raw_df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """
    Runs the full preprocessing pipeline on raw data.
    This is the only function the API will need to import.
    """
    
    # Run all steps in the correct order
    df = rename_columns(raw_df)
    df = convert_to_numeric(df)
    df = apply_log_transformation(df)
    df = apply_imputation(df, artifact['medians'])
    df = apply_feature_engineering(df)
    df = apply_clipping(df, artifact['quantiles'])
    df = drop_correlated_features(df)
    
    return df