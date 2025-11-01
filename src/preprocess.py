import pandas as pd
import numpy as np
import os
import logging
import argparse
import joblib
from sklearn.model_selection import train_test_split

# Logger Setup
# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename="logs/preprocess.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

# Cleaning functions

def renamed_columns(df):
    """
    Cleans and standardizes column names.
    """
    try:
        df.columns = (
            df.columns.str.replace(r'\s+', '_', regex=True)   # replace spaces/newlines with _
                      .str.replace(r'[^\w]', '', regex=True)  # remove special chars
                      .str.lower()                            # convert to lowercase
        )
        logger.info("✅ Columns renamed successfully")
        return df
    except Exception as e:
        logger.error(f"An Unexpected Error has occured in renamed_columns: {e}")
        raise e

def drop_columns(df, cols_to_drop):
    """
    Safely drops specified columns if they exist.
    """
    try:
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        updated_df = df.drop(columns=existing_cols_to_drop)
        logger.info(f"✅ Columns dropped successfully: {existing_cols_to_drop}")
        return updated_df
    except Exception as e:
        logger.error(f"An Unexpected Error has occured during column dropping: {e}")
        raise e

def calculate_wqi(row):
    """
    Calculates the Water Quality Index (WQI) for a row.
    Assumes all required columns are present and numeric.
    """
    try:
        wqi = 100 - (
            4 * row['log_bod_avg'] +
            3 * row['log_fecal_coliform_avg'] +
            2 * row['log_nitrate_avg'] +
            1.5 * row['temperature_avg'] -
            2 * row['dissolved_oxygen_avg'] +
            10 * abs(7.5 - row['ph_avg'])
        )
        return np.clip(wqi, 0, 100)
    except Exception:
        # Return NaN if any input is NaN
        return np.nan

def classify_wqi(value):
    """
    Classifies a WQI value into a category.
    """
    if pd.isna(value):
        return 'Missing'
    if value > 80:
        return 'Excellent'
    elif value > 60:
        return 'Good'
    elif value > 40:
        return 'Moderate'
    elif value > 20:
        return 'Poor'
    else:
        return 'Very Poor or Unsafe'

# === 2. THE "FIT" FUNCTION (Learns from Training Data) ===

def fit_preprocessor(train_df):
    """
    Learns medians and quantiles *only* from the training data.
    Returns an 'artifact' dictionary containing these learned values.
    """
    logger.info("Starting to fit preprocessor...")
    artifact = {
        'medians': {},
        'quantiles': {}
    }
    
    # Define which columns get which treatment
    # These are the *original* column names (after renaming)
    normal_cols = [
        'temperature_cmin', 'temperature_cmax',
        'dissolved_oxygenmglmin', 'dissolved_oxygenmglmax',
        'phmin', 'phmax'
    ]
    
    # After rename_columns, 'conductivity_µmhocmmin' becomes 'conductivity_mhocmmin'
    skewed_cols = [
        'conductivity_mhocmmin', 'conductivity_mhocmmax',
        'bod_mglmin', 'bod_mglmax',
        'nitrate_n_mglmin', 'nitrate_n_mglmax',
        'fecal_coliform_mpn100mlmin', 'fecal_coliform_mpn100mlmax',
        'total_coliform_mpn100mlmin', 'total_coliform_mpn100mlmax'
    ]
    
    # Create a temporary copy for fitting
    fit_df = train_df.copy()

    # --- Step 1: Convert to numeric ---
    all_numeric_cols = normal_cols + skewed_cols
    for col in all_numeric_cols:
        if col in fit_df.columns:
            fit_df[col] = pd.to_numeric(fit_df[col], errors='coerce')

    # --- Step 2: Learn Medians ---
    # For normal columns, learn the median directly
    for col in normal_cols:
        if col in fit_df.columns:
            median_val = fit_df[col].median()
            artifact['medians'][col] = median_val
            
    # For skewed columns, learn the median *of the log-transformed data*
    for col in skewed_cols:
        if col in fit_df.columns:
            log_col_name = f'log_{col}'
            log_median_val = np.log1p(fit_df[col]).median()
            artifact['medians'][log_col_name] = log_median_val

    logger.info("✅ Medians learned.")

    # --- Step 3: Learn Quantiles (for Winsorization) ---
    # We must first apply imputation to the temp df to get accurate quantiles
    # This is a temporary step just for fitting
    
    # Impute normal cols
    for col in normal_cols:
        if col in fit_df.columns:
            fit_df[col] = fit_df[col].fillna(artifact['medians'][col])
            
    # Impute skewed cols (by creating log col, filling, then using it)
    for col in skewed_cols:
        if col in fit_df.columns:
            log_col_name = f'log_{col}'
            fit_df[log_col_name] = np.log1p(fit_df[col])
            fit_df[log_col_name] = fit_df[log_col_name].fillna(artifact['medians'][log_col_name])

    # --- Step 4: Feature Engineering (temp) ---
    # Create avg/range features to learn their quantiles
    feature_pairs = {
        'temperature': ('temperature_cmin', 'temperature_cmax'),
        'dissolved_oxygen': ('dissolved_oxygenmglmin', 'dissolved_oxygenmglmax'),
        'ph': ('phmin', 'phmax'),
        'log_fecal_coliform': ('log_fecal_coliform_mpn100mlmin', 'log_fecal_coliform_mpn100mlmax'),
        'log_total_coliform': ('log_total_coliform_mpn100mlmin', 'log_total_coliform_mpn100mlmax'),
        'log_nitrate': ('log_nitrate_n_mglmin', 'log_nitrate_n_mglmax'),
        'log_conductivity': ('log_conductivity_mhocmmin', 'log_conductivity_mhocmmax'),
        'log_bod': ('log_bod_mglmin', 'log_bod_mglmax')
    }
    
    final_numeric_features = []
    for prefix, (min_col, max_col) in feature_pairs.items():
         # Use log col if it exists, otherwise original
        min_col_to_use = f'log_{min_col}' if f'log_{min_col}' in fit_df.columns else min_col
        max_col_to_use = f'log_{max_col}' if f'log_{max_col}' in fit_df.columns else max_col

        # Check if the columns to use are actually in the dataframe
        if min_col_to_use in fit_df.columns and max_col_to_use in fit_df.columns:
            avg_col_name = f'{prefix}_avg'
            range_col_name = f'{prefix}_range'
            
            fit_df[avg_col_name] = (fit_df[min_col_to_use] + fit_df[max_col_to_use]) / 2.0
            fit_df[range_col_name] = (fit_df[max_col_to_use] - fit_df[min_col_to_use]).abs() # Use .abs()
            
            final_numeric_features.extend([avg_col_name, range_col_name])
        else:
            logger.warning(f"Skipping feature engineering for {prefix}: Columns {min_col_to_use} or {max_col_to_use} not found.")


    # --- Step 5: Learn Quantiles for final features ---
    for col in final_numeric_features:
        if col in fit_df.columns:
            lower = fit_df[col].quantile(0.01)
            upper = fit_df[col].quantile(0.99)
            if pd.isna(lower) or pd.isna(upper):
                logger.warning(f"Quantiles for {col} are NaN. Skipping clipping for this column.")
            else:
                artifact['quantiles'][col] = (lower, upper)
            
    logger.info("✅ Quantiles learned.")
    logger.info("✅ Preprocessor fitting complete.")
    return artifact

# === 3. THE "TRANSFORM" FUNCTION (Applies to any data) ===

def transform_data(df, artifact):
    """
    Applies all preprocessing steps to a new DataFrame
    using the *learned* values from the 'artifact'.
    """
    logger.info(f"Transforming data with {df.shape[0]} rows...")
    proc_df = df.copy()
    
    # --- Step 1: Convert to numeric ---
    # Get all columns that we have medians for
    all_median_cols_original = list(set([c.replace('log_', '') for c in artifact['medians'].keys()]))
    
    for col in all_median_cols_original:
        if col in proc_df.columns:
            proc_df[col] = pd.to_numeric(proc_df[col], errors='coerce')

    # --- Step 2: Log Transformation ---
    skewed_cols = [
        'conductivity_mhocmmin', 'conductivity_mhocmmax',
        'bod_mglmin', 'bod_mglmax',
        'nitrate_n_mglmin', 'nitrate_n_mglmax',
        'fecal_coliform_mpn100mlmin', 'fecal_coliform_mpn100mlmax',
        'total_coliform_mpn100mlmin', 'total_coliform_mpn100mlmax'
    ]
    for col in skewed_cols:
        if col in proc_df.columns:
            proc_df[f'log_{col}'] = np.log1p(proc_df[col])
            
    logger.info("  Log transformation applied.")

    # Imputation (using saved medians) 
    for col_name, median_val in artifact['medians'].items():
        if col_name in proc_df.columns:
            proc_df[col_name] = proc_df[col_name].fillna(median_val)
            
    logger.info("  Imputation applied.")

    # Feature Engineering (Avg/Range)
    feature_pairs = {
        'temperature': ('temperature_cmin', 'temperature_cmax'),
        'dissolved_oxygen': ('dissolved_oxygenmglmin', 'dissolved_oxygenmglmax'),
        'ph': ('phmin', 'phmax'),
        'log_fecal_coliform': ('log_fecal_coliform_mpn100mlmin', 'log_fecal_coliform_mpn100mlmax'),
        'log_total_coliform': ('log_total_coliform_mpn100mlmin', 'log_total_coliform_mpn100mlmax'),
        'log_nitrate': ('log_nitrate_n_mglmin', 'log_nitrate_n_mglmax'),
        'log_conductivity': ('log_conductivity_mhocmmin', 'log_conductivity_mhocmmax'),
        'log_bod': ('log_bod_mglmin', 'log_bod_mglmax')
    }
    
    original_min_max_cols = []
    final_feature_cols = ['state_name'] # Start with state_name
    
    for prefix, (min_col, max_col) in feature_pairs.items():
         # Use log col if it exists, otherwise original
        min_col_to_use = f'log_{min_col}' if f'log_{min_col}' in proc_df.columns else min_col
        max_col_to_use = f'log_{max_col}' if f'log_{max_col}' in proc_df.columns else max_col
        
        # Add original cols to drop list
        original_min_max_cols.extend([min_col, max_col])

        if min_col_to_use in proc_df.columns and max_col_to_use in proc_df.columns:
            avg_col_name = f'{prefix}_avg'
            range_col_name = f'{prefix}_range'
            
            proc_df[avg_col_name] = (proc_df[min_col_to_use] + proc_df[max_col_to_use]) / 2.0
            proc_df[range_col_name] = (proc_df[max_col_to_use] - proc_df[min_col_to_use]).abs()
            
            final_feature_cols.extend([avg_col_name, range_col_name])

    logger.info("  Feature engineering applied.")

    # Clipping (Winsorization) 
    for col_name, (lower, upper) in artifact['quantiles'].items():
        if col_name in proc_df.columns:
            proc_df[col_name] = proc_df[col_name].clip(lower, upper)
            
    logger.info("  Clipping applied.")
    
    # Drop highly correlated features (from your notebook) 
    cols_to_drop = ['log_total_coliform_range', 'log_total_coliform_avg']
    proc_df = drop_columns(proc_df, cols_to_drop)

    # Step 7: Create Target Variable (WQI) 
    proc_df['WQI'] = proc_df.apply(calculate_wqi, axis=1)
    proc_df['WQI_Category'] = proc_df['WQI'].apply(classify_wqi)
    
    logger.info("  Target variables (WQI, WQI_Category) created.")
    
    # Final Column Selection 
    # Keep only the engineered features, state, and target
    # Get the unique list of columns to keep
    final_cols_to_keep = list(dict.fromkeys(
        [col for col in proc_df.columns if col in final_feature_cols or col in ['WQI', 'WQI_Category']]
    ))
    
    # Ensure state_name is in the list if it's not already
    if 'state_name' not in final_cols_to_keep and 'state_name' in proc_df.columns:
        final_cols_to_keep.insert(0, 'state_name')
        
    final_df = proc_df[final_cols_to_keep].copy()
    
    # Drop rows where WQI could not be calculated (due to NaNs)
    final_df = final_df.dropna(subset=['WQI'])
    
    logger.info("  Final column selection and NaN WQI removal complete.")
    logger.info(f"✅ Transformation complete. Final shape: {final_df.shape}")
    
    return final_df

# THE MAIN EXECUTION FUNCTION 

def main(input_path, output_dir, artifacts_dir):
    logger.info(f"--- Starting preprocessing from {input_path} ---")
    
    try:
        #  Load Data 
        df = pd.read_csv(input_path)
        logger.info(f"Raw data loaded. Shape: {df.shape}")
        
        # Initial "Global" Cleaning 
        df = renamed_columns(df)
        
        # Drop columns with >60% missing data
        cols_to_drop = ['fecal_streptococci_mpn100mlmin', 'fecal_streptococci_mpn100mlmax']
        df = drop_columns(df, cols_to_drop)
        
        # Drop rows with missing state_name (as it's a key feature)
        df = df.dropna(subset=['state_name'])
        logger.info(f"Data after initial cleaning. Shape: {df.shape}")

        # Split Data (Prevent Data Leakage) 
        # FIX: Removed stratify=df['state_name'] because some states have only 1 sample
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        logger.info(f"Train set shape: {train_df.shape}, Test set shape: {test_df.shape}")
        
        # "Fit" on Training Data 
        # Learn medians and quantiles *only* from train_df
        preprocessor_artifact = fit_preprocessor(train_df)
        
        # "Transform" both datasets 
        train_processed = transform_data(train_df, preprocessor_artifact)
        test_processed = transform_data(test_df, preprocessor_artifact)
        
        # Save Artifacts 
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Save processed data
        train_path = os.path.join(output_dir, 'train_processed.csv')
        test_path = os.path.join(output_dir, 'test_processed.csv')
        train_processed.to_csv(train_path, index=False)
        test_processed.to_csv(test_path, index=False)
        logger.info(f"✅ Processed train data saved to {train_path}")
        logger.info(f"✅ Processed test data saved to {test_path}")
        
        # Save the "learned" artifact
        artifact_path = os.path.join(artifacts_dir, 'preprocessor_artifact.joblib')
        joblib.dump(preprocessor_artifact, artifact_path)
        logger.info(f"✅ Preprocessor artifact saved to {artifact_path}")
        
        logger.info("--- Preprocessing script finished successfully ---")
        
        print(f"\n--- Preprocessing Complete ---")
        print(f"Processed train data saved to: {train_path}")
        print(f"Processed test data saved to: {test_path}")
        print(f"Preprocessor artifact saved to: {artifact_path}")
        print("Log file saved to: logs/preprocess.log")

    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {input_path}")
        print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}")
        print(f"An unexpected error occurred: {e}")
        raise e

# --- Script Execution ---
if __name__ == "__main__":
    # This allows you to run the script from the command line
    parser = argparse.ArgumentParser(description="Preprocess raw water quality data.")
    
    parser.add_argument("--input_path", type=str, default="data/raw/raw_data.csv", help="Path to the raw data CSV file.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save processed train/test CSVs.")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to save the fitted preprocessor artifact.")
    
    args = parser.parse_args()
    
    # Create the output directories from the arguments
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    
    main(input_path=args.input_path, output_dir=args.output_dir, artifacts_dir=args.artifacts_dir)