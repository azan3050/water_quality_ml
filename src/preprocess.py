# src/preprocess.py

import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
import argparse
from datetime import datetime

# Ensure 'src' package is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main processing pipeline and helpers
from src.processing import (
    process_raw_data,
    rename_columns,
    convert_to_numeric,
    calculate_wqi,
    classify_wqi
)

# Logging setup
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "preprocess.log"),
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


# Utility functions
def fit_preprocessor(df_numeric):
    """
    Compute preprocessing artifacts (medians, quantiles)
    from a *numerically-converted* dataframe.
    """
    logger.info("Fitting preprocessing artifacts...")
    try:
        # 1. Calculate Medians for imputation
        from src.processing import SKEWED_COLS
        
        df_for_fit = df_numeric.copy()
        
        # Apply log transform to temp df to get correct medians
        for col in SKEWED_COLS:
            if col in df_for_fit.columns:
                df_for_fit[f'log_{col}'] = np.log1p(df_for_fit[col])
        
        medians = df_for_fit.median(numeric_only=True)
        
        # 2. Calculate Quantiles for clipping
        quantiles = {
            col: (df_for_fit[col].quantile(0.01), df_for_fit[col].quantile(0.99))
            for col in df_for_fit.select_dtypes(include="number").columns
            if not df_for_fit[col].isnull().all()
        }
        
        artifact = {
            "medians": medians,
            "quantiles": quantiles,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info("✅ Artifact fitting complete.")
        return artifact

    except Exception as e:
        logger.error(f"❌ Error during artifact fitting: {e}")
        raise e


def transform_data(df, artifact):
    """
    Transforms raw data using the main pipeline from processing.py
    and then calculates the target variables (WQI, WQI_Category).
    """
    logger.info("Transforming data using 'process_raw_data'...")
    
    # 1. Run the main processing pipeline
    proc_df = process_raw_data(df, artifact)
    
    # 2. Create the target variables
    logger.info("Calculating WQI and WQI_Category...")
    proc_df['WQI'] = proc_df.apply(calculate_wqi, axis=1)
    proc_df['WQI_Category'] = proc_df['WQI'].apply(classify_wqi)
    
    # 3. Drop rows where WQI could not be calculated (critical!)
    initial_rows = proc_df.shape[0]
    proc_df = proc_df.dropna(subset=['WQI'])
    final_rows = proc_df.shape[0]
    logger.info(f"Dropped {initial_rows - final_rows} rows with missing WQI.")
    
    logger.info("✅ Data transformation complete.")
    #print(proc_df.columns)
    return proc_df


# Main script execution
def main(input_path, output_dir, artifacts_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    try:
        logger.info(f"🚀 Reading raw data from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} cols")

        # 1. Perform initial conversion BEFORE fitting
        logger.info("Performing initial rename and numeric conversion...")
        df_renamed = rename_columns(df)
        df_numeric = convert_to_numeric(df_renamed)
        
        # 2. Fit the preprocessor on the *numeric* data
        artifact = fit_preprocessor(df_numeric)
        artifact_path = os.path.join(artifacts_dir, "preprocessor_artifact.joblib")
        joblib.dump(artifact, artifact_path)
        logger.info(f"💾 Saved preprocessing artifacts at: {artifact_path}")

        # 3. Transform the data (using the original raw df)
        processed_df = transform_data(df, artifact)

        # --- 4. Final Feature Selection (Your Request) ---
        logger.info("Selecting final features for model training...")
        
        # This list MUST match the FINAL_FEATURES list in train.py
        FINAL_FEATURES = [
            'state_name',  # The categorical feature
            
            # Average features
            'temperature_avg',
            'dissolved_oxygen_avg',
            'ph_avg',
            'log_fecal_coliform_avg',
            'log_nitrate_avg',
            'log_conductivity_avg', # This will now be created
            'log_bod_avg',
            
            # Range features
            'temperature_range',
            'dissolved_oxygen_range',
            'ph_range',
            'log_fecal_coliform_range',
            'log_nitrate_range',
            'log_conductivity_range', # This will now be created
            'log_bod_range'
        ]
        
        TARGETS = ['WQI', 'WQI_Category']
        
        # Select *only* the columns we need for the final dataset
        final_cols_to_keep = FINAL_FEATURES + TARGETS
        
        # Check for missing columns (in case process_raw_data failed)
        missing_cols = [col for col in final_cols_to_keep if col not in processed_df.columns]
        if missing_cols:
            logger.error(f"❌ CRITICAL: The following columns are missing after processing: {missing_cols}")
            raise KeyError(f"Columns not found in processed data: {missing_cols}")
            
        processed_df_clean = processed_df[final_cols_to_keep]
        logger.info(f"✅ Cleaned data to final shape: {processed_df_clean.shape}")
        
        # 5. Split into Train and Test sets
        logger.info("Splitting data into train and test sets...")
        train_df = processed_df_clean.sample(frac=0.8, random_state=42)
        test_df = processed_df_clean.drop(train_df.index)
        
        train_path = os.path.join(output_dir, "train_processed.csv")
        test_path = os.path.join(output_dir, "test_processed.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"✅ Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        logger.info(f"✅ Cleaned/Processed data saved to {output_dir}")
        logger.info("🏁 Preprocessing script finished successfully.")

    except Exception as e:
        logger.error(f"❌ An error occurred in main: {e}", exc_info=True)
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw water quality data.")
    
    parser.add_argument(
        "--input_path", 
        type=str, 
        default="data/raw/raw_data.csv", 
        help="Path to the raw data CSV file."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed", 
        help="Directory to save processed train/test CSVs."
    )
    parser.add_argument(
        "--artifacts_dir", 
        type=str, 
        default="artifacts", 
        help="Directory to save the preprocessor artifact."
    )

    args = parser.parse_args()
    main(
        input_path=args.input_path,
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir
    )