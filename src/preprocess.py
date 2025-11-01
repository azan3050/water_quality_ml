import os
import sys
import pandas as pd
import joblib
import logging
from datetime import datetime

# Ensure 'src' package is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing import (
    rename_columns,
    apply_log_transformation,
    apply_imputation,
    apply_feature_engineering,
    apply_clipping,
    drop_correlated_features,
)
from src.processing import calculate_wqi, classify_wqi



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
def fit_preprocessor(df):
    """
    Compute preprocessing artifacts such as median values and quantile limits.
    These will be used for consistent preprocessing in future data.
    """
    logger.info("Fitting preprocessing artifacts...")

    medians = df.median(numeric_only=True)
    quantiles = {
        col: (df[col].quantile(0.01), df[col].quantile(0.99))
        for col in df.select_dtypes(include="number").columns
    }

    artifact = {
        "medians": medians,
        "quantiles": quantiles,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    logger.info("✅ Preprocessing artifacts computed successfully.")
    return artifact


def transform_data(df, artifact):
    """
    Transform raw dataframe using the preprocessor artifacts and helper functions.
    """
    logger.info(f"Starting transformation on data with {df.shape[0]} rows...")
    proc_df = df.copy()

    try:
        proc_df = rename_columns(proc_df)
        proc_df = apply_log_transformation(proc_df)
        proc_df = apply_imputation(proc_df, artifact["medians"])
        proc_df = apply_feature_engineering(proc_df)
        proc_df = apply_clipping(proc_df, artifact["quantiles"])
        proc_df = drop_correlated_features(proc_df)

        # Compute target variable (WQI) and category
        proc_df["WQI"] = proc_df.apply(calculate_wqi, axis=1)
        proc_df["WQI_Category"] = proc_df["WQI"].apply(classify_wqi)
        proc_df = proc_df.dropna(subset=["WQI"])

        logger.info(f"✅ Transformation complete. Final shape: {proc_df.shape}")
        return proc_df

    except Exception as e:
        logger.error(f"❌ Error during transformation: {str(e)}", exc_info=True)
        raise


# Main preprocessing pipeline
def main(input_path="data/raw/raw_data.csv", output_dir="data/processed", artifacts_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    try:
        logger.info(f"🚀 Reading raw data from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Raw data loaded successfully: {df.shape[0]} rows, {df.shape[1]} cols")

        # Compute and save preprocessing artifacts
        artifact = fit_preprocessor(df)
        artifact_path = os.path.join(artifacts_dir, "preprocessor_artifact.joblib")
        joblib.dump(artifact, artifact_path)
        logger.info(f"💾 Saved preprocessing artifacts at: {artifact_path}")

        # Transform data
        processed_df = transform_data(df, artifact)
        processed_path = os.path.join(output_dir, "processed_data.csv")
        processed_df.to_csv(processed_path, index=False)
        logger.info(f"✅ Processed data saved at: {processed_path}")

        # Train-test split (optional)
        train_df = processed_df.sample(frac=0.8, random_state=42)
        test_df = processed_df.drop(train_df.index)
        train_df.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)

        logger.info(f"📊 Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        logger.info("🎯 Preprocessing pipeline completed successfully.")

    except Exception as e:
        logger.error(f"❌ Preprocessing pipeline failed: {str(e)}", exc_info=True)
        raise


# -------------------------------
# Script entry point
# -------------------------------
if __name__ == "__main__":
    main()