import pandas as pd
import numpy as np
import os
import logging
import joblib
import argparse
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing import process_raw_data

# Logger setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/predict.log",
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


# Load artifacts
def load_artifacts(artifacts_dir):
    try:
        preprocessor_artifact_path = os.path.join(artifacts_dir, "preprocessor_artifact.joblib")
        model_path = os.path.join(artifacts_dir, "model_pipeline_xgb.joblib")
        label_encoder_path = os.path.join(artifacts_dir, "label_encoder.joblib")

        preprocessor_artifact = joblib.load(preprocessor_artifact_path)
        model_pipeline = joblib.load(model_path)
        label_encoder = joblib.load(label_encoder_path)

        logger.info(f"✅ Loaded preprocessor artifact: {preprocessor_artifact_path}")
        logger.info(f"✅ Loaded model pipeline: {model_path}")
        logger.info(f"✅ Loaded label encoder: {label_encoder_path}")

        return preprocessor_artifact, model_pipeline, label_encoder

    except FileNotFoundError as e:
        logger.error(f"❌ Artifact not found: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading artifacts: {e}")
        raise


# Predict function
def predict(processed_data_df, model_pipeline, label_encoder):
    """
    Makes a WQI category prediction on new, preprocessed data.
    """
    try:
        logger.info(f"Received {processed_data_df.shape[0]} row(s) for prediction.")

        y_pred_encoded = model_pipeline.predict(processed_data_df)
        y_pred_label = label_encoder.inverse_transform(y_pred_encoded)

        if hasattr(model_pipeline, "predict_proba"):
            proba = model_pipeline.predict_proba(processed_data_df)
            confidence = np.max(proba, axis=1)
        else:
            confidence = np.ones_like(y_pred_encoded, dtype=float)

        results = pd.DataFrame({
            "Predicted_WQI_Category": y_pred_label,
            "Confidence": np.round(confidence * 100, 2)
        })

        logger.info(f"✅ Prediction successful:\n{results}")
        return results

    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}", exc_info=True)
        raise


# Script entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Water Quality Index Category")

    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="artifacts",
        help="Directory containing saved model and encoder artifacts."
    )

    args = parser.parse_args()

    # Example RAW input, mirroring the structure of the API request
    new_sample = {
        "State Name": "MAHARASHTRA",
        "Temperature (°C)(Min)": 26.0,
        "Temperature (°C)(Max)": 30.0,
        "Dissolved Oxygen(mg/L)(min)": 4.5,
        "Dissolved Oxygen(mg/L)(max)": 5.5,
        "pH(min)": 7.8,
        "pH(max)": 8.2,
        "BOD\n(mg/L)(min)": 6.0,
        "BOD\n(mg/L)(max)": 10.0,
        "Nitrate N (mg/L)(Min)": 2.0,
        "Nitrate N (mg/L)(Max)": 3.5,
        "Fecal Coliform (MPN/100ml)(Min)": 800,
        "Fecal Coliform (MPN/100ml)(Max)": 1500,
        "Conductivity (µmho/cm)(Min)": 800,
        "Conductivity (µmho/cm)(Max)": 1200
}

    raw_df = pd.DataFrame([new_sample])

    try:
        preprocessor_artifact, model_pipeline, label_encoder = load_artifacts(args.artifacts_dir)
        processed_df = process_raw_data(raw_df, preprocessor_artifact)
        results = predict(processed_df, model_pipeline, label_encoder)

        print("\n--- 🌊 Water Quality Prediction ---")
        print(results.to_string(index=False))
        print("-----------------------------------")
        logger.info("Prediction script finished successfully.")

    except Exception as e:
        print(f"An error occurred. Check logs/predict.log for details.")
        logger.error(f"Prediction failed: {e}")