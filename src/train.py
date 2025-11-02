import os
import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Logger setup
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/train.log",
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


# Main training function
def main(train_data_path, test_data_path, artifacts_dir):
    logger.info("🚀 Starting Training Script...")

    try:
        # Load preprocessed train & test data
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        logger.info(f"Loaded Train {train_df.shape}, Test {test_df.shape}")

        # Define features & target
        TARGET = "WQI_Category"

        FINAL_FEATURES = [
            'state_name',  # The categorical feature
            
            # Average features
            'temperature_avg',
            'dissolved_oxygen_avg',
            'ph_avg',
            'log_fecal_coliform_avg',
            'log_nitrate_avg',
            'log_conductivity_avg',
            'log_bod_avg',
            
            # Range features
            'temperature_range',
            'dissolved_oxygen_range',
            'ph_range',
            'log_fecal_coliform_range',
            'log_nitrate_range',
            'log_conductivity_range',
            'log_bod_range'
        ]

        # Select *only* these features
        X_train = train_df[FINAL_FEATURES]
        y_train = train_df[TARGET]
        
        X_test = test_df[FINAL_FEATURES]
        y_test = test_df[TARGET]

        logger.info("✅ Split into features (X) and target (y)")

        # Encode target labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)

        os.makedirs(artifacts_dir, exist_ok=True)
        label_encoder_path = os.path.join(artifacts_dir, "label_encoder.joblib")
        joblib.dump(le, label_encoder_path)
        logger.info(f"💾 LabelEncoder saved at: {label_encoder_path}")

        # Preprocessing for categorical data
        categorical_features = ["state_name"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            ],
            remainder="passthrough", 
        )

        logger.info("✅ ColumnTransformer created for state_name")

        # Define XGBoost model
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )

        # Create model pipeline
        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        logger.info("✅ Model pipeline (preprocessor + classifier) created")

        # Train the model
        logger.info("🎯 Training the XGBoost model...")
        model_pipeline.fit(X_train, y_train_encoded)
        logger.info("✅ Model training completed successfully.")

        # Save model artifacts
        os.makedirs(artifacts_dir, exist_ok=True)
        model_path = os.path.join(artifacts_dir, "model_pipeline_xgb.joblib")
        joblib.dump(model_pipeline, model_path)
        logger.info(f"💾 Model pipeline saved at: {model_path}")

        # Model evaluation
        logger.info("📊 Evaluating model on test data...")

        y_pred_encoded = model_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        y_pred_labels = le.inverse_transform(y_pred_encoded)

        report = classification_report(y_test, y_pred_labels, zero_division=0)

        logger.info(f"✅ Test Accuracy: {accuracy:.4f}")
        logger.info(f"\nClassification Report:\n{report}")

        print("\n--- XGBoost Model Evaluation ---")
        print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(report)
        print("--------------------------------")

        logger.info("🏁 Training script finished successfully.")

    except FileNotFoundError:
        logger.error("❌ Data files not found.")
        print("Error: Data files not found.")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
        raise e


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a WQI Classification Model")

    parser.add_argument(
        "--train_data_path",
        type=str,
        default="data/processed/train_processed.csv",
        help="Path to the training data CSV file.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="data/processed/test_processed.csv",
        help="Path to the testing data CSV file.",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="artifacts",
        help="Directory to save the model pipeline.",
    )

    args = parser.parse_args()
    main(
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        artifacts_dir=args.artifacts_dir,
    )