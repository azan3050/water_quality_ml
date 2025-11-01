import os
import pandas as pd 
import numpy as np
import argparse
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# Logger Setup
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    filename='logs/train.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


# main function
def main(train_data_path, test_data_path, artifacts_dir):
    logger.info(f"--- Starting Training Script ---")
    
    try:
        # Load Data
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        logger.info(f"Loaded Train Data {train_df.shape} and Test Data {test_df.shape}")
        
        # Define Target and Features
        TARGET = 'WQI_Category'
        
        # Dropping WQI value as it hints the answer to eliminate data leakage
        features_to_drop = [TARGET, 'WQI']
        
        X_train = train_df.drop(columns=[TARGET] + features_to_drop)
        y_train = train_df[TARGET]
        
        X_test = test_df.drop(columns=features_to_drop)
        y_test = test_df[TARGET]
        
        logger.info("Separated features X and Target y")
        
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.fit_transform(y_test)
        
        logger.info(f"LabelEncoder classes: {list(le.classes_)}")
        
        os.makedirs(artifacts_dir, exist_ok=True)
        label_encoder_path = os.path.join(artifacts_dir, 'label_encoder.joblib')
        joblib.dump(le, label_encoder_path)
        logger.info(f"✅ LabelEncoder saved to {label_encoder_path}")
        
        # Preprocessing Pipeline for model
        categorical_features = ['state_name']
        
        # Preprocessor for State Name
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder= 'passthrough'
        )
        logger.info("Created Column Transformer for state_name")
        
        # XGBoost Classification Model
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        # Model Pipelne
        model_pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ]
        )    
        
        logger.info("Created full model pipeline (preprocessor + classifier)")
        
        # Training model
        logger.info("Starting model training...")
        model_pipeline.fit(X_train, y_train_encoded)
        logger.info("✅ Model training complete.")

        # Save the Model Pipeline 
        os.makedirs(artifacts_dir, exist_ok=True)
        model_path = os.path.join(artifacts_dir, 'model_pipeline_xgb.joblib')
        joblib.dump(model_pipeline, model_path)
        logger.info(f"✅ Model pipeline saved to {model_path}")

        # Model Evaluation
        logger.info("Starting model evaluation on test set...")
        # predict() will return encoded integers (0, 1, 2...)
        y_pred_encoded = model_pipeline.predict(X_test)
        
        # Get accuracy using the encoded labels
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        
        # For the classification report, we want to see the *names*
        # So we decode the predictions
        y_pred_labels = le.inverse_transform(y_pred_encoded)
        
        # Compare the original y_test (strings) with the decoded y_pred (strings)
        report = classification_report(y_test, y_pred_labels, zero_division=0)
        
        logger.info(f"Test Set Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
        print(f"\n--- XGBoost Model Evaluation ---")
        print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(report)
        print(f"--------------------------------")
        
        logger.info("--- XGBoost training script finished successfully ---")

    except FileNotFoundError:
        logger.error(f"Error: Data files not found.")
        print(f"Error: Data files not found.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")
        raise e

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train a WQI Classification Model")
    
    parser.add_argument("--train_data_path", type=str, default="data/processed/train_processed.csv", help="Path to the training data CSV file.")
    
    parser.add_argument("--test_data_path", type=str, default="data/processed/test_processed.csv", help="Path to the testing data CSV file.")
    
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to save the model pipeline.")
    
    args = parser.parse_args()
    
    main(train_data_path=args.train_data_path, test_data_path=args.test_data_path, artifacts_dir=args.artifacts_dir)