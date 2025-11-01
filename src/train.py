import os
import pandas as pd 
import numpy as np
import argparse
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
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
        
        # Preprocessing Pipeline for model
        categorical_features = ['state_name']
        
        numerical_features = [col for col in X_train.columns if col not in categorical_features]
        
        # Preprocessor for State Name
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder= 'passthrough'
        )
        logger.info("Created Column Transformer for state_name")
        
        # Random Forest Model
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
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
        model_pipeline.fit(X_train, y_train)
        logger.info("✅ Model training complete.")

        # Save the Model Pipeline 
        os.makedirs(artifacts_dir, exist_ok=True)
        model_path = os.path.join(artifacts_dir, 'model_pipeline.joblib')
        joblib.dump(model_pipeline, model_path)
        logger.info(f"✅ Model pipeline saved to {model_path}")

        # Evaluate the Model 
        logger.info("Starting model evaluation on test set...")
        y_pred = model_pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logger.info(f"Test Set Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
        print(f"\n--- Model Evaluation ---")
        print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"------------------------")
        
        logger.info("--- Training script finished successfully ---")

    except FileNotFoundError:
        logger.error(f"Error: Data files not found at {train_data_path} or {test_data_path}")
        print(f"Error: Data files not found. Make sure these files exist:\n{train_data_path}\n{test_data_path}")
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