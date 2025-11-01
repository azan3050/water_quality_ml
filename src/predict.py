import pandas as pd
import numpy as np
import os
import logging
import argparse
import joblib

# --- Logger Setup ---
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename="logs/predict.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

def load_artifacts(artifacts_dir):
    """
    Loads the preprocessor and model pipeline artifacts.
    """
    try:
        preprocessor_path = os.path.join(artifacts_dir, 'preprocessor_artifact.joblib')
        model_path = os.path.join(artifacts_dir, 'model_pipeline.joblib')
        
        preprocessor_artifact = joblib.load(preprocessor_path)
        model_pipeline = joblib.load(model_path)
        
        logger.info(f"Loaded preprocessor from {preprocessor_path}")
        logger.info(f"Loaded model pipeline from {model_path}")
        return preprocessor_artifact, model_pipeline
        
    except FileNotFoundError as e:
        logger.error(f"Artifact not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        raise

def predict(raw_data_df, preprocessor_artifact, model_pipeline):
    """
    Makes a WQI prediction on new, raw data.
    """
    try:
        logger.info(f"Received {raw_data_df.shape[0]} row(s) for prediction.")
        
        # --- Step 1: Apply initial preprocessing ---
        # We need the transform_data function from preprocess.py
        # For simplicity, let's re-define the necessary parts here.
        # In a real app, you would import this from preprocess.py
        
        # (This is a simplified version of your transform_data function)
        
        # 1a. Rename columns
        proc_df = raw_data_df.copy()
        proc_df.columns = (
            proc_df.columns.str.replace(r'\s+', '_', regex=True)
                           .str.replace(r'[^\w]', '', regex=True)
                           .str.lower()
        )
        
        # 1b. Apply log transforms
        skewed_cols = [
            'conductivity_mhocmmin', 'conductivity_mhocmmax', 'bod_mglmin', 'bod_mglmax',
            'nitrate_n_mglmin', 'nitrate_n_mglmax', 'fecal_coliform_mpn100mlmin', 
            'fecal_coliform_mpn100mlmax', 'total_coliform_mpn100mlmin', 'total_coliform_mpn100mlmax'
        ]
        for col in skewed_cols:
            if col in proc_df.columns:
                proc_df[f'log_{col}'] = np.log1p(pd.to_numeric(proc_df[col], errors='coerce'))
        
        # 1c. Impute NaNs using saved medians
        for col_name, median_val in preprocessor_artifact['medians'].items():
            if col_name in proc_df.columns:
                proc_df[col_name] = pd.to_numeric(proc_df[col_name], errors='coerce').fillna(median_val)
            # Handle case where original (non-log) col needs filling
            elif col_name.replace('log_', '') in proc_df.columns:
                original_col = col_name.replace('log_', '')
                proc_df[original_col] = pd.to_numeric(proc_df[original_col], errors='coerce').fillna(median_val) # Note: This assumes median is for non-log
        
        logger.info("  Raw data cleaned and imputed.")

        # 1d. Feature Engineering
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
        for prefix, (min_col, max_col) in feature_pairs.items():
            min_col_to_use = f'log_{min_col}' if f'log_{min_col}' in proc_df.columns else min_col
            max_col_to_use = f'log_{max_col}' if f'log_{max_col}' in proc_df.columns else max_col
            if min_col_to_use in proc_df.columns and max_col_to_use in proc_df.columns:
                proc_df[f'{prefix}_avg'] = (proc_df[min_col_to_use] + proc_df[max_col_to_use]) / 2.0
                proc_df[f'{prefix}_range'] = (proc_df[max_col_to_use] - proc_df[min_col_to_use]).abs()

        # 1e. Clipping
        for col_name, (lower, upper) in preprocessor_artifact['quantiles'].items():
            if col_name in proc_df.columns:
                proc_df[col_name] = proc_df[col_name].clip(lower, upper)
        
        logger.info("  Feature engineering and clipping applied.")

        # --- Step 2: Make Prediction with Model Pipeline ---
        # The model_pipeline is expecting the "state_name" column
        # and all the engineered features (e.g., 'temperature_avg')
        
        # Ensures all columns needed by the model is present
        model_features = model_pipeline.named_steps['preprocessor'].feature_names_in_
        
        # Create a dataframe with all required features, filling missing ones with NaN
        # (The OneHotEncoder will handle 'state_name', and numerics are already imputed)
        final_input_df = pd.DataFrame(columns=model_features)
        for col in model_features:
            if col in proc_df.columns:
                final_input_df[col] = proc_df[col]
            else:
                final_input_df[col] = np.nan # or 0
        
        prediction = model_pipeline.predict(final_input_df)
        
        logger.info(f"Prediction successful: {prediction}")
        return prediction

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

# --- Script Execution ---
if __name__ == "__main__":
    
    # Example raw data
    
    new_sample = {
        'Station Code': '9999',
        'Monitoring Location': 'TEST RIVER',
        'State Name': 'PUNJAB', # A state the model has seen
        'Temperature (°C)(Min)': 20.0,
        'Temperature (°C)(Max)': 25.0,
        'Dissolved Oxygen(mg/L)(min)': 7.1,
        'Dissolved Oxygen(mg/L)(max)': 8.2,
        'pH(min)': 7.0,
        'pH(max)': 7.4,
        'Conductivity (µmho/cm)(min)': 120,
        'Conductivity (µmho/cm)(max)': 150,
        'BOD\n(mg/L)(min)': 1.5,
        'BOD\n(mg/L)(max)': 2.5,
        'Nitrate N (mg/L)(Min)': 0.8,
        'Nitrate N (mg/L)(Max)': 1.2,
        'Fecal Coliform (MPN/100ml)(Min)': 80,
        'Fecal Coliform (MPN/100ml)(Max)': 120,
        'Total Coliform (MPN/100ml)(Min)': 200,
        'Total Coliform (MPN/100ml)(Max)': 300,
        # Note: We don't include Fecal Streptococci (we dropped it)
    }
    
    # Convert dictionary to DataFrame
    raw_df = pd.DataFrame([new_sample])

    # --- Load Artifacts and Predict ---
    try:
        preprocessor, model = load_artifacts(artifacts_dir='artifacts')
        
        prediction = predict(raw_df, preprocessor, model)
        
        print("\n--- Prediction Result ---")
        print(f"Raw Input Data (State): {new_sample['State Name']}")
        print(f"Predicted WQI Category: {prediction[0]}")
        print("-------------------------")
        logger.info("Prediction script finished successfully.")

    except Exception as e:
        print(f"\nAn error occurred. Check 'logs/predict.log' for details.")
        logger.error(f"Prediction script failed: {e}")
