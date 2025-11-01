import pandas as pd
import numpy as np
import os
import logging
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# --- NEW Imports ---
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
# --------------------

from fastapi.middleware.cors import CORSMiddleware
from processing import process_raw_data

# --- 1. Setup Logger ---
# (Same as before)
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- 2. Pydantic Model for Input Data ---
# (Same as before)
class WaterQualityInput(BaseModel):
    station_code: Optional[str] = Field(None, alias="Station Code")
    monitoring_location: Optional[str] = Field(None, alias="Monitoring Location")
    state_name: str = Field(..., alias="State Name")
    temperature_cmin: Optional[float] = Field(None, alias="Temperature (°C)(Min)")
    temperature_cmax: Optional[float] = Field(None, alias="Temperature (°C)(Max)")
    dissolved_oxygenmglmin: Optional[float] = Field(None, alias="Dissolved Oxygen(mg/L)(min)")
    dissolved_oxygenmglmax: Optional[float] = Field(None, alias="Dissolved Oxygen(mg/L)(max)")
    phmin: Optional[float] = Field(None, alias="pH(min)")
    phmax: Optional[float] = Field(None, alias="pH(max)")
    conductivity_mhocmmin: Optional[float] = Field(None, alias="Conductivity (µmho/cm)(min)")
    conductivity_mhocmmax: Optional[float] = Field(None, alias="Conductivity (µmho/cm)(max)")
    bod_mglmin: Optional[float] = Field(None, alias="BOD\n(mg/L)(min)")
    bod_mglmax: Optional[float] = Field(None, alias="BOD\n(mg/L)(max)")
    nitrate_n_mglmin: Optional[float] = Field(None, alias="Nitrate N (mg/L)(Min)")
    nitrate_n_mglmax: Optional[float] = Field(None, alias="Nitrate N (mg/L)(Max)")
    fecal_coliform_mpn100mlmin: Optional[float] = Field(None, alias="Fecal Coliform (MPN/100ml)(Min)")
    fecal_coliform_mpn100mlmax: Optional[float] = Field(None, alias="Fecal Coliform (MPN/100ml)(Max)")
    total_coliform_mpn100mlmin: Optional[float] = Field(None, alias="Total Coliform (MPN/100ml)(Min)")
    total_coliform_mpn100mlmax: Optional[float] = Field(None, alias="Total Coliform (MPN/100ml)(Max)")

    class Config:
        populate_by_name = True

# --- 3. Load Artifacts on Startup ---
# (Same as before)
try:
    preprocessor_artifact_path = os.path.join('artifacts', 'preprocessor_artifact.joblib')
    model_pipeline_path = os.path.join('artifacts', 'model_pipeline_xgb.joblib')
    label_encoder_path = os.path.join('artifacts', 'label_encoder.joblib')
    
    PREPROCESSOR_ARTIFACT = joblib.load(preprocessor_artifact_path)
    MODEL_PIPELINE = joblib.load(model_pipeline_path)
    LABEL_ENCODER = joblib.load(label_encoder_path)
    
    logger.info("✅ Artifacts loaded successfully on startup.")

except FileNotFoundError:
    logger.error("Artifact files not found. Check 'artifacts' directory.")
    PREPROCESSOR_ARTIFACT = None
    MODEL_PIPELINE = None
    LABEL_ENCODER = None

except Exception as e:
    logger.error(f"Error loading artifacts: {e}")
    PREPROCESSOR_ARTIFACT = None
    MODEL_PIPELINE = None

# --- 4. Initialize FastAPI App ---
app = FastAPI(
    title="Water Quality Prediction API",
    description="An API to predict WQI Category based on water measurements.",
    version="1.0.0"
)

# --- 5. Add CORS Middleware ---
# (Same as before)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# --- 6. API Endpoints ---
@app.on_event("startup")
async def check_artifacts():
    if PREPROCESSOR_ARTIFACT is None or MODEL_PIPELINE is None:
        logger.error("API is starting in a failed state: Artifacts not loaded.")
    else:
        logger.info("API started successfully with artifacts loaded.")

@app.post("/predict", tags=["Prediction"])
async def predict_wqi(data: WaterQualityInput):
    """
    Takes raw water quality measurements and returns a predicted WQI Category.
    """
    if PREPROCESSOR_ARTIFACT is None or MODEL_PIPELINE is None or LABEL_ENCODER is None: # <-- Check all
        logger.error("Prediction failed: Artifacts are not loaded.")
        raise HTTPException(status_code=500, detail="Internal Server Error: Model artifacts not loaded.")
    try:
        # 1. Convert to DataFrame
        input_data = data.model_dump(by_alias=True)
        raw_df = pd.DataFrame([input_data])
        logger.info(f"Received prediction request for state: {data.state_name}")
        
        # 2. Process data
        processed_df = process_raw_data(raw_df, PREPROCESSOR_ARTIFACT)
        
        # 3. Make prediction (will return an integer, e.g., [2])
        prediction_encoded = MODEL_PIPELINE.predict(processed_df)
        
        # 4. NEW: Decode the integer prediction to a string label
        predicted_category = LABEL_ENCODER.inverse_transform(prediction_encoded)[0]
        
        logger.info(f"Prediction successful: {predicted_category}")
        
        # 5. Return the result
        return {
            "prediction": predicted_category,
            "input_data": input_data
        }

    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# --- 7. NEW: Mount Static Files & Serve UI ---

# This line tells FastAPI to serve all files in the 'templates' folder
# under the path '/'. So, 'templates/style.css' will be available at '/style.css'.
app.mount("/", StaticFiles(directory="src/templates", html=True), name="static")

# This endpoint will serve your index.html as the home page
@app.get("/", tags=["UI"])
async def read_index():
    """Serves the main HTML page for the UI."""
    return FileResponse(os.path.join('src/templates', 'index.html'))

# --- 8. Run the App (for development) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server for development...")
    uvicorn.run(app, host="127.0.0.1", port=8000)