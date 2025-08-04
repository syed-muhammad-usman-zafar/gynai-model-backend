#connects the FastAPI application to the Gynai model backend
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import PatientInput, PredictionResponse, ModelInfo, HealthResponse
from app.services.ml_service import DeliveryModePredictor
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GynAI Delivery Mode Prediction API",
    description="Clinical decision support system for predicting childbirth delivery modes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ML model
predictor = DeliveryModePredictor()
model_loaded = False

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model_loaded
    model_dir = "models/trained_models"
    
    if os.path.exists(model_dir):
        success = predictor.load_model(model_dir)
        if success:
            model_loaded = True
            logger.info("Model loaded successfully")
        else:
            logger.warning("Failed to load model - will need to train first")
    else:
        logger.warning("Model directory not found - will need to train first")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "GynAI Delivery Mode Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_delivery_mode(patient_data: PatientInput):
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please contact administrator."
        )
    
    try:
        # Convert Pydantic model to dict - using uppercase field names
        patient_dict = {
            "Mother_Age": patient_data.Mother_Age,
            "Gravida": patient_data.Gravida,
            "Parity": patient_data.Parity,
            "Gestation_Weeks": patient_data.Gestation_Weeks,
            "Previous_CS": patient_data.Previous_CS
        }
        
        # Make prediction
        result = predictor.predict(patient_dict)
        
        # Log prediction for monitoring
        logger.info(f"Prediction made: {result['predicted_delivery_type']} with confidence {result['confidence_score']:.3f}")
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please contact administrator."
        )
    
    try:
        model_info = predictor.model_info
        return ModelInfo(**model_info)
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model info: {str(e)}"
        )

@app.post("/model/train", tags=["Admin"])
async def train_model():
    """
    Train a new model (Admin endpoint)
    This endpoint should be protected in production
    """
    try:
        global model_loaded
        
        data_path = "data/maternal_data_clean.csv"
        model_dir = "models/trained_models"
        
        if not os.path.exists(data_path):
            raise HTTPException(
                status_code=404,
                detail="Training data not found"
            )
        
        logger.info("Starting model training...")
        model_info = predictor.train_model(data_path)
        predictor.save_model(model_dir)
        
        model_loaded = True
        logger.info("Model training completed successfully")
        
        return {
            "message": "Model trained successfully",
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Model training failed: {str(e)}"
        )

@app.get("/model/feature-importance", tags=["Model"])
async def get_feature_importance():
    """Get feature importance from the loaded model"""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please contact administrator."
        )
    
    try:
        feature_importance = predictor.model_info.get('feature_importance', {})
        if not feature_importance:
            raise HTTPException(
                status_code=404,
                detail="Feature importance not available for this model"
            )
        
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "feature_importance": dict(sorted_features),
            "model_name": predictor.model_info.get('model_name', 'Unknown')
        }
        
    except Exception as e:
        logger.error(f"Error retrieving feature importance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve feature importance: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
