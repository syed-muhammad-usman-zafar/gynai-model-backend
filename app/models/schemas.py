#loads the schemas for the Gynai model backend
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict
from enum import Enum

class DeliveryType(str, Enum):
    CESAREAN = "Cesarean"
    VAGINAL = "Vaginal"
    ASSISTED_VAGINAL = "Assisted_Vaginal"
    UNKNOWN = "Unknown"

class PatientInput(BaseModel):
    Mother_Age: float = Field(..., ge=12, le=60, description="Mother's age in years", example=28.0)
    Gravida: float = Field(..., ge=1, le=20, description="Total number of pregnancies", example=2.0)
    Parity: float = Field(..., ge=0, le=15, description="Number of previous live births", example=1.0)
    Gestation_Weeks: float = Field(..., ge=20, le=45, description="Gestational age in weeks", example=38.5)
    Previous_CS: float = Field(..., ge=0, le=10, description="Number of previous cesarean sections", example=1.0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Mother_Age": 28.0,
                "Gravida": 2.0,
                "Parity": 1.0,
                "Gestation_Weeks": 38.5,
                "Previous_CS": 1.0
            }
        }
    )

class PredictionResponse(BaseModel):
    predicted_delivery_type: DeliveryType
    probabilities: Dict[str, float]
    confidence_score: float
    confidence_level: Dict[str, str]
    risk_factors: List[str]
    recommendations: List[str]

class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    version: str
    accuracy: float
    training_date: str
    features: List[str]
    target_classes: List[str]

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_loaded: bool
    timestamp: str
