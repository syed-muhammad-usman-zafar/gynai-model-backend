# GynAI Model Backend Configuration


API_TITLE = "GynAI Delivery Mode Prediction API"
API_DESCRIPTION = "Clinical decision support system for predicting childbirth delivery modes"
API_VERSION = "1.0.0"
API_HOST = "0.0.0.0"
API_PORT = 8000


MODEL_DIR = "models/trained_models"
DATA_DIR = "data"
LOGS_DIR = "logs"

FEATURE_NAMES = [
    "Mother_Age", 
    "Gravida", 
    "Parity", 
    "Gestation_Weeks", 
    "Previous_CS"
]


RISK_THRESHOLDS = {
    "teenage_pregnancy": 18,
    "advanced_maternal_age": 35,
    "preterm_delivery": 37,
    "post_term_pregnancy": 42,
    "grand_multiparous": 5,
    "multiple_pregnancies": 3
}

# Model Performance Thresholds
MIN_CONFIDENCE_SCORE = 0.6
HIGH_CONFIDENCE_THRESHOLD = 0.8


DELIVERY_TYPES = [
    "Cesarean",
    "Vaginal", 
    "Assisted_Vaginal"
]
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# CORS Configuration (for production, restrict origins)
ALLOWED_ORIGINS = ["*"]  # Configure for production
ALLOWED_METHODS = ["GET", "POST"]
ALLOWED_HEADERS = ["*"]
