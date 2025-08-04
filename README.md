# GynAI Model Backend

A FastAPI backend for the GynAI clinical support system that predicts childbirth delivery modes for gynecologists.

## Features

- **Delivery Mode Prediction**: ML model to predict Cesarean, Vaginal, or Assisted Vaginal delivery
- **RESTful API**: FastAPI-based endpoints for model inference
- **Clinical Decision Support**: Provides probability scores and recommendations
- **Data Validation**: Robust input validation for medical data

## Project Structure

```
gynai-model-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   ├── prediction.py    # ML model logic
│   │   └── schemas.py       # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   └── ml_service.py    # Model training and prediction services
│   └── utils/
│       ├── __init__.py
│       └── data_preprocessing.py
├── data/
│   └── maternal_data_clean.csv
├── models/
│   └── trained_models/      # Saved model files
├── notebooks/
│   └── model_development.ipynb
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python -m app.services.ml_service
```

3. Run the FastAPI server:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

- `POST /predict` - Predict delivery mode for a patient
- `GET /model/info` - Get model information and performance metrics
- `GET /health` - Health check endpoint

## Model Features

The model uses the following maternal health indicators:
- Mother's age
- Gravida (total pregnancies)
- Parity (previous live births)
- Gestational weeks
- Previous cesarean sections

## Usage Example

```python
import requests

patient_data = {
    "mother_age": 28.0,
    "gravida": 2.0,
    "parity": 1.0,
    "gestation_weeks": 38.5,
    "previous_cs": 1.0
}

response = requests.post("http://localhost:8000/predict", json=patient_data)
prediction = response.json()
```


