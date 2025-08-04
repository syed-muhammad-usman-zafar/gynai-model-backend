#!/usr/bin/env python3
"""
GynAI Model Training Script
This script trains the delivery mode prediction model and saves it for FastAPI integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ml_service import DeliveryModePredictor

def main():
    """Main training function"""
    print("GynAI Model Training")
    print("=" * 40)
    
    # Initialize predictor
    predictor = DeliveryModePredictor()
    
    # Set paths
    data_path = "data/maternal_data_clean.csv"
    model_dir = "models/trained_models"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return False
    
    try:
        # Train model
        print("Starting model training...")
        model_info = predictor.train_model(data_path)
        
        # Save model
        print("Saving model...")
        predictor.save_model(model_dir)

        print("\nModel training completed successfully!")
        print(f"Model: {model_info['model_name']}")
        print(f"Accuracy: {model_info['accuracy']:.4f}")
        print(f"Training Date: {model_info['training_date']}")
        print(f"Model saved to: {model_dir}")

        # Test prediction
        print("\nTesting prediction...")
        test_patient = {
            'Mother_Age': 28.0,
            'Gravida': 2.0,
            'Parity': 1.0,
            'Gestation_Weeks': 38.5,
            'Previous_CS': 1.0
        }
        
        result = predictor.predict(test_patient)
        print(f"Test prediction: {result['predicted_delivery_type']} (confidence: {result['confidence_score']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
