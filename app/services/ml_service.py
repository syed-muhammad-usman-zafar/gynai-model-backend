#actual model called by FASTAPI when user makes a request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
import os
from datetime import datetime
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

class DeliveryModePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = ['Mother_Age', 'Gravida', 'Parity', 'Gestation_Weeks', 'Previous_CS']
        self.model_info = {}
        
    def load_and_preprocess_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(data_path)
        df_clean = df[df['Delivery_Type'] != 'Unknown'].copy()
        imputer = SimpleImputer(strategy='median')
        X = df_clean[self.feature_names].copy()
        y = df_clean['Delivery_Type'].copy()
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X), 
            columns=self.feature_names,
            index=X.index
        )
        self.imputer = imputer
        
        return X_imputed, y
    
    def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X_engineered = X.copy()
        X_engineered['Age_Category'] = pd.cut(
            X['Mother_Age'], 
            bins=[0, 20, 35, 50], 
            labels=['Young', 'Normal', 'Advanced']
        ).cat.codes
        
        # high risk pregnancy indicators 
        X_engineered['High_Risk_Age'] = (X['Mother_Age'] < 18) | (X['Mother_Age'] > 35)
        X_engineered['Previous_CS_Risk'] = X['Previous_CS'] > 0
        X_engineered['Preterm'] = X['Gestation_Weeks'] < 37
        X_engineered['Multiple_Pregnancies'] = X['Gravida'] > 3
        
        # Parity related features
        X_engineered['Nulliparous'] = X['Parity'] == 0
        X_engineered['Grand_Multiparous'] = X['Parity'] >= 5
        

        bool_cols = X_engineered.select_dtypes(include=['bool']).columns
        X_engineered[bool_cols] = X_engineered[bool_cols].astype(int)
        
        return X_engineered
    
    def train_model(self, data_path: str) -> Dict[str, Any]:
        print("Loading and preprocessing data")
        X, y = self.load_and_preprocess_data(data_path)
        
        print("Engineering features")
        X_engineered = self.engineer_features(X)
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
 
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
        }
        
        best_model = None
        best_score = 0
        best_model_name = ""
        
        print("Training models")
        for name, model in models.items():
            print(f"Training {name}")
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            print(f"{name} accuracy: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name
        
        self.model = best_model
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        self.model_info = {
            'model_name': best_model_name,
            'version': '1.0',
            'accuracy': best_score,
            'training_date': datetime.now().isoformat(),
            'features': list(X_engineered.columns),
            'target_classes': list(self.label_encoder.classes_),
            'feature_importance': None
        }
    
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(X_engineered.columns, self.model.feature_importances_))
            self.model_info['feature_importance'] = importance_dict
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Test Accuracy: {best_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return self.model_info
    
    def predict(self, patient_data: Dict[str, float]) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # No field mapping needed - using consistent uppercase naming
        input_df = pd.DataFrame([patient_data])
        for feature in self.feature_names:
            if feature not in input_df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        input_imputed = pd.DataFrame(
            self.imputer.transform(input_df[self.feature_names]),
            columns=self.feature_names
        )
        input_engineered = self.engineer_features(input_imputed)
        
        input_scaled = self.scaler.transform(input_engineered)

        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        prob_dict = dict(zip(self.label_encoder.classes_, probabilities))
    
        confidence_score = max(probabilities)
        risk_factors = self._identify_risk_factors(patient_data)
        recommendations = self._generate_recommendations(patient_data, predicted_class, risk_factors)
        
        # Add confidence-based alerts
        confidence_level = self._assess_confidence_level(confidence_score)
        
        return {
            'predicted_delivery_type': predicted_class,
            'probabilities': prob_dict,
            'confidence_score': confidence_score,
            'confidence_level': confidence_level,
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }
    
    def _identify_risk_factors(self, patient_data: Dict[str, float]) -> List[str]:
        """Identify risk factors based on patient data"""
        risk_factors = []
        
        age = patient_data.get('Mother_Age', 0)
        gravida = patient_data.get('Gravida', 0)
        parity = patient_data.get('Parity', 0)
        gestation_weeks = patient_data.get('Gestation_Weeks', 0)
        previous_cs = patient_data.get('Previous_CS', 0)
        
        if age < 18:
            risk_factors.append("Teenage pregnancy")
        elif age > 35:
            risk_factors.append("Advanced maternal age")
        
        if previous_cs > 0:
            risk_factors.append("Previous cesarean section")
        
        if gestation_weeks < 37:
            risk_factors.append("Preterm delivery risk")
        elif gestation_weeks > 42:
            risk_factors.append("Post-term pregnancy")
        
        if parity == 0:
            risk_factors.append("First-time mother (nulliparous)")
        elif parity >= 5:
            risk_factors.append("Grand multiparous")
        
        if gravida > 5:
            risk_factors.append("Multiple previous pregnancies")
        
        return risk_factors
    
    def _assess_confidence_level(self, confidence_score: float) -> Dict[str, Any]:
        """Assess prediction confidence level and provide guidance"""
        # Import thresholds (you could import from config.py)
        MIN_CONFIDENCE = 0.6
        HIGH_CONFIDENCE = 0.8
        
        if confidence_score >= HIGH_CONFIDENCE:
            return {
                "level": "High",
                "description": "Prediction is highly reliable",
                "clinical_action": "Safe to follow model recommendation",
                "color": "green"
            }
        elif confidence_score >= MIN_CONFIDENCE:
            return {
                "level": "Moderate", 
                "description": "Prediction is moderately reliable",
                "clinical_action": "Consider additional clinical assessment",
                "color": "yellow"
            }
        else:
            return {
                "level": "Low",
                "description": "Prediction has low confidence",
                "clinical_action": "Require senior clinician review",
                "color": "red"
            }
    
    def _generate_recommendations(self, patient_data: Dict[str, float], 
                                predicted_delivery: str, risk_factors: List[str]) -> List[str]:
        recommendations = []
        
        if predicted_delivery == "Cesarean":
            recommendations.append("Consider cesarean section planning")
            recommendations.append("Discuss surgical risks and benefits with patient")
            recommendations.append("Ensure anesthesia consultation")
        
        elif predicted_delivery == "Assisted_Vaginal":
            recommendations.append("Prepare for potential instrumental delivery")
            recommendations.append("Monitor labor progress closely")
            recommendations.append("Have vacuum/forceps readily available")
        
        else:  # Vaginal delivery
            recommendations.append("Plan for vaginal delivery")
            recommendations.append("Monitor labor progress")
        
        # Risk-specific recommendations
        if "Advanced maternal age" in risk_factors:
            recommendations.append("Enhanced fetal monitoring recommended")
        
        if "Previous cesarean section" in risk_factors:
            recommendations.append("Consider VBAC counseling if appropriate")
            recommendations.append("Monitor for uterine rupture signs")
        
        if "Preterm delivery risk" in risk_factors:
            recommendations.append("Consider corticosteroids for fetal lung maturity")
            recommendations.append("NICU consultation recommended")
        
        if "Teenage pregnancy" in risk_factors:
            recommendations.append("Provide additional counseling and support")
            recommendations.append("Monitor for pregnancy complications closely")
        
        if "Grand multiparous" in risk_factors:
            recommendations.append("Monitor for uterine rupture risk")
            recommendations.append("Prepare for potential bleeding complications")
        
        return recommendations
    
    def save_model(self, model_dir: str):
        """Save the trained model and preprocessing objects"""
        os.makedirs(model_dir, exist_ok=True)
    
        joblib.dump(self.model, os.path.join(model_dir, 'delivery_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))
        joblib.dump(self.imputer, os.path.join(model_dir, 'imputer.pkl'))
        joblib.dump(self.model_info, os.path.join(model_dir, 'model_info.pkl'))
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir: str):
        """Load a pre-trained model"""
        try:
            self.model = joblib.load(os.path.join(model_dir, 'delivery_model.pkl'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
            self.imputer = joblib.load(os.path.join(model_dir, 'imputer.pkl'))
            self.model_info = joblib.load(os.path.join(model_dir, 'model_info.pkl'))
            print(f"Model loaded from {model_dir}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

if __name__ == "__main__":
    # Train the model
    predictor = DeliveryModePredictor()
    
    data_path = "../data/maternal_data_clean.csv"
    model_dir = "../models/trained_models"
    
    # Train and save model
    model_info = predictor.train_model(data_path)
    predictor.save_model(model_dir)
    
    # Test prediction
    test_patient = {
        'Mother_Age': 28.0,
        'Gravida': 2.0,
        'Parity': 1.0,
        'Gestation_Weeks': 38.5,
        'Previous_CS': 1.0
    }
    
    result = predictor.predict(test_patient)
    print("\nTest Prediction:")
    print(f"Predicted delivery type: {result['predicted_delivery_type']}")
    print(f"Confidence: {result['confidence_score']:.3f}")
    print(f"Risk factors: {result['risk_factors']}")
    print(f"Recommendations: {result['recommendations']}")
