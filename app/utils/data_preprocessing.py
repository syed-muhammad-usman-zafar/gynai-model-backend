import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

def load_maternal_data(file_path: str) -> pd.DataFrame:
    """Load maternal health data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze data quality and return summary statistics"""
    quality_report = {
        'total_records': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'delivery_type_distribution': df['Delivery_Type'].value_counts().to_dict(),
        'summary_statistics': df.describe().to_dict()
    }
    
    # Check for data inconsistencies
    inconsistencies = []
    
    # Check if parity > gravida (impossible)
    invalid_parity = df[df['Parity'] > df['Gravida']]
    if not invalid_parity.empty:
        inconsistencies.append(f"Found {len(invalid_parity)} records where Parity > Gravida")
    
    # Check for unrealistic gestational weeks
    unrealistic_gestation = df[(df['Gestation_Weeks'] < 20) | (df['Gestation_Weeks'] > 45)]
    if not unrealistic_gestation.empty:
        inconsistencies.append(f"Found {len(unrealistic_gestation)} records with unrealistic gestational weeks")
    
    # Check for unrealistic maternal age
    unrealistic_age = df[(df['Mother_Age'] < 12) | (df['Mother_Age'] > 60)]
    if not unrealistic_age.empty:
        inconsistencies.append(f"Found {len(unrealistic_age)} records with unrealistic maternal age")
    
    quality_report['inconsistencies'] = inconsistencies
    
    return quality_report

def preprocess_maternal_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess maternal data"""
    df_processed = df.copy()
    
    # Remove duplicate patient records
    df_processed = df_processed.drop_duplicates(subset=['Patient_ID'])
    
    # Handle missing values in numerical columns
    numerical_cols = ['Mother_Age', 'Gravida', 'Parity', 'Gestation_Weeks', 'Previous_CS']
    
    for col in numerical_cols:
        if col in df_processed.columns:
            # Fill missing values with median
            median_value = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_value)
    
    # Ensure data consistency
    # Parity cannot be greater than Gravida
    inconsistent_parity = df_processed['Parity'] > df_processed['Gravida']
    df_processed.loc[inconsistent_parity, 'Parity'] = df_processed.loc[inconsistent_parity, 'Gravida']
    
    # Previous CS cannot be greater than Parity
    inconsistent_cs = df_processed['Previous_CS'] > df_processed['Parity']
    df_processed.loc[inconsistent_cs, 'Previous_CS'] = df_processed.loc[inconsistent_cs, 'Parity']
    
    return df_processed

def validate_patient_input(patient_data: Dict[str, float]) -> Tuple[bool, str]:
    """Validate patient input data"""
    required_fields = ['mother_age', 'gravida', 'parity', 'gestation_weeks', 'previous_cs']
    
    # Check required fields
    for field in required_fields:
        if field not in patient_data:
            return False, f"Missing required field: {field}"
        
        if patient_data[field] is None:
            return False, f"Field {field} cannot be None"
    
    # Validate ranges
    validations = {
        'mother_age': (12, 60, "Mother's age should be between 12 and 60 years"),
        'gravida': (1, 20, "Gravida should be between 1 and 20"),
        'parity': (0, 15, "Parity should be between 0 and 15"),
        'gestation_weeks': (20, 45, "Gestational weeks should be between 20 and 45"),
        'previous_cs': (0, 10, "Previous CS should be between 0 and 10")
    }
    
    for field, (min_val, max_val, error_msg) in validations.items():
        value = patient_data[field]
        if not (min_val <= value <= max_val):
            return False, error_msg
    
    # Logical validations
    if patient_data['parity'] > patient_data['gravida']:
        return False, "Parity cannot be greater than Gravida"
    
    if patient_data['previous_cs'] > patient_data['parity']:
        return False, "Previous CS cannot be greater than Parity"
    
    return True, "Valid input"

def calculate_risk_score(patient_data: Dict[str, float]) -> Dict[str, Any]:
    """Calculate a composite risk score for the patient"""
    risk_score = 0
    risk_factors = []
    
    age = patient_data.get('mother_age', 0)
    gravida = patient_data.get('gravida', 0)
    parity = patient_data.get('parity', 0)
    gestation_weeks = patient_data.get('gestation_weeks', 0)
    previous_cs = patient_data.get('previous_cs', 0)
    
    # Age-related risk
    if age < 18:
        risk_score += 2
        risk_factors.append("Teenage pregnancy")
    elif age > 35:
        risk_score += 1 if age <= 40 else 2
        risk_factors.append("Advanced maternal age")
    
    # Previous cesarean risk
    if previous_cs > 0:
        risk_score += previous_cs * 2
        risk_factors.append(f"Previous cesarean section(s): {int(previous_cs)}")
    
    # Gestational age risk
    if gestation_weeks < 37:
        risk_score += 3
        risk_factors.append("Preterm delivery risk")
    elif gestation_weeks > 42:
        risk_score += 2
        risk_factors.append("Post-term pregnancy")
    
    # Parity-related risk
    if parity == 0:
        risk_score += 1
        risk_factors.append("Nulliparous (first pregnancy)")
    elif parity >= 5:
        risk_score += 2
        risk_factors.append("Grand multiparous")
    
    # Multiple pregnancies
    if gravida > 5:
        risk_score += 1
        risk_factors.append("Multiple previous pregnancies")
    
    # Categorize risk level
    if risk_score <= 2:
        risk_level = "Low"
    elif risk_score <= 5:
        risk_level = "Moderate"
    else:
        risk_level = "High"
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_factors': risk_factors
    }

if __name__ == "__main__":
    # Example usage
    data_path = "../data/maternal_data_clean.csv"
    
    # Load and analyze data
    df = load_maternal_data(data_path)
    quality_report = check_data_quality(df)
    
    print("Data Quality Report:")
    print(f"Total records: {quality_report['total_records']}")
    print(f"Missing values: {quality_report['missing_values']}")
    print(f"Delivery type distribution: {quality_report['delivery_type_distribution']}")
    
    if quality_report['inconsistencies']:
        print("Data inconsistencies found:")
        for inconsistency in quality_report['inconsistencies']:
            print(f"- {inconsistency}")
    
    # Preprocess data
    df_clean = preprocess_maternal_data(df)
    print(f"\nAfter preprocessing: {len(df_clean)} records")
