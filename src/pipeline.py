import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from clearml import Task, Logger

def run_pipeline():
    # Initialize ClearML Task
    task = Task.init(
        project_name="MTN Customer Segmentation", 
        task_name="KMeans_Production_Pipeline"
    )
    logger = Logger.current_logger()

    # Paths
    RAW_PATH = "data/raw/mtn_customer_churn.csv"
    PROCESSED_PATH = "data/processed/mtn_segmented_customers.csv"
    SCALER_BIN_PATH = "models/scaler.joblib"
    MODEL_BIN_PATH = "models/kmeans_model.joblib"
    
    print("Starting Production Data Pipeline...")

    # DATA CLEANING 
    df = pd.read_csv(RAW_PATH)
    
    # Parse dates explicitly
    df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], format='%b-%y', errors='coerce')
    
    # Standardize string categories
    string_cols = ['State', 'MTN Device', 'Gender', 'Customer Review', 'Subscription Plan']
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()
    df['State'] = df['State'].str.title()
    
    # Handle churn reason nulls contextually
    df['Reasons for Churn'] = df['Reasons for Churn'].fillna('Active Customer')
    
    # Enforce revenue mathematical integrity
    df['Total Revenue'] = df['Unit Price'] * df['Number of Times Purchased']
    print("Data cleaning and mathematical validation complete.")

    # FEATURE ENGINEERING & SCALING 
    device_tier_map = {
        'Mobile SIM Card': 1,
        'Broadband MiFi': 2,
        '4G Router': 3,
        '5G Broadband Router': 4
    }
    df['Device_Tier'] = df['MTN Device'].map(device_tier_map)
    
    cluster_features = [
        'Age', 'Customer Tenure in months', 'Satisfaction Rate', 
        'Unit Price', 'Data Usage', 'Total Revenue', 'Device_Tier'
    ]
    
    X = df[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled perfectly for distance calculation.")

    # MODEL TRAINING & ARTIFACT LOGGING 
    optimal_k = 4
    # Log hyperparameters to ClearML dashboard automatically
    task.connect({"num_clusters": optimal_k, "random_state": 42})
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Log summary performance metrics to ClearML
    logger.report_single_value(name="Inertia / WCSS", value=kmeans.inertia_)
    
    # Save processed output
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Production model fitted. Segmented dataset saved to {PROCESSED_PATH}")
    
    #  BINARY EXPORT & CLEARML ARTIFACT UPLOAD 
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, SCALER_BIN_PATH)
    joblib.dump(kmeans, MODEL_BIN_PATH)
    
    print("Uploading binary artifacts to ClearML...")
    task.upload_artifact(name="standard_scaler", artifact_object=SCALER_BIN_PATH)
    task.upload_artifact(name="kmeans_model", artifact_object=MODEL_BIN_PATH)
    print("Artifact registration complete!")
    
    # Log final cluster balance metrics to ClearML console
    print("\n--- Final Segment Balance ---")
    print(df['Cluster'].value_counts())
    
    task.close()

if __name__ == "__main__":
    run_pipeline()