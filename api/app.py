from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from clearml import Task

app = FastAPI(
    title="MTN Segment Prediction API",
    description="Decoupled production endpoint pulling serialized model binaries directly from ClearML.",
    version="1.2.0"
)

# Global placeholders for clean in-memory model execution
SCALER = None
KMEANS_MODEL = None

class CustomerInput(BaseModel):
    Age: float
    Customer_Tenure_Months: float
    Satisfaction_Rate: float
    Unit_Price: float
    Data_Usage_GB: float
    Total_Revenue: float
    Device_Tier: int  

@app.on_event("startup")
def pull_clearml_artifacts():
    """Downloads and deserializes the production scaler and model directly from ClearML."""
    global SCALER, KMEANS_MODEL
    try:
        print("🔗 Connecting to ClearML tracking server...")
        # Grab the latest completed task matching our production pipeline name
        task = Task.get_task(
            project_name="MTN Customer Segmentation",
            task_name="KMeans_Production_Pipeline",
            task_filter={"status": ["completed"]}
        )
        
        print(f"Fetching binary artifacts from Task ID: {task.id}")
        
        # Download local temporary file copies from the ClearML server storage paths
        scaler_local_path = task.artifacts["standard_scaler"].get_local_copy()
        model_local_path = task.artifacts["kmeans_model"].get_local_copy()
        
        # Unpack binaries directly into memory
        SCALER = joblib.load(scaler_local_path)
        KMEANS_MODEL = joblib.load(model_local_path)
        
        print("Prediction engine running cleanly on active production binary anchors.")
        
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        raise RuntimeError("Could not synchronize application state with ClearML storage.")

@app.post("/predict/segment", summary="Predict the subscriber persona segment for a single customer record")
def predict_customer_segment(customer: CustomerInput):
    if KMEANS_MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Model engine state is uninitialized.")
        
    try:
        # Arrange incoming parameters into an identical row vector structure
        raw_input = np.array([[
            customer.Age,
            customer.Customer_Tenure_Months,
            customer.Satisfaction_Rate,
            customer.Unit_Price,
            customer.Data_Usage_GB,
            customer.Total_Revenue,
            customer.Device_Tier
        ]])
        
        # Scale inputs using the retrieved binary scaler coefficients
        scaled_input = SCALER.transform(raw_input)
        cluster_idx = int(KMEANS_MODEL.predict(scaled_input)[0])
        
        persona_mapping = {
            0: "The Retail Mobile Base",
            1: "The Maturing Loyalists",
            2: "The Enterprise Power Users",
            3: "The Heavy Data Hubs"
        }
        
        return {
            "predicted_cluster_id": cluster_idx,
            "assigned_persona": persona_mapping[cluster_idx]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction calculation error: {str(e)}")