# MTN Customer Segmentation Portal 

An end-to-end, production-grade MLOps system that cleans subscriber telemetry data, performs automated clustering, tracks experiment artifacts securely using ClearML, and hosts a decoupled FastAPI backend alongside an interactive Streamlit executive dashboard.

 **Live Portal App:** [Streamlit Dashboard](https://mtn-customer-segmentation-8d7liyjy9guesrukxynirq.streamlit.app/)  
 **Production API Gateway:** [FastAPI Backend Engine](https://mtn-customer-segmentation.onrender.com/docs)
**Dashboard:** [Looker Studio Dashboard](https://datastudio.google.com/s/nyxfz0bALq0) 

---

## System Architecture

This project is built around a lightweight, containerless local-to-cloud development model ("bare metal") optimized for raw environment speed and high inference efficiency. 

1. **Orchestration & ETL (`src/pipeline.py`)**: Imports raw telecom data, parses purchase timestamps, enforces numerical integrity across billing rules, engines categorical hardware variables, scales vectors using a `StandardScaler`, and fits an optimal 4-cluster `KMeans` segmentation strategy.
2. **Experiment Tracking & Artifact Registry (ClearML)**: Automatically registers hyperparameters (e.g., $K=4$), performance metrics (WCSS / Inertia), and serializes the trained mathematical states into standalone `.joblib` binaries uploaded directly to remote cloud storage.
3. **Application Layer (`api/app.py`)**: A high-performance FastAPI service that handshakes securely with ClearML on startup, downloads the pre-calibrated binary anchors straight into memory, and hosts an inference pipeline vector endpoint at `/predict/segment`.
4. **Presentation Layer (`app/dashboard.py`)**: A highly interactive, executive-ready Streamlit dashboard that lets analysts alter subscriber features on the fly and stream real-time persona assignments via clean API requests.

---

## Repository Structure

```text
mtn-customer-segmentation/
├── api/
│   └── app.py               # Live FastAPI Backend Engine 
├── app/
│   └── dashboard.py         # Streamlit Presentation Frontend
├── src/
│   └── pipeline.py          # ETL, Training, & ClearML Artifact Upload Script
├── data/
│   ├── raw/                 # Baseline subscriber source data tables (Git ignored)
│   └── processed/           # Cleaned and clustered tables (Git ignored)
├── models/                  # Local binary artifact cache and structural logs
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   └── 02_customer_segmentation.ipynb
├── .gitignore               # Strict exclusion matrix tracking local venv & keys
├── requirements.txt         # Production library dependency manifest
└── README.md                # Enterprise systems documentation


##### Identified Subscriber Personas

The underlying clustering model maps incoming subscriber records into 4 distinct commercial archetypes based on transactional footprint:

Cluster 0: The Retail Mobile Base — High-volume, standard consumers utilizing standard mobile SIM networks for basic data/voice allocations.

Cluster 1: The Maturing Loyalists — Mid-tier hardware configurations with highly stable tenure vectors but shorter active tracking cycles.

Cluster 2: The Enterprise Power Users — High-value corporate subscriber bands pulling massive 5G data traffic and pushing substantial revenue margins (ARPU).

Cluster 3: The Heavy Data Hubs — Power users anchored to large-scale 4G installations running near-continuous active downloads.

###### Local Environment Installation

Follow these steps to run the complete pipeline loop locally on your machine.

1. Clone the Workspace & Configure Dependencies

git clone [https://github.com/jameskingsley/mtn-customer-segmentation.git](https://github.com/jameskingsley/mtn-customer-segmentation.git)
cd mtn-customer-segmentation

# Initialize local Python virtual environment
python -m venv venv
# Activate on Windows systems
.\venv\Scripts\activate
# Activate on macOS/Linux
source venv/bin/activate

# Install exact structural libraries
pip install -r requirements.txt

###### Configure Your Tracking 

Ensure you have initialized your credentials with the ClearML server locally by ensuring your tracking configuration environment variable matrix or local .clearml.conf file matches your workspace setup:

clearml-init

###### Run the Automation Training Pipeline

Executing this script will clear data tables, run feature scaling, register metrics, and push the binary joblib models straight to the ClearML central registry:

python src/pipeline.py

###### Boot Up the Decoupled Servers Local Loop

Open two separate terminal splits inside your active environment to run both backend and frontend applications simultaneously:
# Terminal Split 1: Start your FastAPI Server
python -m uvicorn api.app:app --reload

# Terminal Split 2: Start your Streamlit Web Dashboard
streamlit run app/dashboard.py

###### Cloud Deployment Framework

FastAPI Service Configuration (Render)

Runtime Environment: Native Python 3 (Containerless/Bare-Metal execution)

Build Command: pip install -r requirements.txt

Start Command: python -m uvicorn api.app:app --host 0.0.0.0 --port $PORT

Environment Configuration Matrix:

CLEARML_API_HOST = https://api.clear.ml

CLEARML_WEB_HOST = https://app.clear.ml

CLEARML_FILES_HOST = https://files.clear.ml

CLEARML_API_ACCESS_KEY = <Your-ClearML-Access-Key>

CLEARML_API_SECRET_KEY = <Your-ClearML-Secret-Key>

###### Core Technology Stack

Core Backend Framework: FastAPI (Asynchronous Python Web framework)

Presentation Layer: Streamlit (Interactive Frontend Layout Engine)

MLOps Tracker: ClearML (Centralized Experiment Logging & Artifact Registry)

Mathematical Processing: Scikit-Learn, Pandas, NumPy, Joblib

Dashboard: Looker Studio

ASGI Server Host Engine: Uvicorn