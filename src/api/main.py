#! D:\Python\myvenv\Scripts\python.exe

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Counter, Histogram
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import os

app = FastAPI()
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Configuration
EXPERIMENT_NAME = "Default"
RUN_NAME = "spiffy-seal-538"
MODEL_ARTIFACT_PATH = "model"  # Default path where models are logged
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

API_REQUESTS = Counter("api_requests_total", "Total API requests")
API_ERRORS = Counter("api_errors_total", "Total API errors")
REQUEST_DURATION = Histogram("request_duration_seconds", "Request duration")

class TextInput(BaseModel):
    text: str

@app.on_event("startup")
def load_model():
    global model
    try:
        # Initialize MLflow client
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        # Get experiment by name
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found")

        # Search for the specific run by name
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.`mlflow.runName` = '{RUN_NAME}'"
        )

        if not runs:
            raise ValueError(f"Run '{RUN_NAME}' not found in experiment '{EXPERIMENT_NAME}'")

        # Get the first matching run
        run = runs[0]
        
        # Construct model URI
        model_uri = f"runs:/{run.info.run_id}/{MODEL_ARTIFACT_PATH}"
        
        # Load the model
        model = mlflow.pyfunc.load_model(model_uri)

    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

@app.post("/analyze")
@REQUEST_DURATION.time()
def analyze_sentiment(input_data: TextInput):
    API_REQUESTS.inc()
    try:
        result = model.predict([input_data.text])[0]
        return {
            "text": input_data.text,
            "sentiment": result["label"].upper(),
            "score": float(result["score"])
        }
    except Exception as e:
        API_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
