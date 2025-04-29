#! D:\Python\myvenv\Scripts\python.exe
import streamlit as st
import mlflow
import pandas as pd
import os
from mlflow.tracking import MlflowClient
import logging
from datetime import datetime
import torch

# Add to top of your Streamlit app
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"sentiment_app_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("sentiment-analysis-ui")

# Configuration
MODEL_NAME = "sentiment-analysis"
ALIAS = "champion"  # Replace "Production" stage with "champion" alias
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

def load_production_model():
    """Load the production model from MLflow Registry using alias instead of stage"""
    logger.info(f"Attempting to load model {MODEL_NAME}@{ALIAS} from MLflow Registry")
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
        # Use transformers-specific loading
        model = mlflow.transformers.load_model(
            model_uri=f"models:/{MODEL_NAME}@{ALIAS}",
            device_map="auto",  # Let accelerate handle device mapping
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            return_type="pipeline"
        )
        
        # Explicit device placement
        if torch.cuda.is_available():
            model.model = model.model.to("cuda:0")
        else:
            model.model = model.model.cpu()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# App initialization
logger.info("Starting Sentiment Analysis Streamlit application")

# Streamlit UI
st.title("Real-time Sentiment Analysis")
st.markdown("Analyze customer reviews using our production ML model")

# Load model once per session
if 'model' not in st.session_state:
    logger.info("First run - loading model for this session")
    st.session_state.model = load_production_model()
    if not st.session_state.model:
        logger.warning("Failed to load model for session")

if st.session_state.model:
    text_input = st.text_area("Enter customer review:", height=150)
    
    if st.button("Analyze"):
        truncated_text = text_input[:50] + "..." if len(text_input) > 50 else text_input
        logger.info(f"Processing text: '{truncated_text}'")
        with st.spinner("Processing..."):
            try:
                start_time = datetime.now()
                prediction = st.session_state.model.predict(text_input)[0]
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                label = prediction["label"].upper()
                score = prediction["score"]
                logger.info(f"Prediction result: {label} with confidence {score:.4f} (took {processing_time:.2f}s)")
                st.subheader("Results")
                col1, col2 = st.columns(2)
                col1.metric("Sentiment", label)
                col2.metric("Confidence", f"{score:.2%}")
                
                if label == "LABEL_1":
                    st.success("✅ Positive sentiment detected")
                    logger.info("Displayed positive sentiment result to user")
                else:
                    st.error(f"⚠️ Negative sentiment detected with label: {label}")
                    logger.info("Displayed negative sentiment result to user")
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

else:
    logger.warning("Application running without a model loaded")


