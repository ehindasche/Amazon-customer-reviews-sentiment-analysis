#! D:\Python\myvenv\Scripts\python.exe
import streamlit as st
import mlflow
import pandas as pd
import os
from mlflow.tracking import MlflowClient
import logging
from datetime import datetime
import torch
import shutil
from pathlib import Path
# Add configuration
FEEDBACK_DIR = "feedback_data"
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "user_feedback.csv")
os.makedirs(FEEDBACK_DIR, exist_ok=True)

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
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

def load_latest_model():
    """Load the latest version of the model from MLflow Registry"""
    logger.info(f"Attempting to load latest version of model {MODEL_NAME}")
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
        
        # Initialize MLflow client
        client = MlflowClient()
        
        # Get the latest version (sorted by version number descending)
        latest_version = client.get_latest_versions(MODEL_NAME)[0]
        model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
        
        logger.info(f"Loading model version {latest_version.version}")
        
        # Use transformers-specific loading with device_map="auto"
        model = mlflow.transformers.load_model(
            model_uri=model_uri,
            device_map="auto",  # Let accelerate handle device mapping
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            return_type="pipeline"
        )
        
        return model, latest_version.version
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logger.error(f"Model loading failed: {str(e)}", exc_info=True)
        return None, None

def retrain_model():
    """Retrain model with original data + feedback"""
    try:
        logger.info("Starting retraining process")
        original_train = "data/processed/train.csv"
        original_test = "data/processed/test.csv"
        updated_train = "data/processed/updated_train.csv"

        if os.path.exists(FEEDBACK_FILE):
            feedback_df = pd.read_csv(FEEDBACK_FILE)
            if not feedback_df.empty:
                original_train_df = pd.read_csv(original_train)
                feedback_df['label'] = (feedback_df['label'] == 'label_1').astype(int)
                merged_train = pd.concat([original_train_df, feedback_df[["label","text"]]])
                merged_train = merged_train.reset_index().rename(columns={"index": "Id"})
                merged_train.to_csv(updated_train, index=False)
                
                retrain_cmd = (
                    f"python src/models/train_model.py "
                    f"--train-data {updated_train} "
                    f"--test-data {original_test} "
                    "--model-name distilbert-base-uncased "
                    "--output-dir models/sentiment"
                )
                exit_code = os.system(retrain_cmd)
                
                if exit_code == 0:
                    logger.info("Retraining completed successfully")
                    # Archive feedback
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    shutil.move(FEEDBACK_FILE, f"{FEEDBACK_FILE}.{timestamp}.bak")
                    return True
        return False
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}", exc_info=True)
        return False

# App initialization
logger.info("Starting Sentiment Analysis Streamlit application")

# Streamlit UI
st.title("Real-time Sentiment Analysis")
st.markdown("Analyze customer reviews using our production ML model")

# Load model once per session
if 'model' not in st.session_state:
    logger.info("First run - loading model for this session")
    st.session_state.model, st.session_state.model_version = load_latest_model()
    if st.session_state.model:
        st.sidebar.success(f"Loaded model version {st.session_state.model_version}")
    else:
        logger.warning("Failed to load model for session")

if st.session_state.model:
    # Display model version in sidebar
    st.sidebar.markdown(f"**Model Version:** {st.session_state.model_version}")
    
    text_input = st.text_area("Enter customer review:", height=150, key="review_input")
    
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
                
                # Store prediction results in session state
                st.session_state.prediction = {
                    "label": label,
                    "score": score,
                    "text": text_input,
                    "timestamp": datetime.now().isoformat()
                }
                
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
                logger.error(f"Prediction error: {str(e)}", exc_info=True)

    # Show feedback form if we have a prediction in session state
    if 'prediction' in st.session_state:
        st.markdown("---")
        with st.form("feedback_form"):
            st.subheader("Provide Feedback")
            actual_sentiment = st.selectbox(
                "What's the correct sentiment?",
                ("LABEL_1", "LABEL_0"),
                index=1 if st.session_state.prediction["label"] == "LABEL_1" else 0,
                key="sentiment_select"
            )
            feedback_text = st.text_area("Additional comments (optional):", key="feedback_text")
            
            if st.form_submit_button("Submit Feedback"):
                try:
                    # Create feedback entry
                    feedback_data = {
                        "text": st.session_state.prediction["text"],
                        "label": actual_sentiment.lower(),
                        "feedback_text": feedback_text,
                        "timestamp": datetime.now().isoformat(),
                        "original_prediction": st.session_state.prediction["label"],
                        "original_confidence": st.session_state.prediction["score"],
                        "model_version": st.session_state.model_version
                    }
                    
                    # Save to CSV
                    feedback_df = pd.DataFrame([feedback_data])
                    header = not os.path.exists(FEEDBACK_FILE)
                    feedback_df.to_csv(FEEDBACK_FILE, mode='a', header=header, index=False)
                    
                    st.success("Thank you for your feedback! This will help improve our model.")
                    logger.info(f"Feedback saved: {feedback_data}")
                    
                    # Check if retraining needed
                    if os.path.exists(FEEDBACK_FILE):
                        feedback_count = len(pd.read_csv(FEEDBACK_FILE))
                        if feedback_count >= 2:  # Retrain after 2 feedbacks
                            with st.spinner("Retraining model with new feedback..."):
                                if retrain_model():
                                    # Reload the latest model after retraining
                                    st.session_state.model, st.session_state.model_version = load_latest_model()
                                    st.success("Model updated successfully!")
                                    logger.info("Model retrained and reloaded successfully")
                                    # Clear prediction to avoid confusion
                                    del st.session_state.prediction
                                    st.experimental_rerun()
                                else:
                                    st.warning("Retraining completed but no model update was needed")
                                    
                except Exception as e:
                    st.error(f"Failed to save feedback: {str(e)}")
                    logger.error(f"Feedback submission failed: {str(e)}", exc_info=True)