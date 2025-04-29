#! D:\Python\myvenv\Scripts\python.exe

import argparse
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import os
# With this version-check alternative:
from accelerate import __version__ as accelerate_version
from packaging import version
from mlflow.tracking import MlflowClient

IS_ACCELERATE_AVAILABLE = version.parse(accelerate_version) >= version.parse("0.29.0")
if not IS_ACCELERATE_AVAILABLE:
    raise ImportError("Accelerate 0.29.0+ required")


def train_model(train_data_path, test_data_path, model_name="distilbert-base-uncased", output_dir="models"):
    # Start MLflow run
    mlflow.start_run()
    
    # Log parameters
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("train_data", train_data_path)
    mlflow.log_param("test_data", test_data_path)
    
    # Load data
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)
    
    # Define training arguments
    # src/models/train_model.py (Line 42)
    # In src/models/train_model.py (Lines 42-55)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        # Updated parameters for compatibility:
        eval_strategy="epoch",        # Changed from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,                    # Add mixed-precision training
        gradient_accumulation_steps=2,
        report_to="mlflow"            # Ensure MLflow integration
    )


    
    # Define compute_metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    trainer.train()
    
    # Evaluate model
    eval_results = trainer.evaluate()
    
    # Save model
    model_path = os.path.join(output_dir, "final-model")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Add this in the training code before mlflow.end_run()
    mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": tokenizer
        },
        artifact_path="model",
        task="text-classification",
        registered_model_name="sentiment-analysis"  # Added model registration
    )

    # After registration, set an alias
    client = MlflowClient()
    # Find the latest version
    latest_version = client.search_model_versions(f"name='sentiment-analysis'")[0].version
    # Set the alias
    client.set_registered_model_alias(name='sentiment-analysis', alias="champion", version=latest_version)
    # End the MLflow run
    mlflow.end_run()
    
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data')
    parser.add_argument('--model-name', type=str, default="distilbert-base-uncased", help='Pretrained model name')
    parser.add_argument('--output-dir', type=str, default="models", help='Output directory for model artifacts')
    
    args = parser.parse_args()
    train_model(args.train_data, args.test_data, args.model_name, args.output_dir)
