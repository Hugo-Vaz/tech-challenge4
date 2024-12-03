import boto3
import pickle
import os
import logging
import sys
import torch
from pathlib import Path

# Add the Model directory to sys.path dynamically
current_dir = Path(__file__).resolve().parent
model_dir = current_dir.parent.parent / "Model"
sys.path.append(str(model_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET_NAME = "techchallenge4"
MODEL_KEY = "modelo/lstm.sav"


def load_model_and_scaler():
    """
    Load the LSTM model and scaler from MinIO/S3 storage.
    """
    try:
        logger.info("Configuring the MinIO client...")
        s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_URL", "http://localhost:9000"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        )
        logger.info("MinIO client configured successfully. Connecting to bucket '%s'.", BUCKET_NAME)

        logger.info("Downloading the file '%s' from bucket '%s'...", MODEL_KEY, BUCKET_NAME)
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
        model_data = pickle.loads(response["Body"].read())
        logger.info("File downloaded successfully. Starting deserialization...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)

        model = model_data["Modelo"].to(device)
        scaler = model_data["Scaler"]

        logger.info("Model and scaler loaded successfully.")
        return model, scaler, device

    except boto3.exceptions.S3UploadFailedError as e:
        logger.error("upload failed: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise