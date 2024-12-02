import boto3
import pickle
import os
import torch

BUCKET_NAME = "techchallenge4"
MODEL_KEY = "modelo/lstm.sav"

def load_model_and_scaler():
    # Configurar o cliente MinIO
    s3_client = boto3.client(
        's3',
        endpoint_url=os.getenv('MINIO_URL', 'http://localhost:9000'),
        aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
        aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
    )

    # Baixar o modelo do MinIO
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
    model_data = pickle.loads(response['Body'].read())

    # Configurar o dispositivo (GPU ou CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_data["Modelo"].to(device)
    scaler = model_data["Scaler"]
    return model, scaler, device
