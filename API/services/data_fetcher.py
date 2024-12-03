import boto3
import os
import pickle
from datetime import datetime, timedelta

def fetch_last_30_days_data(target_date: str):
    """
    Busca os últimos 30 dias de dados do bucket S3.
    """
    # Configurações do S3
    s3_client = boto3.client(
        's3',
        endpoint_url=os.getenv('MINIO_URL', 'http://localhost:9000'),
        aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
        aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin')
    )

    bucket_name = "techchallenge4"
    file_key = "modelo/lstm_data.sav"  # Nome do arquivo com os dados históricos

    # Baixa o arquivo do S3
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        data = pickle.loads(response['Body'].read())
    except Exception as e:
        raise RuntimeError(f"Erro ao buscar dados históricos no S3: {str(e)}")

    # Filtra os 30 dias anteriores à data-alvo
    target_datetime = datetime.strptime(target_date, "%Y-%m-%d")
    start_date = target_datetime - timedelta(days=30)

    filtered_data = [
        item for item in data
        if start_date <= datetime.strptime(item['data'], "%Y-%m-%d") <= target_datetime
    ]

    if not filtered_data:
        raise ValueError("Nenhum dado histórico encontrado para os últimos 30 dias.")

    return filtered_data
