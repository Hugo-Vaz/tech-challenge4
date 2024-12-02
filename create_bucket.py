import boto3
import os

def configure_minio_client():
    """
    Configura o cliente MinIO usando as variáveis de ambiente.
    """
    return boto3.client(
        's3',
        endpoint_url=os.getenv('MINIO_URL', 'http://localhost:9000'),  # URL do MinIO
        aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),  # Usuário
        aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin')  # Senha
    )

def ensure_bucket_exists(client, bucket_name):
    """
    Verifica se o bucket existe no MinIO e o cria se necessário.
    """
    try:
        # Verificar se o bucket existe
        buckets = client.list_buckets().get("Buckets", [])
        if any(bucket["Name"] == bucket_name for bucket in buckets):
            print(f"Bucket '{bucket_name}' já existe.")
        else:
            print(f"Bucket '{bucket_name}' não encontrado. Criando...")
            client.create_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' criado com sucesso.")
    except Exception as e:
        print(f"Erro ao verificar/criar bucket: {e}")

if __name__ == "__main__":
    # Nome do bucket que deseja criar
    BUCKET_NAME = "techchallenge4"

    # Configurar o cliente MinIO
    minio_client = configure_minio_client()

    # Garantir que o bucket exista
    ensure_bucket_exists(minio_client, BUCKET_NAME)
