import pickle
import boto3
import os


class SaveModel:
    def __init__(self):
        """
        Inicializa o cliente MinIO/S3 usando as configurações fornecidas nas variáveis de ambiente.
        """
        self.boto_client = self._configure_minio_client()

    def _configure_minio_client(self):
        """
        Configura o cliente MinIO/S3.
        """
        return boto3.client(
            's3',
            endpoint_url=os.getenv('MINIO_URL', 'http://localhost:9000'),  # URL do MinIO
            aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),  # Usuário
            aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin')  # Senha
        )

    def _ensure_bucket_exists(self, bucket_name):
        """
        Verifica se o bucket existe no MinIO. Se não existir, ele será criado.
        """
        try:
            self.boto_client.head_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' já existe.")
        except self.boto_client.exceptions.NoSuchBucket:
            print(f"Bucket '{bucket_name}' não encontrado. Criando...")
            self.boto_client.create_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' criado com sucesso.")
        except Exception as e:
            print(f"Erro ao verificar/criar bucket: {e}")
            raise


    def _serialize_model(self, ml_model, scaler):
        """
        Serializa o modelo e o scaler em formato binário.
        """
        complete_model = {
            "Modelo": ml_model,
            "Scaler": scaler
        }
        return pickle.dumps(complete_model)

    def _upload_to_minio(self, bucket_name, file_key, file_body):
        """
        Faz o upload do arquivo para o bucket MinIO.
        """
        # Garantir que o bucket exista
        self._ensure_bucket_exists(bucket_name)

        # Fazer upload do arquivo
        self.boto_client.put_object(
            Bucket=bucket_name,
            Key=file_key,
            Body=file_body
        )
        print(f"Arquivo salvo com sucesso no bucket '{bucket_name}' com o caminho '{file_key}'.")

    def save(self, ml_model, scaler, bucket_name="techchallenge4", file_key="modelo/lstm.sav"):
        """
        Método principal para salvar o modelo e o scaler no MinIO.
        """
        modelo_serializado = self._serialize_model(ml_model, scaler)
        
        self._upload_to_minio(bucket_name, file_key, modelo_serializado)
