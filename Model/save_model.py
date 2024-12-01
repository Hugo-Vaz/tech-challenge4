import pickle
import boto3
import os

class SaveModel:
    def __init__(self):
        self.boto_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('s3_access_key_id'),
        aws_secret_access_key=os.getenv('s3_secret_access_key'),
        aws_session_token=os.getenv('s3_session_token')
    )
        pass

    def save(self, ml_model, scaler):
        complete_model = {
            "Modelo":ml_model,
            "Scaler":scaler
        }

        modelo_buffer = pickle.dumps(complete_model)
        self.boto_client.put_object(Bucket="techchallenge4", Key=f"modelo/lstm.sav", Body=modelo_buffer)

