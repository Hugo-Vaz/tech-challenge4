from datetime import datetime, timedelta, date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np

from API.services.model_loader import load_model_and_scaler
from Model.data_importer import ImportStockData
from Model.data_creator import CreateLTSMData

# Inicializa o roteador para este endpoint
router = APIRouter()

# Define a estrutura de entrada para cada registro histórico
class StockData(BaseModel):
    fechamento: float = Field(..., description="Preço de fechamento do dia")
    abertura: float = Field(..., description="Preço de abertura do dia")
    maxima: float = Field(..., description="Preço máximo do dia")
    minima: float = Field(..., description="Preço mínimo do dia")
    volume: float = Field(..., description="Volume de transações do dia")
    data: str = Field(..., description="Data do registro no formato YYYY-MM-DD")

# Define a estrutura do payload de entrada do endpoint
class PredictionRequest(BaseModel):
    predictions: list[StockData] = Field(..., description="Lista de objetos para previsão")

# Define a estrutura da resposta do endpoint
class PredictionResponse(BaseModel):
    predictions: list[StockData] = Field(..., description="Lista de listas de valores previstos")

@router.post("/", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Endpoint para prever valores de fechamento de ações.
    Para cada registro no payload, busca os 20 registros anteriores no Yahoo Finance
    e usa-os para gerar previsões.
    """
    try:
        # Carregar o modelo treinado e o scaler no MinIO/S3
        model, scaler, device = load_model_and_scaler()
        importer = ImportStockData()
        creator = CreateLTSMData()

        # Lista para armazenar previsões de cada item do payload
        all_predictions = []

        # Iterar sobre cada item no payload
        for prediction in request.predictions:

            parsed_data = datetime.strptime(prediction.data, '%Y-%m-%d').date()

            start_dt = parsed_data - timedelta(days=60)
            # Buscar os 20 registros anteriores à data informada no payload
            symbol = "PETR4.SA"  # Substitua pelo símbolo relevante
            stock_data = importer.process_prediction_data(
                symbol=symbol,
                start=f"{start_dt:%Y-%m-%d}",  # Data inicial é 20 dias antes da informada no item
                end=f"{parsed_data:%Y-%m-%d}",  # Data final é a informada no item
                abertura=prediction.abertura,
                min=prediction.minima,
                max=prediction.maxima,
                volume=prediction.volume
            )

            scaled_data = scaler.fit_transform(stock_data)

            x, y = creator.split_features_test(scaled_data, 20)

            outputs = model(x)

            new_array = outputs.detach().cpu().numpy()
            ultimo = new_array[-1]
            new_array=ultimo.reshape(-1,1)
            prediction_copies_array = np.repeat(ultimo,6, axis=-1)
            reshaped = np.reshape(prediction_copies_array,(len(new_array),6))
            scaled_pred = scaler.inverse_transform(reshaped)

            prediction.fechamento = float(scaled_pred[-1][0])

            all_predictions.append(prediction)

        # Retornar a lista de previsões no formato esperado
        return PredictionResponse(predictions=all_predictions)

    except HTTPException as e:
        # Propaga erros do cliente (HTTP 4xx)
        raise e
    except Exception as e:
        # Captura e retorna erros internos do servidor (HTTP 5xx)
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {str(e)}")
