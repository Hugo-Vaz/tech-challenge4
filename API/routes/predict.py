from datetime import datetime, timedelta, date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from API.services.model_loader import load_model_and_scaler
from Model.data_importer import ImportStockData
from Model.data_creator import CreateLTSMData

# Inicializa o roteador para este endpoint
router = APIRouter()

# Define a estrutura de entrada para cada registro histórico
class StockData(BaseModel):
    abertura: float = Field(..., description="Preço de abertura do dia")
    maxima: float = Field(..., description="Preço máximo do dia")
    minima: float = Field(..., description="Preço mínimo do dia")
    volume: float = Field(..., description="Volume de transações do dia")
    data: str = Field(..., description="Data do registro no formato YYYY-MM-DD")

# Define a estrutura do payload de entrada do endpoint
class PredictionRequest(BaseModel):
    prediction_dates: list[date] = Field(..., description="Lista de datas para previsão")

# Define a estrutura da resposta do endpoint
class PredictionResponse(BaseModel):
    predictions: List[List[float]] = Field(..., description="Lista de listas de valores previstos")

@router.post("/", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Endpoint para prever valores de fechamento de ações.
    Para cada registro no payload, busca os 20 registros anteriores no S3
    e usa-os para gerar previsões.
    """
    try:
        # Carregar o modelo treinado e o escalador do S3
        model, scaler, device = load_model_and_scaler()
        importer = ImportStockData()
        creator = CreateLTSMData()

        # Lista para armazenar previsões de cada item do payload
        all_predictions = []

        # Iterar sobre cada item no payload
        for prediction_date in request.prediction_dates:
            start_dt = prediction_date - timedelta(days=60)

            # Buscar os 20 registros anteriores à data informada no payload
            symbol = "PETR4.SA"  # Substitua pelo símbolo relevante
            stock_data, _ = importer.get_stock_data(
                symbol=symbol,
                start=f"{start_dt:%Y-%m-%d}",  # Data inicial é 20 dias antes da informada no item
                end=f"{prediction_date:%Y-%m-%d}"  # Data final é a informada no item
            )

            _, x_test, _, y_test = creator.build_data(stock_data, 20, test_size=0)
            test_dataset = TensorDataset(x_test, y_test)
            test_loader = DataLoader(dataset=test_dataset, batch_size=5, shuffle=False)

            for (sequences, _) in test_loader:
                sequences = sequences.to(device)
                # Forward pass
                outputs = model(sequences)

            np_out = outputs.detach().cpu().numpy()
            scaler.inverse_transform(np_out)

            print("Done")

            # # Adicionar as previsões à lista de todas as previsões
            # all_predictions.append(predictions)

        # Retornar a lista de previsões no formato esperado
        # return PredictionResponse(predictions=all_predictions)

    except HTTPException as e:
        # Propaga erros do cliente (HTTP 4xx)
        raise e
    except Exception as e:
        # Captura e retorna erros internos do servidor (HTTP 5xx)
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {str(e)}")
