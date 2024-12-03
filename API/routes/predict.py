from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from API.services.model_loader import load_model_and_scaler
from API.services.prediction import make_prediction
from API.services.data_fetcher import fetch_last_30_days_data  # Novo serviço

router = APIRouter()

# Modelo e scaler carregados na inicialização
model, scaler, device = load_model_and_scaler()

# Modelo de dados de entrada
class StockData(BaseModel):
    abertura: float
    maxima: float
    minima: float
    volume: float
    data: str  # Data como string (ISO 8601: YYYY-MM-DD)

@router.post("/")
def predict(stock_data: list[StockData]):
    try:
        # Validar dados de entrada
        if len(stock_data) < 1:
            raise ValueError("O corpo da solicitação deve conter pelo menos um item.")

        # Pega a última data do payload e busca os 30 dias anteriores no S3
        target_date = stock_data[-1].data
        historical_data = fetch_last_30_days_data(target_date)

        # Adiciona os dados fornecidos pelo usuário ao histórico
        full_data = historical_data + stock_data

        # Faz a previsão
        predictions = make_prediction(full_data, model, scaler, device)
        return {"predictions": predictions}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
