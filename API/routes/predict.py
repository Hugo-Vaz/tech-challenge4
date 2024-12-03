from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from API.services.model_loader import load_model_and_scaler
from API.services.prediction import make_prediction

router = APIRouter()

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
        # Carregar o modelo e o scaler sob demanda
        model, scaler, device = load_model_and_scaler()

        # Validar dados de entrada
        if len(stock_data) < 1:
            raise ValueError("O corpo da solicitação deve conter pelo menos um item.")

        # Faz a previsão
        predictions = make_prediction(stock_data, model, scaler, device)
        return {"predictions": predictions}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
