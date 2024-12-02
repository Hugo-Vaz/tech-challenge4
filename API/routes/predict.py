from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from API.services.model_loader import load_model_and_scaler
from API.services.prediction import make_prediction

router = APIRouter()

# Modelo e scaler carregados na inicialização
model, scaler, device = load_model_and_scaler()

# Modelo de dados de entrada
class StockData(BaseModel):
    abertura: float
    maxima: float
    minima: float
    volume: float
    data: str  # Data como string (não usada no modelo)

@router.post("/")
def predict(stock_data: list[StockData]):
    try:
        predictions = make_prediction(stock_data, model, scaler, device)
        return {"predictions": predictions}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
