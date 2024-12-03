from fastapi import APIRouter, HTTPException
from Model.train_model import train

router = APIRouter()

@router.post("/")
def train_model():
    """
    Treina o modelo LSTM usando o arquivo train_model.py.
    """
    try:
        # Executar o script train_model.py
        train()

        # Retornar o output do treinamento
        return {
            "status": "Success",
        }
    except Exception as e:
        # Capturar erros no subprocesso
        raise HTTPException(
            status_code=500,
            detail={
                "status": "Error",
                "output": str(e)
            }
        )
