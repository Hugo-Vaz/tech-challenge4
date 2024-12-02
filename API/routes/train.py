from fastapi import APIRouter, HTTPException
import subprocess

router = APIRouter()

@router.post("/")
def train_model():
    """
    Treina o modelo LSTM usando o arquivo train_model.py.
    """
    try:
        # Executar o script train_model.py
        result = subprocess.run(
            ["python", "Model/train_model.py"],
            text=True,
            capture_output=True,
            check=True
        )

        # Retornar o output do treinamento
        return {
            "status": "Success",
            "output": result.stdout
        }
    except subprocess.CalledProcessError as e:
        # Capturar erros no subprocesso
        raise HTTPException(
            status_code=500,
            detail={
                "status": "Error",
                "output": e.stdout,
                "error": e.stderr
            }
        )
