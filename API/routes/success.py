from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def success_message():

    return {"message": "deu certo"}
