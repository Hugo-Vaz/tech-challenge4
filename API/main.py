from fastapi import FastAPI
from API.routes import health, train, success

app = FastAPI(
    title="Tech Challenge 4 API",
    version="1.0.0",
    description="API para prever o valor de fechamento da Petrobras (PETR4.SA).",
    docs_url="/swagger",  # Altera o caminho do Swagger para /swagger
    redoc_url=None        # Desativa o Redoc (opcional)
)

# Registrar rotas
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(train.router, prefix="/train", tags=["Training"])
app.include_router(success.router, prefix="/success", tags=["Success"])
