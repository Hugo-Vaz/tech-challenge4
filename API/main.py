from fastapi import FastAPI
from API.routes import health, train, predict, success

app = FastAPI(
    title="Tech Challenge 4 API",
    version="1.0.0",
    description="API for PETR4.SA stock price prediction.",
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(train.router, prefix="/train", tags=["Training"])
app.include_router(success.router, prefix="/success", tags=["Utility"])
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])


# Ponto de entrada principal
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="debug")