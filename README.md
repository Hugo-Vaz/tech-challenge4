# Tech Challenge 4

Repositório para o quarto desafio da pós gradução em Engenharia de ML

## Vídeo Explicação
[![Youtube video](https://img.youtube.com/vi/L6rS8jcr85I/0.jpg)](https://youtu.be/L6rS8jcr85I)

# Objetivo

Criaremos um modelo preditivo de redes neurais Long Short Term Memory (LSTM) para predizer o valor de fechamento da bolsa de valores, escolhemos a empresa Petrobras (PETR4.SA), utilizaremos o Pytorch para o desenvolvimento do modelo LSTM.

1 - Utilizaremos a api do Yahoo, através do pacote yFinance para extrair os dados.
2 - Adicionamos o MLFlow, para logar os testes, validações e o modelo. Contudo após o treinamento, o modelo e o scaler serão salvos no formato .sav no MinIO/S3, fazendo uso do modulo [pickle](https://docs.python.org/3/library/pickle.html#module-pickle) do Python. O motivo disso, é que foi o mesmo processo que utilizamos no Tech Challenge anterior.

# API

Como o treinamento foi feito baseado em todas as features disponiveis: Close (sendo esse o target/alvo), High, Low, Open e Volume.
Log para conseguirmos prever o valor de fechamento vamos precisar da data (para buscar os dados dos 20 dias mais recentes, visto que 20 foi nosso n_past definido), Hig,Low, Open e volume (onde todos são valores decimais)

## No endpoint (post) do predict:

- O payload pode conter vários registros (um ou mais).
- Para cada registro no payload:
- Buscar os últimos 20 registros históricos no bucket MinIO/S3 com base na data fornecida nesse item do payload.
- Usar esses 20 registros históricos para fazer a previsão.
- Retornar as previsões correspondentes para cada item do payload.

### Payload esperado

```json
{
  "predictions": [
    {
      "fechamento": 0,
      "abertura": 19.5,
      "maxima": 19.22,
      "minima": 16.41,
      "volume": 24.01,
      "data": "2024-12-02"
    }
  ]
}
```

# Getting Started

```bash
# Instalar bibliotecas necessarias para rodar o projeto
py -m pip install -r requirements.txt

# Configurar variaveis a serem usadas pelo container do MinIO/S3
export MINIO_URL=http://localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin

# Criar um container docker do MinIO/S3 (onde serao salvo os modelos)
docker-compose up --build -d

# Cria o bucket dos modelos do MinIO/S3

py create_bucket.py

# Rodar a API localmente
uvicorn API.main:app --host 0.0.0.0 --port 8000 --reload
```


