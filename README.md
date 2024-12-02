# tech-challenge4

Repositório para o quarto desafio da pós gradução em Engenharia de ML

# Objetivo

Criaremos um modelo preditivo de redes neurais Long Short Term Memory (LSTM) para predizer o valor de fechamento da bolsa de valores, escolhemos a empresa Petrobras (PETR4.SA), utilizaremos o Pytorch para o desenvolvimento do modelo LSTM.

1 - Utilizaremos a api do Yahoo, através do pacote yFinance para extrair os dados.
2 - Adicionamos o MLFlow, para logar os testes, validações e o modelo. Contudo após o treinamento, o modelo e o scaler serão salvos no formato .sav na S3, fazendo uso do modulo [pickle](https://docs.python.org/3/library/pickle.html#module-pickle) do Python. O motivo disso, é que foi o mesmo processo que utilizamos no Tech Challenge anterior.

# API

Como o treinamento foi feito baseado em todas as features disponiveis: Close (sendo esse o target/alvo), High, Low, Open e Volume.
Log para conseguirmos prever o valor de fechamento vamos precisar da data (para buscar os dados dos 20 dias mais recentes, visto que 20 foi nosso n_past definido), Hig,Low, Open e volume (onde todos são valores decimais)

# Docker compose

para criar o bucket rode o comando: docker-compose up --build

# Criar o bucket no minIO

Rode o create_bucket.py!

# Rodar a api

Instalar requirements: pip install -r requirements.txt
Rodar APi: uvicorn API.main:app --host 0.0.0.0 --port 8000 --reload
