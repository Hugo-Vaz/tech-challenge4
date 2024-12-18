import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

class ImportStockData:
    def __init__(self):
        pass
    # Faz o download dos dados utilizando a biblioteca yfinance, params de entrada são:
    # symbol (stock symbol)
    # start (data da primeira entrada)
    # end (data de corte/última entrada)
    def load_stock(self,symbol,start,end):
        data = yf.download(symbol, start=start, end=end)
        return data

    # Primeira etapa de pre processamento dos dados, extrai e utilizando o MinMaxScaler da scikit-learn,
    # cria uma escala consistente dos preços de treinamento, para facilitar/tornar coerente o treino do modelo
    def pre_process_stock(self,data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        return scaled_data, scaler

    def get_stock_data(self,symbol, start = "2020-01-01",end = "2024-12-31"):
        data = self.load_stock(symbol,start,end)
        return self.pre_process_stock(data)
    
    def process_prediction_data(self, symbol, start = "2020-01-01", end = "2024-12-31", abertura=0, max=0, min=0, volume=0):
        data = self.load_stock(symbol, start, end)
        data.loc[len(data)] = [0, 0, max, min, abertura, volume]

        return data