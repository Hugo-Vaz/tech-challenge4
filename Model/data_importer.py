import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

class ImportStockData:
    def __init__(self):
        pass
    #Faz o download dos dados utilizando a biblioteca yfinance, params de entrada são:
    # symbol (stock symbol)
    # start (data da primeira entrada)
    # end (data de corte/última entrada)
    def load_stock(self,symbol,start,end):
        data = yf.download(symbol, start=start, end=end)        
        return data
    
    #Primeira etapa de pre processamento dos dados, extrai e utilizando o MinMaxScaler da scikit-learn, 
    # cria uma escala consistente dos preços de treinamento, para facilitar/tornar coerente o treino do modelo
    def pre_process_stock(self,data):
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        close_prices_scaled = scaler.fit_transform(close_prices)

        return close_prices_scaled
    
    def get_stock_data(self,symbol, start = "2020-01-01",end = "2024-12-31"):
        data = self.load_stock(symbol,start,end)
        return self.pre_process_stock(data)