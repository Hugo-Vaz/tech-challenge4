import numpy as np
import torch

class CreateLTSMData:
    def __init__(self):
        pass

    #Aqui de fato prepararemos os nossos dados para ser usado no modelo, tanto pra o treino quanto para o teste
    #scaled_stock são os ticks/dados da bolsa da empresa/symbol escolhida
    #n_past é o números de vezes/camadas que iremos olhar no passado para prever o proximo alvo, por exemplo se n_past é 20, 
    #olharemos os dados de x 0 - 20 para prever y 21
    #test_size é o percentual que iremos usar para teste
    def build_data(self,scaled_stock_data,n_past, test_size):        
        #pegaremos o percentual que usaremos, pra teste, encontraremos qual valor isso representa em nosso dataframe
        #assim separaremos em dois novos dataframes de treino e teste
        test_split=round(len(scaled_stock_data)*test_size)
        train_data = scaled_stock_data[:test_split]
        test_data = scaled_stock_data[test_split:]
        x_train,y_train = self.split_features_test(train_data,n_past)
        x_test,y_test = self.split_features_test(test_data,n_past)

        return x_train,x_test,y_train,y_test

    
    #Aqui criaremos de fato, os dados que serão utilizados no modelo dataX conterá todas as features (High, Low, Open, Volume and Close)
    #Enquanto dataY conterá apenas o target/alvo (Close)
    #baseado no n_past, o X será um "array de arrays" com o tamanho do dataframe, onde cada iteração, serão n_past entradas com n_features como colunas
    def split_features_test(self,dataframe, n_past):
        dataX = []
        dataY = []
        for i in range(n_past, len(dataframe)):
            dataX.append(dataframe[i-n_past:i, 1:dataframe.shape[1]])
            dataY.append(dataframe[i,1])
    
        return torch.tensor(np.array(dataX),dtype=torch.float32),torch.tensor(np.array(dataY),dtype=torch.float32)       
    