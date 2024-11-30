import numpy as np
from sklearn.model_selection import train_test_split

class CreateLTSMData:
    def __init__(self):
        pass

    def build_data(self,scaled_stock_data,sequence_length, test_size, seed):        
        x, y = [], []
        for i in range(len(scaled_stock_data) - sequence_length):
            x.append(scaled_stock_data[i:(i + sequence_length), 0])
            y.append(scaled_stock_data[i + sequence_length, 0])
        return self.train_test_split(np.array(x), np.array(y),test_size,seed)
    
    def train_test_split(self,x,y,test_size,seed):
        return train_test_split(x, y, test_size=test_size, random_state=seed)
    
    