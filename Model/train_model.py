import data_importer
import data_creator
import lstm_model

SEED = 42
sequence_lenth = 20
importer = data_importer.ImportStockData()
creator = data_creator.CreateLTSMData()

stock_data = importer.get_stock_data(symbol="PETR4.SA",start="2020-01-01",end="2024-12-31")
X_train, X_test, y_train, y_test = creator.build_data(stock_data,sequence_lenth,test_size=0.33, seed=SEED)

print(stock_data)