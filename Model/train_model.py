import data_importer
import data_creator
import lstm_model
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

SEED = 42
sequence_lenth = 20
input_length = 1
importer = data_importer.ImportStockData()
creator = data_creator.CreateLTSMData()
#HyperParams
hidden_size = 100     # Number of hidden units in the LSTM
num_layers = 3       # Number of LSTM layers
output_size = 1      # Number of output units (e.g., regression output)
num_epochs = 150
batch_size = 64
learning_rate = 0.001

stock_data = importer.get_stock_data(symbol="PETR4.SA",start="2020-01-01",end="2024-12-31")
x_train, x_test, y_train, y_test = creator.build_data(stock_data,sequence_lenth,test_size=0.33, seed=SEED)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#Usa a GPU, caso disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = lstm_model.LSTM(input_length, hidden_size, num_layers, output_size,device).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

mlflow.set_experiment("LSTM Yfinance experiment")
with mlflow.start_run():
    # Log model parameters   
    mlflow.log_param("input_size", input_length)
    mlflow.log_param("hidden_size", hidden_size)
    mlflow.log_param("num_layers", num_layers)
    mlflow.log_param("output_size", output_size)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)           
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Log metrics every 100 batches
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                mlflow.log_metric("train_loss", running_loss / (i+1), step=epoch * len(train_loader) + i)

    # Save the model
    mlflow.pytorch.log_model(model, "lstm_yfinance_data_model")

    # Evaluate the model
    model.evaluate_model(model, criterion, test_loader,device)
