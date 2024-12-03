import torch
import torch.nn as nn
import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def evaluate_model(self, model, criterion, test_loader, device):
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                outputs = np.reshape(np.repeat(outputs, 5, axis=-1), (len(outputs), 5))[:, 0]
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        average_test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {average_test_loss:.4f}")
        mlflow.log_metric("test_loss", average_test_loss)