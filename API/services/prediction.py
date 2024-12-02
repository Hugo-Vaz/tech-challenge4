import numpy as np
import torch

def make_prediction(stock_data, model, scaler, device):
    # Processar os dados de entrada
    input_data = np.array([[d.abertura, d.maxima, d.minima, d.volume] for d in stock_data])
    scaled_input = scaler.transform(input_data)

    # Gerar sequências
    n_past = 20
    if len(scaled_input) < n_past:
        raise ValueError("Mínimo de 20 entradas é necessário para previsão.")

    sequences = [
        scaled_input[i:i + n_past] for i in range(len(scaled_input) - n_past + 1)
    ]
    sequences = torch.tensor(np.array(sequences), dtype=torch.float32).to(device)

    # Fazer a previsão
    model.eval()
    with torch.no_grad():
        predictions = model(sequences)

    # Reverter a escala
    inverse_scaled_predictions = scaler.inverse_transform(
        np.hstack((np.zeros((predictions.shape[0], 4)), predictions.cpu().numpy()))
    )
    return inverse_scaled_predictions[:, -1].tolist()
