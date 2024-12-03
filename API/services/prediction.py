import numpy as np
import torch

def make_prediction(stock_data, model, scaler, device):
    """
    Realiza previsões usando o modelo LSTM e os dados fornecidos.

    Args:
        stock_data (list): Lista de dados contendo as features de entrada.
        model: Modelo LSTM carregado.
        scaler: Scaler treinado para normalização.
        device: Dispositivo ('cuda' ou 'cpu').

    Returns:
        list: Lista de previsões revertidas para a escala original.
    """
    try:
        # Processar os dados de entrada
        input_data = np.array([[d.abertura, d.maxima, d.minima, d.volume] for d in stock_data])
        scaled_input = scaler.transform(input_data)

        # Verificar o tamanho mínimo necessário para gerar sequências
        n_past = 20
        if len(scaled_input) < n_past:
            raise ValueError("Mínimo de 20 entradas é necessário para previsão.")

        # Criar sequências a partir dos dados escalados
        sequences = [
            scaled_input[i:i + n_past] for i in range(len(scaled_input) - n_past + 1)
        ]
        sequences = torch.tensor(np.array(sequences), dtype=torch.float32).to(device)

        # Fazer a previsão com o modelo
        model.eval()
        with torch.no_grad():
            predictions = model(sequences)

        # Reverter a escala para os valores originais
        # Adiciona zeros temporários para completar as features esperadas pelo scaler
        dummy_features = np.zeros((predictions.shape[0], 4))
        scaled_predictions = np.hstack((dummy_features, predictions.cpu().numpy()))
        inverse_scaled_predictions = scaler.inverse_transform(scaled_predictions)

        # Retorna apenas os valores de interesse (última coluna, que é a previsão)
        return inverse_scaled_predictions[:, -1].tolist()

    except ValueError as ve:
        raise ve
    except Exception as e:
        raise RuntimeError(f"Erro inesperado em 'make_prediction': {e}")
