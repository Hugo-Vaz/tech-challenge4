
import numpy as np
import torch

def make_prediction(stock_data, model, scaler, device, n_past=20, n_future=1):
    """
    Realiza previsões usando o modelo LSTM e os dados fornecidos.
    
    Args:
        stock_data (list): Lista de dados contendo as features de entrada.
        model: Modelo LSTM carregado.
        scaler: Scaler treinado para normalização.
        device: Dispositivo ('cuda' ou 'cpu').
        n_past (int): Número de registros históricos usados para previsão.
        n_future (int): Número de passos futuros a prever.
    
    Returns:
        list: Lista de previsões revertidas para a escala original.
    """
    try:
        # Processar os dados de entrada
        input_data = np.array([[d.abertura, d.maxima, d.minima, d.volume] for d in stock_data])
        scaled_input = scaler.transform(input_data)
        
        # Garantir que há dados suficientes para a previsão
        if len(scaled_input) < n_past:
            raise ValueError(f"Mínimo de {n_past} entradas é necessário para previsão.")
        
        # Criar sequência inicial
        initial_sequence = scaled_input[-n_past:]
        
        # Prever múltiplos passos futuros
        predictions = []
        current_sequence = initial_sequence
        
        for _ in range(n_future):
            # Adicionar dimensão para prever
            input_tensor = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Realizar previsão
            predicted = model(input_tensor).detach().cpu().numpy()[0]
            
            # Guardar previsão
            predictions.append(predicted)
            
            # Atualizar sequência
            next_input = np.vstack([current_sequence[1:], predicted])
            current_sequence = next_input
        
        # Reverter previsões para escala original
        predictions = np.array(predictions).reshape(-1, 1)
        predictions_scaled = np.repeat(predictions, scaled_input.shape[1], axis=-1)
        predictions_original = scaler.inverse_transform(predictions_scaled)[:, 0]
        
        return predictions_original.tolist()
    
    except Exception as e:
        raise RuntimeError(f"Erro ao fazer previsão: {e}")
