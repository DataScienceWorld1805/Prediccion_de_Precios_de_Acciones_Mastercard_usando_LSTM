"""
Script para usar el modelo LSTM entrenado para hacer predicciones
de precios de acciones Mastercard.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import load_model
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FUNCIONES PARA CALCULAR INDICADORES TÉCNICOS
# ============================================================================

def calculate_rsi(prices, period=14):
    """Calcula el RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calcula el MACD (Moving Average Convergence Divergence)"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def prepare_data(df, feature_columns):
    """Prepara los datos calculando indicadores técnicos"""
    # Calcular indicadores
    df['RSI'] = calculate_rsi(df['Close'], period=14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
    
    # Medias móviles
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_14'] = df['Close'].rolling(window=14).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # Indicadores adicionales
    df['Price_Change'] = df['Close'].pct_change()
    df['Volatility'] = df['Price_Change'].rolling(window=14).std()
    
    # Eliminar filas con NaN
    df = df.dropna().reset_index(drop=True)
    
    # Seleccionar features
    data = df[feature_columns].values
    
    return df, data

def predict_future_days(model, last_sequence, scaler, feature_columns, sequence_length, days=30):
    """Predice los próximos días usando la última secuencia"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # Predecir siguiente día
        next_pred = model.predict(current_sequence.reshape(1, sequence_length, len(feature_columns)), verbose=0)
        
        # Para desnormalizar, necesitamos valores aproximados de otras features
        # Usaremos los valores del último día de la secuencia
        last_day = current_sequence[-1].copy()
        pred_array = np.zeros((1, len(feature_columns)))
        pred_array[0, :] = last_day
        pred_array[0, 3] = next_pred[0, 0]  # Actualizar Close (índice 3)
        
        # Desnormalizar
        pred_denorm = scaler.inverse_transform(pred_array)
        close_price = pred_denorm[0, 3]
        predictions.append(close_price)
        
        # Actualizar secuencia: agregar predicción y eliminar primer día
        new_day = last_day.copy()
        new_day[3] = next_pred[0, 0]  # Close predicho
        # Estimar otros valores basándose en la tendencia
        price_change = (next_pred[0, 0] - last_day[3]) / last_day[3] if last_day[3] > 0 else 0
        new_day[0] = last_day[3]  # Open ≈ Close anterior
        new_day[1] = next_pred[0, 0] * (1 + abs(price_change) * 0.5)  # High
        new_day[2] = next_pred[0, 0] * (1 - abs(price_change) * 0.5)  # Low
        
        # Actualizar secuencia
        current_sequence = np.vstack([current_sequence[1:], new_day])
    
    return np.array(predictions)

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def hacer_predicciones(dias_futuros=30, datos_csv='../Datasets_Mastercard/Mastercard_stock_history.csv', 
                       mostrar_grafico=True):
    """
    Carga el modelo entrenado y hace predicciones para los próximos días.
    
    Parámetros:
    -----------
    dias_futuros : int
        Número de días futuros a predecir (default: 30)
    datos_csv : str
        Ruta al archivo CSV con datos históricos (default: '../Datasets_Mastercard/Mastercard_stock_history.csv')
    mostrar_grafico : bool
        Si mostrar el gráfico de predicciones (default: True)
    
    Returns:
    --------
    tuple: (predicciones, fechas_futuras, ultimo_precio, metadatos)
    """
    
    print("=" * 60)
    print("USANDO MODELO LSTM ENTRENADO - PREDICCIÓN MASTERCARD")
    print("=" * 60)
    
    # Obtener directorio del script actual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Verificar que existen los archivos necesarios
    modelo_path = os.path.join(script_dir, 'modelo_lstm.keras')
    scaler_path = os.path.join(script_dir, 'scaler.pkl')
    metadata_path = os.path.join(script_dir, 'metadatos.json')
    
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"No se encuentra el modelo en '{modelo_path}'. "
                              "Asegúrate de ejecutar primero el script de entrenamiento.")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No se encuentra el scaler en '{scaler_path}'. "
                              "Asegúrate de ejecutar primero el script de entrenamiento.")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No se encuentran los metadatos en '{metadata_path}'. "
                              "Asegúrate de ejecutar primero el script de entrenamiento.")
    
    # Cargar modelo
    print("\n[1/4] Cargando modelo entrenado...")
    model = load_model(modelo_path)
    print(f"   [OK] Modelo cargado desde '{modelo_path}'")
    
    # Cargar scaler
    print("\n[2/4] Cargando scaler...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"   [OK] Scaler cargado desde '{scaler_path}'")
    
    # Cargar metadatos
    print("\n[3/4] Cargando metadatos...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    feature_columns = metadata['feature_columns']
    sequence_length = metadata['sequence_length']
    print(f"   [OK] Metadatos cargados")
    print(f"   • Sequence Length: {sequence_length}")
    print(f"   • Features: {len(feature_columns)}")
    print(f"   • Métricas del modelo entrenado:")
    print(f"     - RMSE: ${metadata['rmse']:.2f}")
    print(f"     - MAE: ${metadata['mae']:.2f}")
    print(f"     - MAPE: {metadata['mape']:.2f}%")
    
    # Cargar y preparar datos
    print(f"\n[4/4] Cargando y preparando datos desde '{datos_csv}'...")
    
    # Si la ruta es relativa, intentar desde el directorio del script primero
    if not os.path.isabs(datos_csv) and not os.path.exists(datos_csv):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        possible_path = os.path.join(parent_dir, 'Datasets_Mastercard', 'Mastercard_stock_history.csv')
        if os.path.exists(possible_path):
            datos_csv = possible_path
    
    if not os.path.exists(datos_csv):
        raise FileNotFoundError(f"No se encuentra el archivo de datos en '{datos_csv}'. "
                              f"Busca el archivo Mastercard_stock_history.csv en Datasets_Mastercard/")
    
    df = pd.read_csv(datos_csv)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Preparar datos con indicadores técnicos
    df_prepared, data = prepare_data(df, feature_columns)
    
    # Normalizar datos
    data_scaled = scaler.transform(data)
    
    # Obtener última secuencia
    last_sequence = data_scaled[-sequence_length:]
    
    print(f"   [OK] Datos cargados: {len(df)} registros")
    print(f"   [OK] Periodo: {df['Date'].min().date()} a {df['Date'].max().date()}")
    print(f"   [OK] Último precio conocido: ${df['Close'].iloc[-1]:.2f}")
    
    # Hacer predicciones
    print(f"\n[PREDICIENDO] Generando predicciones para {dias_futuros} días futuros...")
    future_predictions = predict_future_days(
        model, last_sequence, scaler, feature_columns, sequence_length, days=dias_futuros
    )
    
    # Crear fechas futuras
    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                 periods=dias_futuros, freq='D')
    
    print(f"   [OK] Predicciones generadas")
    print(f"\n   Último precio conocido: ${df['Close'].iloc[-1]:.2f}")
    print(f"   Predicción día 1: ${future_predictions[0]:.2f}")
    print(f"   Predicción día {dias_futuros}: ${future_predictions[-1]:.2f}")
    print(f"   Cambio proyectado: {((future_predictions[-1] / df['Close'].iloc[-1]) - 1) * 100:.2f}%")
    
    # Visualización
    if mostrar_grafico:
        print("\n[VISUALIZACIÓN] Generando gráfico...")
        plt.figure(figsize=(16, 8))
        
        # Últimos 200 días históricos
        last_200_days = min(200, len(df))
        recent_dates = df['Date'].iloc[-last_200_days:]
        recent_prices = df['Close'].iloc[-last_200_days:]
        
        plt.plot(recent_dates, recent_prices, label='Historial', 
                linewidth=2, color='blue')
        plt.plot(future_dates, future_predictions, label='Predicciones Futuras', 
                linewidth=2, color='red', linestyle='--', marker='o', markersize=4)
        plt.axvline(x=last_date, color='green', linestyle=':', linewidth=2, 
                   label='Inicio Predicción')
        plt.title(f'Predicción de Precios Mastercard - Últimos {last_200_days} días + {dias_futuros} días futuros', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Precio de Cierre (USD)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Guardar gráfico
        script_dir = os.path.dirname(os.path.abspath(__file__))
        grafico_path = os.path.join(script_dir, 'predicciones_modelo.png')
        plt.savefig(grafico_path, dpi=300, bbox_inches='tight')
        print(f"   [OK] Gráfico guardado en '{grafico_path}'")
        plt.show()
    
    # Guardar predicciones en CSV
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_predictions
    })
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'predicciones_generadas.csv')
    predictions_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Predicciones guardadas en '{csv_path}'")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN DE PREDICCIÓN")
    print("=" * 60)
    print(f"[OK] Modelo usado exitosamente")
    print(f"[OK] Predicciones generadas para {dias_futuros} días futuros")
    print(f"[OK] Último precio: ${df['Close'].iloc[-1]:.2f}")
    print(f"[OK] Precio predicho (día {dias_futuros}): ${future_predictions[-1]:.2f}")
    print("=" * 60)
    
    return future_predictions, future_dates, df['Close'].iloc[-1], metadata

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Ejecutar predicciones con parámetros por defecto
    # Puedes modificar estos parámetros según necesites
    try:
        predicciones, fechas, ultimo_precio, metadatos = hacer_predicciones(
            dias_futuros=30,
            datos_csv='../Datasets_Mastercard/Mastercard_stock_history.csv',
            mostrar_grafico=True
        )
        
        # Ejemplo de uso adicional: acceder a las predicciones
        print("\n📊 Primeras 5 predicciones:")
        for i in range(min(5, len(predicciones))):
            print(f"   {fechas[i].strftime('%Y-%m-%d')}: ${predicciones[i]:.2f}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nAsegúrate de que:")
        print("  1. El modelo esté entrenado (ejecuta stock_price_prediction_lstm.py)")
        print("  2. Los archivos modelo_lstm.keras, scaler.pkl y metadatos.json existan en Modelo_Entrenado/")
        print("  3. El archivo de datos históricos exista en la ruta especificada")

