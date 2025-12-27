"""
Predicción de Precios de Acciones Mastercard usando LSTM
Incluye indicadores técnicos: RSI, MACD, medias móviles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración
SEQUENCE_LENGTH = 60  # Días históricos para predecir
PREDICTION_DAYS = 30  # Días futuros a predecir
TEST_SIZE = 0.2  # 20% para testing

print("=" * 60)
print("PREDICCIÓN DE PRECIOS DE ACCIONES MASTERCARD - LSTM")
print("=" * 60)

# ============================================================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# ============================================================================
print("\n[1/6] Cargando datos...")
df = pd.read_csv('Datasets_Mastercard/Mastercard_stock_history.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"   [OK] Datos cargados: {len(df)} registros")
print(f"   [OK] Periodo: {df['Date'].min().date()} a {df['Date'].max().date()}")

# ============================================================================
# 2. CÁLCULO DE INDICADORES TÉCNICOS
# ============================================================================
print("\n[2/6] Calculando indicadores técnicos...")

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

# Eliminar filas con NaN (debido a cálculos de medias móviles)
df = df.dropna().reset_index(drop=True)

print(f"   [OK] Indicadores calculados: RSI, MACD, 5 medias moviles, volatilidad")
print(f"   [OK] Datos despues de limpieza: {len(df)} registros")

# ============================================================================
# 3. PREPARACIÓN DE DATOS PARA LSTM
# ============================================================================
print("\n[3/6] Preparando datos para LSTM...")

# Seleccionar features
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                   'MA_7', 'MA_14', 'MA_30', 'MA_50', 'MA_200',
                   'Price_Change', 'Volatility']

# Crear dataset con features
data = df[feature_columns].values

# Normalizar datos
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Crear secuencias para LSTM
def create_sequences(data, seq_length, prediction_days=1):
    X, y = [], []
    for i in range(seq_length, len(data) - prediction_days + 1):
        X.append(data[i-seq_length:i])
        y.append(data[i:i+prediction_days, 3])  # Columna 3 es 'Close'
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, SEQUENCE_LENGTH, 1)

# Dividir en train y test
split_idx = int(len(X) * (1 - TEST_SIZE))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"   [OK] Secuencias creadas: {len(X)} secuencias de {SEQUENCE_LENGTH} dias")
print(f"   [OK] Train: {len(X_train)} secuencias")
print(f"   [OK] Test: {len(X_test)} secuencias")
print(f"   [OK] Features: {X.shape[2]} caracteristicas")

# ============================================================================
# 4. CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO LSTM
# ============================================================================
print("\n[4/6] Construyendo y entrenando modelo LSTM...")

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(feature_columns))),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

# Entrenar modelo
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("   [OK] Modelo entrenado exitosamente")

# ============================================================================
# 5. EVALUACIÓN DEL MODELO
# ============================================================================
print("\n[5/6] Evaluando modelo...")

# Predicciones en test set
y_pred = model.predict(X_test)

# Desnormalizar predicciones y valores reales
# Necesitamos crear arrays completos para desnormalizar correctamente
y_test_denorm = np.zeros((len(y_test), len(feature_columns)))
y_test_denorm[:, 3] = y_test.flatten()  # Columna Close
y_test_denorm = scaler.inverse_transform(y_test_denorm)[:, 3]

y_pred_denorm = np.zeros((len(y_pred), len(feature_columns)))
y_pred_denorm[:, 3] = y_pred.flatten()
y_pred_denorm = scaler.inverse_transform(y_pred_denorm)[:, 3]

# Métricas
mse = mean_squared_error(y_test_denorm, y_pred_denorm)
mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test_denorm - y_pred_denorm) / y_test_denorm)) * 100

print(f"\n   Métricas de evaluación:")
print(f"   • RMSE: ${rmse:.2f}")
print(f"   • MAE: ${mae:.2f}")
print(f"   • MAPE: {mape:.2f}%")

# ============================================================================
# 6. PREDICCIÓN PARA PRÓXIMOS DÍAS
# ============================================================================
print(f"\n[6/6] Prediciendo próximos {PREDICTION_DAYS} días...")

def predict_future_days(model, last_sequence, scaler, feature_columns, days=PREDICTION_DAYS):
    """Predice los próximos días usando la última secuencia"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # Predecir siguiente día
        next_pred = model.predict(current_sequence.reshape(1, SEQUENCE_LENGTH, len(feature_columns)), verbose=0)
        
        # Desnormalizar solo el precio Close
        pred_array = np.zeros((1, len(feature_columns)))
        pred_array[0, 3] = next_pred[0, 0]  # Close está en índice 3
        
        # Para desnormalizar, necesitamos valores aproximados de otras features
        # Usaremos los valores del último día de la secuencia
        last_day = current_sequence[-1].copy()
        pred_array[0, :] = last_day
        pred_array[0, 3] = next_pred[0, 0]  # Actualizar Close
        
        # Desnormalizar
        pred_denorm = scaler.inverse_transform(pred_array)
        close_price = pred_denorm[0, 3]
        predictions.append(close_price)
        
        # Actualizar secuencia: agregar predicción y eliminar primer día
        # Estimar otros valores basándose en la tendencia
        new_day = last_day.copy()
        new_day[3] = next_pred[0, 0]  # Close predicho
        # Estimar Open, High, Low basándose en Close
        price_change = (next_pred[0, 0] - last_day[3]) / last_day[3] if last_day[3] > 0 else 0
        new_day[0] = last_day[3]  # Open ≈ Close anterior
        new_day[1] = next_pred[0, 0] * (1 + abs(price_change) * 0.5)  # High
        new_day[2] = next_pred[0, 0] * (1 - abs(price_change) * 0.5)  # Low
        
        # Actualizar secuencia
        current_sequence = np.vstack([current_sequence[1:], new_day])
    
    return np.array(predictions)

# Obtener última secuencia
last_sequence = data_scaled[-SEQUENCE_LENGTH:]

# Predecir
future_predictions = predict_future_days(model, last_sequence, scaler, feature_columns, PREDICTION_DAYS)

# Crear fechas futuras
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=PREDICTION_DAYS, freq='D')

print(f"   [OK] Predicciones generadas para {PREDICTION_DAYS} dias")
print(f"\n   Ultimo precio conocido: ${df['Close'].iloc[-1]:.2f}")
print(f"   Prediccion dia 1: ${future_predictions[0]:.2f}")
print(f"   Prediccion dia {PREDICTION_DAYS}: ${future_predictions[-1]:.2f}")
print(f"   Cambio proyectado: {((future_predictions[-1] / df['Close'].iloc[-1]) - 1) * 100:.2f}%")

# ============================================================================
# Guardar modelo y componentes necesarios en Modelo_Entrenado/
# ============================================================================
print("\n[Guardando modelo en Modelo_Entrenado/...]")
os.makedirs('Modelo_Entrenado', exist_ok=True)

# Guardar modelo
model_path = 'Modelo_Entrenado/modelo_lstm.keras'
model.save(model_path)
print(f"   [OK] Modelo guardado en '{model_path}'")

# Guardar scaler
scaler_path = 'Modelo_Entrenado/scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"   [OK] Scaler guardado en '{scaler_path}'")

# Guardar metadatos
metadata = {
    'feature_columns': feature_columns,
    'sequence_length': SEQUENCE_LENGTH,
    'prediction_days': PREDICTION_DAYS,
    'test_size': TEST_SIZE,
    'rmse': float(rmse),
    'mae': float(mae),
    'mape': float(mape),
    'last_date': last_date.strftime('%Y-%m-%d'),
    'last_price': float(df['Close'].iloc[-1])
}

metadata_path = 'Modelo_Entrenado/metadatos.json'
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)
print(f"   [OK] Metadatos guardados en '{metadata_path}'")

# ============================================================================
# 7. VISUALIZACIONES
# ============================================================================
print("\n[7/7] Generando visualizaciones...")

fig = plt.figure(figsize=(20, 12))

# Gráfico 1: Historial de entrenamiento
plt.subplot(2, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Perdida del Modelo durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Perdida (MSE)')
plt.legend()
plt.grid(True)

# Gráfico 2: Predicciones vs Valores Reales (Test Set)
plt.subplot(2, 3, 2)
test_dates = df['Date'].iloc[split_idx + SEQUENCE_LENGTH:split_idx + SEQUENCE_LENGTH + len(y_test)]
plt.plot(test_dates, y_test_denorm, label='Valores Reales', linewidth=2)
plt.plot(test_dates, y_pred_denorm, label='Predicciones', linewidth=2, alpha=0.7)
plt.title(f'Predicciones vs Valores Reales (Test Set)\nRMSE: ${rmse:.2f} | MAE: ${mae:.2f}')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# Gráfico 3: Últimos 200 días + Predicciones Futuras
plt.subplot(2, 3, 3)
last_200_days = 200
recent_dates = df['Date'].iloc[-last_200_days:]
recent_prices = df['Close'].iloc[-last_200_days:]

plt.plot(recent_dates, recent_prices, label='Historial', linewidth=2, color='blue')
plt.plot(future_dates, future_predictions, label='Predicciones Futuras', 
         linewidth=2, color='red', linestyle='--', marker='o', markersize=4)
plt.axvline(x=last_date, color='green', linestyle=':', linewidth=2, label='Inicio Predicción')
plt.title(f'Ultimos {last_200_days} Dias + Predicciones ({PREDICTION_DAYS} dias)')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# Gráfico 4: Indicadores Técnicos - RSI
plt.subplot(2, 3, 4)
plt.plot(df['Date'].iloc[-500:], df['RSI'].iloc[-500:], color='purple', linewidth=1.5)
plt.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Sobrecompra (70)')
plt.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Sobreventa (30)')
plt.title('RSI (Relative Strength Index) - Ultimos 500 dias')
plt.xlabel('Fecha')
plt.ylabel('RSI')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# Gráfico 5: Indicadores Técnicos - MACD
plt.subplot(2, 3, 5)
plt.plot(df['Date'].iloc[-500:], df['MACD'].iloc[-500:], label='MACD', linewidth=1.5)
plt.plot(df['Date'].iloc[-500:], df['MACD_Signal'].iloc[-500:], label='Signal', linewidth=1.5)
plt.bar(df['Date'].iloc[-500:], df['MACD_Hist'].iloc[-500:], label='Histogram', alpha=0.3)
plt.title('MACD - Ultimos 500 dias')
plt.xlabel('Fecha')
plt.ylabel('MACD')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# Gráfico 6: Precio con Medias Móviles
plt.subplot(2, 3, 6)
plt.plot(df['Date'].iloc[-500:], df['Close'].iloc[-500:], label='Precio Close', linewidth=2, color='black')
plt.plot(df['Date'].iloc[-500:], df['MA_7'].iloc[-500:], label='MA 7', linewidth=1, alpha=0.7)
plt.plot(df['Date'].iloc[-500:], df['MA_14'].iloc[-500:], label='MA 14', linewidth=1, alpha=0.7)
plt.plot(df['Date'].iloc[-500:], df['MA_30'].iloc[-500:], label='MA 30', linewidth=1, alpha=0.7)
plt.plot(df['Date'].iloc[-500:], df['MA_50'].iloc[-500:], label='MA 50', linewidth=1, alpha=0.7)
plt.title('Precio con Medias Moviles - Ultimos 500 dias')
plt.xlabel('Fecha')
plt.ylabel('Precio (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('prediccion_mastercard_lstm.png', dpi=300, bbox_inches='tight')
print("   [OK] Graficos guardados en 'prediccion_mastercard_lstm.png'")

# Guardar predicciones en CSV
predictions_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Close': future_predictions
})
predictions_df.to_csv('Analisis_de_Predicciones/predicciones_futuras.csv', index=False)
print("   [OK] Predicciones guardadas en 'Analisis_de_Predicciones/predicciones_futuras.csv'")

# Resumen final
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"[OK] Modelo LSTM entrenado con {len(feature_columns)} features")
print(f"[OK] RMSE: ${rmse:.2f} | MAE: ${mae:.2f} | MAPE: {mape:.2f}%")
print(f"[OK] Predicciones generadas para {PREDICTION_DAYS} dias futuros")
print(f"[OK] Ultimo precio: ${df['Close'].iloc[-1]:.2f}")
print(f"[OK] Precio predicho (dia {PREDICTION_DAYS}): ${future_predictions[-1]:.2f}")
print("=" * 60)

plt.show()

