# Predicción de Precios de Acciones Mastercard usando Red Neuronal (LSTM)

Este proyecto implementa un modelo de red neuronal LSTM (Long Short-Term Memory) para predecir los precios de cierre 
de las acciones de Mastercard. Incluye un análisis completo de indicadores técnicos, evaluación del modelo con múltiples 
métricas, y predicciones futuras con análisis detallado.

## 📊 Fuente de Datos

Los datos utilizados en este proyecto provienen del siguiente dataset de Kaggle:

**Dataset:** [Mastercard Stock Data - Latest and Updated](https://www.kaggle.com/datasets/kalilurrahman/mastercard-stock-data-latest-and-updated)

El dataset incluye información histórica completa de Mastercard desde 2006 hasta 2025, incluyendo:
- Precios históricos (Open, High, Low, Close)
- Volumen de transacciones
- Dividendos
- Stock Splits
- Información adicional de la acción

## 📋 Características del Proyecto

### Modelo LSTM
- **Arquitectura**: Red neuronal con 3 capas LSTM y dropout para prevenir overfitting
- **Capas**: 
  - 3 capas LSTM (50 unidades cada una)
  - Dropout (0.2) entre capas
  - Capas densas para la salida
- **Optimización**: Adam optimizer con callbacks (EarlyStopping, ReduceLROnPlateau)
- **Normalización**: MinMaxScaler para todas las features

### Indicadores Técnicos Implementados
- **RSI** (Relative Strength Index) - Periodo 14 días
- **MACD** (Moving Average Convergence Divergence) - Fast: 12, Slow: 26, Signal: 9
- **Medias Móviles**: 5 períodos diferentes (7, 14, 30, 50, 200 días)
- **Volatilidad**: Desviación estándar de cambios porcentuales (14 días)
- **Price Change**: Cambio porcentual diario

### Features Utilizadas
El modelo utiliza 16 características:
- Precios: Open, High, Low, Close
- Volumen
- RSI, MACD, MACD_Signal, MACD_Hist
- 5 Medias Móviles (MA_7, MA_14, MA_30, MA_50, MA_200)
- Price_Change, Volatility

### Predicción
- **Secuencia histórica**: 60 días para predecir el siguiente día
- **Predicción futura**: Hasta 30 días en el futuro
- **Método**: Predicción iterativa usando la última secuencia conocida

## 🚀 Instalación

1. Clonar o descargar el repositorio
2. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

### Requisitos del Sistema
- Python 3.8 o superior
- TensorFlow 2.13+
- Pandas 2.0+
- NumPy 1.24+
- Matplotlib 3.7+
- Scikit-learn 1.3+

## 📁 Estructura del Proyecto

```
MasterCard_Data/
│
├── Datasets_Mastercard/              # Datasets originales
│   ├── Mastercard_stock_history.csv  # Datos históricos principales
│   ├── Mastercard_stock_action.csv
│   ├── Mastercard_stock_dividends.csv
│   ├── Mastercard_stock_info.csv
│   └── Mastercard_stock_splits.csv
│
├── Analisis_de_Predicciones/         # Análisis detallado de predicciones
│   ├── analisis_predicciones.py      # Script de análisis
│   ├── analisis_predicciones.png     # Gráficos de análisis
│   └── predicciones_futuras.csv      # Predicciones generadas
│
├── stock_price_prediction_lstm.py    # Script principal de predicción
├── prediccion_mastercard_lstm.png    # Gráficos principales del modelo
├── requirements.txt                   # Dependencias del proyecto
├── referencia_dataset_kaggle.txt     # Referencia al dataset de Kaggle
└── README.md                          # Este archivo
```

## 📊 Uso

### 1. Predicción Principal

Ejecutar el script principal para entrenar el modelo y generar predicciones:

```bash
python stock_price_prediction_lstm.py
```

Este script realiza:
1. ✅ Carga y preprocesamiento de datos históricos
2. ✅ Cálculo de indicadores técnicos (RSI, MACD, medias móviles, volatilidad)
3. ✅ Preparación de secuencias para LSTM (normalización y creación de ventanas)
4. ✅ Construcción y entrenamiento del modelo LSTM
5. ✅ Evaluación con métricas (RMSE, MAE, MAPE)
6. ✅ Predicción de próximos 30 días
7. ✅ Generación de visualizaciones completas

**Archivos generados:**
- `prediccion_mastercard_lstm.png`: Gráficos con 6 visualizaciones:
  - Pérdida del modelo durante entrenamiento
  - Predicciones vs valores reales (test set)
  - Últimos 200 días + predicciones futuras
  - RSI (últimos 500 días)
  - MACD (últimos 500 días)
  - Precio con medias móviles (últimos 500 días)
- `Analisis_de_Predicciones/predicciones_futuras.csv`: Predicciones para los próximos 30 días

### 2. Análisis Detallado de Predicciones

Después de ejecutar el script principal, ejecutar el análisis detallado:

```bash
python Analisis_de_Predicciones/analisis_predicciones.py
```

Este script realiza un análisis exhaustivo de las predicciones:
1. ✅ Estadísticas descriptivas (precio inicial, final, máximo, mínimo, promedio, mediana)
2. ✅ Análisis de tendencia (alcista/bajista, velocidad de cambio)
3. ✅ Comparación con datos históricos (últimos 30 días)
4. ✅ Análisis de riesgo (drawdown máximo, rango de precios, rachas)
5. ✅ Proyecciones adicionales (por semana)
6. ✅ Visualizaciones detalladas (6 gráficos)

**Archivos generados:**
- `Analisis_de_Predicciones/analisis_predicciones.png`: Gráficos de análisis con 6 visualizaciones:
  - Predicciones con tendencia
  - Cambios diarios
  - Comparación con histórico
  - Distribución de cambios porcentuales
  - Drawdown
  - Proyección semanal

## 📈 Métricas del Modelo

El modelo se evalúa usando las siguientes métricas:

- **RMSE** (Root Mean Squared Error): Error cuadrático medio en raíz
- **MAE** (Mean Absolute Error): Error absoluto medio
- **MAPE** (Mean Absolute Percentage Error): Error porcentual absoluto medio

Estas métricas se calculan sobre el conjunto de prueba (20% de los datos) y se muestran tanto en consola como en los gráficos.

## ⚙️ Parámetros Configurables

En `stock_price_prediction_lstm.py` puedes ajustar:

```python
SEQUENCE_LENGTH = 60    # Días históricos para predecir (default: 60)
PREDICTION_DAYS = 30    # Días futuros a predecir (default: 30)
TEST_SIZE = 0.2         # Porcentaje de datos para testing (default: 0.2)
```

### Parámetros del Modelo

```python
# Arquitectura LSTM
LSTM_UNITS = 50         # Unidades en cada capa LSTM
DROPOUT_RATE = 0.2     # Tasa de dropout
BATCH_SIZE = 32         # Tamaño del batch
EPOCHS = 50             # Número máximo de épocas
```

### Parámetros de Indicadores Técnicos

```python
RSI_PERIOD = 14         # Período para RSI
MACD_FAST = 12          # EMA rápida para MACD
MACD_SLOW = 26          # EMA lenta para MACD
MACD_SIGNAL = 9         # Período de señal para MACD
MA_PERIODS = [7, 14, 30, 50, 200]  # Períodos de medias móviles
```

## 📊 Visualizaciones Generadas

### Script Principal (`prediccion_mastercard_lstm.png`)
1. **Pérdida del Modelo**: Evolución de train/validation loss durante el entrenamiento
2. **Predicciones vs Reales**: Comparación en el conjunto de prueba con métricas
3. **Historial + Predicciones**: Últimos 200 días históricos + 30 días predichos
4. **RSI**: Indicador de fuerza relativa (últimos 500 días)
5. **MACD**: Indicador de convergencia/divergencia (últimos 500 días)
6. **Precio con Medias Móviles**: Precio de cierre con todas las medias móviles

### Script de Análisis (`analisis_predicciones.png`)
1. **Predicciones con Tendencia**: Predicciones futuras con línea de tendencia
2. **Cambios Diarios**: Barras de cambios diarios (verde/rojo)
3. **Comparación Histórica**: Últimos 60 días históricos vs 30 días predichos
4. **Distribución de Cambios**: Histograma de cambios porcentuales diarios
5. **Drawdown**: Análisis de caídas máximas desde el inicio
6. **Proyección Semanal**: Precio proyectado por semana

## 🔍 Análisis de Predicciones

El script de análisis proporciona:

### Estadísticas Descriptivas
- Precio inicial y final
- Precio máximo y mínimo
- Precio promedio y mediana
- Desviación estándar
- Cambio total y porcentual
- Volatilidad diaria

### Análisis de Tendencia
- Tipo de tendencia (alcista/bajista)
- Velocidad de cambio
- Día con mayor cambio

### Análisis de Riesgo
- Máximo drawdown
- Rango de precios
- Máxima racha bajista/alcista

### Proyecciones Temporales
- Proyección por semana (semana 1, 2, 4)
- Comparación con datos históricos

## 📝 Notas Importantes

- ⚠️ **Datos históricos**: El modelo utiliza datos desde 2006 hasta 2025
- ⚠️ **Predicciones**: Las predicciones son estimaciones basadas en patrones históricos y **NO deben usarse como único criterio para decisiones de inversión**
- ⚠️ **Rendimiento**: El rendimiento del modelo puede variar según las condiciones del mercado
- ⚠️ **Limitaciones**: Los modelos de predicción de series temporales tienen limitaciones inherentes y no pueden predecir eventos imprevistos o cambios estructurales en el mercado
- ⚠️ **Uso responsable**: Este proyecto es para fines educativos y de investigación

## 📚 Referencias

- **Dataset**: [Mastercard Stock Data - Latest and Updated](https://www.kaggle.com/datasets/kalilurrahman/mastercard-stock-data-latest-and-updated)
- **LSTM**: Long Short-Term Memory networks para series temporales
- **Indicadores Técnicos**: RSI, MACD, Moving Averages

## 📄 Licencia

Este proyecto es de código abierto y está disponible para uso educativo y de investigación.

---

**Desarrollado para análisis de predicción de precios de acciones usando Deep Learning**
