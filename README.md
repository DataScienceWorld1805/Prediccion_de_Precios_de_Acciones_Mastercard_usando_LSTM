# Predicción de Precios de Acciones Mastercard usando Red Neuronal (LSTM)

Este proyecto implementa un modelo de red neuronal LSTM (Long Short-Term Memory) para predecir los precios de cierre de las acciones de Mastercard. Incluye un análisis completo de indicadores técnicos, evaluación del modelo con múltiples métricas, predicciones futuras y análisis detallados en dos flujos de trabajo diferentes.

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
├── Modelo_Entrenado/                 # Modelo entrenado y herramientas
│   ├── modelo_lstm.keras            # Modelo entrenado guardado
│   ├── scaler.pkl                   # Normalizador guardado
│   ├── metadatos.json               # Configuración y métricas del modelo
│   ├── usar_modelo.py               # Script para usar el modelo entrenado
│   ├── analizar_predicciones.py     # Script de análisis avanzado
│   ├── predicciones_generadas.csv   # Predicciones generadas
│   ├── predicciones_modelo.png      # Gráfico de predicciones
│   ├── analisis_predicciones.png    # Gráficos de análisis (9 gráficos)
│   └── informe_predicciones.txt     # Informe de texto detallado
│
├── Analisis_de_Predicciones/        # Análisis básico (generado por script principal)
│   ├── analisis_predicciones.py     # Script de análisis básico
│   ├── analisis_predicciones.png    # Gráficos de análisis (6 gráficos)
│   └── predicciones_futuras.csv     # Predicciones generadas por script principal
│
├── stock_price_prediction_lstm.py   # Script principal de entrenamiento
├── prediccion_mastercard_lstm.png   # Gráficos principales del modelo (6 gráficos)
├── requirements.txt                 # Dependencias del proyecto
└── README.md                        # Este archivo
```

## 📊 Flujos de Trabajo

Este proyecto ofrece **dos flujos de trabajo diferentes** según tus necesidades:

### 🔄 Flujo 1: Entrenamiento Completo + Análisis Básico

Este flujo entrena el modelo desde cero y genera un análisis básico de las predicciones.

#### Paso 1: Entrenar el Modelo

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
7. ✅ Guardado del modelo en `Modelo_Entrenado/`
8. ✅ Generación de visualizaciones completas

**Archivos generados:**
- `prediccion_mastercard_lstm.png`: Gráficos con 6 visualizaciones:
  - Pérdida del modelo durante entrenamiento
  - Predicciones vs valores reales (test set)
  - Últimos 200 días + predicciones futuras
  - RSI (últimos 500 días)
  - MACD (últimos 500 días)
  - Precio con medias móviles (últimos 500 días)
- `Modelo_Entrenado/modelo_lstm.keras`: Modelo entrenado
- `Modelo_Entrenado/scaler.pkl`: Normalizador guardado
- `Modelo_Entrenado/metadatos.json`: Metadatos del modelo
- `Analisis_de_Predicciones/predicciones_futuras.csv`: Predicciones para los próximos 30 días

#### Paso 2: Análisis Básico de Predicciones

Después de ejecutar el script principal, ejecutar el análisis básico:

```bash
python Analisis_de_Predicciones/analisis_predicciones.py
```

Este script realiza un análisis básico de las predicciones:
1. ✅ Estadísticas descriptivas (precio inicial, final, máximo, mínimo, promedio, mediana)
2. ✅ Análisis de tendencia (alcista/bajista, velocidad de cambio)
3. ✅ Comparación con datos históricos (últimos 30 días)
4. ✅ Análisis de riesgo (drawdown máximo, rango de precios, rachas)
5. ✅ Proyecciones adicionales (por semana)
6. ✅ Visualizaciones (6 gráficos)

**Archivos generados:**
- `Analisis_de_Predicciones/analisis_predicciones.png`: Gráficos de análisis con 6 visualizaciones

---

### 🔄 Flujo 2: Uso del Modelo Entrenado + Análisis Avanzado

Este flujo permite usar un modelo ya entrenado (sin necesidad de reentrenar) y genera un análisis más completo.

#### Paso 1: Usar el Modelo Entrenado

Si ya tienes un modelo entrenado en `Modelo_Entrenado/`, puedes generar nuevas predicciones:

```bash
python Modelo_Entrenado/usar_modelo.py
```

Este script:
1. ✅ Carga el modelo entrenado desde `modelo_lstm.keras`
2. ✅ Carga el scaler y metadatos guardados
3. ✅ Genera predicciones para los próximos 30 días
4. ✅ Guarda las predicciones en `predicciones_generadas.csv`
5. ✅ Genera visualización de predicciones

**Archivos generados:**
- `Modelo_Entrenado/predicciones_generadas.csv`: Predicciones para los próximos 30 días
- `Modelo_Entrenado/predicciones_modelo.png`: Gráfico de predicciones

**Parámetros configurables en `usar_modelo.py`:**
```python
hacer_predicciones(
    dias_futuros=30,  # Número de días a predecir
    datos_csv='../Datasets_Mastercard/Mastercard_stock_history.csv',  # Ruta a datos
    mostrar_grafico=True  # Mostrar gráfico
)
```

#### Paso 2: Análisis Avanzado de Predicciones

Ejecutar el análisis avanzado (más completo que el básico):

```bash
python Modelo_Entrenado/analizar_predicciones.py
```

Este script realiza un análisis exhaustivo de las predicciones:
1. ✅ Estadísticas descriptivas completas
2. ✅ Análisis de tendencia detallado
3. ✅ Comparación con datos históricos
4. ✅ Análisis de riesgo (drawdown, volatilidad, rachas)
5. ✅ Análisis de volatilidad detallado (top días más volátiles)
6. ✅ Proyecciones por semana
7. ✅ Visualizaciones avanzadas (9 gráficos)
8. ✅ Genera informe de texto completo

**Archivos generados:**
- `Modelo_Entrenado/analisis_predicciones.png`: Gráficos de análisis con 9 visualizaciones:
  1. Predicciones con tendencia y bandas
  2. Cambios diarios
  3. Comparación con histórico
  4. Distribución de cambios porcentuales
  5. Drawdown
  6. Precio con bandas de volatilidad
  7. Proyección semanal
  8. Métricas comparativas
  9. Resumen de cambios porcentuales
- `Modelo_Entrenado/informe_predicciones.txt`: Informe de texto detallado con todas las métricas

---

## 🔍 Comparación de Flujos

| Característica | Flujo 1: Análisis Básico | Flujo 2: Análisis Avanzado |
|----------------|-------------------------|----------------------------|
| **Ubicación** | `Analisis_de_Predicciones/` | `Modelo_Entrenado/` |
| **Requiere entrenamiento** | Sí (primero ejecutar script principal) | No (usa modelo ya entrenado) |
| **Gráficos generados** | 6 gráficos | 9 gráficos |
| **Informe de texto** | No | Sí |
| **Análisis de volatilidad** | Básico | Detallado (top días) |
| **Información del modelo** | No incluye métricas | Incluye RMSE, MAE, MAPE |
| **Complejidad del código** | Script simple | Código modular con funciones |
| **Reutilización del modelo** | No | Sí (puedes usar modelo guardado) |

### ¿Cuándo usar cada flujo?

**Usa Flujo 1** si:
- Quieres entrenar el modelo desde cero
- Necesitas un análisis rápido y básico
- Es la primera vez que ejecutas el proyecto

**Usa Flujo 2** si:
- Ya tienes un modelo entrenado
- Necesitas un análisis más completo y detallado
- Quieres generar nuevas predicciones sin reentrenar
- Necesitas un informe de texto con todas las métricas

## 📈 Métricas del Modelo

El modelo se evalúa usando las siguientes métricas:

- **RMSE** (Root Mean Squared Error): Error cuadrático medio en raíz
- **MAE** (Mean Absolute Error): Error absoluto medio
- **MAPE** (Mean Absolute Percentage Error): Error porcentual absoluto medio

Estas métricas se calculan sobre el conjunto de prueba (20% de los datos) y se muestran tanto en consola como en los gráficos. También se guardan en `Modelo_Entrenado/metadatos.json` para referencia futura.

## ⚙️ Parámetros Configurables

### Parámetros Principales (en `stock_price_prediction_lstm.py`)

```python
SEQUENCE_LENGTH = 60    # Días históricos para predecir (default: 60)
PREDICTION_DAYS = 30    # Días futuros a predecir (default: 30)
TEST_SIZE = 0.2         # Porcentaje de datos para testing (default: 0.2)
```

### Parámetros del Modelo

```python
# Arquitectura LSTM
LSTM_UNITS = 50         # Unidades en cada capa LSTM
DROPOUT_RATE = 0.2      # Tasa de dropout
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

### Análisis Básico (`Analisis_de_Predicciones/analisis_predicciones.png`)
1. **Predicciones con Tendencia**: Predicciones futuras con línea de tendencia (MA 5)
2. **Cambios Diarios**: Barras de cambios diarios (verde/rojo)
3. **Comparación Histórica**: Últimos 60 días históricos vs 30 días predichos
4. **Distribución de Cambios**: Histograma de cambios porcentuales diarios
5. **Drawdown**: Análisis de caídas máximas desde el inicio
6. **Proyección Semanal**: Precio proyectado por semana

### Análisis Avanzado (`Modelo_Entrenado/analisis_predicciones.png`)
1. **Predicciones con Tendencia**: Predicciones con bandas de rango y líneas de referencia
2. **Cambios Diarios**: Barras de cambios diarios con línea de promedio
3. **Comparación Histórica**: Últimos 60 días históricos vs predicciones
4. **Distribución de Cambios**: Histograma de cambios porcentuales diarios
5. **Drawdown**: Análisis de caídas máximas (en porcentaje)
6. **Bandas de Volatilidad**: Precio con bandas de ±1 y ±2 desviaciones estándar
7. **Proyección Semanal**: Precio promedio por semana
8. **Métricas Comparativas**: Gráfico de barras con máx, mín, promedio y mediana
9. **Resumen de Cambios**: Cambios porcentuales totales y comparativos

### Uso del Modelo (`Modelo_Entrenado/predicciones_modelo.png`)
- Gráfico de líneas mostrando últimos 200 días históricos + predicciones futuras
- Línea vertical indicando el inicio de las predicciones

## 🔍 Análisis de Predicciones

### Estadísticas Descriptivas
- Precio inicial y final
- Precio máximo y mínimo
- Precio promedio y mediana
- Desviación estándar
- Cambio total y porcentual
- Volatilidad diaria

### Análisis de Tendencia
- Tipo de tendencia (alcista/bajista/lateral)
- Velocidad de cambio diario
- Día con mayor cambio
- Máximas rachas alcistas/bajistas

### Análisis de Riesgo
- Máximo drawdown (caída máxima desde el inicio)
- Rango de precios
- Máxima racha bajista/alcista
- Volatilidad diaria y coeficiente de variación

### Proyecciones Temporales
- Proyección por semana (semana 1, 2, 4)
- Comparación con datos históricos
- Diferencia vs último precio conocido

### Informe de Texto (Solo Flujo 2)
El análisis avanzado genera un informe de texto completo (`informe_predicciones.txt`) que incluye:
1. Información del modelo (métricas de rendimiento)
2. Estadísticas descriptivas completas
3. Análisis de cambios detallado
4. Comparación con último precio conocido
5. Análisis de tendencia
6. Análisis de volatilidad (incluyendo top 5 días más volátiles)
7. Resumen ejecutivo con recomendaciones

## 📝 Notas Importantes

- ⚠️ **Datos históricos**: El modelo utiliza datos desde 2006 hasta 2025
- ⚠️ **Predicciones**: Las predicciones son estimaciones basadas en patrones históricos y **NO deben usarse como único criterio para decisiones de inversión**
- ⚠️ **Rendimiento**: El rendimiento del modelo puede variar según las condiciones del mercado
- ⚠️ **Limitaciones**: Los modelos de predicción de series temporales tienen limitaciones inherentes y no pueden predecir eventos imprevistos o cambios estructurales en el mercado
- ⚠️ **Uso responsable**: Este proyecto es para fines educativos y de investigación
- ⚠️ **Modelo guardado**: Una vez entrenado, el modelo se guarda en `Modelo_Entrenado/` y puede reutilizarse sin necesidad de reentrenar
- ⚠️ **Archivos generados**: Asegúrate de tener espacio en disco, ya que se generan varios archivos de imagen y datos

## 📚 Referencias

- **Dataset**: [Mastercard Stock Data - Latest and Updated](https://www.kaggle.com/datasets/kalilurrahman/mastercard-stock-data-latest-and-updated)
- **LSTM**: Long Short-Term Memory networks para series temporales
- **Indicadores Técnicos**: RSI, MACD, Moving Averages
- **TensorFlow/Keras**: Framework de deep learning
- **Scikit-learn**: Biblioteca de machine learning

## 📄 Licencia

Este proyecto es de código abierto y está disponible para uso educativo y de investigación.
---
**Desarrollado para análisis de predicción de precios de acciones usando Deep Learning**
