"""
Análisis detallado de las predicciones futuras de Mastercard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Obtener el directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)  # Directorio raíz del proyecto

print("=" * 70)
print("ANALISIS DETALLADO DE PREDICCIONES FUTURAS - MASTERCARD")
print("=" * 70)

# Cargar predicciones
print("\n[1/6] Cargando predicciones...")
predicciones_path = os.path.join(script_dir, 'predicciones_futuras.csv')
predicciones = pd.read_csv(predicciones_path)
predicciones['Date'] = pd.to_datetime(predicciones['Date'])
predicciones = predicciones.sort_values('Date').reset_index(drop=True)

# Cargar datos históricos para comparación
print("[2/6] Cargando datos historicos...")
historico_path = os.path.join(base_dir, 'Datasets_Mastercard', 'Mastercard_stock_history.csv')
historico = pd.read_csv(historico_path)
historico['Date'] = pd.to_datetime(historico['Date'], utc=True)
historico = historico.sort_values('Date').reset_index(drop=True)

print(f"   [OK] {len(predicciones)} predicciones cargadas")
print(f"   [OK] {len(historico)} registros historicos cargados")

# ============================================================================
# ANÁLISIS ESTADÍSTICO
# ============================================================================
print("\n[3/6] Analisis estadistico de predicciones...")

# Estadísticas básicas
precio_inicial = predicciones['Predicted_Close'].iloc[0]
precio_final = predicciones['Predicted_Close'].iloc[-1]
precio_max = predicciones['Predicted_Close'].max()
precio_min = predicciones['Predicted_Close'].min()
precio_promedio = predicciones['Predicted_Close'].mean()
precio_mediana = predicciones['Predicted_Close'].median()
desviacion_std = predicciones['Predicted_Close'].std()

# Cambios
cambio_total = precio_final - precio_inicial
cambio_porcentual = (cambio_total / precio_inicial) * 100
cambio_diario_promedio = cambio_total / len(predicciones)

# Volatilidad
predicciones['Cambio_Diario'] = predicciones['Predicted_Close'].diff()
predicciones['Cambio_Porcentual'] = predicciones['Predicted_Close'].pct_change() * 100
volatilidad = predicciones['Cambio_Porcentual'].std()

# Tendencias
predicciones['Tendencia'] = predicciones['Predicted_Close'].rolling(window=5).mean()
tendencia_inicial = predicciones['Tendencia'].iloc[5]
tendencia_final = predicciones['Tendencia'].iloc[-1]

print("\n" + "-" * 70)
print("ESTADISTICAS DESCRIPTIVAS")
print("-" * 70)
print(f"Precio inicial (dia 1):        ${precio_inicial:.2f}")
print(f"Precio final (dia 30):        ${precio_final:.2f}")
print(f"Precio maximo:                 ${precio_max:.2f}")
print(f"Precio minimo:                 ${precio_min:.2f}")
print(f"Precio promedio:               ${precio_promedio:.2f}")
print(f"Precio mediana:                ${precio_mediana:.2f}")
print(f"Desviacion estandar:           ${desviacion_std:.2f}")
print(f"\nCambio total:                  ${cambio_total:.2f}")
print(f"Cambio porcentual:              {cambio_porcentual:.2f}%")
print(f"Cambio diario promedio:         ${cambio_diario_promedio:.4f}")
print(f"Volatilidad diaria:            {volatilidad:.4f}%")

# ============================================================================
# ANÁLISIS DE TENDENCIA
# ============================================================================
print("\n[4/6] Analisis de tendencia...")

# Determinar si es tendencia alcista o bajista
if cambio_porcentual > 0:
    tipo_tendencia = "ALCISTA"
    direccion = "subida"
else:
    tipo_tendencia = "BAJISTA"
    direccion = "bajada"

# Velocidad de cambio
velocidad_cambio = abs(cambio_porcentual) / len(predicciones)

# Días con mayor y menor cambio
max_cambio_dia = predicciones.loc[predicciones['Cambio_Diario'].abs().idxmax()]
min_cambio_dia = predicciones.loc[predicciones['Cambio_Diario'].abs().idxmin()]

print("\n" + "-" * 70)
print("ANALISIS DE TENDENCIA")
print("-" * 70)
print(f"Tipo de tendencia:             {tipo_tendencia}")
print(f"Direccion proyectada:          {direccion}")
print(f"Velocidad de cambio:           {velocidad_cambio:.4f}% por dia")
print(f"\nMayor cambio diario:")
print(f"  Fecha: {max_cambio_dia['Date'].date()}")
print(f"  Cambio: ${max_cambio_dia['Cambio_Diario']:.2f}")
print(f"  Precio: ${max_cambio_dia['Predicted_Close']:.2f}")

# ============================================================================
# COMPARACIÓN CON DATOS HISTÓRICOS
# ============================================================================
print("\n[5/6] Comparacion con datos historicos...")

# Obtener últimos 30 días históricos
ultimos_30_historicos = historico.tail(30)
precio_historico_inicial = ultimos_30_historicos['Close'].iloc[0]
precio_historico_final = ultimos_30_historicos['Close'].iloc[-1]
cambio_historico = precio_historico_final - precio_historico_inicial
cambio_historico_pct = (cambio_historico / precio_historico_inicial) * 100

# Comparar volatilidad
volatilidad_historica = ultimos_30_historicos['Close'].pct_change().std() * 100

# Último precio conocido
ultimo_precio_conocido = historico['Close'].iloc[-1]
diferencia_inicial = precio_inicial - ultimo_precio_conocido
diferencia_inicial_pct = (diferencia_inicial / ultimo_precio_conocido) * 100

print("\n" + "-" * 70)
print("COMPARACION CON DATOS HISTORICOS")
print("-" * 70)
print(f"Ultimo precio conocido:        ${ultimo_precio_conocido:.2f}")
print(f"Primera prediccion:            ${precio_inicial:.2f}")
print(f"Diferencia inicial:            ${diferencia_inicial:.2f} ({diferencia_inicial_pct:.2f}%)")
print(f"\nUltimos 30 dias historicos:")
print(f"  Precio inicial:               ${precio_historico_inicial:.2f}")
print(f"  Precio final:                ${precio_historico_final:.2f}")
print(f"  Cambio:                      ${cambio_historico:.2f} ({cambio_historico_pct:.2f}%)")
print(f"  Volatilidad historica:        {volatilidad_historica:.4f}%")
print(f"  Volatilidad predicha:         {volatilidad:.4f}%")

# ============================================================================
# ANÁLISIS DE RIESGO
# ============================================================================
print("\n[6/6] Analisis de riesgo...")

# Calcular drawdown (caída máxima desde el inicio)
predicciones['Drawdown'] = predicciones['Predicted_Close'] - precio_inicial
max_drawdown = predicciones['Drawdown'].min()
max_drawdown_pct = (max_drawdown / precio_inicial) * 100

# Rango de precios
rango_precios = precio_max - precio_min
rango_porcentual = (rango_precios / precio_inicial) * 100

# Días consecutivos de bajada/subida
predicciones['Direccion'] = np.where(predicciones['Cambio_Diario'] > 0, 1, 
                                     np.where(predicciones['Cambio_Diario'] < 0, -1, 0))
racha_actual = 0
max_racha_bajista = 0
max_racha_alcista = 0
racha_bajista_actual = 0
racha_alcista_actual = 0

for direccion in predicciones['Direccion']:
    if direccion == -1:  # Bajista
        racha_bajista_actual += 1
        racha_alcista_actual = 0
        max_racha_bajista = max(max_racha_bajista, racha_bajista_actual)
    elif direccion == 1:  # Alcista
        racha_alcista_actual += 1
        racha_bajista_actual = 0
        max_racha_alcista = max(max_racha_alcista, racha_alcista_actual)

print("\n" + "-" * 70)
print("ANALISIS DE RIESGO")
print("-" * 70)
print(f"Maximo drawdown:               ${max_drawdown:.2f} ({max_drawdown_pct:.2f}%)")
print(f"Rango de precios:              ${rango_precios:.2f} ({rango_porcentual:.2f}%)")
print(f"Maxima racha bajista:           {max_racha_bajista} dias")
print(f"Maxima racha alcista:          {max_racha_alcista} dias")

# ============================================================================
# PROYECCIONES ADICIONALES
# ============================================================================
print("\n[7/7] Proyecciones adicionales...")

# Proyección semanal
semanas = len(predicciones) // 7
precio_semana_1 = predicciones['Predicted_Close'].iloc[6] if len(predicciones) > 6 else precio_final
precio_semana_2 = predicciones['Predicted_Close'].iloc[13] if len(predicciones) > 13 else precio_final
precio_semana_4 = predicciones['Predicted_Close'].iloc[27] if len(predicciones) > 27 else precio_final

cambio_semana_1 = ((precio_semana_1 - precio_inicial) / precio_inicial) * 100
cambio_semana_2 = ((precio_semana_2 - precio_inicial) / precio_inicial) * 100
cambio_semana_4 = ((precio_semana_4 - precio_inicial) / precio_inicial) * 100

print("\n" + "-" * 70)
print("PROYECCIONES POR SEMANA")
print("-" * 70)
print(f"Semana 1 (dia 7):               ${precio_semana_1:.2f} ({cambio_semana_1:+.2f}%)")
print(f"Semana 2 (dia 14):              ${precio_semana_2:.2f} ({cambio_semana_2:+.2f}%)")
print(f"Semana 4 (dia 28):              ${precio_semana_4:.2f} ({cambio_semana_4:+.2f}%)")
print(f"Mes completo (dia 30):         ${precio_final:.2f} ({cambio_porcentual:+.2f}%)")

# ============================================================================
# VISUALIZACIONES
# ============================================================================
print("\n[8/8] Generando visualizaciones...")

fig = plt.figure(figsize=(18, 12))

# Gráfico 1: Predicciones con tendencia
plt.subplot(2, 3, 1)
plt.plot(predicciones['Date'], predicciones['Predicted_Close'], 
         label='Predicciones', linewidth=2, color='red', marker='o', markersize=3)
plt.plot(predicciones['Date'], predicciones['Tendencia'], 
         label='Tendencia (MA 5)', linewidth=2, color='blue', linestyle='--')
plt.axhline(y=precio_inicial, color='green', linestyle=':', alpha=0.7, label='Precio inicial')
plt.title('Predicciones de Precio - 30 Dias Futuros', fontsize=12, fontweight='bold')
plt.xlabel('Fecha')
plt.ylabel('Precio (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Gráfico 2: Cambios diarios
plt.subplot(2, 3, 2)
colores = ['red' if x < 0 else 'green' for x in predicciones['Cambio_Diario'].iloc[1:]]
plt.bar(predicciones['Date'].iloc[1:], predicciones['Cambio_Diario'].iloc[1:], 
        color=colores, alpha=0.6)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.title('Cambios Diarios en el Precio', fontsize=12, fontweight='bold')
plt.xlabel('Fecha')
plt.ylabel('Cambio (USD)')
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45)

# Gráfico 3: Comparación con histórico
plt.subplot(2, 3, 3)
ultimos_60_historicos = historico.tail(60)
plt.plot(ultimos_60_historicos['Date'], ultimos_60_historicos['Close'], 
         label='Historico (60 dias)', linewidth=2, color='blue', alpha=0.7)
plt.plot(predicciones['Date'], predicciones['Predicted_Close'], 
         label='Predicciones (30 dias)', linewidth=2, color='red', marker='o', markersize=3)
plt.axvline(x=historico['Date'].iloc[-1], color='green', linestyle='--', 
            linewidth=2, label='Inicio prediccion')
plt.title('Comparacion: Historico vs Predicciones', fontsize=12, fontweight='bold')
plt.xlabel('Fecha')
plt.ylabel('Precio (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Gráfico 4: Distribución de cambios porcentuales
plt.subplot(2, 3, 4)
cambios_pct = predicciones['Cambio_Porcentual'].dropna()
plt.hist(cambios_pct, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Sin cambio')
plt.axvline(x=cambios_pct.mean(), color='green', linestyle='--', 
            linewidth=2, label=f'Media: {cambios_pct.mean():.4f}%')
plt.title('Distribucion de Cambios Porcentuales Diarios', fontsize=12, fontweight='bold')
plt.xlabel('Cambio Porcentual (%)')
plt.ylabel('Frecuencia')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Gráfico 5: Drawdown
plt.subplot(2, 3, 5)
plt.fill_between(predicciones['Date'], predicciones['Drawdown'], 0, 
                 where=(predicciones['Drawdown'] < 0), color='red', alpha=0.3, label='Drawdown')
plt.plot(predicciones['Date'], predicciones['Drawdown'], 
         linewidth=2, color='darkred')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.title(f'Drawdown (Max: ${max_drawdown:.2f})', fontsize=12, fontweight='bold')
plt.xlabel('Fecha')
plt.ylabel('Drawdown (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Gráfico 6: Proyección semanal
plt.subplot(2, 3, 6)
semanas_data = ['Inicial', 'Sem 1', 'Sem 2', 'Sem 4', 'Final']
precios_semanas = [precio_inicial, precio_semana_1, precio_semana_2, precio_semana_4, precio_final]
colores_semanas = ['blue', 'orange', 'green', 'purple', 'red']
plt.plot(semanas_data, precios_semanas, marker='o', linewidth=2, markersize=8, color='darkblue')
for i, (sem, precio, color) in enumerate(zip(semanas_data, precios_semanas, colores_semanas)):
    plt.bar(sem, precio, alpha=0.3, color=color)
    plt.text(i, precio + 2, f'${precio:.2f}', ha='center', fontweight='bold')
plt.title('Proyeccion Semanal del Precio', fontsize=12, fontweight='bold')
plt.xlabel('Periodo')
plt.ylabel('Precio (USD)')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = os.path.join(script_dir, 'analisis_predicciones.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"   [OK] Graficos guardados en '{output_path}'")

# ============================================================================
# RESUMEN EJECUTIVO
# ============================================================================
print("\n" + "=" * 70)
print("RESUMEN EJECUTIVO")
print("=" * 70)
print(f"\n1. TENDENCIA GENERAL:")
print(f"   - Tipo: {tipo_tendencia}")
print(f"   - Cambio proyectado en 30 dias: {cambio_porcentual:.2f}%")
print(f"   - Velocidad: {velocidad_cambio:.4f}% por dia")

print(f"\n2. NIVELES DE PRECIO:")
print(f"   - Rango: ${precio_min:.2f} - ${precio_max:.2f}")
print(f"   - Promedio: ${precio_promedio:.2f}")
print(f"   - Mediana: ${precio_mediana:.2f}")

print(f"\n3. RIESGO:")
print(f"   - Maximo drawdown: {max_drawdown_pct:.2f}%")
print(f"   - Volatilidad diaria: {volatilidad:.4f}%")
print(f"   - Rango de variacion: {rango_porcentual:.2f}%")

print(f"\n4. PROYECCIONES TEMPORALES:")
print(f"   - Semana 1: ${precio_semana_1:.2f} ({cambio_semana_1:+.2f}%)")
print(f"   - Semana 2: ${precio_semana_2:.2f} ({cambio_semana_2:+.2f}%)")
print(f"   - Mes completo: ${precio_final:.2f} ({cambio_porcentual:+.2f}%)")

print(f"\n5. COMPARACION CON HISTORICO:")
print(f"   - Diferencia inicial: {diferencia_inicial_pct:+.2f}%")
print(f"   - Volatilidad historica: {volatilidad_historica:.4f}%")
print(f"   - Volatilidad predicha: {volatilidad:.4f}%")

print("\n" + "=" * 70)
print("Analisis completado exitosamente!")
print("=" * 70)

plt.show()

