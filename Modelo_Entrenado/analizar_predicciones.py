"""
Análisis completo del archivo predicciones_generadas.csv
Genera un informe detallado con estadísticas, tendencias y visualizaciones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime, timedelta

# Configuración de matplotlib para español
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (16, 10)

def cargar_datos():
    """Carga las predicciones y metadatos"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cargar predicciones
    predicciones_path = os.path.join(script_dir, 'predicciones_generadas.csv')
    if not os.path.exists(predicciones_path):
        raise FileNotFoundError(f"No se encuentra el archivo de predicciones en '{predicciones_path}'")
    
    df_pred = pd.read_csv(predicciones_path)
    df_pred['Date'] = pd.to_datetime(df_pred['Date'])
    df_pred = df_pred.sort_values('Date').reset_index(drop=True)
    
    # Cargar metadatos
    metadata_path = os.path.join(script_dir, 'metadatos.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No se encuentran los metadatos en '{metadata_path}'")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return df_pred, metadata

def cargar_datos_historicos(metadata):
    """Carga los últimos datos históricos para comparación"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    datos_path = os.path.join(parent_dir, 'Datasets_Mastercard', 'Mastercard_stock_history.csv')
    
    if not os.path.exists(datos_path):
        print(f"⚠️  No se encontraron datos históricos en '{datos_path}'")
        return None
    
    df_hist = pd.read_csv(datos_path)
    df_hist['Date'] = pd.to_datetime(df_hist['Date'], utc=True).dt.tz_localize(None)
    df_hist = df_hist.sort_values('Date').reset_index(drop=True)
    
    # Filtrar últimos 60 días antes de la predicción
    last_date = pd.to_datetime(metadata['last_date'])
    df_hist_recent = df_hist[df_hist['Date'] <= last_date].tail(60)
    
    return df_hist_recent

def calcular_estadisticas(df_pred, metadata):
    """Calcula estadísticas descriptivas de las predicciones"""
    pred_prices = df_pred['Predicted_Close'].values
    
    stats = {
        'precio_inicial': pred_prices[0],
        'precio_final': pred_prices[-1],
        'precio_maximo': np.max(pred_prices),
        'precio_minimo': np.min(pred_prices),
        'precio_promedio': np.mean(pred_prices),
        'precio_mediana': np.median(pred_prices),
        'desviacion_estandar': np.std(pred_prices),
        'cambio_total': pred_prices[-1] - pred_prices[0],
        'cambio_porcentual': ((pred_prices[-1] / pred_prices[0]) - 1) * 100,
        'rango': np.max(pred_prices) - np.min(pred_prices),
        'volatilidad_diaria': np.std(np.diff(pred_prices)),
        'dia_maximo': df_pred.loc[np.argmax(pred_prices), 'Date'],
        'dia_minimo': df_pred.loc[np.argmin(pred_prices), 'Date'],
    }
    
    # Cambios diarios
    cambios_diarios = np.diff(pred_prices)
    cambios_porcentuales = (cambios_diarios / pred_prices[:-1]) * 100
    
    stats['cambio_diario_promedio'] = np.mean(cambios_diarios)
    stats['cambio_diario_std'] = np.std(cambios_diarios)
    stats['mayor_subida_diaria'] = np.max(cambios_diarios)
    stats['mayor_bajada_diaria'] = np.min(cambios_diarios)
    stats['mayor_subida_porcentual'] = np.max(cambios_porcentuales)
    stats['mayor_bajada_porcentual'] = np.min(cambios_porcentuales)
    
    # Comparación con último precio conocido
    ultimo_precio = metadata['last_price']
    stats['diferencia_vs_inicial'] = pred_prices[0] - ultimo_precio
    stats['diferencia_porcentual_vs_inicial'] = ((pred_prices[0] / ultimo_precio) - 1) * 100
    stats['diferencia_vs_final'] = pred_prices[-1] - ultimo_precio
    stats['diferencia_porcentual_vs_final'] = ((pred_prices[-1] / ultimo_precio) - 1) * 100
    
    return stats, cambios_diarios, cambios_porcentuales

def analizar_tendencia(df_pred, stats):
    """Analiza la tendencia de las predicciones"""
    pred_prices = df_pred['Predicted_Close'].values
    
    # Tendencia general
    if stats['cambio_porcentual'] > 0:
        tendencia = "ALCISTA"
    elif stats['cambio_porcentual'] < 0:
        tendencia = "BAJISTA"
    else:
        tendencia = "LATERAL"
    
    # Velocidad de cambio
    velocidad_cambio = abs(stats['cambio_porcentual']) / len(pred_prices)
    
    # Análisis de rachas
    cambios = np.diff(pred_prices)
    racha_alcista = 0
    racha_bajista = 0
    max_racha_alcista = 0
    max_racha_bajista = 0
    
    for cambio in cambios:
        if cambio > 0:
            racha_alcista += 1
            racha_bajista = 0
            max_racha_alcista = max(max_racha_alcista, racha_alcista)
        else:
            racha_bajista += 1
            racha_alcista = 0
            max_racha_bajista = max(max_racha_bajista, racha_bajista)
    
    # Drawdown (caída máxima desde el inicio)
    running_max = np.maximum.accumulate(pred_prices)
    drawdown = ((pred_prices - running_max) / running_max) * 100
    max_drawdown = np.min(drawdown)
    
    analisis = {
        'tendencia': tendencia,
        'velocidad_cambio_diario': velocidad_cambio,
        'max_racha_alcista': max_racha_alcista,
        'max_racha_bajista': max_racha_bajista,
        'max_drawdown': max_drawdown,
        'drawdown_dia': df_pred.loc[np.argmin(drawdown), 'Date'],
    }
    
    return analisis, drawdown

def generar_visualizaciones(df_pred, df_hist, stats, cambios_diarios, cambios_porcentuales, 
                           drawdown, metadata, analisis):
    """Genera visualizaciones del análisis"""
    
    fig = plt.figure(figsize=(20, 14))
    
    # Gráfico 1: Predicciones con tendencia
    plt.subplot(3, 3, 1)
    plt.plot(df_pred['Date'], df_pred['Predicted_Close'], 
            linewidth=2.5, color='#2E86AB', label='Predicciones', marker='o', markersize=3)
    plt.axhline(y=stats['precio_promedio'], color='orange', linestyle='--', 
               linewidth=1.5, label=f'Promedio: ${stats["precio_promedio"]:.2f}', alpha=0.7)
    plt.axhline(y=metadata['last_price'], color='green', linestyle=':', 
               linewidth=2, label=f'Último precio: ${metadata["last_price"]:.2f}', alpha=0.8)
    plt.fill_between(df_pred['Date'], stats['precio_minimo'], stats['precio_maximo'], 
                     alpha=0.2, color='gray', label='Rango')
    plt.title(f'Predicciones de Precio - Tendencia {analisis["tendencia"]}', 
             fontsize=12, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Precio (USD)')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Gráfico 2: Cambios diarios
    plt.subplot(3, 3, 2)
    colores = ['green' if x > 0 else 'red' for x in cambios_diarios]
    plt.bar(range(len(cambios_diarios)), cambios_diarios, color=colores, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    plt.axhline(y=stats['cambio_diario_promedio'], color='blue', linestyle='--', 
               linewidth=1.5, label=f'Promedio: ${stats["cambio_diario_promedio"]:.2f}')
    plt.title('Cambios Diarios en las Predicciones', fontsize=12, fontweight='bold')
    plt.xlabel('Día')
    plt.ylabel('Cambio (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 3: Comparación con histórico (si está disponible)
    plt.subplot(3, 3, 3)
    if df_hist is not None and len(df_hist) > 0:
        plt.plot(df_hist['Date'], df_hist['Close'], 
                linewidth=2, color='blue', label='Histórico (últimos 60 días)', alpha=0.7)
        plt.plot(df_pred['Date'], df_pred['Predicted_Close'], 
                linewidth=2.5, color='red', label='Predicciones', linestyle='--', marker='o', markersize=3)
        plt.axvline(x=df_hist['Date'].iloc[-1], color='green', linestyle=':', 
                   linewidth=2, label='Inicio predicción')
    else:
        plt.text(0.5, 0.5, 'Datos históricos\nno disponibles', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.title('Comparación: Histórico vs Predicciones', fontsize=12, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Precio (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Gráfico 4: Distribución de cambios porcentuales
    plt.subplot(3, 3, 4)
    plt.hist(cambios_porcentuales, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='-', linewidth=1.5)
    plt.axvline(x=np.mean(cambios_porcentuales), color='green', linestyle='--', 
               linewidth=1.5, label=f'Media: {np.mean(cambios_porcentuales):.2f}%')
    plt.title('Distribución de Cambios Porcentuales Diarios', fontsize=12, fontweight='bold')
    plt.xlabel('Cambio Porcentual (%)')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 5: Drawdown
    plt.subplot(3, 3, 5)
    plt.fill_between(df_pred['Date'], drawdown, 0, color='red', alpha=0.3)
    plt.plot(df_pred['Date'], drawdown, linewidth=2, color='darkred')
    plt.axhline(y=analisis['max_drawdown'], color='black', linestyle='--', 
               linewidth=1.5, label=f'Máx Drawdown: {analisis["max_drawdown"]:.2f}%')
    plt.title('Drawdown (Caída desde Máximo)', fontsize=12, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Gráfico 6: Precio con bandas de volatilidad
    plt.subplot(3, 3, 6)
    pred_prices = df_pred['Predicted_Close'].values
    media = np.mean(pred_prices)
    std = np.std(pred_prices)
    plt.plot(df_pred['Date'], pred_prices, linewidth=2, color='#2E86AB', label='Predicción')
    plt.fill_between(df_pred['Date'], media - std, media + std, 
                     alpha=0.2, color='orange', label=f'±1 Desv. Est. (${std:.2f})')
    plt.fill_between(df_pred['Date'], media - 2*std, media + 2*std, 
                     alpha=0.1, color='orange', label='±2 Desv. Est.')
    plt.title('Predicciones con Bandas de Volatilidad', fontsize=12, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Precio (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Gráfico 7: Proyección semanal
    plt.subplot(3, 3, 7)
    df_pred['Semana'] = df_pred['Date'].dt.isocalendar().week
    df_pred['Año-Semana'] = df_pred['Date'].dt.strftime('%Y-W%U')
    semanal = df_pred.groupby('Año-Semana')['Predicted_Close'].mean()
    semanas = range(len(semanal))
    plt.bar(semanas, semanal.values, color='teal', alpha=0.7, edgecolor='black')
    plt.plot(semanas, semanal.values, color='darkblue', marker='o', linewidth=2, markersize=6)
    plt.title('Precio Promedio por Semana', fontsize=12, fontweight='bold')
    plt.xlabel('Semana')
    plt.ylabel('Precio Promedio (USD)')
    plt.xticks(semanas, semanal.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Gráfico 8: Métricas comparativas
    plt.subplot(3, 3, 8)
    metricas = ['Máximo', 'Mínimo', 'Promedio', 'Mediana']
    valores = [stats['precio_maximo'], stats['precio_minimo'], 
              stats['precio_promedio'], stats['precio_mediana']]
    colores_met = ['green', 'red', 'blue', 'orange']
    bars = plt.bar(metricas, valores, color=colores_met, alpha=0.7, edgecolor='black')
    plt.title('Métricas de Precio', fontsize=12, fontweight='bold')
    plt.ylabel('Precio (USD)')
    plt.grid(True, alpha=0.3, axis='y')
    # Agregar valores en las barras
    for bar, val in zip(bars, valores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Gráfico 9: Resumen de cambios
    plt.subplot(3, 3, 9)
    cambios_resumen = {
        'Total': stats['cambio_porcentual'],
        'vs Inicial': stats['diferencia_porcentual_vs_inicial'],
        'vs Final': stats['diferencia_porcentual_vs_final'],
    }
    colores_cambios = ['green' if v > 0 else 'red' for v in cambios_resumen.values()]
    bars = plt.barh(list(cambios_resumen.keys()), list(cambios_resumen.values()), 
                    color=colores_cambios, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.title('Cambios Porcentuales (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Cambio (%)')
    plt.grid(True, alpha=0.3, axis='x')
    # Agregar valores
    for bar, val in zip(bars, cambios_resumen.values()):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.2f}%', ha='left' if width > 0 else 'right', 
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar gráfico
    script_dir = os.path.dirname(os.path.abspath(__file__))
    grafico_path = os.path.join(script_dir, 'analisis_predicciones.png')
    plt.savefig(grafico_path, dpi=300, bbox_inches='tight')
    print(f"   [OK] Gráficos guardados en '{grafico_path}'")
    
    return grafico_path

def generar_informe_texto(df_pred, stats, analisis, metadata, cambios_diarios):
    """Genera un informe de texto detallado"""
    
    informe = []
    informe.append("=" * 80)
    informe.append("INFORME DE ANÁLISIS DE PREDICCIONES - MASTERCARD")
    informe.append("=" * 80)
    informe.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    informe.append(f"Periodo analizado: {df_pred['Date'].iloc[0].strftime('%Y-%m-%d')} a {df_pred['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    informe.append(f"Días predichos: {len(df_pred)}")
    informe.append("")
    
    informe.append("=" * 80)
    informe.append("1. INFORMACIÓN DEL MODELO")
    informe.append("=" * 80)
    informe.append(f"Último precio conocido: ${metadata['last_price']:.2f}")
    informe.append(f"Fecha último dato: {metadata['last_date']}")
    informe.append(f"RMSE del modelo: ${metadata['rmse']:.2f}")
    informe.append(f"MAE del modelo: ${metadata['mae']:.2f}")
    informe.append(f"MAPE del modelo: {metadata['mape']:.2f}%")
    informe.append("")
    
    informe.append("=" * 80)
    informe.append("2. ESTADÍSTICAS DESCRIPTIVAS")
    informe.append("=" * 80)
    informe.append(f"Precio inicial (día 1):        ${stats['precio_inicial']:.2f}")
    informe.append(f"Precio final (día {len(df_pred)}):           ${stats['precio_final']:.2f}")
    informe.append(f"Precio máximo:                 ${stats['precio_maximo']:.2f} (día: {stats['dia_maximo'].strftime('%Y-%m-%d')})")
    informe.append(f"Precio mínimo:                 ${stats['precio_minimo']:.2f} (día: {stats['dia_minimo'].strftime('%Y-%m-%d')})")
    informe.append(f"Precio promedio:               ${stats['precio_promedio']:.2f}")
    informe.append(f"Precio mediana:                ${stats['precio_mediana']:.2f}")
    informe.append(f"Desviación estándar:           ${stats['desviacion_estandar']:.2f}")
    informe.append(f"Rango (máx - mín):             ${stats['rango']:.2f}")
    informe.append("")
    
    informe.append("=" * 80)
    informe.append("3. ANÁLISIS DE CAMBIOS")
    informe.append("=" * 80)
    informe.append(f"Cambio total:                  ${stats['cambio_total']:.2f}")
    informe.append(f"Cambio porcentual total:       {stats['cambio_porcentual']:.2f}%")
    informe.append(f"Cambio diario promedio:        ${stats['cambio_diario_promedio']:.2f}")
    informe.append(f"Desviación estándar diaria:    ${stats['cambio_diario_std']:.2f}")
    informe.append(f"Mayor subida diaria:           ${stats['mayor_subida_diaria']:.2f} ({stats['mayor_subida_porcentual']:.2f}%)")
    informe.append(f"Mayor bajada diaria:           ${stats['mayor_bajada_diaria']:.2f} ({stats['mayor_bajada_porcentual']:.2f}%)")
    informe.append("")
    
    informe.append("=" * 80)
    informe.append("4. COMPARACIÓN CON ÚLTIMO PRECIO CONOCIDO")
    informe.append("=" * 80)
    informe.append(f"Diferencia precio inicial vs último conocido:  ${stats['diferencia_vs_inicial']:.2f} ({stats['diferencia_porcentual_vs_inicial']:.2f}%)")
    informe.append(f"Diferencia precio final vs último conocido:    ${stats['diferencia_vs_final']:.2f} ({stats['diferencia_porcentual_vs_final']:.2f}%)")
    informe.append("")
    
    informe.append("=" * 80)
    informe.append("5. ANÁLISIS DE TENDENCIA")
    informe.append("=" * 80)
    informe.append(f"Tendencia general:             {analisis['tendencia']}")
    informe.append(f"Velocidad de cambio diario:    {analisis['velocidad_cambio_diario']:.3f}%")
    informe.append(f"Máxima racha alcista:          {analisis['max_racha_alcista']} días")
    informe.append(f"Máxima racha bajista:          {analisis['max_racha_bajista']} días")
    informe.append(f"Máximo drawdown:               {analisis['max_drawdown']:.2f}%")
    informe.append(f"Día de máximo drawdown:        {analisis['drawdown_dia'].strftime('%Y-%m-%d')}")
    informe.append("")
    
    informe.append("=" * 80)
    informe.append("6. ANÁLISIS DE VOLATILIDAD")
    informe.append("=" * 80)
    informe.append(f"Volatilidad diaria (std):      ${stats['volatilidad_diaria']:.2f}")
    informe.append(f"Coeficiente de variación:      {(stats['desviacion_estandar'] / stats['precio_promedio']) * 100:.2f}%")
    
    # Análisis de días con mayor volatilidad
    cambios_abs = np.abs(cambios_diarios)
    top_5_volatiles = np.argsort(cambios_abs)[-5:][::-1]
    informe.append("\nTop 5 días más volátiles:")
    for i, idx in enumerate(top_5_volatiles, 1):
        fecha = df_pred['Date'].iloc[idx]
        cambio = cambios_diarios[idx]
        cambio_pct = (cambio / df_pred['Predicted_Close'].iloc[idx-1]) * 100 if idx > 0 else 0
        informe.append(f"  {i}. {fecha.strftime('%Y-%m-%d')}: ${cambio:.2f} ({cambio_pct:.2f}%)")
    informe.append("")
    
    informe.append("=" * 80)
    informe.append("7. RESUMEN EJECUTIVO")
    informe.append("=" * 80)
    
    if analisis['tendencia'] == 'BAJISTA':
        informe.append("[!] TENDENCIA BAJISTA DETECTADA")
        informe.append(f"   El modelo predice una caida del {abs(stats['cambio_porcentual']):.2f}% en {len(df_pred)} dias.")
    elif analisis['tendencia'] == 'ALCISTA':
        informe.append("[+] TENDENCIA ALCISTA DETECTADA")
        informe.append(f"   El modelo predice un aumento del {stats['cambio_porcentual']:.2f}% en {len(df_pred)} dias.")
    else:
        informe.append("[=] TENDENCIA LATERAL")
        informe.append("   El modelo predice estabilidad en el precio.")
    
    informe.append(f"\nPrecio proyectado final: ${stats['precio_final']:.2f}")
    informe.append(f"Rango de precios: ${stats['precio_minimo']:.2f} - ${stats['precio_maximo']:.2f}")
    informe.append(f"Maxima caida desde inicio: {analisis['max_drawdown']:.2f}%")
    
    informe.append("\n" + "=" * 80)
    informe.append("FIN DEL INFORME")
    informe.append("=" * 80)
    
    return "\n".join(informe)

def main():
    """Función principal"""
    print("=" * 80)
    print("ANÁLISIS DE PREDICCIONES GENERADAS")
    print("=" * 80)
    
    try:
        # Cargar datos
        print("\n[1/5] Cargando datos...")
        df_pred, metadata = cargar_datos()
        print(f"   [OK] {len(df_pred)} predicciones cargadas")
        
        df_hist = cargar_datos_historicos(metadata)
        if df_hist is not None:
            print(f"   [OK] {len(df_hist)} días históricos cargados para comparación")
        
        # Calcular estadísticas
        print("\n[2/5] Calculando estadísticas...")
        stats, cambios_diarios, cambios_porcentuales = calcular_estadisticas(df_pred, metadata)
        print("   [OK] Estadísticas calculadas")
        
        # Analizar tendencia
        print("\n[3/5] Analizando tendencias...")
        analisis, drawdown = analizar_tendencia(df_pred, stats)
        print(f"   [OK] Tendencia detectada: {analisis['tendencia']}")
        
        # Generar visualizaciones
        print("\n[4/5] Generando visualizaciones...")
        grafico_path = generar_visualizaciones(df_pred, df_hist, stats, cambios_diarios, 
                                              cambios_porcentuales, drawdown, metadata, analisis)
        print(f"   [OK] Visualizaciones completadas")
        
        # Generar informe de texto
        print("\n[5/5] Generando informe de texto...")
        informe_texto = generar_informe_texto(df_pred, stats, analisis, metadata, cambios_diarios)
        
        # Guardar informe
        script_dir = os.path.dirname(os.path.abspath(__file__))
        informe_path = os.path.join(script_dir, 'informe_predicciones.txt')
        with open(informe_path, 'w', encoding='utf-8') as f:
            f.write(informe_texto)
        print(f"   [OK] Informe guardado en '{informe_path}'")
        
        # Mostrar informe en consola (sin emojis para evitar problemas de encoding)
        informe_consola = informe_texto
        # Reemplazar caracteres especiales para consola Windows
        informe_consola = informe_consola.replace('💰', '[PRECIO]')
        informe_consola = informe_consola.replace('📊', '[RANGO]')
        informe_consola = informe_consola.replace('📉', '[DRAWDOWN]')
        informe_consola = informe_consola.replace('⚠️', '[!]')
        informe_consola = informe_consola.replace('📈', '[+]')
        informe_consola = informe_consola.replace('➡️', '[=]')
        print("\n" + informe_consola)
        
        print("\n" + "=" * 80)
        print("ANALISIS COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        print(f"Informe de texto: {informe_path}")
        print(f"Graficos: {grafico_path}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

