# Jarvis Trading Bot

Proyecto de orquestación de inversiones automatizadas con enfoque en aprendizaje estadístico y transparencia algorítmica.

## Objetivos
1. **Entrenamiento y Simulación:** Fase de 1-2 meses utilizando Paper Trading y datos históricos.
2. **Diversificación:** Foco en activos Tech (S&P 500, Nasdaq) y activos correlacionados.
3. **Interoperabilidad:** Datos almacenados en formatos abiertos (CSV/Parquet) para análisis profundo en R.
4. **Transparencia:** Reportes diarios detallados con razonamiento técnico y glosario dinámico.

## Estructura del Proyecto
- `src/`: Lógica del bot en Python.
- `data/`: Almacenamiento de series temporales y glosario de términos.
- `notebooks/`: Análisis exploratorio y validación de estrategias.
- `docs/`: Documentación técnica y bitácora de optimizaciones sugeridas por Gaia.

## 5. Análisis en R
El bot expone los datos en la carpeta `data/`. El archivo `R/analysis.R` contiene funciones pre-construidas para cargar la cartera y visualizar el historial de mercado.

```r
source("R/analysis.R")
plot_market_history()
```
