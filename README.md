# App consolidada · Modelamiento predictivo y analítico de ventas por SKU (Grupo 5)

Universidad de Lima · Modelamiento predictivo de datos · Sección 805

Una sola app web (Streamlit) que integra las 4 dimensiones del proyecto:

| Página | Contenido |
|---|---|
| Inicio y objetivos | Problema, arquitectura, 4 objetivos específicos con meta numérica y resultado calculado en vivo |
| Datos y análisis descriptivo | Media, mediana, moda, desviación, mín., máx.; correlación; estandarización z-score; selección de SKU; PCA |
| 1 · Forecast | Seasonal Naive, Fourier + Ridge, SARIMAX estacional, DHR; backtest; regularización Ridge; pronóstico con escenario |
| 2 · Elasticidad | Log-log OLS con controles, IC 95%, VIF, Lasso, simulador de precio |
| 3 · Inventarios | SS, ROP y S con el forecast campeón y su error fuera de muestra; sensibilidad al nivel de servicio |
| 4 · Tablero BI integrador | Réplica de las 4 páginas del Power BI (forecast, inventarios, elasticidad, mapa ejecutivo) + descarga de tablas BI_* |
| Escenarios prescriptivos | Comparación editable de escenarios de precio y nivel de servicio con recomendación |
| Metodología y referencias | Justificación de algoritmos, regularización, bosquejo de código real y bibliografía |

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Publicar como web (Streamlit Community Cloud, gratis)

1. Crear un repositorio en GitHub (puede ser público) y subir **todo el contenido de esta carpeta**
   (incluye `data/`, `artifacts/`, `models/` y `.streamlit/`). Pesa ~5 MB.
2. Entrar a https://share.streamlit.io con la cuenta de GitHub → **Create app** → elegir el repositorio,
   rama `main` y archivo principal `streamlit_app.py`.
3. En **Advanced settings** elegir **Python 3.12**.
4. **Deploy**. En 2–4 minutos queda una URL del tipo `https://<nombre>.streamlit.app` para presentar.

> La primera carga tarda unos segundos (lee la base y reconstruye los modelos); luego queda en caché.

## Notas técnicas

- Los `.pkl` originales de SARIMAX pesaban ~60 MB cada uno (≈ 570 MB en total) y no se pueden subir a
  GitHub. Ahora se guardan solo los parámetros estimados (`models/*_params.json`) y el modelo se
  reconstruye con `SARIMAX(...).filter(params)`, sin reentrenar: los pronósticos son **idénticos** a los
  de los `.pkl` (diferencia máxima verificada = 0.0 en los 10 SKU y 4 modelos).
- Si cambia la base o la metodología, volver a ejecutar el notebook de modelamiento y exportar los
  parámetros con `forecast_core.export_light_params(...)`.
- La base `data/weekly_df_final_for_modeling.csv` es la misma hoja del Excel original (sin las 5 columnas
  «Unnamed» vacías).
