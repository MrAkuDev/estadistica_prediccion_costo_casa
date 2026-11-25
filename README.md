# Predicción del Valor de Propiedades Inmobiliarias

Proyecto final de estadística aplicada: análisis exploratorio, modelado con **Regresión Lineal estándar (`LinearRegression` de `scikit-learn`)**, informe técnico en LaTeX y aplicación interactiva con simulador de precios en **Streamlit**.

## Estructura del Proyecto

- `app.py` + `src/`: Aplicación modular de Streamlit lista para deployar (valuador interactivo de precios, exploración del mercado y evaluación del modelo lineal).
- `train_model.py`: Script reproducible de limpieza de datos, entrenamiento de `LinearRegression`, cálculo de métricas y generación de gráficas.
- `models/linear_regression_model.joblib`: Modelo `LinearRegression` de `scikit-learn` entrenado (2.2 KB).
- `models/metrics_and_eda.json`: Estadísticas descriptivas, matriz de correlación, coeficientes y métricas de comparación.
- `data/sample_realtor_clean.csv`: Muestra ligera representativa (8,000 registros) para las gráficas interactivas en Streamlit.
- `informe/informe_tecnico.tex`: Informe técnico completo en LaTeX (con figuras en `informe/figuras/`).
- `requirements.txt`: Dependencias para Streamlit Community Cloud.

> **Nota sobre el dataset:** El archivo original `realtor-data.zip.csv` (~178 MB) está excluido en `.gitignore` para permitir subir el repositorio a GitHub y desplegar en Streamlit Cloud sin exceder el límite de 100 MB.

## Ejecución Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Si deseas reentrenar los modelos y regenerar las figuras desde `realtor-data.zip.csv`:

```bash
python train_model.py
```
