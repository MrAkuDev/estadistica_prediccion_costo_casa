# Predicción del Valor de Propiedades Inmobiliarias

Proyecto final de estadística aplicada: análisis exploratorio, modelado con **Regresión Lineal (`scikit-learn`)**, comparación de modelos de Machine Learning, informe técnico en LaTeX y aplicación interactiva con simulador de precios en **Streamlit**.

## Estructura del Proyecto

- `app.py`: Aplicación de Streamlit lista para deployar (incluye simulador interactivo de precios con Regresión Lineal, dashboard de EDA y comparación de modelos).
- `train_model.py`: Script reproducible de limpieza de datos, entrenamiento de modelos, cálculo de métricas y generación de gráficas.
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

## Despliegue en Streamlit Community Cloud

1. Haz commit y push de los archivos del repositorio a GitHub.
2. Ingresa a [share.streamlit.io](https://share.streamlit.io), selecciona el repositorio y define `app.py` como archivo principal.
3. Haz clic en **Deploy**.
