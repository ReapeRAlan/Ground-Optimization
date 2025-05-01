
# 🌱 Sentinel Ground System

## 📌 Descripción

Sentinel Ground System es una aplicación inteligente diseñada para recomendar cultivos óptimos según las condiciones actuales del suelo, utilizando técnicas avanzadas de Machine Learning (Random Forest) y análisis de datos agrícolas. Además, proporciona recomendaciones específicas para ajustar el suelo y hacerlo más compatible con un cultivo deseado.

---

## 🚀 Características principales

- **Predicción de cultivos** basados en parámetros del suelo.
- **Recomendaciones detalladas** para mejorar la compatibilidad del suelo con cultivos específicos.
- **Visualizaciones interactivas** incluyendo gráficos radar comparativos y tablas detalladas.
- **Historial de predicciones** y capacidad de exportar recomendaciones a CSV.

---

## 🛠️ Tecnologías utilizadas

- **Python**
- **Streamlit**
- **Scikit-learn**
- **Pandas y NumPy**
- **Matplotlib y Seaborn**
- **Joblib (persistencia de modelos)**

---

## 📂 Estructura del proyecto

```
.
├── datasets
│   ├── crop_dataset2.csv
│   └── Crop_Predication_dataset.xlsx
├── modelo_cultivo.pkl
├── app.py (Streamlit application)
├── training.py (Script de entrenamiento y evaluación del modelo)
└── requirements.txt
```

---

## 📥 Instalación

1. Clona este repositorio:

```bash
git clone https://github.com/tuusuario/sentinel-ground-system.git
cd sentinel-ground-system
```

2. Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

---

## 🖥️ Uso

### Ejecutar la aplicación Streamlit

```bash
streamlit run app.py
```

- Accede a la interfaz web en [http://localhost:8501](http://localhost:8501).

### Entrenamiento del modelo

Para entrenar un nuevo modelo desde cero o evaluar su rendimiento:

```bash
python training.py
```

Esto generará automáticamente un archivo `modelo_cultivo.pkl`.

---

## 📊 Ejemplo de uso

1. Ingresa parámetros del suelo:
    - Nitrógeno, Fósforo, Potasio, pH, Conductividad eléctrica, Humedad.

2. Selecciona el cultivo deseado.

3. Establece un umbral para identificar diferencias significativas.

4. Haz clic en **Predict and Evaluate Compatibility**.

---

## 📸 Visualizaciones disponibles

- **Gráfico Radar**: Compara visualmente los parámetros actuales del suelo con los parámetros ideales para el cultivo seleccionado.
- **Tabla Comparativa**: Muestra diferencias numéricas claras entre el suelo actual y las condiciones ideales.

---

## 📌 Notas adicionales

- El modelo actual usa Random Forest con una precisión reportada superior al 90%.
- Los datasets iniciales se encuentran en la carpeta `datasets`.

---

## 📃 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

