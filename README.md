
# 🌱 Sentinel Ground System — Sistema Inteligente de Optimización de Suelos

> Plataforma de recomendación agrícola impulsada por Machine Learning que predice el cultivo óptimo según las condiciones del suelo y genera recomendaciones de ajuste para maximizar la compatibilidad con cualquier cultivo deseado.

---

## 📌 Descripción

**Sentinel Ground System** es una aplicación web interactiva construida con Python y Streamlit que utiliza un modelo de clasificación **Random Forest** para:

1. **Predecir el cultivo más adecuado** para las condiciones actuales de un suelo dado.
2. **Recomendar ajustes específicos** (aumentar o reducir parámetros) para adaptar el suelo a un cultivo objetivo.
3. **Visualizar comparaciones** entre el estado actual del suelo y las condiciones ideales mediante gráficos radar y tablas detalladas.

El sistema trabaja con seis parámetros clave del suelo:

| Parámetro | Unidad | Descripción |
|-----------|--------|-------------|
| **Nitrógeno (N)** | mg/kg | Esencial para el crecimiento vegetal |
| **Fósforo (P)** | mg/kg | Importante para el desarrollo de raíces y flores |
| **Potasio (K)** | mg/kg | Ayuda a la resistencia contra enfermedades y estrés hídrico |
| **pH** | — | Afecta la disponibilidad de nutrientes |
| **Conductividad Eléctrica (EC)** | µS/cm | Indica la salinidad del suelo |
| **Humedad** | % | Crucial para el crecimiento de las plantas |

---

## 🚀 Características principales

- 🔮 **Predicción de cultivos**: Modelo Random Forest (100 estimadores) entrenado con datasets agrícolas reales.
- 🔄 **Recomendaciones de ajuste**: Comparación automática entre el suelo actual y las condiciones ideales para un cultivo deseado, con umbral configurable.
- 📊 **Gráfico Radar interactivo**: Visualización polar que compara los parámetros actuales vs. ideales del suelo.
- 📋 **Tabla comparativa**: Muestra las diferencias numéricas exactas entre los valores actuales e ideales.
- 📜 **Historial de predicciones**: Registro en sesión de todas las predicciones realizadas.
- 📥 **Exportación CSV**: Descarga de las recomendaciones generadas en formato CSV.
- ❓ **Guía de uso integrada**: Sección de ayuda dentro de la propia aplicación.
- ⚡ **Caché inteligente**: Uso de `@st.cache_resource` y `@st.cache_data` para optimizar tiempos de carga.

---

## 🛠️ Tecnologías utilizadas

| Tecnología | Uso |
|------------|-----|
| **Python 3.7+** | Lenguaje principal |
| **Streamlit** | Framework para la interfaz web interactiva |
| **Scikit-learn** | Modelo de Machine Learning (RandomForestClassifier) |
| **Pandas** | Manipulación y análisis de datos |
| **NumPy** | Cálculos numéricos |
| **Matplotlib** | Generación de gráficos (radar chart) |
| **Seaborn** | Visualizaciones estadísticas (heatmaps de correlación) |
| **Joblib** | Persistencia del modelo entrenado (`.pkl`) |
| **OpenPyXL** | Lectura de datasets en formato Excel (`.xlsx`) |

---

## 📂 Estructura del proyecto

```
Ground-Optimization/
├── datasets/                              # Directorio de datos (no incluido en el repo)
│   ├── crop_dataset2.csv                  # Dataset principal de cultivos (CSV)
│   └── Crop_Predication_dataset.xlsx      # Dataset complementario (Excel)
├── app.py                                 # Aplicación web Streamlit (interfaz de usuario)
├── Main.py                                # Script de entrenamiento, evaluación y pruebas del modelo
├── normalizer.py                          # Exploración de datos, normalización y análisis de correlación
├── modelo_cultivo.pkl                     # Modelo entrenado (generado automáticamente, no versionado)
├── .gitignore                             # Excluye archivos .pkl del repositorio
└── README.md                              # Este archivo
```

### Descripción de cada archivo

- **`app.py`**: Aplicación principal de Streamlit. Carga el modelo entrenado y el dataset, presenta una interfaz con controles para ingresar los parámetros del suelo, ejecuta predicciones, genera visualizaciones (radar chart y tabla comparativa), y permite exportar resultados.

- **`Main.py`**: Script de entrenamiento y evaluación. Carga los datasets, normaliza los datos con `MinMaxScaler`, entrena un modelo `RandomForestClassifier` con división 80/20, guarda el modelo como `modelo_cultivo.pkl`, y ejecuta una prueba de predicción de ejemplo.

- **`normalizer.py`**: Script de exploración y preprocesamiento de datos. Carga ambos datasets, muestra información estructural, aplica normalización Min-Max, genera un mapa de calor de correlación, y entrena un modelo de prueba para verificar la precisión.

---

## 📥 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/ReapeRAlan/Ground-Optimization.git
cd Ground-Optimization
```

### 2. Instalar dependencias

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib openpyxl
```

> **Nota:** Actualmente las dependencias se instalan manualmente. Un archivo `requirements.txt` para gestión formal de dependencias está planificado en el [roadmap](#-proyección-y-roadmap).

### 3. Preparar los datasets

Asegúrate de que la carpeta `datasets/` contenga los archivos:

- `crop_dataset2.csv` — Dataset principal con columnas: `Crop`, `N (mg/kg)`, `P (mg/kg)`, `K (mg/kg)`, `pH`, `EC(uS/cm)`, `MOISTURE (%)`
- `Crop_Predication_dataset.xlsx` — Dataset complementario en formato Excel

### 4. Entrenar el modelo

```bash
python Main.py
```

Esto generará automáticamente el archivo `modelo_cultivo.pkl`.

---

## 🖥️ Uso

### Ejecutar la aplicación web

```bash
streamlit run app.py
```

Accede a la interfaz en [http://localhost:8501](http://localhost:8501).

### Flujo de uso

1. **Ingresa los parámetros del suelo** en la barra lateral izquierda:
   - Nitrógeno (0–300 mg/kg)
   - Fósforo (0–150 mg/kg)
   - Potasio (0–300 mg/kg)
   - pH (3.0–9.0)
   - Conductividad Eléctrica (0.1–5.0 µS/cm)
   - Humedad (0–100%)

2. **Selecciona el cultivo deseado** del menú desplegable.

3. **Ajusta el umbral de diferencia significativa** (1–10) para controlar la sensibilidad de las recomendaciones.

4. **Haz clic en** "🔎 Predict and Evaluate Compatibility".

5. **Revisa los resultados**:
   - Cultivo recomendado por el modelo
   - Recomendaciones de ajuste (⬆️ aumentar / ⬇️ reducir)
   - Gráfico radar comparativo
   - Tabla con valores exactos

6. **Descarga las recomendaciones** en CSV si lo deseas.

---

## 📊 Visualizaciones disponibles

### Gráfico Radar

Compara visualmente los 6 parámetros del suelo actual (azul) con las condiciones ideales para el cultivo seleccionado (verde) en un gráfico polar.

### Tabla Comparativa

Muestra los valores numéricos exactos de cada parámetro para el suelo actual y el ideal, facilitando la identificación de las brechas más importantes.

### Mapa de Calor de Correlación

Disponible en `normalizer.py`, permite entender las relaciones estadísticas entre los diferentes parámetros del suelo.

---

## 🧠 Modelo de Machine Learning

| Aspecto | Detalle |
|---------|---------|
| **Algoritmo** | Random Forest (RandomForestClassifier) |
| **Estimadores** | 100 árboles de decisión |
| **División de datos** | 80% entrenamiento / 20% prueba |
| **Normalización** | MinMaxScaler (en preprocesamiento) |
| **Precisión reportada** | Superior al 90% |
| **Persistencia** | Serialización con Joblib (`.pkl`) |
| **Paralelización** | `n_jobs=-1` (uso de todos los núcleos del CPU) |

---

## 🔮 Proyección y Roadmap

El proyecto **Sentinel Ground System** está diseñado como una base escalable para convertirse en una plataforma completa de agricultura de precisión. Las siguientes funcionalidades están proyectadas para futuras versiones:

### Corto plazo

- [ ] **Archivo `requirements.txt`**: Gestión formal de dependencias para facilitar la instalación y el despliegue.
- [ ] **Validación de datos de entrada**: Manejo robusto de errores cuando faltan datasets o el modelo no está entrenado.
- [ ] **Tests automatizados**: Suite de pruebas unitarias con `pytest` para validar predicciones, recomendaciones y carga de datos.
- [ ] **CI/CD con GitHub Actions**: Pipeline automatizado para linting, testing y despliegue.
- [ ] **Dockerización**: Contenedor Docker para despliegue consistente en cualquier entorno.

### Mediano plazo

- [ ] **Integración con sensores IoT**: Lectura automática de datos de suelo desde sensores conectados (NPK, pH, humedad) en tiempo real.
- [ ] **API REST**: Exposición de los endpoints de predicción y recomendación como una API independiente (con FastAPI o Flask) para integración con otros sistemas.
- [ ] **Base de datos**: Migración del almacenamiento de datos de CSV a una base de datos (PostgreSQL o SQLite) para persistencia, historial y análisis a largo plazo.
- [ ] **Múltiples modelos ML**: Comparación de algoritmos (Gradient Boosting, SVM, Redes Neuronales) para seleccionar el más preciso según el contexto.
- [ ] **Soporte multilenguaje**: Interfaz en español e inglés con internacionalización (i18n).

### Largo plazo

- [ ] **Análisis geoespacial**: Integración con datos satelitales y mapas GIS para recomendaciones basadas en la ubicación geográfica.
- [ ] **Predicción climática**: Incorporación de datos meteorológicos para ajustar las recomendaciones según pronósticos climáticos.
- [ ] **App móvil**: Versión móvil (React Native o Flutter) para uso en campo sin necesidad de un ordenador.
- [ ] **Dashboard de monitoreo**: Panel en tiempo real con métricas históricas del suelo, tendencias y alertas automáticas.
- [ ] **Comunidad y marketplace**: Plataforma donde agricultores compartan datos anónimos de suelo para mejorar los modelos colaborativamente.

---

## 📌 Notas técnicas

- El archivo `.gitignore` excluye los archivos `.pkl` del repositorio para evitar versionar modelos pesados.
- La carpeta `datasets/` debe ser proporcionada manualmente (no está incluida en el repositorio).
- La aplicación Streamlit utiliza `st.session_state` para mantener el historial de predicciones durante la sesión activa.
- El modelo debe ser entrenado ejecutando `Main.py` antes de lanzar la aplicación por primera vez.

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Haz un fork del repositorio.
2. Crea una rama para tu feature: `git checkout -b feature/mi-nueva-funcionalidad`
3. Realiza tus cambios y haz commit: `git commit -m "Añadir nueva funcionalidad"`
4. Sube tu rama: `git push origin feature/mi-nueva-funcionalidad`
5. Abre un Pull Request.

---

## 📃 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

