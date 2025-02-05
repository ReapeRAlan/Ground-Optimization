import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# 📌 Función para cargar el modelo y el dataset con caché
@st.cache_resource  # Usar cache_resource para el modelo (objeto no serializable)
def cargar_modelo():
    model_path = "modelo_cultivo.pkl"
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

@st.cache_data  # Usar cache_data para el dataset (datos serializables)
def cargar_dataset():
    csv_file = "datasets/crop_dataset2.csv"
    df_csv = pd.read_csv(csv_file)
    return df_csv

# 📌 Cargar modelo y dataset
model = cargar_modelo()
df_csv = cargar_dataset()

# 📌 Función para recomendar ajustes en el suelo
def recomendar_ajustes(suelo_actual, cultivo_deseado, umbral):
    cultivos_similares = df_csv[df_csv["Crop"] == cultivo_deseado].drop(columns=["Crop"]).mean()
    
    if cultivos_similares.empty:
        return f"Not enough information about {cultivo_deseado} in the dataset."

    recomendaciones = {}

    for param in suelo_actual.keys():
        valor_actual = suelo_actual[param]
        valor_ideal = cultivos_similares[param]
        
        if abs(valor_actual - valor_ideal) > umbral:  # Umbral de diferencia significativa
            ajuste = "⬆️ Increase" if valor_actual < valor_ideal else "⬇️ Decrease"
            recomendaciones[param] = f"{ajuste} (Current: {valor_actual:.2f}, Ideal: {valor_ideal:.2f})"

    return recomendaciones if recomendaciones else "The soil is already suitable for this crop."

# 📌 Crear Interfaz en Streamlit
st.title("🌱 Sentinel Ground System")
st.sidebar.header("🔍 Enter Soil Data")

# 📥 Entrada de datos por el usuario en el sidebar
with st.sidebar:
    st.markdown("### 🛠️ Soil Configuration")
    N = st.number_input("Nitrogen (mg/kg)", min_value=0, max_value=300, value=140, 
                        help="Nitrogen is essential for plant growth.")
    P = st.number_input("Phosphorus (mg/kg)", min_value=0, max_value=150, value=45, 
                        help="Phosphorus is important for root and flower development.")
    K = st.number_input("Potassium (mg/kg)", min_value=0, max_value=300, value=120, 
                        help="Potassium helps in disease resistance and water stress.")
    pH = st.number_input("pH", min_value=3.0, max_value=9.0, value=6.5, step=0.1, 
                         help="pH affects nutrient availability in the soil.")
    EC = st.number_input("Electrical Conductivity (uS/cm)", min_value=0.1, max_value=5.0, value=1.2, step=0.1, 
                         help="Electrical conductivity indicates soil salinity.")
    moisture = st.number_input("Moisture (%)", min_value=0, max_value=100, value=55, 
                               help="Moisture is crucial for plant growth.")
    cultivo_deseado = st.selectbox("🌾 Select a crop:", df_csv["Crop"].unique())
    umbral = st.slider("Significant difference threshold", min_value=1, max_value=10, value=5, 
                       help="Adjust the threshold to consider significant differences in soil parameters.")

# 📊 Convertir datos ingresados en un DataFrame
suelo_nuevo = pd.DataFrame([[N, P, K, pH, EC, moisture]], 
                           columns=['N (mg/kg)', 'P (mg/kg)', 'K (mg/kg)', 'pH', 'EC(uS/cm)', 'MOISTURE (%)'])

# 🔍 Predicción y recomendaciones
if st.button("🔎 Predict and Evaluate Compatibility", key="predict_button"):
    if model is None:
        st.error("Could not load the model. Please check the model file.")
    else:
        with st.spinner("Making predictions and generating recommendations..."):
            cultivo_recomendado = model.predict(suelo_nuevo)[0]
            st.success(f"🌾 **Recommended crop based on soil:** {cultivo_recomendado}")

            # Agregar la predicción al historial
            resultado = f"Soil: N={N}, P={P}, K={K}, pH={pH}, EC={EC}, Moisture={moisture} -> Recommended crop: {cultivo_recomendado}"
            if "historial" not in st.session_state:
                st.session_state.historial = []
            st.session_state.historial.append(resultado)

            recomendaciones = recomendar_ajustes(suelo_nuevo.iloc[0], cultivo_deseado, umbral)

            if isinstance(recomendaciones, dict):
                st.info(f"🔄 Recommended adjustments for growing **{cultivo_deseado}**:")
                for param, ajuste in recomendaciones.items():
                    if abs(float(ajuste.split("(")[1].split(",")[0].split(":")[1]) - float(ajuste.split("Ideal:")[1].split(")")[0])) > umbral:
                        st.error(f"⚠️ {param}: {ajuste}")
                    else:
                        st.success(f"✅ {param}: {ajuste}")
            else:
                st.success("✅ The soil is already suitable for this crop.")

            # 📊 Gráfico de radar para comparar suelo actual vs. ideal
            st.subheader("📊 Current Soil vs. Ideal Soil Comparison (Radar Chart)")
            valores_actuales = suelo_nuevo.iloc[0].values
            valores_ideales = df_csv[df_csv["Crop"] == cultivo_deseado].drop(columns=["Crop"]).mean().values

            labels = suelo_nuevo.columns
            num_vars = len(labels)

            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            valores_actuales = np.concatenate((valores_actuales, [valores_actuales[0]]))  # Cerrar el círculo
            valores_ideales = np.concatenate((valores_ideales, [valores_ideales[0]]))  # Cerrar el círculo
            angles += angles[:1]  # Cerrar el círculo

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.fill(angles, valores_actuales, color='blue', alpha=0.25, label="Current Soil")
            ax.fill(angles, valores_ideales, color='green', alpha=0.25, label="Ideal Soil")
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_title(f"Current Soil vs. Ideal Soil for {cultivo_deseado}", size=14, y=1.1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
            st.pyplot(fig)

            # 📊 Tabla comparativa
            st.subheader("📋 Comparison Table")
            comparacion_df = pd.DataFrame({
                "Parameter": labels,
                "Current Soil": valores_actuales[:-1],
                "Ideal Soil": valores_ideales[:-1]
            })
            st.dataframe(comparacion_df)

            # 📥 Exportar recomendaciones
            if isinstance(recomendaciones, dict):
                recomendaciones_df = pd.DataFrame(recomendaciones.items(), columns=["Parameter", "Recommendation"])
                csv = recomendaciones_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download recommendations as CSV",
                    data=csv,
                    file_name="recommendations.csv",
                    mime="text/csv",
                    key="download_button"  # Clave única para este botón
                )

# 📜 Ver historial de predicciones
if "historial" not in st.session_state:
    st.session_state.historial = []

if st.checkbox("📜 View prediction history", key="history_checkbox"):
    st.subheader("📜 Prediction History")
    for i, item in enumerate(st.session_state.historial, 1):
        st.write(f"{i}. {item}")

# ❓ ¿Cómo usar esta aplicación?
if st.checkbox("❓ How to use this application?"):
    st.write("""
    1. Enter the soil values in the corresponding fields.
    2. Select the desired crop.
    3. Adjust the significant difference threshold if necessary.
    4. Press the 'Predict' button to get recommendations.
    5. Explore the charts and download the recommendations in CSV if desired.
    """)