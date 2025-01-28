import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 📌 Cargar modelo entrenado
model_path = "modelo_cultivo.pkl"
model = joblib.load(model_path)

# 📂 Cargar dataset para recomendaciones
csv_file = "datasets/crop_dataset2.csv"
df_csv = pd.read_csv(csv_file)

# 📌 Función para recomendar ajustes en el suelo
def recomendar_ajustes(suelo_actual, cultivo_deseado):
    cultivos_similares = df_csv[df_csv["Crop"] == cultivo_deseado].drop(columns=["Crop"]).mean()
    
    if cultivos_similares.empty:
        return f"No hay suficiente información sobre {cultivo_deseado} en el dataset."

    recomendaciones = {}

    for param in suelo_actual.keys():
        valor_actual = suelo_actual[param]
        valor_ideal = cultivos_similares[param]
        
        if abs(valor_actual - valor_ideal) > 5:  # Umbral de diferencia significativa
            ajuste = "⬆️ Aumentar" if valor_actual < valor_ideal else "⬇️ Reducir"
            recomendaciones[param] = f"{ajuste} (Actual: {valor_actual:.2f}, Ideal: {valor_ideal:.2f})"

    return recomendaciones if recomendaciones else "El suelo ya es adecuado para este cultivo."

# 📌 Crear Interfaz en Streamlit
st.title("🌱 Sistema de Recomendación para Agricultura Inteligente")
st.subheader("🔍 Ingresa los datos del suelo y selecciona un cultivo")

# 📥 Entrada de datos por el usuario
N = st.number_input("Nitrógeno (mg/kg)", min_value=0, max_value=300, value=140)
P = st.number_input("Fósforo (mg/kg)", min_value=0, max_value=150, value=45)
K = st.number_input("Potasio (mg/kg)", min_value=0, max_value=300, value=120)
pH = st.number_input("pH", min_value=3.0, max_value=9.0, value=6.5, step=0.1)
EC = st.number_input("Conductividad Eléctrica (uS/cm)", min_value=0.1, max_value=5.0, value=1.2, step=0.1)
moisture = st.number_input("Humedad (%)", min_value=0, max_value=100, value=55)

# Selección del cultivo deseado
cultivo_deseado = st.selectbox("🌾 Selecciona un cultivo:", df_csv["Crop"].unique())

# 📊 Convertir datos ingresados en un DataFrame
suelo_nuevo = pd.DataFrame([[N, P, K, pH, EC, moisture]], 
                           columns=['N (mg/kg)', 'P (mg/kg)', 'K (mg/kg)', 'pH', 'EC(uS/cm)', 'MOISTURE (%)'])

# 🔍 Predicción y recomendaciones
if st.button("🔎 Predecir y Evaluar Compatibilidad"):
    cultivo_recomendado = model.predict(suelo_nuevo)[0]
    st.success(f"🌾 **Cultivo recomendado según el suelo:** {cultivo_recomendado}")

    recomendaciones = recomendar_ajustes(suelo_nuevo.iloc[0], cultivo_deseado)

    if isinstance(recomendaciones, dict):
        st.info(f"🔄 Ajustes recomendados para cultivar **{cultivo_deseado}**:")
        for param, ajuste in recomendaciones.items():
            st.write(f"- {param}: {ajuste}")
    else:
        st.success("✅ El suelo ya es adecuado para este cultivo.")

    # 📊 Generar gráficos comparativos
    st.subheader("📊 Comparación del suelo actual vs. suelo ideal")
    fig, ax = plt.subplots(figsize=(8,5))
    valores_actuales = suelo_nuevo.iloc[0].values
    valores_ideales = df_csv[df_csv["Crop"] == cultivo_deseado].drop(columns=["Crop"]).mean().values

    x_labels = suelo_nuevo.columns
    width = 0.35
    ax.bar(x_labels, valores_actuales, width, label="Suelo Actual", color="blue")
    ax.bar(x_labels, valores_ideales, width, label="Suelo Ideal", color="green", alpha=0.7)
    
    ax.set_ylabel("Valores")
    ax.set_title(f"Comparación de Suelo Actual vs. Ideal para {cultivo_deseado}")
    ax.legend()
    st.pyplot(fig)

# 👁️ Mostrar datos generales del dataset
if st.checkbox("📌 Mostrar Datos Generales del Dataset"):
    st.write(df_csv.head())
