import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 📂 Cargar los datasets
csv_file = "datasets/crop_dataset2.csv"
xlsx_file = "datasets/Crop_Predication_dataset.xlsx"

df_csv = pd.read_csv(csv_file)
df_xlsx = pd.read_excel(xlsx_file)

# 📌 Normalizar los datos numéricos
scaler = MinMaxScaler()
df_csv_scaled = pd.DataFrame(scaler.fit_transform(df_csv.iloc[:, 1:]), columns=df_csv.columns[1:])
df_csv_scaled["Crop"] = df_csv["Crop"]

# 📌 Preparar datos para entrenamiento
X = df_csv.drop(columns=["Crop"])
y = df_csv["Crop"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Cargar modelo si ya existe, sino, entrenar uno nuevo
if os.path.exists("modelo_cultivo.pkl"):
    model = joblib.load("modelo_cultivo.pkl")
    print("✅ Modelo cargado desde 'modelo_cultivo.pkl'")
else:
    print("⚠️ No se encontró un modelo guardado. Entrenando un nuevo modelo...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, "modelo_cultivo.pkl")
    print("✅ Modelo entrenado y guardado como 'modelo_cultivo.pkl'")

# 🔍 Función para recomendar ajustes del suelo
def recomendar_ajustes(suelo_actual, cultivo_deseado):
    cultivos_similares = df_csv[df_csv["Crop"] == cultivo_deseado].drop(columns=["Crop"]).mean()
    
    if cultivos_similares.empty:
        return f"No hay suficiente información sobre {cultivo_deseado} en el dataset."

    recomendaciones = {}

    for param in X.columns:
        valor_actual = suelo_actual[param].values[0]
        valor_ideal = cultivos_similares[param]
        
        if abs(valor_actual - valor_ideal) > 5:  # Umbral de ajuste significativo
            ajuste = "⬆️ Aumentar" if valor_actual < valor_ideal else "⬇️ Reducir"
            recomendaciones[param] = f"{ajuste} (Actual: {valor_actual:.2f}, Ideal: {valor_ideal:.2f})"

    return recomendaciones if recomendaciones else "El suelo ya es adecuado para este cultivo."

# 🔍 Prueba de predicción
suelo_nuevo = pd.DataFrame([[140, 45, 120, 6.5, 1.2, 55]], 
                           columns=['N (mg/kg)', 'P (mg/kg)', 'K (mg/kg)', 'pH', 'EC(uS/cm)', 'MOISTURE (%)'])

cultivo_recomendado = model.predict(suelo_nuevo)[0]
print(f"\n🌱 Cultivo recomendado según el suelo: {cultivo_recomendado}")

# 🔄 Sugerencia de ajustes para otro cultivo
cultivo_deseado = "Maize (corn)"  # Cambia por el cultivo que quieras
print(f"\n🔄 Recomendaciones para adaptar el suelo a {cultivo_deseado}:")
print(recomendar_ajustes(suelo_nuevo, cultivo_deseado))
