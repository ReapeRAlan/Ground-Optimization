from sklearn.preprocessing import MinMaxScaler

import pandas as pd

# Cargar archivos
csv_file = "datasets/crop_dataset2.csv"
xlsx_file = "datasets/Crop_Predication_dataset.xlsx"

df_csv = pd.read_csv(csv_file)
df_xlsx = pd.read_excel(xlsx_file)

# Ver las primeras filas
print("CSV Dataset:")
print(df_csv.head())

print("\nExcel Dataset:")
print(df_xlsx.head())

# Revisar información de las columnas
print("\nCSV Info:")
print(df_csv.info())

print("\nExcel Info:")
print(df_xlsx.info())




# Normalizar datos numéricos del CSV
scaler = MinMaxScaler()
df_csv_scaled = pd.DataFrame(scaler.fit_transform(df_csv.iloc[:, 1:]), columns=df_csv.columns[1:])

# Normalizar datos del Excel
df_xlsx_scaled = pd.DataFrame(scaler.fit_transform(df_xlsx.iloc[:, :-1]), columns=df_xlsx.columns[:-1])

# Agregar etiquetas de cultivos de nuevo
df_xlsx_scaled['label'] = df_xlsx['label']
import seaborn as sns
import matplotlib.pyplot as plt

# Matriz de correlación sin la columna de cultivos
plt.figure(figsize=(10,6))
sns.heatmap(df_csv.drop(columns=["Crop"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación del Dataset CSV")
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Definir características (X) y variable objetivo (y)
X = df_csv.drop(columns=["Crop"])  # Eliminar la columna de cultivos
y = df_csv["Crop"]

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")
