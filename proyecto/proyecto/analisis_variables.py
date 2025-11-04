import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 🔹 Cargar el archivo Excel
data = pd.read_excel("DEMALE-HSJM_2025_data (1).xlsx")

# 🔹 Mostrar columnas para verificar
print("Columnas del dataset:\n", data.columns.tolist(), "\n")

# 🔹 Asegurarte de que la columna objetivo se llama 'diagnosis'
target_col = 'diagnosis'

# 🔹 Codificar texto a números si hay datos categóricos
le = LabelEncoder()
data[target_col] = le.fit_transform(data[target_col])

# 🔹 Separar variables predictoras y objetivo
X = data.drop(columns=[target_col])
y = data[target_col]

# 🔹 Entrenar modelo de bosque aleatorio
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 🔹 Obtener importancia de las variables
importances = model.feature_importances_
feat_importances = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': importances
}).sort_values(by='Importancia', ascending=False)

# 🔹 Mostrar las 15 más importantes
print("🔝 Variables más importantes:")
print(feat_importances.head(15))

# 🔹 Graficar
plt.figure(figsize=(10,6))
plt.barh(feat_importances['Variable'].head(15), feat_importances['Importancia'].head(15))
plt.gca().invert_yaxis()
plt.title('Importancia de las Variables (Top 15)')
plt.xlabel('Importancia')
plt.tight_layout()
plt.show()