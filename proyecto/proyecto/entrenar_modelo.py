import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Crear carpeta models si no existe
os.makedirs("models", exist_ok=True)

# Cargar dataset
data = pd.read_excel("DEMALE-HSJM_2025_data (1).xlsx")

# Variables seleccionadas
variables = [
    "AST (SGOT)", "ALT (SGPT)", "total_proteins", "direct_bilirubin",
    "total_bilirubin", "lymphocytes", "hemoglobin", "hematocrit",
    "age", "urea", "red_blood_cells", "monocytes",
    "white_blood_cells", "creatinine", "ALP (alkaline_phosphatase)"
]

objetivo = "diagnosis"

# Eliminar nulos y separar
data = data[variables + [objetivo]].dropna()
X = data[variables]
y = data[objetivo]

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo
modelo = LogisticRegression(max_iter=500, random_state=42)
modelo.fit(X_train_scaled, y_train)

# Evaluar
accuracy = accuracy_score(y_test, modelo.predict(X_test_scaled))
print(f"Precisión del modelo: {accuracy:.2f}")

# Guardar modelo y escalador
with open("models/modelo_logistica.pkl", "wb") as f:
    pickle.dump(modelo, f)
with open("models/escalador.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(" Modelo y escalador guardados correctamente en 'models/'")