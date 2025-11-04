import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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

data = data[variables + [objetivo]].dropna()
X = data[variables]
y = data[objetivo]

# División
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Red neuronal
rna = MLPClassifier(hidden_layer_sizes=(15, 10), max_iter=800, random_state=42)
rna.fit(X_train_scaled, y_train)

accuracy = accuracy_score(y_test, rna.predict(X_test_scaled))
print(f"Precisión del modelo RNA: {accuracy:.2f}")

# Guardar
with open("models/modelo_rna.pkl", "wb") as f:
    pickle.dump(rna, f)
with open("models/escalador_rna.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Modelo RNA guardado correctamente en 'models/'")