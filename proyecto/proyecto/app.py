from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
app.secret_key = "clave_secreta"

# Cargar modelos
models = {
    "rna": {
        "modelo": pickle.load(open("models/modelo_rna.pkl", "rb")),
        "scaler": pickle.load(open("models/escalador_rna.pkl", "rb"))
    },
    "logistica": {
        "modelo": pickle.load(open("models/modelo_logistica.pkl", "rb")),
        "scaler": pickle.load(open("models/escalador.pkl", "rb"))
    }
}

# Variables del modelo
variables = [
    "AST (SGOT)", "ALT (SGPT)", "total_proteins", "direct_bilirubin",
    "total_bilirubin", "lymphocytes", "hemoglobin", "hematocrit",
    "age", "urea", "red_blood_cells", "monocytes",
    "white_blood_cells", "creatinine", "ALP (alkaline_phosphatase)"
]

nombres_variables = {
    "AST (SGOT)": "AST (SGOT)",
    "ALT (SGPT)": "ALT (SGPT)",
    "total_proteins": "Proteínas Totales",
    "direct_bilirubin": "Bilirrubina Directa",
    "total_bilirubin": "Bilirrubina Total",
    "lymphocytes": "Linfocitos",
    "hemoglobin": "Hemoglobina",
    "hematocrit": "Hematocrito",
    "age": "Edad",
    "urea": "Urea",
    "red_blood_cells": "Glóbulos Rojos",
    "monocytes": "Monocitos",
    "white_blood_cells": "Glóbulos Blancos",
    "creatinine": "Creatinina",
    "ALP (alkaline_phosphatase)": "Fosfatasa Alcalina (ALP)"
}

# Dataset base para rangos
data = pd.read_excel("DEMALE-HSJM_2025_data (1).xlsx")
rangos = {}
for v in variables:
    min_val = float(data[v].min())
    max_val = float(data[v].max())
    step_val = 1 if v == "age" else round((max_val - min_val) / 100, 2)
    rangos[v] = {"min": min_val, "max": max_val, "step": step_val}

# Rutas
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/individual', methods=['GET', 'POST'])
def individual():
    resultado = None
    if request.method == 'POST':
        try:
            modelo_seleccionado = request.form.get("modelo")
            if modelo_seleccionado not in models:
                flash("Modelo no válido", "error")
                return redirect(url_for('individual'))

            datos = [float(request.form.get(v)) for v in variables]
            df = pd.DataFrame([datos], columns=variables)

            scaler = models[modelo_seleccionado]["scaler"]
            modelo = models[modelo_seleccionado]["modelo"]

            pred = modelo.predict(scaler.transform(df))[0]
            resultado = "🩸 Positivo para enfermedad" if pred == 1 else "💚 Negativo para enfermedad"

        except Exception as e:
            flash(f"Error: {str(e)}", "error")

    return render_template('individual.html', resultado=resultado, columnas=variables, nombres=nombres_variables, rangos=rangos)

@app.route('/lotes', methods=['GET', 'POST'])
def lotes():
    tabla = None
    metricas = None

    if request.method == 'POST':
        try:
            modelo_seleccionado = request.form.get("modelo")
            archivo = request.files.get("dataset")

            if not archivo:
                flash("⚠️ Debes subir un archivo antes de predecir.", "error")
                return redirect(url_for('lotes'))

            if modelo_seleccionado not in models:
                flash("⚠️ Modelo no válido o no encontrado.", "error")
                return redirect(url_for('lotes'))

            # Leer archivo
            if archivo.filename.endswith('.csv'):
                df = pd.read_csv(archivo)
            elif archivo.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(archivo)
            else:
                flash("⚠️ Formato no soportado. Usa un archivo .csv o .xlsx", "error")
                return redirect(url_for('lotes'))

            # Normalizar nombres y mapear a las esperadas
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            mapa_columnas = {
                "ast": "AST (SGOT)", "ast_(sgot)": "AST (SGOT)",
                "alt": "ALT (SGPT)", "alt_(sgpt)": "ALT (SGPT)",
                "proteinas_totales": "total_proteins", "total_proteins": "total_proteins",
                "bilirrubina_directa": "direct_bilirubin", "direct_bilirubin": "direct_bilirubin",
                "bilirrubina_total": "total_bilirubin", "total_bilirubin": "total_bilirubin",
                "linfocitos": "lymphocytes", "hemoglobina": "hemoglobin",
                "hematocrito": "hematocrit", "edad": "age",
                "urea": "urea", "globulos_rojos": "red_blood_cells",
                "monocitos": "monocytes", "globulos_blancos": "white_blood_cells",
                "creatinina": "creatinine", "fosfatasa_alcalina": "ALP (alkaline_phosphatase)",
                "alp": "ALP (alkaline_phosphatase)"
            }

            df_normalizado = pd.DataFrame()
            for col_modelo in variables:
                encontrado = None
                for col in df.columns:
                    if col in mapa_columnas and mapa_columnas[col] == col_modelo:
                        encontrado = col
                        break
                    if col == col_modelo.lower().replace(" ", "_"):
                        encontrado = col
                        break
                if encontrado:
                    df_normalizado[col_modelo] = df[encontrado]
                else:
                    df_normalizado[col_modelo] = 0

            df = df_normalizado

            # Rellenar valores vacíos con la media
            df[variables] = df[variables].fillna(df[variables].mean())

            # Escalar y predecir
            scaler = models[modelo_seleccionado]["scaler"]
            modelo = models[modelo_seleccionado]["modelo"]
            X = scaler.transform(df[variables])
            predicciones = modelo.predict(X)
            df["Predicción"] = ["🩸 Positivo" if p == 1 else "💚 Negativo" for p in predicciones]

            # Métricas si hay etiqueta real
            posibles_columnas_y = ["target", "real", "diagnóstico", "etiqueta", "label"]
            col_y = next((c for c in posibles_columnas_y if c in df.columns), None)

            if col_y:
                y_real = df[col_y]
                y_pred = predicciones

                acc = accuracy_score(y_real, y_pred)
                prec = precision_score(y_real, y_pred)
                rec = recall_score(y_real, y_pred)
                f1 = f1_score(y_real, y_pred)

                metricas = {
                    "accuracy": round(acc * 100, 2),
                    "precision": round(prec * 100, 2),
                    "recall": round(rec * 100, 2),
                    "f1": round(f1 * 100, 2)
                }

            # Guardar resultados
            os.makedirs("uploads", exist_ok=True)
            ruta_salida = os.path.join("uploads", "resultados_prediccion.xlsx")
            df.to_excel(ruta_salida, index=False)

            tabla = df.head(40).to_html(classes="table table-striped text-center", index=False, justify="center")
            flash("✅ Predicción completada correctamente.", "success")

        except Exception as e:
            flash(f"❌ Ocurrió un error durante la predicción: {str(e)}", "error")
            return redirect(url_for('lotes'))

    return render_template('lotes.html', tabla=tabla, metricas=metricas)

@app.route('/descargar_resultados')
def descargar_resultados():
    ruta_salida = os.path.join("uploads", "resultados_prediccion.xlsx")
    if not os.path.exists(ruta_salida):
        flash("⚠️ No hay resultados disponibles para descargar.", "error")
        return redirect(url_for('lotes'))
    return send_file(ruta_salida, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)