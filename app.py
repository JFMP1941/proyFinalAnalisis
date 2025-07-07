import json
import pandas as pd
from collections import defaultdict
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import os

# ───────────────────────────────
# 1. Carga y entrenamiento
# ───────────────────────────────
df2 = pd.read_csv('SINIESTROS_FINAL.csv')
df2.drop(columns=['Unnamed: 0'], inplace=True)
target = 'Gravedad_Indicador_Tradicional'
df2 = df2.dropna(subset=[target])

X = df2.drop(columns=[target])
y = df2[target]

# Pipeline original para calcular importancias
categorical_features = ['Clase', 'Sexo', 'Dia_Semana_Acc', 'Localidad']
numeric_features     = [c for c in X.columns if c not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
      ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)
pipeline_full = Pipeline([
    ('preproc', preprocessor),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline_full.fit(X, y)

# Agregar importancias por variable original
ohe = pipeline_full.named_steps['preproc'].named_transformers_['cat']
ohe_names = ohe.get_feature_names_out(categorical_features)
all_names = list(ohe_names) + numeric_features
importances = pipeline_full.named_steps['rf'].feature_importances_

agg_imp = defaultdict(float)
for name, imp in zip(all_names, importances):
    matched = False
    for cat in categorical_features:
        if name.startswith(cat + "_"):
            agg_imp[cat] += imp
            matched = True
            break
    if not matched:
        agg_imp[name] += imp

# Top‑9 originales
top9 = pd.Series(agg_imp).sort_values(ascending=False).head(9)
top9_features = top9.index.tolist()

# Guardar lista (opcional)
with open('top9_features.json', 'w') as f:
    json.dump(top9_features, f, indent=2)

# Reentreno RF sólo con top‑9
X9 = df2[top9_features]
rf9 = RandomForestClassifier(n_estimators=100, random_state=42)
rf9.fit(X9, y)

# ───────────────────────────────
# 2. Servidor Flask
# ───────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    # Pasa la lista de features al template para generar inputs
    return render_template('index.html', features=top9_features)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Construye un DataFrame con los 9 valores
    ejemplo = pd.DataFrame([{ feat: data.get(feat, 0) for feat in top9_features }])
    pred_class = int(rf9.predict(ejemplo)[0])
    pred_proba = rf9.predict_proba(ejemplo)[0].tolist()
    return jsonify(prediccion=pred_class, probabilidades=pred_proba)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
