"""Entrena la regresión lineal: antigüedad (años) -> PCI."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Carga y limpieza
df = pd.read_csv("datos/pavimento.csv", parse_dates=["fecha"])
print(df.info())
print(df.describe())
df = df.drop_duplicates().dropna(subset=["antiguedad", "pci", "fecha"])
df = df[(df["pci"] >= 0) & (df["pci"] <= 100) & (df["antiguedad"] > 0)]
print(f"\nRegistros válidos: {len(df)}")
assert len(df) >= 50, "Se requieren al menos 50 registros"

# 2. División CRONOLÓGICA: 80 % más antiguo = entrenamiento, 20 % más reciente = prueba
df = df.sort_values("fecha").reset_index(drop=True)
corte = int(len(df) * 0.8)
train, test = df.iloc[:corte], df.iloc[corte:]
X_train, y_train = train[["antiguedad"]], train["pci"]
X_test, y_test = test[["antiguedad"]], test["pci"]

# 3. Entrenamiento
modelo = LinearRegression().fit(X_train, y_train)

# 4. Evaluación en PRUEBA
pred = modelo.predict(X_test)
metricas = {
    "r2": float(r2_score(y_test, pred)),
    "mae": float(mean_absolute_error(y_test, pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    "pendiente": float(modelo.coef_[0]),
    "intercepto": float(modelo.intercept_),
    "n_train": int(len(train)),
    "n_test": int(len(test)),
    "edad_min": int(df["antiguedad"].min()),
    "edad_max": int(df["antiguedad"].max()),
}
print("\n=== MÉTRICAS (conjunto de prueba) ===")
for k, v in metricas.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# 5. Guardado
joblib.dump(modelo, "modelo.pkl")
with open("metricas.json", "w") as f:
    json.dump(metricas, f, indent=2)
print("\nModelo guardado en modelo.pkl y métricas en metricas.json")