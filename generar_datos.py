"""Genera un dataset SIMULADO de inspecciones de pavimento rígido (Huánuco)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 60

tramos = ["Jr. Dos de Mayo", "Av. Universitaria", "Jr. Huallayco", "Av. Los Portales",
          "Jr. Abtao", "Av. Esteban Pavletich", "Jr. Ayacucho", "Av. Malecón Alomía Robles"]
fallas = ["Grietas longitudinales", "Grietas transversales", "Grietas de esquina",
          "Losa dividida", "Despostillamiento", "Escalonamiento"]
severidades = ["Baja", "Media", "Alta"]
transitos = ["Bajo", "Medio", "Alto"]

antiguedad = rng.integers(1, 11, N)
pci = 101 - 8.0 * antiguedad + rng.normal(0, 4, N)
pci = np.clip(pci, 0, 100).round(1)

# severidad coherente con el PCI
sev = np.where(pci > 70, "Baja", np.where(pci > 40, "Media", "Alta"))

df = pd.DataFrame({
    "tramo": rng.choice(tramos, N),
    "fecha": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 700, N), unit="D"),
    "antiguedad": antiguedad,
    "transito": rng.choice(transitos, N),
    "tipo_falla": rng.choice(fallas, N),
    "severidad": sev,
    "pci": pci,
    "origen": "SIMULADO",
})
df = df.sort_values("fecha").reset_index(drop=True)
df.to_csv("datos/pavimento.csv", index=False)
print(f"Dataset SIMULADO generado: {len(df)} registros -> datos/pavimento.csv")