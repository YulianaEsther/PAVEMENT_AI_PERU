"""PAVEMENT AI - Predicción del PCI en pavimento rígido (Huánuco) con regresión lineal."""
import json
import os
import uuid
from datetime import date

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from deteccion import analizar_imagen

st.set_page_config(page_title="PAVEMENT AI", page_icon="🛣️", layout="wide")

RUTA_DATOS = "datos/pavimento.csv"
RUTA_REGISTRO = "datos/registro_patologias.csv"
CARPETA_FOTOS = "fotos"
PATOLOGIAS = ["Grietas longitudinales", "Grietas transversales", "Grietas de esquina",
              "Losa dividida", "Despostillamiento", "Escalonamiento"]
SEVERIDADES = ["Baja", "Media", "Alta"]
COLUMNAS_REG = ["id", "tramo", "fecha", "patologia", "severidad", "foto", "observaciones"]
os.makedirs(CARPETA_FOTOS, exist_ok=True)

# ---------- estilo profesional (tema oscuro) ----------
_t = pio.templates["plotly_dark"]
_t.layout.paper_bgcolor = "rgba(0,0,0,0)"
_t.layout.plot_bgcolor = "rgba(0,0,0,0)"
_t.layout.colorway = ["#22b8ff", "#f5a623", "#ff5d6c", "#34d399", "#a78bfa", "#fbbf24"]
_t.layout.font = dict(family="Inter, Segoe UI, sans-serif", color="#cfe0f1")
_t.layout.xaxis.gridcolor = "#1f3550"
_t.layout.yaxis.gridcolor = "#1f3550"
pio.templates.default = "plotly_dark"
COLOR_SEV = {"Baja": "#34d399", "Media": "#f5a623", "Alta": "#ff5d6c"}


def grafico(fig, use_container_width=True):
    st.plotly_chart(fig, use_container_width=use_container_width, theme=None)


st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg, #0f2f52 0%, #0b1623 70%); border: 1px solid #1f3550;
       border-left: 5px solid #22b8ff; border-radius: 14px; padding: 26px 30px; margin-bottom: 18px;}
.hero small {color: #22b8ff; letter-spacing: 3px; font-weight: 600;}
.hero h1 {margin: 6px 0 4px 0; font-size: 2.6rem; color: #ffffff; letter-spacing: 1px;}
.hero p {margin: 0; color: #9db4cc;}
div[data-testid="stMetric"] {background: #122338; border: 1px solid #1f3550; border-radius: 12px;
       padding: 16px 18px; box-shadow: 0 4px 14px rgba(0,0,0,.25);}
div[data-testid="stMetricLabel"] {color: #9db4cc;}
div[data-testid="stMetricValue"] {color: #22b8ff; font-weight: 700;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {background: #122338; border-radius: 8px 8px 0 0; padding: 10px 18px;}
.stTabs [aria-selected="true"] {background: #1a3a5c; color: #22b8ff;}
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
       border-radius: 8px; font-weight: 600; border: 1px solid #22b8ff;}
h2, h3 {color: #ffffff; font-weight: 600;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------- utilidades ----------
@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load("modelo.pkl")
    except Exception:
        return None


@st.cache_data
def cargar_metricas():
    try:
        with open("metricas.json") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def cargar_datos():
    try:
        return pd.read_csv(RUTA_DATOS, parse_dates=["fecha"])
    except Exception:
        return None


def cargar_registro():
    if os.path.exists(RUTA_REGISTRO):
        try:
            return pd.read_csv(RUTA_REGISTRO)
        except Exception:
            pass
    return pd.DataFrame(columns=COLUMNAS_REG)


def condicion(pci):
    if pci > 85: return "Bueno"
    if pci > 70: return "Satisfactorio"
    if pci > 55: return "Regular"
    if pci > 40: return "Pobre"
    if pci > 25: return "Muy pobre"
    if pci > 10: return "Grave"
    return "Fallado"


modelo = cargar_modelo()
metricas = cargar_metricas()
datos = cargar_datos()

st.markdown("""
<div class="hero">
  <small>SENATI · INTELIGENCIA ARTIFICIAL · MACHINE LEARNING</small>
  <h1>🛣️ PAVEMENT AI</h1>
  <p>Predicción del deterioro del pavimento rígido mediante regresión lineal · Huánuco, Perú</p>
</div>
""", unsafe_allow_html=True)
st.info("Datos de entrenamiento **SIMULADOS** con fines educativos. El modelo no reemplaza "
        "la metodología PCI ni una inspección estructural.")

if modelo is None or metricas is None or datos is None:
    st.error("Falta el modelo o los datos. Ejecuta primero: `python generar_datos.py` y `python entrenar.py`.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📝 Registro de patologías",
                                   "🔮 Simulador PCI", "⬇️ Exportar"])

# ---------- 1. DASHBOARD ----------
with tab1:
    st.subheader("Rendimiento del modelo (conjunto de prueba)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{metricas['r2']:.3f}")
    c2.metric("MAE", f"{metricas['mae']:.2f} pts")
    c3.metric("RMSE", f"{metricas['rmse']:.2f} pts")
    c4.metric("Pendiente", f"{metricas['pendiente']:.2f} PCI/año")
    st.caption(f"PCI = {metricas['intercepto']:.2f} + ({metricas['pendiente']:.2f}) × antigüedad · "
               f"Entrenamiento: {metricas['n_train']} registros · Prueba: {metricas['n_test']} registros")

    fig = px.scatter(datos, x="antiguedad", y="pci", color="severidad", color_discrete_map=COLOR_SEV,
                     hover_data=["tramo", "tipo_falla", "fecha"],
                     labels={"antiguedad": "Antigüedad (años)", "pci": "PCI observado"},
                     title="Datos históricos y recta de regresión")
    xs = list(range(metricas["edad_min"], metricas["edad_max"] + 1))
    ys = modelo.predict(pd.DataFrame({"antiguedad": xs}))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Regresión lineal",
                             line=dict(color="#f5a623", dash="dash", width=3)))
    grafico(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        cnt = datos["tipo_falla"].value_counts().reset_index()
        cnt.columns = ["Patología", "Cantidad"]
        grafico(px.bar(cnt, x="Patología", y="Cantidad", title="Patologías en el dataset"),
                        use_container_width=True)
    with col_b:
        datos_c = datos.assign(Condición=datos["pci"].apply(condicion))
        cc = datos_c["Condición"].value_counts().reset_index()
        cc.columns = ["Condición", "Cantidad"]
        grafico(px.pie(cc, names="Condición", values="Cantidad",
                               title="Condición de los tramos"), use_container_width=True)

# ---------- 2. REGISTRO ----------
with tab2:
    # --- 2a. Detector de daños en imágenes ---
    st.subheader("🔍 Detector de daños en imágenes")
    st.caption("Suba una fotografía del pavimento: el sistema resalta las grietas (OpenCV), calcula el "
               "porcentaje de área afectada y sugiere la severidad.")
    archivo = st.file_uploader("Fotografía del pavimento (png, jpg, jpeg)", type=["png", "jpg", "jpeg"],
                               key="foto_detector")
    if archivo is not None:
        try:
            contenido = archivo.getvalue()
            res = analizar_imagen(contenido)
            st.session_state["foto"] = {"bytes": contenido, "ext": archivo.name.rsplit(".", 1)[-1].lower(),
                                        "severidad": res["severidad"]}
            ci1, ci2 = st.columns(2)
            ci1.image(res["original"], caption="Imagen original", use_container_width=True)
            ci2.image(res["detectada"], caption="Daño detectado (rojo)", use_container_width=True)
            d1, d2, d3 = st.columns(3)
            d1.metric("Área agrietada", f"{res['porcentaje']:.2f} %")
            d2.metric("Grietas detectadas", res["num_grietas"])
            d3.metric("Severidad sugerida", res["severidad"])
            if res["num_grietas"] == 0:
                st.success("No se detectaron grietas visibles en la imagen.")
            st.caption("⚠️ Análisis automático referencial: sombras, manchas o juntas pueden generar "
                       "falsos positivos. No reemplaza la inspección técnica.")
        except Exception as e:
            st.session_state.pop("foto", None)
            st.error(f"No se pudo analizar la imagen: {e}")
    else:
        st.session_state.pop("foto", None)

    st.divider()

    # --- 2b. Formulario de registro ---
    st.subheader("Registrar patología")
    foto_sel = st.session_state.get("foto")
    sev_defecto = SEVERIDADES.index(foto_sel["severidad"]) if foto_sel else 0
    with st.form("form_patologia", clear_on_submit=True):
        tramo = st.text_input("Tramo / vía (obligatorio)", max_chars=80)
        fecha = st.date_input("Fecha de inspección", value=date.today(), max_value=date.today())
        patologia = st.selectbox("Tipo de patología", PATOLOGIAS)
        severidad = st.selectbox("Severidad", SEVERIDADES, index=sev_defecto)
        obs = st.text_area("Observaciones", max_chars=300)
        if foto_sel:
            st.caption("📎 Se adjuntará la fotografía analizada.")
        enviar = st.form_submit_button("Guardar registro")

    if enviar:
        if not tramo.strip():
            st.error("El tramo es obligatorio.")
        else:
            try:
                nombre_foto = ""
                if foto_sel:
                    nombre_foto = f"{uuid.uuid4().hex}.{foto_sel['ext']}"
                    with open(os.path.join(CARPETA_FOTOS, nombre_foto), "wb") as f:
                        f.write(foto_sel["bytes"])
                nuevo = pd.DataFrame([{
                    "id": uuid.uuid4().hex[:8], "tramo": tramo.strip(), "fecha": fecha,
                    "patologia": patologia, "severidad": severidad,
                    "foto": nombre_foto, "observaciones": obs.strip()}])
                reg = pd.concat([cargar_registro(), nuevo], ignore_index=True)
                reg.to_csv(RUTA_REGISTRO, index=False)
                st.success("Registro guardado correctamente.")
            except Exception as e:
                st.error(f"No se pudo guardar el registro: {e}")

    st.subheader("Registros guardados")
    reg = cargar_registro()
    if reg.empty:
        st.info("Aún no hay registros.")
    else:
        st.dataframe(reg, use_container_width=True)
        fila = reg[reg["foto"].notna() & (reg["foto"] != "")]
        if not fila.empty:
            st.caption("Última fotografía registrada")
            ruta = os.path.join(CARPETA_FOTOS, fila.iloc[-1]["foto"])
            if os.path.exists(ruta):
                st.image(ruta, width=300)

# ---------- 3. SIMULADOR ----------
with tab3:
    st.subheader("Simulador de PCI futuro")
    c1, c2 = st.columns(2)
    edad = c1.number_input("Antigüedad actual (años)", min_value=1, max_value=30, value=5, step=1)
    horizonte = c2.number_input("Horizonte de predicción (años)", min_value=1, max_value=10, value=3, step=1)

    if st.button("Ejecutar predicción", type="primary"):
        try:
            futuro = int(edad + horizonte)
            pci_act = float(min(100, max(0, modelo.predict(pd.DataFrame({"antiguedad": [edad]}))[0])))
            pci_fut = float(min(100, max(0, modelo.predict(pd.DataFrame({"antiguedad": [futuro]}))[0])))
            m1, m2, m3 = st.columns(3)
            m1.metric("PCI actual estimado", f"{pci_act:.1f}")
            m2.metric("PCI futuro estimado", f"{pci_fut:.1f}", f"{pci_fut - pci_act:.1f}")
            m3.metric("Condición futura", condicion(pci_fut))

            if futuro > metricas["edad_max"] or edad < metricas["edad_min"]:
                st.warning(f"⚠️ Extrapolación: el modelo se entrenó con antigüedades de "
                           f"{metricas['edad_min']} a {metricas['edad_max']} años. "
                           "La predicción fuera de ese rango es poco confiable.")

            rango = list(range(1, max(futuro, metricas["edad_max"]) + 3))
            linea = [min(100, max(0, v)) for v in modelo.predict(pd.DataFrame({"antiguedad": rango}))]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=datos["antiguedad"], y=datos["pci"], mode="markers", name="Historial"))
            fig.add_trace(go.Scatter(x=rango, y=linea, mode="lines", name="Modelo",
                                     line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=[futuro], y=[pci_fut], mode="markers", name="Predicción",
                                     marker=dict(size=14, color="#ff5d6c", line=dict(width=2, color="white"))))
            fig.update_layout(xaxis_title="Antigüedad (años)", yaxis_title="PCI", yaxis_range=[0, 105])
            grafico(fig, use_container_width=True)

            st.session_state["ultima"] = {"antiguedad_actual": int(edad), "horizonte": int(horizonte),
                                          "pci_actual": round(pci_act, 1), "pci_futuro": round(pci_fut, 1),
                                          "condicion_futura": condicion(pci_fut)}
        except Exception as e:
            st.error(f"Error al predecir: {e}")

# ---------- 4. EXPORTAR ----------
with tab4:
    st.subheader("Exportar resultados")
    reg = cargar_registro()
    st.download_button("Descargar registro de patologías (CSV)",
                       reg.to_csv(index=False).encode("utf-8"), "registro_patologias.csv", "text/csv")
    st.download_button("Descargar dataset de entrenamiento (CSV)",
                       datos.to_csv(index=False).encode("utf-8"), "pavimento_simulado.csv", "text/csv")
    if "ultima" in st.session_state:
        st.download_button("Descargar última predicción (CSV)",
                           pd.DataFrame([st.session_state["ultima"]]).to_csv(index=False).encode("utf-8"),
                           "prediccion_pci.csv", "text/csv")
    else:
        st.info("Ejecuta una predicción en el simulador para poder exportarla.")