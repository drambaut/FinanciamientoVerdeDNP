# app.py
# -*- coding: utf-8 -*-
import os
from io import BytesIO

import pandas as pd
import streamlit as st

import modelo  # <-- tu módulo intacto con Procesa_CATS (usa 'clave' y exporta archivos)

# ============================================================
# CONFIGURACIÓN DE PÁGINA (logo y título)
# ============================================================
DNP_LOGO = "https://terridata.dnp.gov.co/assets/homev3/img/png/logo-dnp.png"
st.set_page_config(
    page_title="Clasificador Financiemiento verde (MÓDULO/ BIO/ CC/ GRD)",
    page_icon=DNP_LOGO,
    layout="wide",
)

# ============================================================
# PALETA Y ESTILOS (look & feel DNP)
# ============================================================
PRIMARY = "#004884"   # Azul institucional GOV.CO
ACCENT  = "#1B9DD9"   # Azul claro de apoyo
INK     = "#0B1F32"   # Azul muy oscuro para títulos
GRAYBG  = "#F5F7FA"   # Fondo gris claro
BORDER  = "#D9E2EC"   # Bordes suaves

CUSTOM_CSS = f"""
<style>
/* Fondo general y tipografía */
html, body, [data-testid="stAppViewContainer"] {{
    background: {GRAYBG};
}}
[data-testid="stHeader"] {{ background: transparent; }}

/* Cabecera con franja institucional */
.dnp-topbar {{
    background: linear-gradient(90deg, {PRIMARY} 0%, {PRIMARY} 70%, {ACCENT} 100%);
    color: white;
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.06);
}}
.dnp-title {{
    font-weight: 700;
    font-size: 1.35rem;
    line-height: 1.2;
    margin: 0;
    color: #ffffff;
}}
.dnp-subtitle {{
    margin: 2px 0 0 0;
    opacity: 0.9;
}}

/* Layout y tarjetas */
.block-container {{
    padding-top: 1.2rem;
}}
.dnp-card {{
    background: #fff;
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.03);
}}

/* Tablas */
[data-testid="stVerticalBlock"] .stDataFrame {{
    background: white;
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 6px;
}}

/* Botones */
.stButton>button {{
    background: {PRIMARY};
    color: #fff;
    border: 1px solid {PRIMARY};
    border-radius: 12px;
    padding: 0.6rem 0.9rem;
    font-weight: 600;
    transition: all .15s ease-in-out;
}}
.stButton>button:hover {{
    filter: brightness(1.05);
    transform: translateY(-1px);
}}

/* Botón de descarga */
[data-testid="stDownloadButton"] > button {{
    border-radius: 12px;
    border: 1px solid {PRIMARY};
}}

/* Badges */
.badge {{
    display: inline-block;
    background: #fff;
    color: {PRIMARY};
    border: 1px solid {PRIMARY};
    padding: 2px 8px;
    font-size: 0.80rem;
    border-radius: 999px;
    margin-right: 6px;
}}

/* Separadores suaves */
hr {{
    border: none;
    border-top: 1px solid {BORDER};
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# UTILIDADES (descarga + guardado local por archivo)
# ============================================================
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convierte DataFrame a bytes Excel (xlsxwriter)."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

def save_and_download(df: pd.DataFrame, filename: str, export_dir: str):
    """Guarda en ./exports/<archivo_base>/ y devuelve bytes para descarga."""
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, filename)
    df.to_excel(path, index=False)
    return to_excel_bytes(df)

# ============================================================
# CABECERA MARCA (título en blanco)
# ============================================================
with st.container():
    st.markdown(
        f"""
        <div class="dnp-topbar">
          <div style="display:flex; gap:14px; align-items:center;">
            <img src="{DNP_LOGO}" alt="DNP" style="height:120px; background:#ffffff; border-radius:8px; padding:6px;">
            <div>
              <h1 class="dnp-title">Clasificador Financiemiento verde (MÓDULO/ BIO/ CC/ GRD)</h1>
              <div class="dnp-subtitle">
                <span class="badge">DNP</span>
                <span class="badge">Financiamiento verde</span>
                <span class="badge">Zero-shot (mDeBERTa)</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# INPUT DE ARCHIVO + CHECK DE CLAVE
# ============================================================
with st.container():
    st.markdown('<div class="dnp-card">', unsafe_allow_html=True)
    st.markdown("**Sube tu archivo de proyectos (.xlsx) con columnas `bpin` y `texto`.**")
    uploaded_file = st.file_uploader("📂 Cargar archivo Excel", type=["xlsx"], label_visibility="collapsed")

    # Checkbox para encender la CLAVE = "SI"
    clave_checked = st.checkbox("Procesar 'MÓDULO' antes de las ramas (Clave = \"SI\")", value=True)
    clave = "SI" if clave_checked else "NO"

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# BOTONES (misma fila)
# ============================================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    run_modulo = st.button("Módulo", use_container_width=True)
with col2:
    run_grd = st.button("GRD (Gestión de Riesgo de Desastres)", use_container_width=True)
with col3:
    run_bio = st.button("BIO (Biodiversidad)", use_container_width=True)
with col4:
    run_cc = st.button("CC (Cambio Climático)", use_container_width=True)

st.write("")  # separación

# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================
if uploaded_file:
    data = pd.read_excel(uploaded_file)
    missing = [c for c in ("texto", "bpin") if c not in data.columns]
    if missing:
        st.error(f"El archivo debe contener las columnas requeridas: {', '.join(missing)}.")
    else:
        # Carpeta dinámica basada en el nombre del archivo
        file_base = os.path.splitext(uploaded_file.name)[0]
        export_dir = os.path.join("exports", file_base)
        os.makedirs(export_dir, exist_ok=True)

        if run_modulo or run_grd or run_bio or run_cc:
            # Ejecutar pipeline completo de tu módulo (sin tocar modelo.py)
            with st.spinner("🔎 Procesando con Procesa_CATS..."):
                # Nota: Pasamos 'clave' según el checkbox
                modulo_results, CC_resultados, GRD_resultados, BIO_resultados = modelo.Procesa_CATS(
                    data, export_dir, clave=clave, umbral=0.25
                )

            st.success(f"✅ Listo. Resultados guardados en `{export_dir}`")
            st.write("")

            # -----------------------
            # Helper para mostrar/descargar
            # -----------------------
            def mostrar_df(df, filename, titulo):
                if df is None or df.empty:
                    st.warning(f"⚠️ No se generó {titulo} o está vacío.")
                    return
                with st.container():
                    st.markdown('<div class="dnp-card">', unsafe_allow_html=True)
                    st.subheader(titulo)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    excel_bytes = save_and_download(df, filename, export_dir)
                    st.download_button(
                        label=f"📥 Descargar {filename}",
                        data=excel_bytes,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

            # -----------------------
            # Mostrar según botón
            # -----------------------
            if run_modulo:
                # Si 'clave' es NO, modulo_results = data_prueba (sin clasificación)
                mostrar_df(modulo_results, "modulo_results.xlsx", "Resultados del Módulo")
                # Intentamos cargar también la versión “mayor” si existe (cuando clave = SI)
                mayor_path = os.path.join(export_dir, "modulo_results_mayor.xlsx")
                if os.path.exists(mayor_path):
                    try:
                        mayor_df = pd.read_excel(mayor_path)
                        mostrar_df(mayor_df, "modulo_results_mayor.xlsx", "Módulo (versión mayor)")
                    except Exception:
                        st.warning(
                            "No fue posible leer `modulo_results_mayor.xlsx` para previsualización, pero sí se guardó en disco."
                        )

            elif run_grd:
                # En tu modelo, GRD no siempre genera cat3; se maneja con get().
                for key, titulo in [
                    ("cat1", "GRD - Categoría 1"),
                    ("cat2", "GRD - Categoría 2"),
                    ("cat3", "GRD - Categoría 3"),
                ]:
                    df = GRD_resultados.get(key, pd.DataFrame())
                    mostrar_df(df, f"GRD_{key}_results.xlsx", titulo)
                    # Versión “mayor” si existe
                    mayor_fn = f"GRD_{key}_results_mayor.xlsx"
                    mayor_path = os.path.join(export_dir, mayor_fn)
                    if os.path.exists(mayor_path):
                        try:
                            mayor_df = pd.read_excel(mayor_path)
                            mostrar_df(mayor_df, mayor_fn, f"{titulo} (versión mayor)")
                        except Exception:
                            st.warning(
                                f"No fue posible leer `{mayor_fn}` para previsualización, pero sí se guardó en disco."
                            )

            elif run_bio:
                # Tu función exporta BIO_cat1_results.xlsx
                mostrar_df(BIO_resultados, "BIO_cat1_results.xlsx", "BIO - Categoría 1")
                # (Tu versión mayor está comentada en el modelo; si decides activarla, se mostrará aquí)
                mayor_path = os.path.join(export_dir, "BIO_results_mayor.xlsx")
                if os.path.exists(mayor_path):
                    try:
                        mayor_df = pd.read_excel(mayor_path)
                        mostrar_df(mayor_df, "BIO_results_mayor.xlsx", "BIO (versión mayor)")
                    except Exception:
                        st.warning(
                            "No fue posible leer `BIO_results_mayor.xlsx` para previsualización, pero sí se guardó en disco."
                        )

            elif run_cc:
                for key, titulo in [
                    ("cat1", "CC - Categoría 1"),
                    ("cat2", "CC - Categoría 2"),
                    ("cat3", "CC - Categoría 3"),
                ]:
                    df = CC_resultados.get(key, pd.DataFrame())
                    mostrar_df(df, f"CC_{key}_results.xlsx", titulo)
                    mayor_fn = f"CC_{key}_results_mayor.xlsx"
                    mayor_path = os.path.join(export_dir, mayor_fn)
                    if os.path.exists(mayor_path):
                        try:
                            mayor_df = pd.read_excel(mayor_path)
                            mostrar_df(mayor_df, mayor_fn, f"{titulo} (versión mayor)")
                        except Exception:
                            st.warning(
                                f"No fue posible leer `{mayor_fn}` para previsualización, pero sí se guardó en disco."
                            )
else:
    st.info("👆 Sube un archivo Excel para comenzar.")

# ============================================================
# PIE DE PÁGINA LIGERO
# ============================================================
st.write("")
st.markdown(
    """
    <div style="text-align:center; opacity:.7; font-size:.9rem; margin-top:12px;">
      Departamento Nacional de Planeación – Prototipo interno de apoyo a clasificación de financiamiento verde.
    </div>
    """,
    unsafe_allow_html=True,
)
