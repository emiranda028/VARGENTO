# app.py

import streamlit as st
import pandas as pd
from PIL import Image
import io
import pickle
from fpdf import FPDF
import base64
import plotly.express as px

st.set_page_config(layout="wide", page_title="VARGENTO - Análisis VAR Inteligente", page_icon="⚽")

# Cargar modelo y recursos
@st.cache_resource
def cargar_modelo():
    with open("modelo.pkl", "rb") as f:
        modelo = pickle.load(f)
    with open("vectorizador.pkl", "rb") as f:
        vectorizador = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    df = pd.read_csv("VAR_Limpio_Generado.csv", encoding="utf-8")
    return modelo, vectorizador, le, df

modelo, vectorizador, le, df_data = cargar_modelo()

# UI inicial
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("## ⚽ VARGENTO - Asistente VAR Inteligente")
    st.write("Describí la jugada, subí evidencia, y recibí una decisión sugerida con IA.")
with col2:
    st.image("https://media.tenor.com/xOb4uwv-VV8AAAAC/var-checking.gif", use_column_width=True)

st.markdown("---")
st.markdown("### 🎥 Análisis de jugada")

# Entrada de usuario
col1, col2 = st.columns(2)
with col1:
    texto_jugada = st.text_area("✍️ Describí la jugada", placeholder="Ej: Falta dentro del área tras un córner")
    archivo_subido = st.file_uploader("📁 Subí imagen o video (opcional)", type=["jpg", "jpeg", "png", "mp4"])
    link_youtube = st.text_input("🔗 Link de YouTube (opcional):")

with col2:
    st.markdown("##### Resultado")
    if st.button("🔍 Predecir decisión"):
        if not texto_jugada.strip():
            st.warning("⚠️ Ingresá una descripción para analizar.")
        else:
            X_nueva = vectorizador.transform([texto_jugada])
            pred = modelo.predict(X_nueva)[0]
            pred_label = le.inverse_transform([pred])[0]
            st.success(f"📢 Decisión sugerida: **{pred_label}**")

            if archivo_subido:
                if archivo_subido.type.startswith("video"):
                    st.video(archivo_subido)
                elif archivo_subido.type.startswith("image"):
                    img = Image.open(archivo_subido)
                    st.image(img, caption="Imagen de la jugada")

            if link_youtube:
                st.video(link_youtube)

            st.markdown("##### 📥 Exportar reporte")
            if st.button("📄 Descargar PDF"):
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, f"Jugada:\n{texto_jugada}\n\nDecisión sugerida: {pred_label}")
                    pdf_output = io.BytesIO()
                    pdf.output(pdf_output)
                    pdf_output.seek(0)
                    b64 = base64.b64encode(pdf_output.read()).decode('utf-8')
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="reporte_var.pdf">📄 Descargar PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Error al generar PDF: {e}")

# Gráfico de decisiones
st.markdown("---")
st.subheader("📊 Distribución de decisiones en el dataset")
fig = px.histogram(df_data, x="Decision", title="Decisiones registradas")
st.plotly_chart(fig)

st.markdown('<div style="text-align: center; color: gray;">Desarrollado por LTELC - Consultoría en Datos e IA ⚙️</div>', unsafe_allow_html=True)


