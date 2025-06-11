# app.py - VARGENTO

import streamlit as st
import pandas as pd
from PIL import Image
import io
import pickle
from fpdf import FPDF
import base64
import plotly.express as px

st.set_page_config(layout="wide", page_title="VARGENTO - Análisis VAR Inteligente", page_icon="⚽")

# Función para cargar modelo, vectorizador y encoder
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

# Carga todo
modelo, vectorizador, le, df_data = cargar_modelo()

# Encabezado en dos columnas
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("## ⚽ VARGENTO - Asistente VAR Inteligente")
    st.write("""
        Describí la jugada, subí evidencia visual si querés, y recibí una sugerencia de decisión 
        basada en jugadas históricas clasificadas por inteligencia artificial.
    """)
with col2:
    st.image("https://media.tenor.com/xOb4uwv-VV8AAAAC/var-checking.gif", use_column_width=True)

# Mostrar precisión estimada
st.markdown("---")
st.markdown("### 🧠 Precisión del modelo (estimada)")
try:
    X_temp = vectorizador.transform(df_data["descripcion"].astype(str))
    y_temp = df_data["Decision"]
    y_enc = le.transform(y_temp)
    acc = modelo.score(X_temp, y_enc)
    st.markdown(f"📊 Precisión del modelo: **{acc * 100:.2f}%**")
except Exception as e:
    st.warning(f"No se pudo calcular precisión automáticamente. Detalle: {e}")

# Entrada principal
st.markdown("---")
st.markdown("### 🎥 Análisis de jugada")

col1, col2 = st.columns(2)

with col1:
    texto_jugada = st.text_area("✍️ Describí brevemente lo que ocurrió", placeholder="Ej: Centro desde la derecha, mano del defensor al bloquear el remate")
    archivo_subido = st.file_uploader("📁 Subí imagen o video (opcional)", type=["jpg", "jpeg", "png", "mp4"])
    link_youtube = st.text_input("🔗 Link de YouTube (opcional):")

with col2:
    st.markdown("#### Resultado del análisis")

    if st.button("🔍 Predecir decisión"):
        if not texto_jugada.strip():
            st.warning("⚠️ Por favor ingresá una descripción.")
        else:
            try:
                X_nueva = vectorizador.transform([texto_jugada])
                pred = modelo.predict(X_nueva)[0]
                pred_label = le.inverse_transform([pred])[0]
                st.success(f"📢 Decisión sugerida por IA: **{pred_label}**")

                if archivo_subido:
                    if archivo_subido.type.startswith("video"):
                        st.video(archivo_subido)
                    elif archivo_subido.type.startswith("image"):
                        img = Image.open(archivo_subido)
                        st.image(img, caption="📷 Imagen de la jugada")

                if link_youtube:
                    st.video(link_youtube)

                # Exportación a PDF
                st.markdown("##### 📥 Descargar reporte")
                if st.button("📄 Generar PDF"):
                    try:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 10, f"Descripción de la jugada:\n{texto_jugada}\n\nDecisión sugerida: {pred_label}")
                        pdf_output = io.BytesIO()
                        pdf.output(pdf_output)
                        pdf_output.seek(0)
                        b64 = base64.b64encode(pdf_output.read()).decode('utf-8')
                        href = f'<a href="data:application/octet-stream;base64,{b64}" download="reporte_var.pdf">📄 Descargar PDF</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Error al generar PDF: {e}")

            except Exception as e:
                st.error(f"❌ Error durante la predicción: {e}")

# Gráfico de clases
st.markdown("---")
st.markdown("### 📊 Distribución de decisiones en el dataset")

try:
    fig = px.histogram(df_data, x="Decision", title="Frecuencia de cada decisión registrada")
    st.plotly_chart(fig)
except Exception as e:
    st.warning(f"Error al generar gráfico: {e}")

# Tabla de ejemplo (opcional)
with st.expander("🧾 Ver primeras filas del dataset"):
    st.dataframe(df_data.head(20))

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: gray;">Desarrollado por <b>LTELC</b> - Consultoría en Datos e IA ⚙️</div>', unsafe_allow_html=True)



