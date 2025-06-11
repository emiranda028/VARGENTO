# VARGENTO - Plataforma Inteligente de Análisis VAR

import streamlit as st
import pandas as pd
from PIL import Image
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import base64
import plotly.express as px
from collections import Counter

st.set_page_config(layout="wide", page_title="VARGENTO - Análisis VAR Inteligente", page_icon="⚽")

# Cargar modelo y datos
@st.cache_resource
def cargar_modelo():
    try:
        df = pd.read_csv("VAR_Limpio_Generado.csv", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv("VAR_Limpio_Generado.csv", encoding="latin1")

    columnas = [col.lower().strip() for col in df.columns]
    if "descripcion" in columnas:
        col_name = df.columns[columnas.index("descripcion")]
    elif "incident" in columnas:
        col_name = df.columns[columnas.index("incident")]
    else:
        st.error("❌ No se encontró una columna válida con descripciones de jugadas.")
        st.stop()

    if "Decision" not in df.columns:
        df["Decision"] = "Desconocido"

    df = df.dropna(subset=["Decision", col_name])
    df = df[df["Decision"].astype(str).str.strip() != ""]

    conteo = df["Decision"].value_counts()
    clases_validas = conteo[conteo >= 2].index.tolist()
    df_filtrado = df[df["Decision"].isin(clases_validas)]

    if df_filtrado.empty or len(clases_validas) < 2:
        st.error("❌ Se requieren al menos dos clases con 2 ejemplos cada una.")
        st.stop()

    vectorizador = CountVectorizer()
    X = vectorizador.fit_transform(df_filtrado[col_name].astype(str))
    y = df_filtrado["Decision"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    modelo = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    modelo.fit(X_train, y_train_enc)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test_enc, y_pred)

    return modelo, vectorizador, acc, df_filtrado, col_name, le

modelo, vectorizador, acc, df_data, col_name, le = cargar_modelo()

# Encabezado en dos columnas
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("## ⚽ VARGENTO - Asistente VAR Inteligente")
    st.write("Describe la jugada, subí evidencia si querés, y obtené la decisión sugerida basada en IA.")

with col2:
    st.image("https://media.tenor.com/xOb4uwv-VV8AAAAC/var-checking.gif", use_column_width=True)

st.markdown(f"### 🎯 Precisión del modelo: **{acc * 100:.2f}%**")

st.markdown("---")
st.markdown("### 🎥 Análisis de jugada")

# Entradas de usuario: jugada, archivo, video
col1, col2 = st.columns(2)

with col1:
    texto_jugada = st.text_area("✍️ Describí la jugada", placeholder="Ej: Falta dentro del área tras un córner")
    archivo_subido = st.file_uploader("📁 Subí imagen o video", type=["jpg", "jpeg", "png", "mp4"])
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

            # Mostrar multimedia
            if archivo_subido:
                if archivo_subido.type.startswith("video"):
                    st.video(archivo_subido)
                elif archivo_subido.type.startswith("image"):
                    img = Image.open(archivo_subido)
                    st.image(img, caption="Imagen de la jugada")

            if link_youtube:
                st.video(link_youtube)

            # Descargar PDF
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

# Gráfico al final
st.markdown("---")
st.subheader("📊 Distribución de decisiones en el dataset")
fig = px.histogram(df_data, x="Decision", title="Decisiones registradas")
st.plotly_chart(fig)

st.markdown('<div style="text-align: center; color: gray;">Desarrollado por LTELC - Consultoría en Datos e IA ⚙️</div>', unsafe_allow_htm
