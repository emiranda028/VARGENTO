# VARGENTO - Plataforma Inteligente de Análisis VAR

import streamlit as st
import pandas as pd
from PIL import Image
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from fpdf import FPDF
import base64
import plotly.express as px
import os
import streamlit.components.v1 as components
import random
from collections import Counter

st.set_page_config(layout="wide", page_title="VARGENTO - Análisis VAR Inteligente", page_icon="⚽")

st.markdown("""
    <style>
        body { background-color: white !important; }
        .title { font-size: 36px; font-weight: bold; color: #003366; }
        .subtitle { font-size: 20px; color: #333333; margin-bottom: 15px; }
        .footer { font-size: 13px; color: gray; margin-top: 40px; text-align: center; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
    <script>
        var audio = new Audio('https://www.fesliyanstudios.com/play-mp3/4385');
        window.addEventListener('load', function() {
            audio.play().catch(e => console.log('Auto play blocked'));
        });
    </script>
""", unsafe_allow_html=True)

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
        st.error("No se encontró una columna válida para descripción de jugada.")
        st.stop()

    if "Decision" not in df.columns:
        st.warning("⚠️ No se encontró la columna 'Decision'. Se agregará automáticamente.")
        df["Decision"] = "Desconocido"

    df = df.dropna(subset=["Decision", col_name])
    df = df[df["Decision"].astype(str).str.strip() != ""]

    conteo_decisiones = df["Decision"].value_counts()
    st.write("📌 Distribución actual de clases:", conteo_decisiones)

    clases_validas_para_split = conteo_decisiones[conteo_decisiones >= 2].index.tolist()
    df_filtrado = df[df["Decision"].isin(clases_validas_para_split)]

    vectorizador = CountVectorizer()
    X = vectorizador.fit_transform(df_filtrado[col_name].astype(str))
    y = df_filtrado["Decision"]

    clases_en_y = Counter(y)
    st.write("🔍 Clases encontradas en los datos para entrenamiento:", clases_en_y)

    if len(clases_en_y) < 2:
        st.error("❌ El modelo necesita al menos 2 clases distintas para entrenar.")
        st.stop()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError as e:
        st.warning("⚠️ No se pudo usar estratificación. Se usará división aleatoria.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    modelo = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return modelo, vectorizador, acc, df

modelo, vectorizador, acc, df_data = cargar_modelo()

st.markdown("""
### 🧠 Precisión del modelo
La precisión actual del modelo es: **{:.2f}%**
""".format(acc * 100))

st.markdown("---")

st.subheader("📸 Analizar nueva jugada")
texto_jugada = st.text_area("Describí la jugada para que el modelo sugiera una decisión:", "Jugador comete falta dentro del área tras revisión del VAR")

archivo_subido = st.file_uploader("Opcional: subí un video, imagen o archivo MP4 de la jugada", type=["jpg", "jpeg", "png", "mp4"])
link_youtube = st.text_input("O pegá un link de YouTube con la jugada (opcional)")

if st.button("🔍 Predecir decisión"):
    if not texto_jugada.strip():
        st.warning("Por favor, ingresá una descripción de la jugada.")
    else:
        X_nuevo = vectorizador.transform([texto_jugada])
        prediccion = modelo.predict(X_nuevo)[0]
        st.success(f"✅ Decisión sugerida por el modelo: **{prediccion}**")

        if archivo_subido:
            if archivo_subido.type.startswith("video"):
                st.video(archivo_subido)
            elif archivo_subido.type.startswith("image"):
                imagen = Image.open(archivo_subido)
                st.image(imagen, caption="Imagen de la jugada")

        if link_youtube:
            st.video(link_youtube)

st.markdown("---")

st.subheader("📊 Análisis de distribución de decisiones")
fig = px.histogram(df_data, x="Decision", title="Distribución de decisiones en el dataset")
st.plotly_chart(fig)

st.markdown("---")

st.subheader("📤 Descargar resultados")
if st.button("📥 Exportar predicción a PDF"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Reporte de decisión VARGENTO", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Jugada: {texto_jugada}\n\nDecisión sugerida: {prediccion}")

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    b64 = base64.b64encode(pdf_output.read()).decode('utf-8')
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="reporte_var.pdf">📄 Descargar PDF</a>'
    st.markdown(href, unsafe_allow_html=True)

st.markdown("""
<div class="footer">Desarrollado por LTELC - Consultoría en Datos e IA ⚙️</div>
""", unsafe_allow_html=True)
