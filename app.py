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

    df = df.dropna(subset=["Decision"])
    df = df[df["Decision"].astype(str).str.strip() != ""]

    conteo_decisiones = df["Decision"].value_counts()
    st.write("📌 Distribución actual de clases:", conteo_decisiones)

    clases_validas = conteo_decisiones[conteo_decisiones >= 3].index.tolist()
    df = df[df["Decision"].isin(clases_validas)]

    if len(clases_validas) < 2:
        st.warning("⚠️ No hay suficientes clases representadas para entrenar un modelo.")
        ejemplos_sinteticos = pd.DataFrame([
            {"Incident": "remate al arco que termina en gol", "Decision": "Gol"},
            {"Incident": "remate desviado", "Decision": "No gol"},
            {"Incident": "mano clara dentro del área", "Decision": "Penal"},
            {"Incident": "entrada fuerte con plancha", "Decision": "Roja"},
            {"Incident": "protesta reiterada", "Decision": "Amarilla"},
        ])
        df = pd.concat([df, ejemplos_sinteticos], ignore_index=True)

    vectorizador = CountVectorizer()
    X = vectorizador.fit_transform(df[col_name].astype(str))
    y = df["Decision"]

    if len(set(y)) < 2:
        st.error("❌ No hay suficientes clases distintas para entrenar el modelo.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(set(y_train)) < 2:
        st.error("❌ Error: El set de entrenamiento no contiene al menos dos clases distintas.")
        st.stop()

    modelo = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return modelo, vectorizador, acc, df

modelo, vectorizador, acc, df_data = cargar_modelo()

st.header("¿Qué desea chequear?")
texto_input = st.text_area("Describa brevemente la jugada (por ejemplo: 'mano en el área tras un centro')")
uploaded_file = st.file_uploader("Suba una imagen, video MP4 o enlace de YouTube para analizar", type=["jpg", "jpeg", "png", "mp4"])
youtube_link = st.text_input("O ingrese un enlace de YouTube")

if "historial" not in st.session_state:
    st.session_state.historial = []

if st.button("Revisar jugada"):
    if not texto_input:
        st.warning("Por favor describa brevemente la jugada.")
    else:
        X_nuevo = vectorizador.transform([texto_input])
        prediccion = modelo.predict(X_nuevo)[0]
        probas = modelo.predict_proba(X_nuevo)[0]
        confianza = round(max(probas) * 100, 2)

        st.markdown(f"## ✅ Decisión sugerida por VARGENTO: **{prediccion}**")
        st.markdown(f"### 🔍 Precisión estimada del modelo: **{confianza}%**")

        if uploaded_file:
            if uploaded_file.type.startswith("image"):
                image = Image.open(uploaded_file)
                st.image(image, caption="Imagen subida", use_column_width=True)
            elif uploaded_file.type == "video/mp4":
                video_bytes = uploaded_file.read()
                st.video(video_bytes)

        if youtube_link:
            if "youtube.com" in youtube_link or "youtu.be" in youtube_link:
                video_id = youtube_link.split("v=")[-1] if "v=" in youtube_link else youtube_link.split("/")[-1]
                embed_link = f"https://www.youtube.com/embed/{video_id}"
                components.iframe(embed_link, height=315)

        st.session_state.historial.append({
            "Texto": texto_input,
            "Decisión": prediccion,
            "Precisión": confianza
        })

        if st.button("📄 Exportar resultado a PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Informe de jugada analizada por VARGENTO", ln=True, align='C')
            pdf.ln(10)
            pdf.multi_cell(0, 10, txt=f"Descripción: {texto_input}\nDecisión: {prediccion}\nPrecisión: {confianza}%")
            pdf.output("resultado_var.pdf")
            with open("resultado_var.pdf", "rb") as f:
                b64_pdf = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="resultado_var.pdf">📥 Descargar PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

st.subheader("📊 Estadísticas por equipo y árbitro")
if "Team" in df_data.columns:
    equipo_counts = df_data["Team"].value_counts().reset_index()
    equipo_counts.columns = ["Equipo", "Cantidad"]
    fig_eq = px.bar(equipo_counts, x='Equipo', y='Cantidad', title='Jugadas analizadas por equipo', labels={'Cantidad': 'Cantidad de jugadas'})
    st.plotly_chart(fig_eq)

if "VAR used" in df_data.columns:
    uso_var = df_data["VAR used"].value_counts().reset_index()
    uso_var.columns = ["VAR usado", "Cantidad"]
    fig_var = px.pie(uso_var, names='VAR usado', values='Cantidad', title='Distribución del uso del VAR')
    st.plotly_chart(fig_var)

if st.session_state.historial:
    st.subheader("📘 Historial de decisiones")
    df_hist = pd.DataFrame(st.session_state.historial)
    st.dataframe(df_hist)
