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
""", unsafe_allow_html=True)

st.audio("https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg", format="audio/ogg")

with st.expander("🔍 Iniciando revisión VAR..."):
    st.markdown("""
    <div style='font-size: 48px; text-align: center;'>
        🖐️➡️⬇️⬅️⬆️🖐️
    </div>
    <div style='text-align: center;'>
        <em>El árbitro está revisando la jugada...</em>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    try:
        df = pd.read_csv("var.csv", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv("var.csv", encoding="latin1")

    columnas = [col.lower().strip() for col in df.columns]
    if "descripcion" in columnas:
        col_name = df.columns[columnas.index("descripcion")]
    elif "incident" in columnas:
        col_name = df.columns[columnas.index("incident")]
    else:
        st.error(f"No se encontró una columna válida para descripción de jugada ('Descripcion' o 'Incident'). Columnas disponibles: {list(df.columns)}")
        st.stop()

    if "Decision" not in df.columns:
        st.warning("⚠️ No se encontró la columna 'Decision' en el CSV. Se generarán decisiones ficticias para fines de demostración.")
        decisiones = ["Penal", "Tiro libre", "Sin falta", "Amarilla", "Roja"]
        df["Decision"] = [random.choice(decisiones) for _ in range(len(df))]

    vectorizador = CountVectorizer()
    X = vectorizador.fit_transform(df[col_name])
    y = df["Decision"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return modelo, vectorizador, acc, df

st.image("VAR_System_Logo.svg.png", width=200)

st.markdown('<div class="title">📺 VARGENTO</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Plataforma Inteligente de Análisis VAR en tiempo real para decisiones arbitrales</div>', unsafe_allow_html=True)

modelo, vectorizador, acc, df_data = cargar_modelo()

st.markdown('<div class="subtitle">¿Qué desea chequear?</div>', unsafe_allow_html=True)

texto_input = st.text_area("Describa brevemente la jugada (por ejemplo: 'mano en el área tras un centro')")
video_link = st.text_input("(Opcional) Ingrese un enlace de YouTube o suba un video MP4")
uploaded_file = st.file_uploader("(Opcional) Suba una imagen o captura de la jugada", type=["png", "jpg", "jpeg", "mp4"])

if texto_input:
    X_nuevo = vectorizador.transform([texto_input])
    prediccion = modelo.predict(X_nuevo)[0]
    probas = modelo.predict_proba(X_nuevo)[0]
    st.markdown(f"✅ Decisión sugerida por VARGENTO: **{prediccion}**")
    st.markdown(f"🔍 Precisión estimada del modelo: **{acc:.2%}**")

    articulo = "Regla 12 - Faltas e incorrecciones"
    resumen = "Se sanciona con tiro libre directo si un jugador toca el balón con la mano de manera antinatural ampliando su volumen corporal."
    st.markdown("---")
    st.markdown(f"📘 **Referencia reglamentaria:** {articulo}")
    st.markdown(f"📝 {resumen}")

    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.set_text_color(0, 70, 140)
            self.cell(0, 10, 'Reporte VARGENTO - Análisis VAR', ln=True, align='C')

    def generar_pdf(jugada, decision, precision, articulo, resumen, imagen=None):
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"Jugadas analizada: {jugada}\n\nDecisión sugerida: {decision}\n\nPrecisión del modelo: {precision:.2%}\n\nArtículo: {articulo}\n\nDescripción: {resumen}")
        if imagen:
            pdf.image(imagen, x=10, y=pdf.get_y() + 10, w=100)
        pdf.output("reporte_vargento.pdf")

    if st.button("📄 Generar informe en PDF"):
        generar_pdf(texto_input, prediccion, acc, articulo, resumen, uploaded_file if uploaded_file else None)
        with open("reporte_vargento.pdf", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="reporte_vargento.pdf">📥 Descargar informe PDF</a>'
            st.markdown(href, unsafe_allow_html=True)

if 'df_data' in locals():
    st.subheader("📈 Estadísticas por equipo y árbitro")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Jugadas por equipo**")
        equipo_counts = df_data['Team'].value_counts().reset_index()
        equipo_counts.columns = ['Equipo', 'Cantidad']
        st.dataframe(equipo_counts)

    with col2:
        st.markdown("**Jugadas por árbitro**")
        if 'Referee' in df_data.columns:
            arbitro_counts = df_data['Referee'].value_counts().reset_index()
            arbitro_counts.columns = ['Árbitro', 'Cantidad']
            st.dataframe(arbitro_counts)

    st.markdown("**📊 Gráfico: Jugadas por equipo**")
    fig_eq = px.bar(equipo_counts, x='Equipo', y='Cantidad', title='Jugadas analizadas por equipo', labels={'Cantidad': 'Cantidad de jugadas'})
    st.plotly_chart(fig_eq, use_container_width=True)

    if 'Referee' in df_data.columns:
        st.markdown("**📊 Gráfico: Jugadas por árbitro**")
        fig_ref = px.bar(arbitro_counts, x='Árbitro', y='Cantidad', title='Jugadas analizadas por árbitro', labels={'Cantidad': 'Cantidad de jugadas'})
        st.plotly_chart(fig_ref, use_container_width=True)

    st.subheader("🎯 Filtro por tipo de jugada")
    tipos_jugada = df_data['Incident'].unique().tolist()
    tipo_seleccionado = st.selectbox("Seleccione un tipo de jugada para ver estadísticas específicas:", ["Todas"] + tipos_jugada)

    if tipo_seleccionado != "Todas":
        df_filtrado = df_data[df_data['Incident'] == tipo_seleccionado]
    else:
        df_filtrado = df_data

    eq_counts_filtrado = df_filtrado['Team'].value_counts().reset_index()
    eq_counts_filtrado.columns = ['Equipo', 'Cantidad']
    fig_eq_filtrado = px.bar(eq_counts_filtrado, x='Equipo', y='Cantidad', title=f'Jugadas de tipo "{tipo_seleccionado}" por equipo', labels={'Cantidad': 'Cantidad de jugadas'})
    st.plotly_chart(fig_eq_filtrado, use_container_width=True)


    
