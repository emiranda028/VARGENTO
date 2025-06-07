# VARGENTO - Plataforma Inteligente de Análisis VAR

import streamlit as st
import pandas as pd
from PIL import Image
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from fpdf import FPDF
import base64
import os

# Estilo de la app con fondo blanco para mayor legibilidad
st.set_page_config(layout="centered", page_title="VARGENTO", page_icon="⚽")
st.markdown(href, unsafe_allow_html=True)

# Visualizaciones por equipo y árbitro
st.subheader("📈 Estadísticas por equipo y árbitro")

if 'df_data' in locals():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Jugadas por equipo**")
        equipo_counts = df_data['Team'].value_counts().reset_index()
        equipo_counts.columns = ['Equipo', 'Cantidad']
        st.dataframe(equipo_counts)

    with col2:
        st.markdown("**Jugadas por árbitro**")
        arbitro_counts = df_data['Referee'].value_counts().reset_index()
        arbitro_counts.columns = ['Árbitro', 'Cantidad']
        st.dataframe(arbitro_counts)

    # Visualizaciones por equipo y árbitro
    st.subheader("📈 Estadísticas por equipo y árbitro")

    if 'df_data' in locals():
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Jugadas por equipo**")
            equipo_counts = df_data['Team'].value_counts().reset_index()
            equipo_counts.columns = ['Equipo', 'Cantidad']
            st.dataframe(equipo_counts)

        with col2:
            st.markdown("**Jugadas por árbitro**")
            arbitro_counts = df_data['Referee'].value_counts().reset_index()
            arbitro_counts.columns = ['Árbitro', 'Cantidad']
            st.dataframe(arbitro_counts)

import plotly.express as px

# Gráfico por equipo
st.markdown("**📊 Gráfico: Jugadas por equipo**")
fig_eq = px.bar(equipo_counts, x='Equipo', y='Cantidad', title='Jugadas analizadas por equipo', labels={'Cantidad': 'Cantidad de jugadas'})
st.plotly_chart(fig_eq, use_container_width=True)

# Gráfico por árbitro
st.markdown("**📊 Gráfico: Jugadas por árbitro**")
fig_ref = px.bar(arbitro_counts, x='Árbitro', y='Cantidad', title='Jugadas analizadas por árbitro', labels={'Cantidad': 'Cantidad de jugadas'})
st.plotly_chart(fig_ref, use_container_width=True)

# Filtro por tipo de jugada
st.subheader("🎯 Filtro por tipo de jugada")
tipos_jugada = df_data['Incident'].unique().tolist()
tipo_seleccionado = st.selectbox("Seleccione un tipo de jugada para ver estadísticas específicas:", ["Todas"] + tipos_jugada)

if tipo_seleccionado != "Todas":
    df_filtrado = df_data[df_data['Incident'] == tipo_seleccionado]
else:
    df_filtrado = df_data

# Recalcular estadísticas filtradas
eq_counts_filtrado = df_filtrado['Team'].value_counts().reset_index()
eq_counts_filtrado.columns = ['Equipo', 'Cantidad']
fig_eq_filtrado = px.bar(eq_counts_filtrado, x='Equipo', y='Cantidad', title=f'Jugadas de tipo "{tipo_seleccionado}" por equipo', labels={'Cantidad': 'Cantidad de jugadas'})
st.plotly_chart(fig_eq_filtrado, use_container_width=True)

st.image("VAR_System_Logo.svg.png", width=200)

st.title("📺 VARGENTO")
st.subheader("Plataforma Inteligente de Análisis VAR")

st.subheader("¿Qué desea chequear?")
uploaded_file = st.file_uploader("Opcional: suba una imagen o video de la jugada", type=["mp4", "jpg", "jpeg", "png"])

if texto_input:
    X_nuevo = vectorizador.transform([texto_input])
    prediccion = modelo.predict(X_nuevo)[0]
    if prediccion.upper() == "AGAINST":
        prediccion = "Cobrar tiro libre indirecto en contra del equipo que cometió la infracción"
elif prediccion.upper() == "FAVOR":
        prediccion = "Cobrar falta a favor del equipo que sufrió la infracción"
elif prediccion.upper() == "PENAL":
        prediccion = "Cobrar penal a favor del equipo atacado"
elif prediccion.upper() == "NO ACTION":
        prediccion = "No tomar ninguna acción disciplinaria ni técnica"
elif prediccion.upper() == "EXPULSIÓN":
        prediccion = "Mostrar tarjeta roja y expulsar al jugador involucrado"
    st.markdown(f"✅ Decisión sugerida por VARGENTO: **{prediccion}**")
    st.markdown(f"📊 Precisión del modelo: **{acc*100:.2f}%**")

    texto_articulo, articulo, resumen = extraer_articulo(texto_input)
    st.markdown(texto_articulo)

    if st.button("📄 Generar informe en PDF"):
        def generar_pdf(jugada, decision, precision, articulo, resumen, imagen_bytes=None):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Informe de Análisis VAR", ln=True, align='C')
            pdf.ln(10)
            pdf.multi_cell(0, 10, txt=f"Jugada descripta: {jugada}")
            pdf.multi_cell(0, 10, txt=f"Decisión automática: {decision}")
            pdf.multi_cell(0, 10, txt=f"Precisión del modelo: {precision*100:.2f}%")
            pdf.multi_cell(0, 10, txt=f"Reglamento aplicable: {articulo}")
            pdf.multi_cell(0, 10, txt=f"Descripción de la regla: {resumen}")

            if imagen_bytes is not None and hasattr(imagen_bytes, 'read'):
                try:
                    temp_img_path = "temp_jugada.jpg"
                    with open(temp_img_path, "wb") as f:
                        f.write(imagen_bytes.read())
                    pdf.image(temp_img_path, x=10, y=None, w=100)
                    os.remove(temp_img_path)
                except Exception as e:
                    print("Error al procesar imagen:", e)

            pdf.ln(10)
            pdf.set_font("Arial", style="I", size=10)
            pdf.cell(200, 10, txt="Dictamen generado por el sistema VARGENTO - Desarrollado por LTELC", ln=True, align='C')

            pdf_output = f"informe_var.pdf"
            pdf.output(pdf_output)
            with open(pdf_output, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="informe_var.pdf">📄 Descargar informe en PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

        generar_pdf(texto_input, prediccion, acc, articulo, resumen, uploaded_file if uploaded_file else None)



