# VARGENTO - Plataforma Inteligente de Análisis VAR


import streamlit as st
from PIL import Image
import pandas as pd

# Título principal
st.title("VARGENTO - Asistente de Decisiones VAR")

# Carga de video
st.header("1. Subí tu video de jugada")
video_file = st.file_uploader("Elegí un archivo .mp4", type=["mp4"])

if video_file:
    st.video(video_file)
    st.success("Video cargado correctamente. Ahora seleccioná el frame de la jugada clave.")

# Selección de frame clave
st.header("2. Indicá el frame clave de la jugada")
frame_num = st.number_input("Frame de la jugada clave (aproximado)", min_value=0, max_value=5000, step=1)

# Datos simulados para ejemplo
st.header("3. Análisis de jugada")
if st.button("Analizar jugada"):
    st.subheader("Probabilidad según jugadas similares")
    st.write("Tipo de evento detectado: Mano")
    st.write("Zona: Cercanía del área grande")

    st.metric("Cobro probable a favor (penal)", "66.7%")
    st.metric("Cobro probable en contra", "33.3%")

    st.subheader("Jugadas similares históricas")
    df = pd.DataFrame({
        "Equipo": ["Liverpool", "Chelsea", "Arsenal"],
        "Minuto": [88, 90, 12],
        "Evento": ["Mano en el área", "Mano previa a gol", "Mano tras rebote"],
        "Decisión": ["FOR", "FOR", "FOR"]
    })
    st.table(df)

# Footer
st.markdown("---")
st.markdown("Desarrollado por VARGENTO ⚽🇦🇷")
