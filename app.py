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
st.markdown("""
    <style>
        .stApp {
            background-color: white;
            color: black;
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            padding: 2rem;
        }
        .stButton>button {
            background-color: #0052cc;
            color: white;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #003d99;
        }
    </style>
""", unsafe_allow_html=True)

st.image("VAR_System_Logo.svg.png", width=200)
st.image("afa_logo.png", width=100)

st.title("📺 VARGENTO")
st.subheader("Plataforma Inteligente de Análisis VAR")

st.write("Subí un video o una imagen de una jugada para analizarla automáticamente.")

st.markdown("🧠 *VARGENTO es un desarrollo de [LTELC](https://lotengoenlacabeza.com.ar/), consultora en inteligencia de datos y visualización aplicada al fútbol profesional.*")

# Subida de archivo
uploaded_file = st.file_uploader("Subí tu jugada (video .mp4 o imagen .jpg/.png)", type=["mp4", "jpg", "jpeg", "png"])

# ... (resto del código igual) ...

def generar_pdf(jugada, decision, precision, articulo, resumen, imagen_bytes=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.image("afa_logo.png", x=160, y=8, w=30)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Informe de Análisis VAR", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=f"Jugada descripta: {jugada}")
    pdf.multi_cell(0, 10, txt=f"Decisión automática: {decision}")
    pdf.multi_cell(0, 10, txt=f"Precisión del modelo: {precision*100:.2f}%")
    pdf.multi_cell(0, 10, txt=f"Reglamento aplicable: {articulo}")
    pdf.multi_cell(0, 10, txt=f"Descripción de la regla: {resumen}")

    if imagen_bytes:
        temp_img_path = "temp_jugada.jpg"
        with open(temp_img_path, "wb") as f:
            f.write(imagen_bytes.getbuffer())
        pdf.image(temp_img_path, x=10, y=None, w=100)
        os.remove(temp_img_path)

    pdf.ln(10)
    pdf.set_font("Arial", style="I", size=10)
    pdf.cell(200, 10, txt="Dictamen generado por el sistema VARGENTO - Desarrollado por LTELC", ln=True, align='C')

    pdf_output = f"informe_var.pdf"
    pdf.output(pdf_output)
    with open(pdf_output, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="informe_var.pdf">📄 Descargar informe en PDF</a>'
        st.markdown(href, unsafe_allow_html=True)
