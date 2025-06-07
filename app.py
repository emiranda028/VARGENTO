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
        df = pd.read_csv("var_limpio.csv", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv("var_limpio.csv", encoding="latin1")

    columnas = [col.lower().strip() for col in df.columns]
    if "descripcion" in columnas:
        col_name = df.columns[columnas.index("descripcion")]
    elif "incident" in columnas:
        col_name = df.columns[columnas.index("incident")]
    else:
        st.error(f"No se encontró una columna válida para descripción de jugada ('Descripcion' o 'Incident'). Columnas disponibles: {list(df.columns)}")
        st.stop()

    # VALIDACIÓN DE LA COLUMNA 'Decision'
    if "Decision" not in df.columns or df["Decision"].nunique() < 2:
        st.error("La columna 'Decision' no está presente o no tiene suficientes clases para entrenar el modelo.")
        st.write("Columnas disponibles:", df.columns.tolist())
        st.write("Valores únicos en 'Decision':", df["Decision"].unique())
        st.stop()

    st.write("Distribución de decisiones en los datos:")
    st.dataframe(df["Decision"].value_counts())

    vectorizador = CountVectorizer()
    X = vectorizador.fit_transform(df[col_name].astype(str))
    y = df["Decision"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return modelo, vectorizador, acc, df

# ... (el resto del código permanece igual)
