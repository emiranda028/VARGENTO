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

st.title("📺 VARGENTO")
st.subheader("Plataforma Inteligente de Análisis VAR")

st.write("Subí un video o una imagen de una jugada para analizarla automáticamente.")

st.markdown("🧠 *VARGENTO es un desarrollo de [LTELC](https://lotengoenlacabeza.com.ar/), consultora en inteligencia de datos y visualización aplicada al fútbol profesional.*")

# Subida de archivo
uploaded_file = st.file_uploader("Subí tu jugada (video .mp4 o imagen .jpg/.png)", type=["mp4", "jpg", "jpeg", "png"])

# Reglas y artículos FIFA relacionados
articulos_fifa = {
    "mano": (
        "Regla 12 - Faltas e incorrecciones: Infracción por mano (pág. 104)",
        "Se sanciona si un jugador toca deliberadamente el balón con la mano o el brazo."
    ),
    "fuera de juego": (
        "Regla 11 - Fuera de juego (pág. 98)",
        "Un jugador está en fuera de juego si está más cerca de la portería rival que el balón y el penúltimo defensor cuando recibe el balón."
    ),
    "agresión": (
        "Regla 12 - conducta violenta (pág. 108)",
        "Incluye golpes, empujones o agresiones físicas hacia otro jugador."
    ),
    "simulación": (
        "Regla 12 - conducta antideportiva (pág. 109)",
        "Simular una falta o exagerar una caída para engañar al árbitro es sancionable."
    ),
    "penal": (
        "Regla 14 - Tiros penales (pág. 113)",
        "Los penales se ejecutan desde el punto penal tras una falta cometida dentro del área."
    ),
    "gol": (
        "Regla 10 - Determinación del resultado (pág. 92)",
        "Un gol es válido si el balón cruza completamente la línea entre los postes y bajo el travesaño."
    )
}

def extraer_articulo(descripcion):
    descripcion = descripcion.lower()
    for clave, (articulo, resumen) in articulos_fifa.items():
        if clave in descripcion:
            return f"📖 {articulo}\n📝 {resumen}", articulo, resumen
    return ("📖 Regla correspondiente según criterio arbitral.\n📝 El incidente debe analizarse según el contexto del partido.",
            "Regla correspondiente", "Debe analizarse según contexto arbitral")

@st.cache_data
def cargar_modelo():
    df = pd.read_csv("var.csv", encoding='latin1')
    df['Liga'] = df['Team'].apply(lambda x: "Argentina" if x in ["River Plate", "Boca Juniors", "Racing", "Independiente", "San Lorenzo"] else "Inglaterra")
    df = df.dropna(subset=['Incident', 'VAR used'])
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Incident'])
    y = df['VAR used']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, vectorizer, accuracy, df

modelo, vectorizador, acc, df_data = cargar_modelo()

st.markdown("📘 Consultá el reglamento oficial completo de FIFA [aquí](https://digitalhub.fifa.com/m/7ae8d5dc60c7da1/original/Reglas-de-Juego-2023-24.pdf)")

texto_input = st.text_area("Describí brevemente la jugada (por ejemplo: 'mano en el área tras un centro')")

if texto_input:
    X_nuevo = vectorizador.transform([texto_input])
    prediccion = modelo.predict(X_nuevo)[0]
    if prediccion.upper() == "AGAINST":
        prediccion = "Cobrar en contra del equipo"
    elif prediccion.upper() == "FAVOR":
        prediccion = "Cobrar a favor del equipo"
    elif prediccion.upper() == "PENAL":
        prediccion = "Cobrar penal"
    elif prediccion.upper() == "NO ACTION":
        prediccion = "No tomar ninguna acción"
    elif prediccion.upper() == "EXPULSIÓN":
        prediccion = "Expulsar al jugador involucrado"
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



