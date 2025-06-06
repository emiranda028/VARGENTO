# VARGENTO - Plataforma Inteligente de Análisis VAR

import streamlit as st
import pandas as pd
from PIL import Image
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Estilo de la app con estética VAR oficial
st.set_page_config(layout="centered", page_title="VARGENTO", page_icon="⚽")
st.markdown("""
    <style>
        .stApp {
            background-color: #0a2c50;
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        .block-container {
            padding: 2rem;
        }
        .css-10trblm, .css-1d391kg, .css-1v0mbdj p, .css-1v0mbdj h1, .css-1v0mbdj h2, .css-1v0mbdj h3 {
            color: #ffffff !important;
        }
        .stButton>button {
            background-color: #00aaff;
            color: white;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #0077cc;
        }
    </style>
""", unsafe_allow_html=True)

st.image("VAR_System_Logo.svg.png", width=200)

st.title("📺 VARGENTO")
st.subheader("Plataforma Inteligente de Análisis VAR")

st.write("Subí un video o una imagen de una jugada para analizarla automáticamente.")

# Subida de archivo
uploaded_file = st.file_uploader("Subí tu jugada (video .mp4 o imagen .jpg/.png)", type=["mp4", "jpg", "jpeg", "png"])

# Cargar y preparar el modelo predictivo
@st.cache_data
def cargar_modelo():
    df = pd.read_csv("var.csv")
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

if uploaded_file is not None:
    if uploaded_file.type.startswith("video"):
        st.video(uploaded_file)
        frame_number = st.number_input("Ingresá el número de frame clave (ej: 174)", min_value=0, value=174)
        descripcion = st.text_input("Describí brevemente la jugada para análisis predictivo")
        if st.button("🔍 Analizar jugada del video") and descripcion:
            X_new = vectorizador.transform([descripcion])
            prediccion = modelo.predict(X_new)[0]
            st.success(f"Frame seleccionado: {frame_number}")
            st.write(f"🤖 Resultado automático: **{prediccion.upper()}**")
            st.write(f"📊 Precisión del modelo: **{acc*100:.2f}%**")
            jugadas_similares = df_data[df_data['VAR used'].str.upper() == prediccion.upper()]
            st.markdown("### 📂 Jugadas similares en el historial")
            for _, row in jugadas_similares.head(3).iterrows():
                st.markdown(f"- **Partido:** {row['Team']} vs {row['Opponent Team']} ({row['Date']})")
                st.markdown(f"  - 📍 Sitio: {'Local' if row['Site'] == 'H' else 'Visitante'}")
                st.markdown(f"  - 🕒 Minuto: {row['Time']}")
                st.markdown(f"  - ⚠️ Incidente: {row['Incident']}")
                st.markdown(f"  - 📽️ [Ver jugada en video](https://example.com/video_placeholder) *(en desarrollo)*")
                st.markdown("---")
            st.image("VAR_System_Logo.svg.png", caption="Pantalla VAR", use_container_width=True)
            st.caption("Árbitro responsable: Germán Delfino")
        elif not descripcion:
            st.warning("Por favor, describí la jugada para hacer la predicción.")

    elif uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)
        descripcion = st.text_input("Describí brevemente la jugada para análisis predictivo")
        if st.button("🔍 Analizar jugada de la imagen") and descripcion:
            X_new = vectorizador.transform([descripcion])
            prediccion = modelo.predict(X_new)[0]
            st.write(f"🤖 Resultado automático: **{prediccion.upper()}**")
            st.write(f"📊 Precisión del modelo: **{acc*100:.2f}%**")
            jugadas_similares = df_data[df_data['VAR used'].str.upper() == prediccion.upper()]
            st.markdown("### 📂 Jugadas similares en el historial")
            for _, row in jugadas_similares.head(3).iterrows():
                st.markdown(f"- **Partido:** {row['Team']} vs {row['Opponent Team']} ({row['Date']})")
                st.markdown(f"  - 📍 Sitio: {'Local' if row['Site'] == 'H' else 'Visitante'}")
                st.markdown(f"  - 🕒 Minuto: {row['Time']}")
                st.markdown(f"  - ⚠️ Incidente: {row['Incident']}")
                st.markdown(f"  - 📽️ [Ver jugada en video](https://example.com/video_placeholder) *(en desarrollo)*")
                st.markdown("---")
            st.image("VAR_System_Logo.svg.png", caption="Pantalla VAR", use_container_width=True)
            st.caption("Árbitro responsable: Darío Herrera")
        elif not descripcion:
            st.warning("Por favor, describí la jugada para hacer la predicción.")
else:
    st.info("Esperando que subas una jugada para analizar...")

# 📊 Visualizaciones
st.header("📈 Estadísticas del VAR")
if st.checkbox("Mostrar estadísticas por equipo, árbitro y jugada"):
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Por equipo", "Por tipo de jugada", "Por decisión VAR", "Por árbitro", "Por país"])

    with tab1:
        equipo_counts = df_data['Team'].value_counts().reset_index()
        equipo_counts.columns = ['Equipo', 'Cantidad']
        st.bar_chart(data=equipo_counts.set_index('Equipo'))

    with tab2:
        tipo_jugadas = df_data['Incident'].str.extract(r'([A-Za-z ]+)')[0].value_counts().reset_index().head(15)
        tipo_jugadas.columns = ['Tipo de jugada', 'Cantidad']
        st.bar_chart(data=tipo_jugadas.set_index('Tipo de jugada'))

    with tab3:
        decision_counts = df_data['VAR used'].value_counts().reset_index()
        decision_counts.columns = ['Decisión', 'Cantidad']
        st.bar_chart(data=decision_counts.set_index('Decisión'))

    with tab4:
        arbitros = df_data['Incident'].str.extract(r'by ([A-Za-z ]+)')[0].dropna()
        arbitro_counts = arbitros.value_counts().reset_index().head(10)
        arbitro_counts.columns = ['Árbitro', 'Cantidad']
        st.bar_chart(data=arbitro_counts.set_index('Árbitro'))

    with tab5:
        pais_counts = df_data['Liga'].value_counts().reset_index()
        pais_counts.columns = ['País', 'Cantidad']
        st.bar_chart(data=pais_counts.set_index('País'))


