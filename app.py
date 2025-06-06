# VARGENTO - Plataforma Inteligente de Análisis VAR

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Estilo de la app tipo pantalla VAR Argentina
st.set_page_config(layout="centered", page_title="VARGENTO", page_icon="⚽")
st.markdown("""
    <style>
        .main {
            background-color: #111;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .css-1d391kg {
            color: #03fcb1;
            font-size: 40px;
        }
        .css-10trblm {
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

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
            st.write(f"🤖 Predicción automática: **{prediccion.upper()}**")
            st.write(f"📊 Precisión del modelo: **{acc*100:.2f}%**")
            jugadas_similares = df_data[df_data['VAR used'].str.upper() == prediccion.upper()]
            st.dataframe(jugadas_similares.head(5))
            st.caption("Árbitro responsable: Germán Delfino")
        elif not descripcion:
            st.warning("Por favor, describí la jugada para hacer la predicción.")

    elif uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)
        descripcion = st.text_input("Describí brevemente la jugada para análisis predictivo")
        if st.button("🔍 Analizar jugada de la imagen") and descripcion:
            X_new = vectorizador.transform([descripcion])
            prediccion = modelo.predict(X_new)[0]
            st.write(f"🤖 Predicción automática: **{prediccion.upper()}**")
            st.write(f"📊 Precisión del modelo: **{acc*100:.2f}%**")
            jugadas_similares = df_data[df_data['VAR used'].str.upper() == prediccion.upper()]
            st.dataframe(jugadas_similares.head(5))
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
        fig1 = px.bar(equipo_counts, x='Equipo', y='Cantidad', title='Jugadas revisadas por equipo', text='Cantidad')
        st.plotly_chart(fig1)

    with tab2:
        tipo_jugadas = df_data['Incident'].str.extract(r'([A-Za-z ]+)')[0].value_counts().reset_index().head(15)
        tipo_jugadas.columns = ['Tipo de jugada', 'Cantidad']
        fig2 = px.bar(tipo_jugadas, x='Tipo de jugada', y='Cantidad', title='Tipos de jugadas más comunes', text='Cantidad')
        st.plotly_chart(fig2)

    with tab3:
        decision_counts = df_data['VAR used'].value_counts().reset_index()
        decision_counts.columns = ['Decisión', 'Cantidad']
        fig3 = px.pie(decision_counts, names='Decisión', values='Cantidad', title='Decisiones del VAR')
        st.plotly_chart(fig3)

    with tab4:
        arbitros = df_data['Incident'].str.extract(r'by ([A-Za-z ]+)')[0].dropna()
        arbitro_counts = arbitros.value_counts().reset_index().head(10)
        arbitro_counts.columns = ['Árbitro', 'Cantidad']
        fig4 = px.bar(arbitro_counts, x='Árbitro', y='Cantidad', title='Árbitros mencionados en jugadas', text='Cantidad')
        st.plotly_chart(fig4)

    with tab5:
        pais_counts = df_data['Liga'].value_counts().reset_index()
        pais_counts.columns = ['País', 'Cantidad']
        fig5 = px.bar(pais_counts, x='País', y='Cantidad', title='Jugadas revisadas por país/liga', text='Cantidad')
        st.plotly_chart(fig5)

