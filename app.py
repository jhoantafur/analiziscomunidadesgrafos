import streamlit as st
import pandas as pd
import networkx as nx
import re
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.graph_objects as go

#  Configuración de la Página y Estilo
st.set_page_config(page_title="Análisis de Moda en X", page_icon="✨", layout="wide")

st.markdown(
    """
<style>
div[data-testid="metric-container"] {
    background-color: #ECF0F1;
    border: 1px solid #ECF0F1;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: #2C3E50;
}
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #F8F9F9;
}
</style>
""",
    unsafe_allow_html=True,
)


#  Funciones de Carga y Procesamiento
@st.cache_data
def load_and_prepare_data(file_path):
    """Carga los datos y realiza la preparación inicial una sola vez."""
    try:
        df = pd.read_csv(file_path)
        for col in ["Account_Created", "Tweet_DateTime"]:
            df[col] = pd.to_datetime(df[col])

        brands = ["zara", "h&m", "primark", "shein", "asos"]
        df["brand"] = (
            df["FinalCleaned"]
            .str.findall(f'({"|".join(brands)})', flags=re.IGNORECASE)
            .str[0]
            .str.lower()
        )
        df["brand"] = df["brand"].fillna("Otra")

        # Extraemos las menciones como una lista
        df["mentions_list"] = df["Tweet_Content"].apply(
            lambda text: re.findall(r"@(\w+)", str(text))
        )

        df["mentions"] = df["mentions_list"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else ""
        )

        # Eliminamos la columna de lista que ya no necesitamos en el cache
        df = df.drop(columns=["mentions_list"])

        return df
    except FileNotFoundError:
        return None


@st.cache_data
def build_graph(df_selection):
    """Construye un grafo a partir de una selección del DataFrame."""
    G = nx.DiGraph()
    for _, row in df_selection.iterrows():
        author = str(row["User_Handle"]).lower()
        mentioned_users_str = row["mentions"]

        if isinstance(mentioned_users_str, str) and mentioned_users_str:
            mentioned_users = mentioned_users_str.split(",")
            for mentioned_user in mentioned_users:
                G.add_edge(author, str(mentioned_user).lower())
    return G


# Nombres de los tópicos
nombres_topicos = {
    0: "Tópico #1: Reventa y Marketplaces",
    1: "Tópico #2: Novedades y Ofertas",
    2: "Tópico #3: Colaboración H&M con BGYO",
    3: "Tópico #4: Tendencias y Rebajas",
    4: "Tópico #5: Opiniones y Comparativas",
    5: "Tópico #6: Compras Online",
    6: "Tópico #7: Contenido de Influencers",
}

#  Carga de Datos Principal
df_original = load_and_prepare_data("data/datos_finales_analisis.csv")

if df_original is None:
    st.error(
        "Error: No se encontró el archivo 'datos_finales_analisis.csv'. Asegúrate de que esté en la carpeta 'data'."
    )
    st.stop()

#  BARRA LATERAL DE FILTROS
st.sidebar.title("🔬 Panel de Exploración")
st.sidebar.markdown("Usa los filtros para explorar la conversación.")

selected_brand = st.sidebar.selectbox(
    "Filtrar por Marca",
    options=["TODAS"] + sorted(df_original["brand"].unique().tolist()),
)

df_filtered = df_original.copy()
if selected_brand != "TODAS":
    df_filtered = df_original[df_original["brand"] == selected_brand].copy()

min_date = df_original["Tweet_DateTime"].min().date()
max_date = df_original["Tweet_DateTime"].max().date()
date_range = st.sidebar.date_input(
    "Selecciona un Rango de Fechas",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_filtered = df_filtered[
        (df_filtered["Tweet_DateTime"].dt.date >= start_date.date())
        & (df_filtered["Tweet_DateTime"].dt.date <= end_date.date())
    ]

#  TÍTULO PRINCIPAL
st.title(f"✨ Dashboard de Análisis de Moda: {selected_brand.upper()}")
st.markdown(f"Análisis interactivo basado en **{len(df_filtered):,}** tweets.")

#  PESTAÑAS DE ANÁLISIS
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Visión General 📈",
        "Análisis de Tópicos 💬",
        "Perfil de Comunidades 👥",
        "Visualización de Red 🕸️",
    ]
)

with tab1:
    st.header("Métricas Clave de la Conversación")
    st.markdown(
        "Un resumen cuantitativo de la selección actual. Estos números reflejan el volumen y la estructura de la conversación."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tweets Analizados", f"{len(df_filtered):,}")
    col2.metric("Usuarios Únicos", f"{df_filtered['User_Handle'].nunique():,}")

    G_filtered = build_graph(df_filtered)
    col3.metric("Interacciones (Menciones)", f"{G_filtered.number_of_edges():,}")

    density = nx.density(G_filtered) if G_filtered.number_of_nodes() > 1 else 0
    col4.metric("Densidad de la Red", f"{density:.4f}")
    st.caption(
        "La densidad mide cuán conectada está la red (0=ninguna conexión, 1=todos conectados con todos). Un valor bajo es normal en redes grandes."
    )

    st.subheader("Evolución Temporal de la Conversación")
    st.markdown(
        "Este gráfico muestra el volumen de tweets a lo largo del tiempo. Los picos pueden indicar eventos importantes, campañas o lanzamientos."
    )
    if not df_filtered.empty:
        tweets_por_dia = df_filtered.set_index("Tweet_DateTime").resample("D").size()
        st.line_chart(tweets_por_dia)
    else:
        st.warning("No hay datos para el periodo seleccionado.")

with tab2:
    st.header("¿De qué se está hablando? (Análisis de Tópicos)")
    st.markdown(
        "Aquí identificamos los temas de conversación dominantes usando el modelo LDA. El gráfico muestra qué temas son más frecuentes en la selección filtrada."
    )

    with st.expander("ℹ️ ¿Qué es el Modelado de Tópicos?"):
        st.info(
            """
        El Modelado de Tópicos (con LDA) es una técnica de IA que analiza el texto de miles de tweets para descubrir automáticamente grupos de palabras que aparecen juntas con frecuencia. Cada grupo forma un "tópico" o tema de conversación.

        **Ejemplo:** Si las palabras "descuento", "oferta" y "código" aparecen juntas a menudo, el modelo las agrupará en un tópico que podemos interpretar como "Promociones y Descuentos".
        """
        )

    if not df_filtered.empty:
        topic_counts = df_filtered["topic"].map(nombres_topicos).value_counts()
        st.bar_chart(topic_counts)
    else:
        st.warning("No hay datos para mostrar.")

with tab3:
    st.header("¿Quiénes están hablando? (Perfil de Comunidades)")
    st.markdown(
        "Hemos detectado grupos de usuarios que interactúan más entre sí. Aquí puedes seleccionar una comunidad para analizar su 'personalidad': sus temas preferidos y sus miembros más relevantes."
    )

    if not df_filtered.empty:
        communities_in_data = sorted(
            df_filtered["community"].dropna().unique().astype(int)
        )
        if communities_in_data:
            selected_community = st.selectbox(
                "Selecciona una Comunidad para perfilar:", options=communities_in_data
            )

            df_community = df_filtered[df_filtered["community"] == selected_community]

            st.subheader(f"Perfil de la Comunidad #{selected_community}")

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(
                    "Nº de Miembros en la Selección",
                    f"{df_community['User_Handle'].nunique():,}",
                )
                st.metric("Nº de Tweets de la Comunidad", f"{len(df_community):,}")

                st.markdown("**Miembros más activos:**")
                st.dataframe(
                    df_community["User_Handle"].value_counts().head(5),
                    use_container_width=True,
                )

            with col2:
                st.markdown("**Temas de interés principales:**")
                topic_dist = (
                    df_community["topic"]
                    .map(nombres_topicos)
                    .value_counts(normalize=True)
                )

                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=topic_dist.index,
                            values=topic_dist.values,
                            hole=0.3,
                            textinfo="label+percent",
                        )
                    ]
                )
                fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay comunidades para analizar en la selección actual.")
    else:
        st.warning("No hay datos para mostrar.")

with tab4:
    st.header("Mapa Interactivo de la Red")
    st.markdown(
        "Esta es una visualización interactiva del grafo de interacciones. Cada punto (nodo) es un usuario y cada línea (arista) es una mención."
    )

    with st.expander("ℹ️ ¿Cómo interpretar este grafo?"):
        st.info(
            """
        * **Nodos (Círculos):** Representan a los usuarios.
        * **Tamaño del Nodo:** Cuanto más grande es el nodo, más veces ha sido mencionado ese usuario (mayor influencia).
        * **Aristas (Líneas):** Representan una mención de un usuario a otro.
        * **Interactividad:** Puedes hacer zoom, arrastrar los nodos para explorar las conexiones y pasar el ratón sobre ellos para ver sus nombres.
        """
        )

    if not df_filtered.empty:
        G_vis = build_graph(df_filtered)

        if G_vis.number_of_nodes() > 0:
            if G_vis.number_of_nodes() > 500:
                st.warning(
                    f"El grafo seleccionado es muy grande ({G_vis.number_of_nodes()} nodos). Para una mejor experiencia, considera filtrar más por marca o fecha."
                )

            degrees = dict(G_vis.in_degree())
            nx.set_node_attributes(G_vis, degrees, "size")

            net = Network(
                height="700px",
                width="100%",
                bgcolor="#FFFFFF",
                font_color="#333333",
                notebook=True,
                directed=True,
                select_menu=True,
                filter_menu=True,
            )
            net.from_nx(G_vis)

            try:
                html_content = net.generate_html()
                components.html(html_content, height=720, scrolling=True)
            except Exception as e:
                st.error(f"Error al generar el grafo: {e}")
        else:
            st.warning(
                "No hay interacciones (menciones) en la selección actual para dibujar un grafo."
            )
    else:
        st.warning("No hay datos para visualizar.")
