import streamlit as st
import pandas as pd
import networkx as nx
import re
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.graph_objects as go

#  Configuración de la Página y Estilo
st.set_page_config(
    page_title="Análisis Temático de Redes Sociales", page_icon="✨", layout="wide"
)

# Estilo CSS para mejorar la apariencia de las métricas y otros elementos
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
.st-emotion-cache-16txtl3 {
    padding-top: 2rem;
}
</style>
""",
    unsafe_allow_html=True,
)


#  Funciones de Carga y Procesamiento
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

        df["mentions"] = df["Tweet_Content"].apply(
            lambda text: ",".join(re.findall(r"@(\w+)", str(text)))
        )
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


#  Carga de Datos Principal
df_original = load_and_prepare_data("data/datos_finales_analisis.csv")

# Nombres de los tópicos para usar en todo el dashboard
nombres_topicos = {
    0: "Tópico #1: Reventa y Marketplaces",
    1: "Tópico #2: Novedades y Ofertas",
    2: "Tópico #3: Colaboración H&M con BGYO",
    3: "Tópico #4: Tendencias y Rebajas",
    4: "Tópico #5: Opiniones y Comparativas",
    5: "Tópico #6: Compras Online",
    6: "Tópico #7: Contenido de Influencers",
}


#  NAVEGACIÓN PRINCIPAL EN LA BARRA LATERAL
st.sidebar.title("Navegación del Proyecto")
page = st.sidebar.radio(
    "Selecciona una página:", ["Página Principal", "Dashboard de Análisis"]
)
st.sidebar.markdown("")

# -----------------------------------------------------
# PÁGINA PRINCIPAL
# -----------------------------------------------------
if page == "Página Principal":
    st.title(
        "Sistema de Análisis Temático en Redes Sociales Aplicando Teoría de Grafos y Minería de Texto"
    )

    st.image(
        "https://pbs.twimg.com/profile_images/496666199720067072/ddlIRzoR_400x400.jpeg",
    )

    st.markdown(
        """
    Este proyecto representa la culminación de un estudio sobre la dinámica de las conversaciones en redes sociales.
    A través de la aplicación de técnicas avanzadas, hemos desarrollado un sistema capaz de:
    - **Extraer y procesar** volúmenes de datos textuales de un dataset predefinido en Kaggle.
    - **Identificar temas latentes** mediante modelado de tópicos (LDA).
    - **Construir y analizar redes de interacción** para detectar comunidades e influencers.

    El resultado es un dashboard interactivo que permite explorar estos hallazgos de manera intuitiva y visual.
    """
    )

    st.markdown("")

    st.subheader("Realizado por:")
    st.markdown(
        """
    - Yohan Camilo Botello Maldonado
    - David Torres Ovallos
    - Jhoan Sebastian Tafur Rodriguez
    """
    )

    st.caption("Para la asignatura **Aplicaciones Prácticas de Teoría de Grafos**")
    st.caption("Universidad Francisco de Paula Santander, 2025")


# -----------------------------------------------------
# PÁGINA DEL DASHBOARD DE ANÁLISIS
# -----------------------------------------------------
elif page == "Dashboard de Análisis":
    if df_original is None:
        st.error(
            "Error: No se encontró el archivo 'datos_finales_analisis.csv'. No se puede cargar el dashboard."
        )
        st.stop()

    # FILTROS DEL DASHBOARD (AHORA EN LA BARRA LATERAL)
    st.sidebar.header("Filtros del Dashboard")
    selected_brand = st.sidebar.selectbox(
        "Filtrar por Marca",
        options=["TODAS"] + sorted(df_original["brand"].unique().tolist()),
    )

    df_filtered = df_original.copy()
    if selected_brand != "TODAS":
        df_filtered = df_original[df_original["brand"] == selected_brand].copy()

    # TÍTULO DEL DASHBOARD
    st.title(f"✨ Dashboard de Análisis de Moda: {selected_brand.upper()}")
    st.markdown(f"Análisis interactivo basado en **{len(df_filtered):,}** tweets.")

    # PESTAÑAS DE ANÁLISIS
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
            tweets_por_dia = (
                df_filtered.set_index("Tweet_DateTime").resample("D").size()
            )
            st.line_chart(tweets_por_dia)
        else:
            st.warning("No hay datos para el periodo seleccionado.")

    with tab2:
        st.header("¿De qué se está hablando? (Análisis de Tópicos)")
        st.markdown(
            "Aquí identificamos los temas de conversación dominantes. El gráfico muestra qué temas son más frecuentes en la selección filtrada."
        )
        with st.expander("ℹ️ ¿Qué es el Modelado de Tópicos?"):
            st.info(
                """El Modelado de Tópicos (con LDA) es una técnica de IA que analiza el texto para descubrir automáticamente grupos de palabras que aparecen juntas con frecuencia. Cada grupo forma un "tópico"."""
            )
        if not df_filtered.empty:
            topic_counts = df_filtered["topic"].map(nombres_topicos).value_counts()
            st.bar_chart(topic_counts)
        else:
            st.warning("No hay datos para mostrar.")

    with tab3:
        st.header("¿Quiénes están hablando? (Perfil de Comunidades)")
        st.markdown(
            "Aquí puedes seleccionar una comunidad detectada para analizar su 'personalidad': sus temas preferidos y sus miembros más relevantes."
        )
        if not df_filtered.empty:
            communities_in_data = sorted(
                df_filtered["community"].dropna().unique().astype(int)
            )
            if communities_in_data:
                selected_community = st.selectbox(
                    "Selecciona una Comunidad para perfilar:",
                    options=communities_in_data,
                )
                df_community = df_filtered[
                    df_filtered["community"] == selected_community
                ]
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
            "Visualización interactiva del grafo de interacciones. Cada punto (nodo) es un usuario y cada línea (arista) es una mención."
        )
        with st.expander("ℹ️ ¿Cómo interpretar este grafo?"):
            st.info(
                """* **Nodos (Círculos):** Representan a los usuarios.
* **Tamaño del Nodo:** Cuanto más grande es el nodo, más veces ha sido mencionado (mayor influencia).
* **Líneas (Aristas):** Muestran una mención de un usuario a otro.
* **Interactividad:** Puedes hacer zoom, arrastrar los nodos y pasar el ratón sobre ellos para ver sus nombres."""
            )

        if not df_filtered.empty:
            G_vis = build_graph(df_filtered)

            if G_vis.number_of_nodes() > 0:
                G_display = None
                NODE_LIMIT = 200

                if G_vis.number_of_nodes() > NODE_LIMIT:
                    st.warning(
                        f"El grafo es muy grande ({G_vis.number_of_nodes()} nodos). Se mostrará un subgrafo centrado en los {NODE_LIMIT} nodos más mencionados."
                    )

                    # 1. Obtenemos los N nodos más mencionados (los más importantes)
                    top_nodes_by_mentions = sorted(
                        G_vis.in_degree, key=lambda x: x[1], reverse=True
                    )[:NODE_LIMIT]
                    top_nodes_names = {name for name, degree in top_nodes_by_mentions}

                    # 2. Creamos un nuevo grafo vacío
                    G_display = nx.DiGraph()

                    # 3. Recorremos TODAS las aristas del grafo original
                    # y añadimos solo aquellas que apuntan a nuestros nodos importantes.
                    for u, v in G_vis.edges():
                        if v in top_nodes_names:
                            G_display.add_edge(u, v)
                else:
                    G_display = G_vis

                if G_display and G_display.number_of_nodes() > 0:
                    # Asignar tamaño a los nodos basado en cuántas veces fueron mencionados
                    degrees = dict(G_display.in_degree())
                    for node in G_display.nodes():
                        size = degrees.get(node, 0)
                        G_display.nodes[node]["size"] = size * 3 + 15

                    # CONFIGURACIÓN DE LA VISUALIZACIÓN
                    net = Network(
                        height="750px",
                        width="100%",
                        bgcolor="#FFFFFF",
                        font_color="#333333",
                        cdn_resources="in_line",
                        directed=True,
                    )

                    # Añadimos opciones de física para mejorar el layout
                    net.set_options(
                        """
                    var options = {
                      "physics": {
                        "forceAtlas2Based": {
                          "gravitationalConstant": -50,
                          "centralGravity": 0.01,
                          "springLength": 230,
                          "springConstant": 0.08
                        },
                        "minVelocity": 0.75,
                        "solver": "forceAtlas2Based"
                      }
                    }
                    """
                    )

                    net.from_nx(G_display)

                    try:
                        html_content = net.generate_html(name="graph.html", local=True)
                        components.html(html_content, height=770, scrolling=True)
                    except Exception as e:
                        st.error(f"Error al generar el grafo: {e}")
                else:
                    st.warning(
                        "No se encontraron interacciones para los filtros seleccionados."
                    )
            else:
                st.warning(
                    "No hay interacciones (menciones) en la selección actual para dibujar un grafo."
                )
        else:
            st.warning("No hay datos para visualizar.")
