"""
App consolidada · Modelamiento predictivo y analítico de ventas por SKU (Grupo 5).

Ejecutar en local:   streamlit run streamlit_app.py
Despliegue web:      Streamlit Community Cloud apuntando a este archivo (ver README.md).
"""
import streamlit as st

st.set_page_config(
    page_title="Modelamiento predictivo de ventas por SKU · Grupo 5",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import analytics as A  # noqa: E402
import ui  # noqa: E402

ui.setup_style()

pages = {
    "Proyecto": [
        st.Page("views/inicio.py", title="Inicio y objetivos", icon="🏠", default=True),
        st.Page("views/datos.py", title="Datos y análisis descriptivo", icon="🧮"),
    ],
    "Dimensiones analíticas": [
        st.Page("views/forecast.py", title="1 · Forecast de demanda", icon="📈"),
        st.Page("views/elasticidad.py", title="2 · Elasticidad precio-demanda", icon="💰"),
        st.Page("views/inventarios.py", title="3 · Inventarios", icon="📦"),
    ],
    "Integración": [
        st.Page("views/bi.py", title="4 · Tablero BI integrador", icon="🧭"),
        st.Page("views/escenarios.py", title="Escenarios prescriptivos", icon="🎯"),
    ],
    "Soporte": [
        st.Page("views/metodologia.py", title="Metodología y referencias", icon="📚"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### SKU en análisis")
    skus = A.priority_skus()
    if st.session_state.get("sku_global") not in skus:
        st.session_state["sku_global"] = skus[0]
    st.selectbox(
        "SKU", skus, key="sku_global", label_visibility="collapsed",
        help="Se aplica a Forecast, Elasticidad e Inventarios. Son los 10 SKU admisibles "
             "(≥104 semanas comunes) de mayor ingreso histórico.",
    )
    st.caption("Universidad de Lima · Modelamiento predictivo de datos · Sección 805 · Grupo 5")

nav.run()
