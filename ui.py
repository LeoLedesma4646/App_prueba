"""Elementos visuales compartidos (estilos, cabeceras, formatos)."""
from __future__ import annotations

import streamlit as st

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
.page-kicker {font-size: .82rem; letter-spacing: .08em; text-transform: uppercase;
              color: #5B6573; font-weight: 600; margin-bottom: .1rem;}
.formula {font-size: 1.15rem; font-weight: 600; padding: .7rem 1rem; border-radius: 8px;
          background: rgba(31,78,121,.07); border-left: 4px solid #1F4E79; margin: .4rem 0 1rem 0;}
.note {font-size: .88rem; color: #5B6573;}
.cite {font-size: .85rem; color: #5B6573; border-top: 1px solid rgba(0,0,0,.08);
       padding-top: .4rem; margin-top: .6rem;}
div[data-testid="stMetricValue"] {font-size: 1.55rem;}
div[data-testid="stMetricLabel"] p {font-size: .85rem;}
table {font-size: .9rem;}
div[data-testid="stMetric"] {background: rgba(31,78,121,.04); border-radius: 10px;
       padding: .6rem .8rem; border: 1px solid rgba(31,78,121,.10);}
</style>
"""


def setup_style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, kicker: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="page-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def formula(text: str) -> None:
    st.markdown(f'<div class="formula">{text}</div>', unsafe_allow_html=True)


def cite(text: str) -> None:
    st.markdown(f'<div class="cite">📚 {text}</div>', unsafe_allow_html=True)


def selected_sku() -> str:
    return st.session_state.get("sku_global")


PLOT_LAYOUT = dict(template="plotly_white", margin=dict(l=40, r=30, t=60, b=90),
                   legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
                   font=dict(size=12))


def vline(fig, x, text: str | None = None, **kw) -> None:
    """Línea vertical compatible con fechas (evita el error de plotly con Timestamp + annotation_text)."""
    kw.setdefault("line_dash", "dot")
    kw.setdefault("line_color", "#64748B")
    if hasattr(x, "to_pydatetime"):
        x = x.to_pydatetime()
    fig.add_vline(x=x, **kw)
    if text:
        fig.add_annotation(x=x, y=1, yref="paper", text=text, showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(size=11, color="#475569"))
