"""
Monitor de Saúde — Ourinhos/SP
Dashboard interativo para vigilância epidemiológica da dengue.
Execute com:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk


# CONFIGURAÇÃO DA PÁGINA

st.set_page_config(
    page_title="Monitor de Saúde — Ourinhos/SP",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Paleta institucional — Prefeitura de Ourinhos
VERDE = "#00796B"          # destaque principal (saúde)
VERDE_ESCURO = "#004D40"
VERDE_CLARO = "#26A69A"
AZUL = "#1565C0"           # complementar institucional
AZUL_CLARO = "#42A5F5"
CINZA_CARD = "#1A1A2E"
FUNDO = "#0D0D1A"
TEXTO = "#E0E0E0"
TEXTO_SECUNDARIO = "#B0BEC5"
BORDA = "#37474F"

st.markdown(f"""
<style>
    /* ── fundo geral ── */
    .stApp {{
        background-color: {FUNDO};
        color: {TEXTO};
    }}
    /* ── sidebar ── */
    section[data-testid="stSidebar"] {{
        background-color: #111128;
        border-right: 2px solid {VERDE};
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {{
        color: {TEXTO};
    }}
    /* ── cabeçalhos ── */
    h1, h2, h3, h4 {{
        color: {VERDE_ESCURO} !important;
    }}
    /* ── métricas ── */
    [data-testid="stMetric"] {{
        background: {CINZA_CARD};
        border: 1px solid {BORDA};
        border-left: 4px solid {VERDE};
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 0 10px rgba(0,121,107,.15);
    }}
    [data-testid="stMetricLabel"] p {{
        color: {VERDE_CLARO} !important;
        font-weight: 600;
    }}
    [data-testid="stMetricValue"] {{
        color: #FFFFFF !important;
    }}
    /* ── tabs ── */
    .stTabs [data-baseweb="tab"] {{
        color: {TEXTO};
        background-color: {CINZA_CARD};
        border-radius: 8px 8px 0 0;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {VERDE} !important;
        color: #fff !important;
    }}
    /* ── cards de alerta ── */
    .alerta-card {{
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,.12);
        color: #fff;
    }}
    .alerta-baixo  {{ background: linear-gradient(135deg, #1b5e20, #2e7d32); border: 2px solid #4caf50; }}
    .alerta-medio  {{ background: linear-gradient(135deg, #e65100, #f57c00); border: 2px solid #ff9800; }}
    .alerta-alto   {{ background: linear-gradient(135deg, #b71c1c, #d32f2f); border: 2px solid #f44336; }}
    .alerta-critico {{ background: linear-gradient(135deg, #4a0000, #8b0000); border: 2px solid #c62828;
                       animation: pulse-border 1.5s infinite; }}
    @keyframes pulse-border {{
        0%   {{ box-shadow: 0 0 8px rgba(198,40,40,.4); }}
        50%  {{ box-shadow: 0 0 24px rgba(198,40,40,.8); }}
        100% {{ box-shadow: 0 0 8px rgba(198,40,40,.4); }}
    }}
    /* ── divider ── */
    .divider {{
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, {VERDE}, transparent);
        margin: 24px 0;
    }}
    /* ── info boxes ── */
    .info-box {{
        background: {CINZA_CARD};
        border-left: 4px solid {VERDE};
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        color: {TEXTO};
        box-shadow: 0 1px 4px rgba(0,0,0,.2);
    }}
    /* ── mosquito banner ── */
    .mosquito-banner {{
        text-align: center;
        font-size: 48px;
        letter-spacing: 12px;
        margin: 8px 0;
    }}
</style>
""", unsafe_allow_html=True)


# CARREGAMENTO DE DADOS (cache)

@st.cache_data
def carregar_dados():
    df = pd.read_csv("Data/dataset_final_ourinhos.csv", sep=";", encoding="latin-1")
    df["Data_Inicio_Semanas"] = pd.to_datetime(df["Data_Inicio_Semanas"])
    df = df.sort_values("Data_Inicio_Semanas").reset_index(drop=True)
    return df


@st.cache_data
def carregar_previsao():
    df = pd.read_csv("previsao_proximas_semanas.csv", sep=";", encoding="latin-1")
    return df


df = carregar_dados()
df_prev = carregar_previsao()

# Colunas derivadas
if "mes" not in df.columns:
    df["mes"] = df["Data_Inicio_Semanas"].dt.month
if "mes_seno" not in df.columns:
    df["mes_seno"] = np.sin(2 * np.pi * df["mes"] / 12)

# Layout Plotly padrao (tema institucional claro)
PLOTLY_LAYOUT = dict(
    paper_bgcolor=FUNDO,
    plot_bgcolor="#111128",
    font=dict(color=TEXTO, family="Inter, sans-serif"),
    title_font=dict(color=VERDE_CLARO, size=16),
    xaxis=dict(gridcolor="#222244", zerolinecolor="#333"),
    yaxis=dict(gridcolor="#222244", zerolinecolor="#333"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def nivel_risco(casos):
    """Retorna (label, classe_css) com base nos casos previstos."""
    if casos <= 5:
        return "BAIXO", "alerta-baixo"
    elif casos <= 15:
        return "MÉDIO", "alerta-medio"
    elif casos <= 40:
        return "ALTO", "alerta-alto"
    else:
        return "CRÍTICO", "alerta-critico"



# SIDEBAR


# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;">
        <h2 style="margin:0; color:{VERDE_ESCURO};">Prefeitura de Ourinhos</h2>
        <p style="color:{VERDE}; font-size:14px; margin-top:-4px;">Secretaria Municipal de Saúde</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    anos_disponiveis = sorted(df["ano"].unique())
    ano_filtro = st.select_slider(
        "Período (anos)",
        options=anos_disponiveis,
        value=(anos_disponiveis[0], anos_disponiveis[-1]),
    )

    # Filtro por bairro
    bairros_lista = [
        "Todos",
        "Centro", "Vila Brasil", "Jd. Matilde", "Jd. Paulista", "Vila Musa",
        "Jd. Ouro Verde", "Vila Nova", "Jd. Europa", "Jd. Santa Fé",
        "Pq. Minas Gerais", "Vila Margarida", "Jd. São Paulo"
    ]
    bairro_filtro = st.selectbox("Bairro", bairros_lista, index=0)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Painel de notícias (RSS)
    import feedparser
    st.markdown(f"<b style='color:{VERDE_ESCURO}'>Notícias sobre Dengue</b>", unsafe_allow_html=True)
    try:
        feed = feedparser.parse("https://g1.globo.com/rss/g1/saude/")
        noticias = [entry for entry in feed.entries if "dengue" in entry.title.lower()][:3]
        for n in noticias:
            st.markdown(f"<a href='{n.link}' target='_blank' style='color:{VERDE_CLARO};'>{n.title}</a>", unsafe_allow_html=True)
    except Exception:
        st.write("Não foi possível carregar notícias.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <b style="color:{VERDE_ESCURO}">Sobre a Dengue</b><br><br>
        A <b>dengue</b> é uma doença viral transmitida pelo mosquito
        <i>Aedes aegypti</i>. Os sintomas incluem febre alta, dores no corpo,
        dor atrás dos olhos e manchas vermelhas na pele.<br><br>
        <b style="color:{VERDE}">Prevenção:</b><br>
        • Elimine água parada<br>
        • Use repelente diariamente<br>
        • Instale telas nas janelas<br>
        • Mantenha caixas d'água fechadas<br>
        • Descarte pneus e garrafas corretamente
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <b style="color:{VERDE_ESCURO}">Ciclo do Aedes aegypti</b><br><br>
        O mosquito deposita seus ovos em recipientes com água parada.
        Em condições favoráveis (temperatura entre 25 °C e 30 °C e alta
        umidade), o ciclo do ovo ao mosquito adulto leva apenas
        <b>7 a 10 dias</b>.<br><br>
        A fêmea infectada transmite o vírus a cada picada, podendo
        infectar <b>dezenas de pessoas</b> em um único ciclo de vida.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center; margin-top:20px; padding:12px;
                background:#0D2818; border-radius:10px;
                border: 1px solid {VERDE}44;">
        <small style="color:{VERDE_CLARO}">Combata o mosquito —<br>é responsabilidade de todos!</small>
    </div>
    """, unsafe_allow_html=True)



# --- FILTRO DE PERÍODO E BAIRRO ---
df_filtrado = df[(df["ano"] >= ano_filtro[0]) & (df["ano"] <= ano_filtro[1])].copy()
if bairro_filtro != "Todos" and "bairro" in df_filtrado.columns:
    df_filtrado = df_filtrado[df_filtrado["bairro"] == bairro_filtro]

# HEADER
st.markdown(f"""
<div style="text-align:center; padding: 20px 0 8px 0;">
    <h1 style="font-size:36px; margin:0;">
        Prefeitura de Ourinhos — <span style="color:{VERDE};">Vigilância Epidemiológica</span>
    </h1>
    <p style="color:{TEXTO_SECUNDARIO}; font-size:15px; margin-top:4px;">
        Painel de monitoramento da dengue • Dados de {int(df['ano'].min())} a {int(df['ano'].max())}
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# PAINEL DE ALERTA — RISCO SEMANAL

st.markdown("## Painel de Alerta — Risco Semanal")

ultima_semana = df.iloc[-1]
prox_semana = df_prev.iloc[0] if len(df_prev) > 0 else None

col_alert1, col_alert2, col_alert3 = st.columns([1, 2, 1])

with col_alert1:
    st.metric(
        label="Última Semana Registrada",
        value=f"SE {int(ultima_semana['Semana_Epidemiologica'])}/{int(ultima_semana['ano'])}",
    )
    st.metric(
        label="Casos Estimados (última SE)",
        value=int(ultima_semana["casos_est"]),
    )

with col_alert2:
    if prox_semana is not None:
        casos_prev = prox_semana["casos_previstos"]
        label_risco, classe_risco = nivel_risco(casos_prev)
        st.markdown(f"""
        <div class="alerta-card {classe_risco}">
            <span style="font-size:14px; opacity:.85;">NÍVEL DE RISCO — PRÓXIMA SEMANA</span><br>
            <span style="font-size:48px; font-weight:900;">{label_risco}</span><br>
            <span style="font-size:22px; font-weight:600;">{casos_prev:.0f} casos previstos</span><br>
            <span style="font-size:13px; opacity:.7;">
                SE {int(prox_semana['semana_epidemiologica'])}/{int(prox_semana['ano'])}
                • mín {prox_semana['minimo_estimado']} — máx {prox_semana['maximo_estimado']}
            </span>
        </div>
        """, unsafe_allow_html=True)

with col_alert3:
    st.metric(
        label="Temperatura prevista",
        value=f"{prox_semana['temp_ar_usada']:.1f} °C" if prox_semana is not None else "—",
    )
    st.metric(
        label="Chuva prevista",
        value=f"{prox_semana['chuva_usada']:.1f} mm" if prox_semana is not None else "—",
    )

# Mini-cards de previsão para as próximas semanas 
st.markdown("#### Previsão para as próximas semanas")
cols_prev = st.columns(len(df_prev))
for i, (_, row) in enumerate(df_prev.iterrows()):
    lbl, cls = nivel_risco(row["casos_previstos"])
    with cols_prev[i]:
        st.markdown(f"""
        <div class="alerta-card {cls}" style="padding:12px 8px; font-size:12px;">
            <b>SE {int(row['semana_epidemiologica'])}</b><br>
            <span style="font-size:20px; font-weight:800;">{int(row['casos_arredondados'])}</span><br>
            <small>{lbl}</small>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# KPIs RESUMO
st.markdown("## Indicadores Gerais")

total_casos = df_filtrado["casos_est"].sum()
media_semanal = df_filtrado["casos_est"].mean()
max_semanal = df_filtrado["casos_est"].max()
temp_media = df_filtrado["tempmed"].mean() if "tempmed" in df_filtrado.columns else np.nan

k1, k2, k3, k4 = st.columns(4)
k1.metric("Casos no período selecionado", f"{total_casos:,.0f}")
k2.metric("Média semanal", f"{media_semanal:,.1f}")
k3.metric("Pico semanal", f"{max_semanal:,.0f}")
k4.metric("Temp. média (°C)", f"{temp_media:.1f}" if not np.isnan(temp_media) else "—")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ABAS PRINCIPAIS
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Série Temporal",
    "Clima × Dengue",
    "Matriz de Correlação",
    "Mapa de Ourinhos",
    "Sobre a Dengue",
    "Explicação do Modelo",
])

# ABA 1 - SERIE TEMPORAL
with tab1:
    st.markdown("### Série Histórica — Casos Estimados vs Reais")

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_filtrado["Data_Inicio_Semanas"],
        y=df_filtrado["casos_est"],
        name="Casos estimados",
        line=dict(color=VERDE, width=2),
        hovertemplate="SE %{x|%d/%m/%Y}<br>Estimados: %{y:.0f}<extra></extra>",
    ))
    if "casos" in df_filtrado.columns:
        fig_ts.add_trace(go.Scatter(
            x=df_filtrado["Data_Inicio_Semanas"],
            y=df_filtrado["casos"],
            name="Casos notificados",
            line=dict(color="#42A5F5", width=1.5, dash="dot"),
            hovertemplate="SE %{x|%d/%m/%Y}<br>Notificados: %{y:.0f}<extra></extra>",
        ))
    # Faixa estimativa
    if "casos_est_min" in df_filtrado.columns and "casos_est_max" in df_filtrado.columns:
        fig_ts.add_trace(go.Scatter(
            x=df_filtrado["Data_Inicio_Semanas"],
            y=df_filtrado["casos_est_max"],
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_ts.add_trace(go.Scatter(
            x=df_filtrado["Data_Inicio_Semanas"],
            y=df_filtrado["casos_est_min"],
            fill="tonexty",
            fillcolor="rgba(0,121,107,0.10)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
            name="Intervalo estimado",
        ))

    fig_ts.update_layout(**PLOTLY_LAYOUT, title="Casos estimados vs notificados — Ourinhos/SP", height=480)
    st.plotly_chart(fig_ts, use_container_width=True)

    # Previsao futura
    st.markdown("### Previsão das Próximas Semanas")
    fig_prev = go.Figure()

    # Últimas 20 semanas reais
    df_recentes = df.tail(20).copy()
    fig_prev.add_trace(go.Scatter(
        x=df_recentes["Data_Inicio_Semanas"],
        y=df_recentes["casos_est"],
        name="Casos estimados (real)",
        line=dict(color="#42A5F5", width=2),
    ))

    # Previsão
    from epiweeks import Week
    datas_prev = []
    for _, r in df_prev.iterrows():
        try:
            d = Week(int(r["ano"]), int(r["semana_epidemiologica"]), system="cdc").startdate()
            datas_prev.append(pd.Timestamp(d))
        except Exception:
            datas_prev.append(pd.NaT)

    fig_prev.add_trace(go.Scatter(
        x=datas_prev,
        y=df_prev["casos_previstos"],
        name="Previsão (modelo)",
        line=dict(color=VERDE, width=3, dash="dash"),
        marker=dict(size=8, color=VERDE),
    ))
    fig_prev.add_trace(go.Scatter(
        x=datas_prev,
        y=df_prev["maximo_estimado"],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_prev.add_trace(go.Scatter(
        x=datas_prev,
        y=df_prev["minimo_estimado"],
        fill="tonexty",
        fillcolor="rgba(0,121,107,0.15)",
        line=dict(width=0), showlegend=False,
        name="Intervalo previsão",
    ))
    fig_prev.update_layout(**PLOTLY_LAYOUT, title="Previsão de casos — próximas semanas", height=400)
    st.plotly_chart(fig_prev, use_container_width=True)

    # Comparacao anual
    st.markdown("### Comparação Anual de Casos")
    anual = df_filtrado.groupby("ano")["casos_est"].sum().reset_index()
    fig_anual = px.bar(
        anual, x="ano", y="casos_est",
        labels={"ano": "Ano", "casos_est": "Total de casos estimados"},
        color_discrete_sequence=[VERDE],
    )
    fig_anual.update_layout(**PLOTLY_LAYOUT, title="Total de casos estimados por ano", height=380)
    st.plotly_chart(fig_anual, use_container_width=True)


# ABA 2 - CLIMA x DENGUE
with tab2:
    st.markdown("### Relação entre Variáveis Climáticas e Dengue")

    # Temperatura + casos no mesmo grafico de eixo duplo
    fig_clima = make_subplots(specs=[[{"secondary_y": True}]])
    fig_clima.add_trace(
        go.Scatter(
            x=df_filtrado["Data_Inicio_Semanas"],
            y=df_filtrado["casos_est"],
            name="Casos estimados",
            line=dict(color=VERDE, width=2),
        ),
        secondary_y=False,
    )
    if "tempmed" in df_filtrado.columns:
        fig_clima.add_trace(
            go.Scatter(
                x=df_filtrado["Data_Inicio_Semanas"],
                y=df_filtrado["tempmed"],
                name="Temperatura (°C)",
                line=dict(color="#FFA726", width=1.5),
            ),
            secondary_y=True,
        )
    fig_clima.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        title="Casos estimados × Temperatura média do ar",
        height=450,
    )
    fig_clima.update_yaxes(title_text="Casos estimados", secondary_y=False, gridcolor="#222244")
    fig_clima.update_yaxes(title_text="Temperatura (°C)", secondary_y=True, gridcolor="#222244")
    fig_clima.update_xaxes(gridcolor="#222244")
    st.plotly_chart(fig_clima, use_container_width=True)

    # Chuva x Casos
    fig_chuva = make_subplots(specs=[[{"secondary_y": True}]])
    fig_chuva.add_trace(
        go.Bar(
            x=df_filtrado["Data_Inicio_Semanas"],
            y=df_filtrado["chuva"],
            name="Chuva (mm)",
            marker_color="#42A5F5",
            opacity=0.5,
        ),
        secondary_y=False,
    )
    fig_chuva.add_trace(
        go.Scatter(
            x=df_filtrado["Data_Inicio_Semanas"],
            y=df_filtrado["casos_est"],
            name="Casos estimados",
            line=dict(color=VERDE, width=2),
        ),
        secondary_y=True,
    )
    fig_chuva.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        title="Precipitação × Casos estimados",
        height=450,
    )
    fig_chuva.update_yaxes(title_text="Chuva (mm)", secondary_y=False, gridcolor="#222244")
    fig_chuva.update_yaxes(title_text="Casos estimados", secondary_y=True, gridcolor="#222244")
    fig_chuva.update_xaxes(gridcolor="#222244")
    st.plotly_chart(fig_chuva, use_container_width=True)

    # Scatter com trendline
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_sc1 = px.scatter(
            df_filtrado, x="tempmed", y="casos_est",
            trendline="ols",
            labels={"tempmed": "Temperatura média (°C)", "casos_est": "Casos estimados"},
            color_discrete_sequence=[VERDE],
            opacity=0.6,
        )
        fig_sc1.update_layout(**PLOTLY_LAYOUT, title="Temperatura × Casos", height=380)
        st.plotly_chart(fig_sc1, use_container_width=True)

    with col_s2:
        fig_sc2 = px.scatter(
            df_filtrado, x="chuva", y="casos_est",
            trendline="ols",
            labels={"chuva": "Chuva (mm)", "casos_est": "Casos estimados"},
            color_discrete_sequence=["#42A5F5"],
            opacity=0.6,
        )
        fig_sc2.update_layout(**PLOTLY_LAYOUT, title="Chuva × Casos", height=380)
        st.plotly_chart(fig_sc2, use_container_width=True)


# ABA 3 - MATRIZ DE CORRELACAO
with tab3:
    st.markdown("### Matriz de Correlação entre Variáveis do Modelo")

    colunas_corr = [
        "casos_est", "tempmed", "chuva", "umidade",
        "chuva_lag_3", "chuva_lag_4", "temp_lag_4",
        "casos_lag_1", "casos_lag_2", "casos_mm4", "idade_media",
    ]
    colunas_existentes = [c for c in colunas_corr if c in df_filtrado.columns]
    corr = df_filtrado[colunas_existentes].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=["#E8F5E9", VERDE_CLARO, VERDE, VERDE_ESCURO],
        aspect="auto",
    )
    fig_corr.update_layout(
        **PLOTLY_LAYOUT,
        title="Correlação de Pearson entre as variáveis",
        height=600,
        coloraxis_colorbar=dict(title="r"),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown(f"""
    <div class="info-box">
        <b style="color:{VERDE_ESCURO}">Como interpretar:</b>
        valores próximos de <b>+1</b> indicam forte correlação positiva;
        valores próximos de <b>−1</b> indicam forte correlação negativa;
        valores próximos de <b>0</b> indicam ausência de relação linear.
    </div>
    """, unsafe_allow_html=True)


# ABA 4 - MAPA 3D DE OURINHOS
# ABA 4 - MAPA 3D DE OURINHOS

with tab4:
    st.markdown("### Mapa 3D de Ourinhos — SP")
    LAT_OURINHOS = -22.9786
    LON_OURINHOS = -49.8709
    # Dados dos bairros
    if prox_semana is not None:
        lbl_r, _ = nivel_risco(prox_semana["casos_previstos"])
        base_casos = prox_semana["casos_previstos"]
    else:
        lbl_r = "—"
        base_casos = 5
    bairros_lista_map = [
        {"nome": "Centro",          "lat": -22.9786, "lon": -49.8709, "casos": base_casos * 1.0,  "cor": [0, 121, 107, 200]},
        {"nome": "Vila Brasil",     "lat": -22.9730, "lon": -49.8580, "casos": base_casos * 0.7,  "cor": [38, 166, 154, 180]},
        {"nome": "Jd. Matilde",     "lat": -22.9860, "lon": -49.8800, "casos": base_casos * 0.5,  "cor": [38, 166, 154, 180]},
        {"nome": "Jd. Paulista",    "lat": -22.9650, "lon": -49.8650, "casos": base_casos * 0.8,  "cor": [0, 121, 107, 200]},
        {"nome": "Vila Musa",       "lat": -22.9900, "lon": -49.8600, "casos": base_casos * 0.6,  "cor": [38, 166, 154, 180]},
        {"nome": "Jd. Ouro Verde",  "lat": -22.9700, "lon": -49.8850, "casos": base_casos * 0.4,  "cor": [128, 203, 196, 160]},
        {"nome": "Vila Nova",       "lat": -22.9830, "lon": -49.8550, "casos": base_casos * 0.55, "cor": [38, 166, 154, 180]},
        {"nome": "Jd. Europa",      "lat": -22.9680, "lon": -49.8750, "casos": base_casos * 0.35, "cor": [128, 203, 196, 160]},
        {"nome": "Jd. Santa Fé",    "lat": -22.9850, "lon": -49.8650, "casos": base_casos * 0.65, "cor": [38, 166, 154, 180]},
        {"nome": "Pq. Minas Gerais","lat": -22.9750, "lon": -49.8550, "casos": base_casos * 0.45, "cor": [128, 203, 196, 160]},
        {"nome": "Vila Margarida",  "lat": -22.9820, "lon": -49.8760, "casos": base_casos * 0.5,  "cor": [38, 166, 154, 180]},
        {"nome": "Jd. São Paulo",   "lat": -22.9710, "lon": -49.8680, "casos": base_casos * 0.75, "cor": [0, 121, 107, 200]},
    ]
    bairros_data = pd.DataFrame(bairros_lista_map)
    bairros_data["casos"] = bairros_data["casos"].round(0).astype(int)
    bairros_data["elevation"] = bairros_data["casos"] * 80
    # Filtro interativo no mapa
    bairro_selecionado = st.selectbox("Selecione um bairro para destacar no mapa:", ["Todos"] + [b["nome"] for b in bairros_lista_map], index=0)
    if bairro_selecionado != "Todos":
        bairros_data["cor"] = bairros_data.apply(lambda row: row["cor"] if row["nome"] == bairro_selecionado else [180,180,180,80], axis=1)
    # Camada de colunas 3D
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=bairros_data,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=1,
        radius=120,
        get_fill_color="cor",
        pickable=True,
        auto_highlight=True,
        extruded=True,
    )
    # Camada de texto com nomes dos bairros
    text_layer = pdk.Layer(
        "TextLayer",
        data=bairros_data,
        get_position=["lon", "lat"],
        get_text="nome",
        get_size=14,
        get_color=[0, 121, 107, 220],
        get_angle=0,
        get_text_anchor='"middle"',
        get_alignment_baseline='"bottom"',
        get_pixel_offset=[0, -20],
    )
    # Anel de risco ao redor do centro (ScatterplotLayer)
    risco_center = pd.DataFrame([{
        "lat": LAT_OURINHOS,
        "lon": LON_OURINHOS,
        "radius": max(400, base_casos * 60),
    }])
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=risco_center,
        get_position=["lon", "lat"],
        get_radius="radius",
        get_fill_color=[0, 121, 107, 40],
        get_line_color=[0, 121, 107, 160],
        stroked=True,
        line_width_min_pixels=2,
        pickable=False,
    )
    view_state = pdk.ViewState(
        latitude=LAT_OURINHOS,
        longitude=LON_OURINHOS,
        zoom=13.2,
        pitch=55,
        bearing=-15,
    )
    deck = pdk.Deck(
        layers=[scatter_layer, column_layer, text_layer],
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={
            "html": "<b>{nome}</b><br/>Casos estimados: <b>{casos}</b>",
            "style": {
                "backgroundColor": "#1A1A2E",
                "color": "#26A69A",
                "border": "1px solid #00796B",
                "borderRadius": "8px",
                "padding": "8px 12px",
                "fontFamily": "sans-serif",
            },
        },
    )
    st.pydeck_chart(deck, height=600)
    # Legenda e info
    col_map1, col_map2 = st.columns(2)
    with col_map1:
        st.markdown(f"""
        <div class="info-box">
            <b style="color:{VERDE_ESCURO}">Colunas 3D</b><br><br>
            A <b>altura</b> de cada coluna representa a estimativa proporcional
            de casos de dengue naquela região. Quanto mais alta e escura,
            maior o nível de atenção necessário.<br><br>
            <b>Risco atual: {lbl_r}</b> — {base_casos:.0f} casos previstos
        </div>
        """, unsafe_allow_html=True)
    with col_map2:
        st.markdown(f"""
        <div class="info-box">
            <b style="color:{VERDE_ESCURO}">Navegação 3D</b><br><br>
            • <b>Rotacionar:</b> clique + arraste com botão direito<br>
            • <b>Inclinar:</b> Ctrl + clique + arraste<br>
            • <b>Zoom:</b> scroll do mouse<br>
            • <b>Mover:</b> clique + arraste com botão esquerdo<br><br>
            <small style="color:{VERDE}">Passe o mouse sobre as colunas para ver detalhes.</small>
        </div>
        """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        <b style="color:{VERDE_ESCURO}">Nota:</b>
        Os valores por bairro são estimativas proporcionais baseadas na previsão
        municipal do modelo. Dados georreferenciados por bairro serão
        integrados futuramente com o módulo do SINAN para maior precisão.
    </div>
    """, unsafe_allow_html=True)
# ABA 6 - EXPLICAÇÃO DO MODELO
with tab6:
    st.markdown("## Explicação do Modelo Preditivo")
    st.markdown(f"""
    <div class="info-box">
        <b style="color:{VERDE_ESCURO}">Como funciona o modelo?</b><br><br>
        O modelo utiliza algoritmos de aprendizado de máquina (XGBoost Tweedie e Random Forest) para prever o número de casos de dengue nas próximas semanas, considerando variáveis climáticas (chuva, temperatura, umidade), dados epidemiológicos e lags temporais.<br><br>
        <b style="color:{VERDE}">Principais variáveis:</b><br>
        • Chuva semanal<br>
        • Temperatura média semanal<br>
        • Lags de chuva e temperatura<br>
        • Casos de dengue em semanas anteriores<br>
        • Idade média dos casos<br>
        <br>
        <b style="color:{VERDE}">Importância das variáveis:</b><br>
        O modelo avalia a importância de cada variável para a previsão. Em geral, chuva e casos anteriores têm maior peso.<br><br>
        <b style="color:{VERDE}">Limitações:</b><br>
        • Dados incompletos ou atrasados podem afetar a precisão<br>
        • Não considera fatores sociais ou mutações do vírus<br>
        • Previsão é probabilística, não determinística<br>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        <b style="color:{VERDE_ESCURO}">Como interpretar?</b><br><br>
        • As previsões indicam tendência, não valores exatos<br>
        • Use em conjunto com ações de vigilância e prevenção<br>
        • Consulte sempre fontes oficiais de saúde<br>
    </div>
    """, unsafe_allow_html=True)


# ABA 5 - SOBRE A DENGUE
with tab5:
    st.markdown("### Informações sobre a Dengue")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown(f"""
        <div class="info-box" style="min-height:320px;">
            <h3 style="color:{VERDE_ESCURO}; margin-top:0;">O que é a Dengue?</h3>
            A dengue é uma doença febril aguda causada por um arbovírus
            do gênero <i>Flavivirus</i>, transmitido pela picada da fêmea
            do mosquito <b><i>Aedes aegypti</i></b> infectado.<br><br>
            Existem <b>4 sorotipos</b> do vírus (DEN-1 a DEN-4). A infecção
            por um sorotipo confere imunidade permanente para aquele tipo,
            mas uma segunda infecção por sorotipo diferente aumenta o risco
            de dengue grave.<br><br>
            <b style="color:{VERDE}">Período de incubação:</b> 4 a 10 dias<br>
            <b style="color:{VERDE}">Duração:</b> 5 a 7 dias em média
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <h3 style="color:{VERDE_ESCURO}; margin-top:0;">Quando procurar ajuda?</h3>
            Procure atendimento médico imediatamente se apresentar:<br><br>
            • Dor abdominal intensa e contínua<br>
            • Vômitos persistentes<br>
            • Sangramento de mucosas<br>
            • Acúmulo de líquidos (ascite, derrame pleural)<br>
            • Letargia e/ou irritabilidade<br>
            • Queda abrupta de plaquetas<br><br>
            <b style="color:{VERDE_ESCURO}">Estes são sinais de alarme para dengue grave!</b>
        </div>
        """, unsafe_allow_html=True)

    with col_info2:
        st.markdown(f"""
        <div class="info-box" style="min-height:320px;">
            <h3 style="color:{VERDE_ESCURO}; margin-top:0;">Sintomas</h3>
            <b>Dengue Clássica:</b><br>
            • Febre alta (39–40 °C) de início súbito<br>
            • Dor de cabeça intensa<br>
            • Dor atrás dos olhos (retro-orbital)<br>
            • Dores musculares e articulares<br>
            • Manchas vermelhas na pele (exantema)<br>
            • Náuseas e perda de apetite<br>
            • Fadiga e fraqueza<br><br>
            <b>Dengue Grave (hemorrágica):</b><br>
            • Sangramento nasal, gengival<br>
            • Dor abdominal intensa<br>
            • Vômitos persistentes<br>
            • Pele pálida, fria e úmida<br>
            • Dificuldade respiratória
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <h3 style="color:{VERDE_ESCURO}; margin-top:0;">Prevenção</h3>
            <b>Combate ao mosquito:</b><br>
            • Eliminar água parada em vasos, pneus, garrafas<br>
            • Manter caixas d'água tampadas<br>
            • Usar telas em portas e janelas<br>
            • Aplicar repelente regularmente<br>
            • Usar roupas compridas em áreas de risco<br><br>
            <b>Curiosidade:</b> O <i>Aedes aegypti</i> tem hábitos diurnos e
            pica preferencialmente no início da manhã e final da tarde.
            Um único mosquito pode infectar até 300 pessoas em seu ciclo
            de vida.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Ciclo de vida visual
    st.markdown(f"""
    <div style="text-align:center; background:{CINZA_CARD}; border-radius:14px;
                padding:28px; border:1px solid {BORDA}; margin:16px 0;">
        <h3 style="color:{VERDE_ESCURO}; margin-top:0;">Ciclo de Vida do Aedes aegypti</h3>
        <div style="display:flex; justify-content:center; gap:12px; margin:20px 0;">
            <span style="font-size:18px; color:{VERDE}; font-weight:700;">OVO</span>
            <span style="font-size:18px; color:{TEXTO};">→</span>
            <span style="font-size:18px; color:{VERDE}; font-weight:700;">LARVA</span>
            <span style="font-size:18px; color:{TEXTO};">→</span>
            <span style="font-size:18px; color:{VERDE}; font-weight:700;">PUPA</span>
            <span style="font-size:18px; color:{TEXTO};">→</span>
            <span style="font-size:18px; color:{VERDE}; font-weight:700;">ADULTO</span>
        </div>
        <div style="display:flex; justify-content:center; gap:40px; color:{TEXTO};">
            <div><b>Ovo</b><br><small>2-3 dias</small></div>
            <div><b>Larva</b><br><small>5-7 dias</small></div>
            <div><b>Pupa</b><br><small>2-3 dias</small></div>
            <div><b>Adulto</b><br><small>30-45 dias</small></div>
        </div>
        <p style="color:{VERDE}; margin-top:16px; font-size:13px;">
            Ciclo completo: apenas <b>7 a 10 dias</b> em condições tropicais ideais
        </p>
    </div>
    """, unsafe_allow_html=True)


# RODAPE
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; padding:16px; color:{TEXTO_SECUNDARIO}; font-size:13px;">
    <b style="color:{VERDE_ESCURO}">Prefeitura de Ourinhos — Secretaria Municipal de Saúde</b><br>
    Trabalho de Conclusão de Curso • Vigilância Epidemiológica da Dengue<br>
    Dados: InfoDengue (Fiocruz) · INMET · SINAN<br>
    Modelo preditivo: XGBoost Tweedie + Random Forest<br><br>
    Combata o Aedes aegypti — é responsabilidade de todos!
</div>
""", unsafe_allow_html=True)
