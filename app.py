# =============================================================================
# SpVAR + ESDA | Homicidios Colombia
# VERSION_DASHBOARD=2026-10-27-v8-plotly-fix
# Streamlit · sin scikit-learn · numpy puro · compatible Python 3.12–3.14
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="SpVAR Homicidios Colombia",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background: linear-gradient(165deg, #0b1220, #0f172a 50%, #111827); color: #f1f5f9; }
h1, h2, h3, p, label, span, div { color: #f1f5f9 !important; }
[data-testid="stMetricValue"] { color: #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)

DEPTS = [
    "AMAZONAS", "ANTIOQUIA", "ARAUCA", "ATLÁNTICO", "BOLÍVAR", "BOYACÁ",
    "CALDAS", "CAQUETÁ", "CASANARE", "CAUCA", "CESAR", "CHOCÓ", "CÓRDOBA",
    "CUNDINAMARCA", "GUAINÍA", "GUAVIARE", "HUILA", "LA GUAJIRA",
    "MAGDALENA", "META", "NARIÑO", "NORTE DE SANTANDER", "PUTUMAYO",
    "QUINDÍO", "RISARALDA", "SAN ANDRÉS", "SANTANDER", "SUCRE", "TOLIMA",
    "VALLE DEL CAUCA", "VAUPÉS", "VICHADA", "BOGOTÁ D.C.",
]
COORDS = {
    "AMAZONAS": (-71.5, -1.5), "ANTIOQUIA": (-75.5, 6.5), "ARAUCA": (-70.8, 7.0),
    "ATLÁNTICO": (-74.8, 10.7), "BOLÍVAR": (-74.5, 9.0), "BOYACÁ": (-73.5, 5.5),
    "CALDAS": (-75.5, 5.2), "CAQUETÁ": (-74.0, 1.0), "CASANARE": (-71.5, 5.5),
    "CAUCA": (-76.5, 2.5), "CESAR": (-73.5, 9.5), "CHOCÓ": (-77.0, 6.0),
    "CÓRDOBA": (-75.8, 8.5), "CUNDINAMARCA": (-74.2, 4.8), "GUAINÍA": (-69.0, 2.5),
    "GUAVIARE": (-72.5, 2.0), "HUILA": (-75.5, 2.5), "LA GUAJIRA": (-72.5, 11.5),
    "MAGDALENA": (-74.2, 10.5), "META": (-73.5, 3.5), "NARIÑO": (-77.5, 1.5),
    "NORTE DE SANTANDER": (-72.5, 8.0), "PUTUMAYO": (-76.5, 0.5), "QUINDÍO": (-75.7, 4.5),
    "RISARALDA": (-75.8, 5.0), "SAN ANDRÉS": (-81.7, 12.5), "SANTANDER": (-73.0, 7.0),
    "SUCRE": (-75.0, 9.2), "TOLIMA": (-75.0, 4.0), "VALLE DEL CAUCA": (-76.5, 3.5),
    "VAUPÉS": (-70.5, 1.0), "VICHADA": (-69.0, 5.0), "BOGOTÁ D.C.": (-74.1, 4.6),
}
HIGH = {
    "CAUCA", "VALLE DEL CAUCA", "ANTIOQUIA", "NARIÑO", "CHOCÓ",
    "NORTE DE SANTANDER", "ARAUCA", "PUTUMAYO", "CAQUETÁ",
}
VARS = ["Tasa_Homicidios", "Desempleo", "Pobreza", "Urbanizacion"]
N_PERIODS = 10


def knn_W(lon, lat, k=5):
    """KNN row-standardized weights with pure numpy."""
    xy = np.column_stack([lon, lat])
    n = len(xy)
    k = min(k, n - 1)
    # squared distances
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argpartition(d2, k, axis=1)[:, :k]
    W = np.zeros((n, n), dtype=float)
    rows = np.repeat(np.arange(n), k)
    W[rows, nn.ravel()] = 1.0 / k
    return W


@st.cache_data(show_spinner="Calculando modelo…")
def estimate_model():
    rng = np.random.default_rng(42)
    n = len(DEPTS)
    base = {
        "Tasa_Homicidios": rng.lognormal(3.2, 0.5, n),
        "Desempleo": rng.uniform(6.5, 18.0, n),
        "Pobreza": rng.uniform(18.0, 55.0, n),
        "Urbanizacion": rng.uniform(35.0, 92.0, n),
    }
    for i, d in enumerate(DEPTS):
        if d in HIGH:
            base["Tasa_Homicidios"][i] *= 1.4
            base["Pobreza"][i] *= 1.12

    rows, prev = [], {k: v.copy() for k, v in base.items()}
    for t in range(N_PERIODS):
        shock = rng.normal(0, 0.05, (n, 4))
        new = {
            "Tasa_Homicidios": np.clip(
                0.78 * prev["Tasa_Homicidios"]
                + 0.10 * prev["Tasa_Homicidios"].mean()
                + 0.15 * (prev["Pobreza"] - base["Pobreza"]) / 10
                + shock[:, 0] * base["Tasa_Homicidios"] * 0.3,
                4, 95,
            ),
            "Desempleo": np.clip(0.85 * prev["Desempleo"] + shock[:, 1] * 1.2, 4.5, 24),
            "Pobreza": np.clip(0.88 * prev["Pobreza"] - 0.1 * t + shock[:, 2] * 1.5, 12, 65),
            "Urbanizacion": np.clip(0.93 * prev["Urbanizacion"] + 0.02 * t + shock[:, 3], 30, 98),
        }
        for i, d in enumerate(DEPTS):
            lon, lat = COORDS[d]
            rows.append({
                "DEPTO": d, "Periodo": t + 1, "lon": lon, "lat": lat,
                "Tasa_Homicidios": float(new["Tasa_Homicidios"][i]),
                "Desempleo": float(new["Desempleo"][i]),
                "Pobreza": float(new["Pobreza"][i]),
                "Urbanizacion": float(new["Urbanizacion"][i]),
            })
        prev = new

    panel = pd.DataFrame(rows)
    last_df = panel[panel["Periodo"] == N_PERIODS].reset_index(drop=True)

    W = knn_W(last_df["lon"].values, last_df["lat"].values, k=5)
    y = last_df["Tasa_Homicidios"].values.astype(float)
    yc = y - y.mean()
    Moran_I = float(n / W.sum() * (yc @ W @ yc) / (yc @ yc + 1e-12))
    rng2 = np.random.default_rng(0)
    I0 = abs(Moran_I)
    cnt = 0
    for _ in range(99):
        yp = rng2.permutation(y)
        yc2 = yp - yp.mean()
        Ip = abs(n / W.sum() * (yc2 @ W @ yc2) / (yc2 @ yc2 + 1e-12))
        if Ip >= I0:
            cnt += 1
    Moran_p = (cnt + 1) / 100.0

    k = len(VARS)
    cube = np.zeros((N_PERIODS, n, k))
    for t in range(N_PERIODS):
        sub = panel[panel["Periodo"] == t + 1].set_index("DEPTO")
        for j, v in enumerate(VARS):
            cube[t, :, j] = sub.loc[DEPTS, v].values
    mu, sd = cube.reshape(-1, k).mean(0), cube.reshape(-1, k).std(0) + 1e-9
    z = (cube - mu) / sd

    Y_list, X_list = [], []
    for t in range(1, N_PERIODS):
        Ylag = z[t - 1]
        X_list.append(np.hstack([Ylag, W @ Ylag]))
        Y_list.append(z[t])
    Y_all, X_all = np.vstack(Y_list), np.vstack(X_list)

    A_mat = np.zeros((k, k))
    B_mat = np.zeros((k, k))
    for j in range(k):
        coef, *_ = np.linalg.lstsq(X_all, Y_all[:, j], rcond=None)
        A_mat[j] = coef[:k]
        B_mat[j] = coef[k:]

    direct = float(A_mat[0, 0])
    indirect = float(B_mat[0, 0])
    total = direct + indirect
    max_mod = float(np.max(np.abs(np.linalg.eigvals(A_mat + B_mat))))
    stable = max_mod < 1.0

    return {
        "last_df": last_df,
        "W_name": "KNN(k=5)",
        "Moran_I": Moran_I,
        "Moran_p": Moran_p,
        "direct": direct,
        "indirect": indirect,
        "total": total,
        "max_mod": max_mod,
        "stable": stable,
    }


def make_map(df, col, title, scale="YlOrRd"):
    v = df[col].astype(float).values
    s = 10 + 24 * (v - v.min()) / (v.max() - v.min() + 1e-9)
    fig = go.Figure(go.Scattergeo(
        lon=df["lon"].tolist(),
        lat=df["lat"].tolist(),
        text=df["DEPTO"].tolist(),
        mode="markers+text",
        marker=dict(
            size=s.tolist(),
            color=v.tolist(),
            colorscale=scale,
            opacity=0.9,
            colorbar=dict(title=col),
            line=dict(width=0.5, color="#94a3b8"),
        ),
        textposition="top center",
        textfont=dict(size=9, color="#ffffff"),
        hovertemplate="<b>%{text}</b><br>" + col + ": %{marker.color:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=480,
        geo=dict(
            scope="south america",
            lonaxis_range=[-82, -66],
            lataxis_range=[-5, 13.5],
            showland=True,
            landcolor="#1e293b",
            showocean=True,
            oceancolor="#0b1220",
            showcountries=True,
            countrycolor="#475569",
            bgcolor="rgba(0,0,0,0)",
            resolution=50,
        ),
    )
    return fig


def make_3d(df):
    h = (df["Tasa_Homicidios"] - df["Tasa_Homicidios"].min()) / (
        df["Tasa_Homicidios"].max() - df["Tasa_Homicidios"].min() + 1e-9)
    s = 8 + 18 * (df["Pobreza"] - df["Pobreza"].min()) / (
        df["Pobreza"].max() - df["Pobreza"].min() + 1e-9)
    fig = go.Figure(go.Scatter3d(
        x=df["lon"].tolist(),
        y=df["lat"].tolist(),
        z=h.tolist(),
        mode="markers+text",
        marker=dict(
            size=s.tolist(),
            color=df["Desempleo"].tolist(),
            colorscale="RdYlBu_r",
            opacity=0.9,
            colorbar=dict(title="Desempleo"),
        ),
        text=df["DEPTO"].tolist(),
        textposition="top center",
        textfont=dict(size=8, color="#ffffff"),
        hovertemplate="<b>%{text}</b><br>Hom: %{customdata[0]:.1f}<br>Pob: %{customdata[1]:.1f}<extra></extra>",
        customdata=np.column_stack([df["Tasa_Homicidios"], df["Pobreza"]]).tolist(),
    ))
    fig.update_layout(
        title="3D · Altura=Homicidios · Tamaño=Pobreza · Color=Desempleo",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="Lon",
            yaxis_title="Lat",
            zaxis_title="Homicidios",
            bgcolor="rgba(15,23,42,0.9)",
        ),
        font=dict(color="#ffffff"),
    )
    return fig


def make_bars(direct, indirect, total):
    fig = go.Figure(data=[
        go.Bar(name="φ Directo", x=["Homicidios"], y=[direct], marker_color="#f87171",
               text=[f"{direct:.3f}"], textposition="outside"),
        go.Bar(name="θ Indirecto", x=["Homicidios"], y=[indirect], marker_color="#38bdf8",
               text=[f"{indirect:.3f}"], textposition="outside"),
        go.Bar(name="Total", x=["Homicidios"], y=[total], marker_color="#a78bfa",
               text=[f"{total:.3f}"], textposition="outside"),
    ])
    fig.update_layout(
        barmode="group", title="Impactos SpVAR",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.5)",
        font=dict(color="#fff"), height=360, margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(gridcolor="#334155"),
    )
    return fig


m = estimate_model()
df = m["last_df"]

st.title("SpVAR + ESDA — Homicidios departamentales Colombia")
st.caption("VERSION_DASHBOARD=v8 · Streamlit · numpy puro")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matriz W", m["W_name"])
c2.metric("Moran I", f"{m['Moran_I']:.3f}", f"p≈{m['Moran_p']:.2f}")
c3.metric("max|λ|", f"{m['max_mod']:.3f}", "Estable" if m["stable"] else "Inestable")
c4.metric("Impacto total", f"{m['total']:.3f}")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Introducción", "Metodología", "Mapa", "Ficha", "3D", "SpVAR", "Ranking / Política",
])

with tab1:
    st.markdown(f"""
### ¿Qué es este estudio?
Análisis de **dependencia espacial** y **transmisión** de la tasa de homicidios entre departamentos.

- **W:** `{m['W_name']}`
- **Moran I:** {m['Moran_I']:.3f} (p≈{m['Moran_p']:.2f})
- **φ (directo):** {m['direct']:.3f}
- **θ (indirecto):** {m['indirect']:.3f}
- **Estabilidad:** max|λ| = {m['max_mod']:.3f} → **{"ESTABLE" if m["stable"] else "INESTABLE"}**
""")

with tab2:
    st.markdown(f"""
### Metodología
- 33 departamentos · {N_PERIODS} periodos (panel simulado)
- **W:** KNN k=5
- **SpVAR:** `Y_t = A Y_(t-1) + B (W Y)_(t-1) + e`
- Impactos: φ = {m['direct']:.3f}, θ = {m['indirect']:.3f}
""")

with tab3:
    var = st.selectbox(
        "Variable",
        ["Tasa_Homicidios", "Pobreza", "Desempleo", "Urbanizacion"],
        format_func=lambda x: {
            "Tasa_Homicidios": "Tasa de Homicidios",
            "Pobreza": "Pobreza %",
            "Desempleo": "Desempleo %",
            "Urbanizacion": "Urbanización %",
        }[x],
    )
    scales = {
        "Tasa_Homicidios": "YlOrRd", "Pobreza": "Oranges",
        "Desempleo": "Blues", "Urbanizacion": "Purples",
    }
    st.plotly_chart(make_map(df, var, f"{var} por departamento", scales[var]), use_container_width=True)

with tab4:
    dept = st.selectbox("Departamento", DEPTS)
    r = df.loc[df["DEPTO"] == dept].iloc[0]
    st.markdown(f"""
### {dept}
| Indicador | Valor |
|-----------|------:|
| **Homicidios** | {r['Tasa_Homicidios']:.1f} |
| **Pobreza %** | {r['Pobreza']:.1f} |
| **Desempleo %** | {r['Desempleo']:.1f} |
| **Urbanización %** | {r['Urbanizacion']:.1f} |
""")

with tab5:
    st.plotly_chart(make_3d(df), use_container_width=True)

with tab6:
    st.plotly_chart(make_bars(m["direct"], m["indirect"], m["total"]), use_container_width=True)
    st.markdown("| **φ** | Persistencia local |\n| **θ** | Spillover espacial |")

with tab7:
    ranking = (
        df[["DEPTO", "Tasa_Homicidios", "Pobreza", "Desempleo", "Urbanizacion"]]
        .sort_values("Tasa_Homicidios", ascending=False)
        .reset_index(drop=True)
    )
    ranking.index += 1
    ranking = ranking.rename(columns={
        "DEPTO": "Departamento", "Tasa_Homicidios": "Tasa Homicidios",
        "Pobreza": "Pobreza %", "Desempleo": "Desempleo %", "Urbanizacion": "Urbanización %",
    }).round(1)
    st.dataframe(ranking, use_container_width=True)
    st.markdown("""
### Hallazgos y política
1. Dependencia espacial positiva.
2. Persistencia local (φ) y spillover (θ).
3. Sistema estable → coordinación interdepartamental en seguridad.
""")
