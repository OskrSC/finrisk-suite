import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.helpers import gen_portfolio_data, COLORS

st.set_page_config(page_title="Stress Test de Portafolio", page_icon="📈", layout="wide")
st.markdown("""
<style>
[data-testid="stMetric"]{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1rem 1.25rem}
.section-title{font-size:1.1rem;font-weight:700;color:#1e293b;border-left:4px solid #0891b2;padding-left:12px;margin:1.5rem 0 1rem}
.info-box{background:#ecfeff;border:1px solid #a5f3fc;border-radius:10px;padding:12px 16px;font-size:13px;color:#0e7490;margin:8px 0}
.warn-box{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:12px 16px;font-size:13px;color:#92400e;margin:8px 0}
.danger-box{background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:12px 16px;font-size:13px;color:#991b1b;margin:8px 0}
</style>
""", unsafe_allow_html=True)

SCENARIOS = {
    "Base (sin estrés)": {
        "pd_mult": 1.0, "lgd_mult": 1.0, "description":
        "Condiciones normales de mercado. PD y LGD sin ajuste.",
        "color": COLORS["success"],
    },
    "Recesión Leve (2001)": {
        "pd_mult": 1.8, "lgd_mult": 1.2, "description":
        "Desaceleración económica moderada. PD +80%, LGD +20%.",
        "color": COLORS["warning"],
    },
    "Crisis Financiera (2008)": {
        "pd_mult": 3.5, "lgd_mult": 1.5, "description":
        "Colapso crediticio severo. PD +250%, LGD +50%.",
        "color": "#ea580c",
    },
    "Crisis COVID (2020)": {
        "pd_mult": 2.5, "lgd_mult": 1.3, "description":
        "Shock pandémico con parálisis sectorial. PD +150%, LGD +30%.",
        "color": "#7c3aed",
    },
    "Crisis Soberana (Custom)": {
        "pd_mult": None, "lgd_mult": None, "description":
        "Configura tus propios multiplicadores de estrés.",
        "color": COLORS["danger"],
    },
}

ASSET_CLASS_PD_SENS = {
    "Hipotecario":  1.0,
    "Consumo":      1.4,
    "Corporativo":  1.1,
    "PYME":         1.6,
    "Auto":         1.2,
}

@st.cache_data(show_spinner="Generando portafolio de préstamos…")
def load():
    return gen_portfolio_data(500)

df = load()

st.markdown("# 📈 Stress Test de Portafolio de Crédito")
st.markdown("Simula el impacto de **escenarios de crisis** sobre un portafolio de préstamos. Mide pérdida esperada, VaR y capital requerido bajo Basilea III.")
st.markdown("---")

total_balance = df["balance"].sum()
base_el       = (df["pd_base"] * df["lgd"] * df["ead"]).sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Balance total portafolio", f"${total_balance/1e6:.1f}M")
c2.metric("Pérdida Esperada base",    f"${base_el/1e3:.0f}K",   f"{base_el/total_balance:.2%} del portafolio")
c3.metric("Préstamos",                f"{len(df):,}")
c4.metric("Clases de activo",         "5")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Scenario Configuration
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Configuración de Escenario de Estrés</div>', unsafe_allow_html=True)

col_cfg, col_desc = st.columns([2, 1])
with col_cfg:
    scenario_name = st.selectbox("Selecciona escenario", list(SCENARIOS.keys()))
    scen = SCENARIOS[scenario_name]

    if scen["pd_mult"] is None:  # Custom
        c1s, c2s = st.columns(2)
        pd_mult  = c1s.slider("Multiplicador PD",  1.0, 6.0, 2.0, step=0.1)
        lgd_mult = c2s.slider("Multiplicador LGD", 1.0, 2.5, 1.3, step=0.05)
    else:
        pd_mult  = scen["pd_mult"]
        lgd_mult = scen["lgd_mult"]

    # Asset-class sensitivity sliders
    with st.expander("⚙️ Ajustar sensibilidad por clase de activo"):
        sens_overrides = {}
        cc = st.columns(5)
        for i, (cls, base_sens) in enumerate(ASSET_CLASS_PD_SENS.items()):
            sens_overrides[cls] = cc[i].slider(
                cls, 0.5, 3.0, float(base_sens), step=0.1, key=f"sens_{cls}"
            )

with col_desc:
    color = scen["color"]
    st.markdown(f"""
    <div style='background:{color}18;border:2px solid {color};border-radius:12px;padding:16px;margin-top:8px'>
        <div style='font-size:1rem;font-weight:700;color:{color}'>{scenario_name}</div>
        <div style='font-size:12px;color:#475569;margin-top:6px'>{scen['description']}</div>
        <div style='margin-top:12px'>
            <span style='font-size:13px;font-weight:600;color:{color}'>PD ×{pd_mult:.1f}</span>
            &nbsp;&nbsp;
            <span style='font-size:13px;font-weight:600;color:{color}'>LGD ×{lgd_mult:.1f}</span>
        </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Stress Calculation
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Resultados del Stress Test</div>', unsafe_allow_html=True)

stressed = df.copy()
for cls, sens in sens_overrides.items():
    mask = stressed["asset_class"] == cls
    stressed.loc[mask, "pd_stress"]  = (df.loc[mask, "pd_base"]  * pd_mult  * sens).clip(0, 1)
    stressed.loc[mask, "lgd_stress"] = (df.loc[mask, "lgd"]      * lgd_mult).clip(0, 1)

stressed["el_base"]   = df["pd_base"] * df["lgd"] * df["ead"]
stressed["el_stress"] = stressed["pd_stress"] * stressed["lgd_stress"] * df["ead"]
stressed["el_delta"]  = stressed["el_stress"] - stressed["el_base"]

total_el_base   = stressed["el_base"].sum()
total_el_stress = stressed["el_stress"].sum()
el_increase     = total_el_stress - total_el_base
el_pct_balance  = total_el_stress / total_balance

# VaR via Monte Carlo
np.random.seed(42)
N_SIM = 10_000
sim_losses = []
for _ in range(N_SIM):
    defaults   = np.random.binomial(1, stressed["pd_stress"].values)
    loss_given = defaults * stressed["lgd_stress"].values * stressed["ead"].values
    sim_losses.append(loss_given.sum())
sim_losses = np.array(sim_losses)
var_95     = np.percentile(sim_losses, 95)
var_99     = np.percentile(sim_losses, 99)
cvar_95    = sim_losses[sim_losses >= var_95].mean()
capital_req = var_99 * 1.06   # simplified Basel III capital buffer

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("EL Base",           f"${total_el_base/1e3:.0f}K",   f"{total_el_base/total_balance:.2%}")
mc2.metric("EL Estresada",      f"${total_el_stress/1e3:.0f}K", f"{el_pct_balance:.2%}",  delta_color="inverse")
mc3.metric("Incremento EL",     f"${el_increase/1e3:.0f}K",     f"+{el_increase/total_el_base:.0%} vs base", delta_color="inverse")
mc4.metric("VaR 99% (MC)",      f"${var_99/1e3:.0f}K",          "Monte Carlo 10K sims")
mc5.metric("Capital requerido", f"${capital_req/1e3:.0f}K",     "Basilea III aprox.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Charts
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📊 EL por Clase", "📉 Distribución MC", "🔥 Mapa de Calor", "📋 Detalle"])

with tab1:
    by_class = stressed.groupby("asset_class").agg(
        el_base=("el_base","sum"),
        el_stress=("el_stress","sum"),
        balance=("balance","sum")
    ).reset_index()
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="EL Base", x=by_class["asset_class"], y=by_class["el_base"],
        marker_color=COLORS["success"], opacity=0.85,
        hovertemplate="%{x}<br>EL Base: $%{y:,.0f}<extra></extra>"
    ))
    fig_bar.add_trace(go.Bar(
        name="EL Estresada", x=by_class["asset_class"], y=by_class["el_stress"],
        marker_color=color, opacity=0.85,
        hovertemplate="%{x}<br>EL Estrés: $%{y:,.0f}<extra></extra>"
    ))
    fig_bar.update_layout(
        barmode="group", title="Pérdida Esperada Base vs Estresada por Clase",
        xaxis_title="Clase de Activo", yaxis_title="Pérdida Esperada (USD)",
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        legend=dict(x=0.7, y=0.95), margin=dict(t=40,b=40,l=60,r=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # EL as % of balance
    by_class["el_stress_pct"] = by_class["el_stress"] / by_class["balance"]
    fig_pct = go.Figure(go.Bar(
        x=by_class["asset_class"], y=by_class["el_stress_pct"],
        marker_color=[color]*len(by_class),
        text=[f"{v:.2%}" for v in by_class["el_stress_pct"]],
        textposition="outside",
        hovertemplate="%{x}: %{y:.2%}<extra></extra>"
    ))
    fig_pct.update_layout(
        title="EL Estresada como % del Balance por Clase",
        xaxis_title="Clase de Activo", yaxis_title="EL / Balance",
        yaxis_tickformat=".1%",
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        margin=dict(t=40,b=40,l=60,r=20)
    )
    st.plotly_chart(fig_pct, use_container_width=True)

with tab2:
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(
        x=sim_losses / 1e3, nbinsx=80, name="Distribución de pérdidas",
        marker_color=COLORS["primary"], opacity=0.75
    ))
    fig_mc.add_vline(x=total_el_stress/1e3, line_dash="dash",
                     line_color=COLORS["warning"],
                     annotation_text=f"EL Estresada ${total_el_stress/1e3:.0f}K")
    fig_mc.add_vline(x=var_95/1e3, line_dash="dot",
                     line_color=COLORS["danger"],
                     annotation_text=f"VaR 95% ${var_95/1e3:.0f}K")
    fig_mc.add_vline(x=var_99/1e3, line_dash="solid",
                     line_color=COLORS["danger"],
                     annotation_text=f"VaR 99% ${var_99/1e3:.0f}K")
    fig_mc.update_layout(
        title=f"Distribución de Pérdidas — Monte Carlo ({N_SIM:,} simulaciones)",
        xaxis_title="Pérdida total portafolio (USD miles)",
        yaxis_title="Frecuencia",
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        margin=dict(t=40,b=40,l=60,r=20)
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    vc1, vc2, vc3 = st.columns(3)
    vc1.metric("VaR 95%",  f"${var_95/1e3:.0f}K",  "Pérdida máx. 95% escenarios")
    vc2.metric("VaR 99%",  f"${var_99/1e3:.0f}K",  "Pérdida máx. 99% escenarios")
    vc3.metric("CVaR 95%", f"${cvar_95/1e3:.0f}K", "Pérdida media en cola 5%")
    st.markdown('<div class="info-box">El VaR mide la pérdida que no se superará con cierto nivel de confianza. El CVaR (Expected Shortfall) mide el promedio de las pérdidas en la cola, siendo más conservador y recomendado por BCBS.</div>', unsafe_allow_html=True)

with tab3:
    # Heatmap: pd_stress vs lgd_stress by asset class
    pivot = stressed.groupby("asset_class").agg(
        pd_stress=("pd_stress","mean"),
        lgd_stress=("lgd_stress","mean"),
        el_stress=("el_stress","sum")
    ).reset_index()

    # Grid scenarios for sensitivity
    pd_grid  = np.linspace(0.01, 0.40, 15)
    lgd_grid = np.linspace(0.10, 0.90, 15)
    z_grid   = np.zeros((len(lgd_grid), len(pd_grid)))
    for i, lgd_v in enumerate(lgd_grid):
        for j, pd_v in enumerate(pd_grid):
            z_grid[i, j] = (pd_v * lgd_v * df["ead"].mean() * len(df)) / 1e3

    fig_hm = go.Figure(go.Heatmap(
        x=pd_grid, y=lgd_grid, z=z_grid,
        colorscale="RdYlGn_r",
        colorbar=dict(title="EL aprox. (USD K)"),
        hovertemplate="PD: %{x:.1%}<br>LGD: %{y:.1%}<br>EL: $%{z:.0f}K<extra></extra>"
    ))
    # Mark current scenario
    avg_pd  = stressed["pd_stress"].mean()
    avg_lgd = stressed["lgd_stress"].mean()
    fig_hm.add_trace(go.Scatter(
        x=[avg_pd], y=[avg_lgd], mode="markers+text",
        marker=dict(color="white", size=14, symbol="star",
                    line=dict(color="black", width=2)),
        text=[scenario_name], textposition="top right",
        name="Escenario actual"
    ))
    fig_hm.update_layout(
        title="Mapa de Calor: Pérdida Esperada (PD × LGD)",
        xaxis_title="Probabilidad de Default promedio",
        yaxis_title="Loss Given Default promedio",
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        paper_bgcolor="white", margin=dict(t=40,b=60,l=70,r=20)
    )
    st.plotly_chart(fig_hm, use_container_width=True)

with tab4:
    display_cols = ["asset_class","balance","pd_base","pd_stress","lgd","lgd_stress","el_base","el_stress","el_delta"]
    summary = stressed.groupby("asset_class")[display_cols[1:]].mean().reset_index()
    summary.columns = ["Clase","Balance","PD Base","PD Stress","LGD Base","LGD Stress","EL Base","EL Stress","Delta EL"]
    summary[["Balance","EL Base","EL Stress","Delta EL"]] = summary[
        ["Balance","EL Base","EL Stress","Delta EL"]
    ]
    st.dataframe(
        summary.style
        .format({
            "Balance":"${:,.0f}","PD Base":"{:.2%}","PD Stress":"{:.2%}",
            "LGD Base":"{:.2%}","LGD Stress":"{:.2%}",
            "EL Base":"${:,.0f}","EL Stress":"${:,.0f}","Delta EL":"${:,.0f}"
        })
        .background_gradient(subset=["PD Stress","EL Stress","Delta EL"], cmap="Reds"),
        use_container_width=True
    )
    st.markdown("---")

    # Multi-scenario comparison table
    st.markdown("**Comparativa de todos los escenarios:**")
    comparison_rows = []
    for sname, sdata in SCENARIOS.items():
        if sdata["pd_mult"] is None:
            continue
        pm, lm = sdata["pd_mult"], sdata["lgd_mult"]
        el_s = (df["pd_base"].clip(0,1) * pm * df["lgd"].clip(0,1) * lm * df["ead"]).sum()
        comparison_rows.append({
            "Escenario":         sname,
            "PD mult":           f"{pm:.1f}x",
            "LGD mult":          f"{lm:.1f}x",
            "EL Estresada":      el_s,
            "EL / Balance":      el_s / total_balance,
            "Incremento vs Base": (el_s - total_el_base) / total_el_base,
        })
    comp_df = pd.DataFrame(comparison_rows)
    st.dataframe(
        comp_df.style.format({
            "EL Estresada":"${:,.0f}",
            "EL / Balance":"{:.2%}",
            "Incremento vs Base":"{:+.1%}"
        }).background_gradient(subset=["EL / Balance","Incremento vs Base"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True
    )
