import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.helpers import (
    gen_corporate_data, build_rating_model,
    plot_shap_bar, hex_to_rgba, COLORS
)

st.set_page_config(page_title="Rating Corporativo", page_icon="🏢", layout="wide")
st.markdown("""
<style>
[data-testid="stMetric"]{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1rem 1.25rem}
.section-title{font-size:1.1rem;font-weight:700;color:#1e293b;border-left:4px solid #7c3aed;padding-left:12px;margin:1.5rem 0 1rem}
.info-box{background:#f5f3ff;border:1px solid #ddd6fe;border-radius:10px;padding:12px 16px;font-size:13px;color:#5b21b6;margin:8px 0}
</style>
""", unsafe_allow_html=True)

RATING_COLORS = {"AAA":"#16a34a","AA":"#65a30d","A":"#d97706","BBB":"#dc2626"}
RATING_LABELS = {1:"AAA", 2:"AA", 3:"A", 4:"BBB"}

@st.cache_data(show_spinner="Entrenando clasificador de rating…")
def load():
    df = gen_corporate_data(2000)
    return df, *build_rating_model(df)

df, model, scaler, features, X_te, y_te, y_pred, acc, shap_vals, X_te_s = load()

st.markdown("# 🏢 Rating Corporativo de Crédito")
st.markdown("Clasificador **XGBoost multiclase** que replica la metodología de ratings (AAA / AA / A / BBB) usando ratios financieros.")
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy global", f"{acc:.3f}", "4 clases")
c2.metric("Empresas en dataset", f"{len(df):,}")
c3.metric("Variables financieras", "6")
c4.metric("Modelo", "XGBoost multiclase")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Individual Rating
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Perfil Financiero de Empresa</div>', unsafe_allow_html=True)

with st.form("rating_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        dte = st.slider("Deuda/Patrimonio (D/E)", 0.0, 5.0, 1.2, step=0.05)
        roa = st.slider("Retorno sobre activos (ROA)", -0.05, 0.20, 0.06, step=0.005, format="%.3f")
    with c2:
        cr  = st.slider("Razón corriente (Current Ratio)", 0.3, 3.5, 1.4, step=0.05)
        ic  = st.slider("Cobertura de intereses", 0.0, 10.0, 3.5, step=0.25)
    with c3:
        rg  = st.slider("Crecimiento de ingresos", -0.15, 0.30, 0.07, step=0.005, format="%.3f")
        nm  = st.slider("Margen neto", -0.10, 0.40, 0.10, step=0.005, format="%.3f")
    submitted = st.form_submit_button("📊 Calcular Rating", use_container_width=True, type="primary")

if submitted:
    row = pd.DataFrame([{
        "debt_to_equity":    dte, "return_on_assets": roa,
        "current_ratio":     cr,  "interest_coverage": ic,
        "revenue_growth":    rg,  "net_margin":        nm,
    }])
    X_in    = scaler.transform(row[features])
    probs   = model.predict_proba(X_in)[0]          # 4-class probs
    pred_cls = model.predict(X_in)[0] + 1           # back to 1-4
    rating   = RATING_LABELS[pred_cls]
    color    = RATING_COLORS[rating]

    r1, r2, r3 = st.columns([1, 1, 2])
    with r1:
        st.markdown(f"""
        <div style='background:{color}18;border:3px solid {color};border-radius:16px;
                    padding:28px;text-align:center;margin-top:10px'>
            <div style='font-size:3.5rem;font-weight:900;color:{color}'>{rating}</div>
            <div style='font-size:0.9rem;color:#64748b;margin-top:8px'>Rating asignado</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        fig_prob = go.Figure(go.Bar(
            x=[RATING_LABELS[i+1] for i in range(4)],
            y=probs,
            marker_color=[RATING_COLORS[RATING_LABELS[i+1]] for i in range(4)],
            text=[f"{p:.1%}" for p in probs], textposition="outside",
            hovertemplate="%{x}: %{y:.1%}<extra></extra>"
        ))
        fig_prob.update_layout(
            title="Probabilidad por clase",
            yaxis_range=[0, 1], height=280,
            paper_bgcolor="white", plot_bgcolor="#f8fafc",
            margin=dict(t=40,b=20,l=20,r=20), font=dict(size=11)
        )
        st.plotly_chart(fig_prob, use_container_width=True)
    with r3:
        # Radar chart of financial profile
        categories = ["D/E (inv)", "ROA", "Curr. Ratio", "Int. Coverage", "Rev. Growth", "Net Margin"]
        # Normalize for radar (0-1 scale, invert D/E since lower is better)
        vals_norm = [
            1 - min(dte/5, 1),     # D/E inverted
            min(max(roa/0.20, 0), 1),
            min(max(cr/3.5, 0), 1),
            min(max(ic/10, 0), 1),
            min(max((rg+0.15)/0.45, 0), 1),
            min(max((nm+0.10)/0.50, 0), 1),
        ]
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals_norm + [vals_norm[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=hex_to_rgba(color, alpha=0.13),  # rgba() — Plotly no acepta 8-digit hex
            line_color=color, name=rating
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            title="Perfil financiero normalizado",
            height=280, paper_bgcolor="white",
            margin=dict(t=40,b=20,l=30,r=30), font=dict(size=11)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Benchmarks vs rating peers
    st.markdown("**Comparativa con promedios del dataset por rating:**")
    bench = df.groupby("rating_label")[features].mean().round(4)
    bench_display = bench.rename(columns={
        "debt_to_equity":"D/E","return_on_assets":"ROA",
        "current_ratio":"Curr. Ratio","interest_coverage":"Int. Coverage",
        "revenue_growth":"Rev. Growth","net_margin":"Net Margin"
    })
    user_row = pd.DataFrame([{
        "D/E":dte,"ROA":roa,"Curr. Ratio":cr,
        "Int. Coverage":ic,"Rev. Growth":rg,"Net Margin":nm
    }], index=["Tu empresa"])
    comparison = pd.concat([bench_display.reindex(["AAA","AA","A","BBB"]), user_row])
    st.dataframe(
        comparison.style
        .highlight_max(axis=0, color="#dcfce7", subset=bench_display.columns)
        .highlight_min(axis=0, color="#fef2f2", subset=bench_display.columns)
        .format("{:.4f}"),
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Model Performance
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Rendimiento del Clasificador</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔢 Matriz de Confusión", "🧠 SHAP Global", "📊 Distribución del Dataset"])
with tab1:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_te, y_pred, labels=[1,2,3,4])
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=["AAA","AA","A","BBB"], y=["AAA","AA","A","BBB"],
        colorscale=[[0,"#f0fdf4"],[1,"#15803d"]],
        text=cm, texttemplate="%{text}",
        hovertemplate="Actual: %{y}<br>Predicho: %{x}<br>Conteo: %{z}<extra></extra>"
    ))
    fig_cm.update_layout(title="Matriz de Confusión — Rating Corporativo",
                         xaxis_title="Predicho", yaxis_title="Real",
                         paper_bgcolor="white", plot_bgcolor="#f8fafc",
                         margin=dict(t=40,b=40,l=60,r=20))
    st.plotly_chart(fig_cm, use_container_width=True)
    st.markdown('<div class="info-box">Los errores de clasificación adyacentes (ej: AA→A) son menos graves que los errores de salto (ej: AAA→BBB). La mayoría de errores deberían estar cerca de la diagonal.</div>', unsafe_allow_html=True)
with tab2:
    if isinstance(shap_vals, list):
        shap_mean = np.mean([np.abs(sv) for sv in shap_vals], axis=0).mean(axis=0)
    else:
        shap_mean = np.abs(shap_vals).mean(axis=0) if shap_vals.ndim == 2 else np.abs(shap_vals)
    fig_shap = go.Figure(go.Bar(
        x=shap_mean, y=features, orientation="h",
        marker_color=COLORS["primary"],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"
    ))
    fig_shap.update_layout(title="Importancia SHAP — Rating Corporativo",
                           xaxis_title="|SHAP promedio|",
                           paper_bgcolor="white", plot_bgcolor="#f8fafc",
                           margin=dict(t=40,b=40,l=140,r=20))
    st.plotly_chart(fig_shap, use_container_width=True)
with tab3:
    dist = df["rating_label"].value_counts().reindex(["AAA","AA","A","BBB"])
    fig_dist = go.Figure(go.Bar(
        x=dist.index, y=dist.values,
        marker_color=[RATING_COLORS[k] for k in dist.index],
        text=dist.values, textposition="outside"
    ))
    fig_dist.update_layout(title="Distribución de Ratings en Dataset",
                           xaxis_title="Rating", yaxis_title="Empresas",
                           paper_bgcolor="white", plot_bgcolor="#f8fafc",
                           margin=dict(t=40,b=40,l=50,r=20))
    st.plotly_chart(fig_dist, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        box_df = df.melt(id_vars="rating_label", value_vars=["debt_to_equity","return_on_assets"],
                         var_name="ratio", value_name="value")
        fig_box = px.box(box_df, x="rating_label", y="value", color="ratio",
                         category_orders={"rating_label":["AAA","AA","A","BBB"]},
                         title="D/E y ROA por Rating",
                         color_discrete_map={"debt_to_equity":COLORS["danger"],"return_on_assets":COLORS["success"]})
        fig_box.update_layout(paper_bgcolor="white", plot_bgcolor="#f8fafc",
                              margin=dict(t=40,b=40,l=50,r=20))
        st.plotly_chart(fig_box, use_container_width=True)
    with c2:
        scatter_df = df.sample(400, random_state=42)
        fig_sc = px.scatter(scatter_df, x="debt_to_equity", y="return_on_assets",
                            color="rating_label",
                            color_discrete_map=RATING_COLORS,
                            category_orders={"rating_label":["AAA","AA","A","BBB"]},
                            opacity=0.65, title="D/E vs ROA — Separación por Rating")
        fig_sc.update_layout(paper_bgcolor="white", plot_bgcolor="#f8fafc",
                             margin=dict(t=40,b=40,l=60,r=20))
        st.plotly_chart(fig_sc, use_container_width=True)
