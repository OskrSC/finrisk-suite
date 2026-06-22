import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.helpers import (
    gen_credit_data, build_credit_model,
    plot_roc_curve, plot_shap_bar, plot_score_gauge,
    prob_to_score, risk_label_from_prob, COLORS, RISK_COLORS
)

st.set_page_config(page_title="Scoring de Crédito", page_icon="📊", layout="wide")

st.markdown("""
<style>
[data-testid="stMetric"]{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1rem 1.25rem}
.section-title{font-size:1.1rem;font-weight:700;color:#1e293b;border-left:4px solid #2563eb;padding-left:12px;margin:1.5rem 0 1rem}
.info-box{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:12px 16px;font-size:13px;color:#1e40af;margin:8px 0}
</style>
""", unsafe_allow_html=True)

# ── Cache data & model ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Entrenando modelo de scoring…")
def load():
    df = gen_credit_data(3000)
    model, scaler, features, X_te, y_te, y_prob, auc, shap_values, X_te_s = build_credit_model(df)
    return df, model, scaler, features, X_te, y_te, y_prob, auc, shap_values, X_te_s

df, model, scaler, features, X_te, y_te, y_prob, auc, shap_values, X_te_s = load()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Scoring de Crédito")
st.markdown("Motor de aprobación crediticia con **XGBoost + SHAP**. Evalúa solicitudes en tiempo real y explica cada decisión.")
st.markdown("---")

# ── KPIs ──────────────────────────────────────────────────────────────────────
approved_rate = df["approved"].mean()
avg_score     = df["credit_score"].mean()
high_risk_pct = (y_prob > 0.65).mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC-ROC del modelo",  f"{auc:.3f}",            "XGBoost calibrado")
c2.metric("Tasa de aprobación",  f"{approved_rate:.1%}",  "Dataset sintético")
c3.metric("Score crédito medio", f"{avg_score:.0f}",      "Rango 300–850")
c4.metric("Solicitantes riesgo alto", f"{high_risk_pct:.1%}", "Probabilidad > 65%")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Individual Evaluation
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Evaluación de Solicitud Individual</div>', unsafe_allow_html=True)

with st.form("credit_form"):
    st.markdown("Ingresa los datos del solicitante para calcular el score y la decisión crediticia.")
    col1, col2, col3 = st.columns(3)
    with col1:
        credit_score  = st.slider("Score crediticio histórico", 300, 850, 680, step=5)
        annual_income = st.number_input("Ingreso anual (USD)", 10000, 300000, 55000, step=1000)
        loan_amount   = st.number_input("Monto solicitado (USD)", 500, 100000, 20000, step=500)
    with col2:
        dti_ratio     = st.slider("Razón deuda-ingreso (DTI)", 0.05, 0.70, 0.30, step=0.01, format="%.2f")
        emp_years     = st.slider("Años de empleo", 0, 30, 5)
        employment    = st.selectbox("Estado laboral", ["Employed","Self-employed","Unemployed"])
    with col3:
        prev_default  = st.selectbox("¿Default previo?", [0, 1], format_func=lambda x: "Sí" if x else "No")
        num_accounts  = st.slider("Cuentas activas", 1, 15, 4)
        late_payments = st.slider("Pagos tardíos (últimos 24m)", 0, 10, 1)
    submitted = st.form_submit_button("🔍 Evaluar solicitud", use_container_width=True, type="primary")

if submitted:
    emp_map  = {"Employed": 2, "Self-employed": 1, "Unemployed": 0}
    input_df = pd.DataFrame([{
        "credit_score":   credit_score,
        "annual_income":  annual_income,
        "loan_amount":    loan_amount,
        "dti_ratio":      dti_ratio,
        "prev_default":   prev_default,
        "emp_years":      emp_years,
        "num_accounts":   num_accounts,
        "late_payments":  late_payments,
        "employment_enc": emp_map[employment],
    }])
    X_in  = scaler.transform(input_df[features])
    prob  = model.predict_proba(X_in)[0, 1]
    score = prob_to_score(prob)
    risk  = risk_label_from_prob(prob, (0.25, 0.45, 0.65, 0.80))
    decision = "✅ APROBADO" if prob < 0.45 else "❌ RECHAZADO"

    import shap as _shap
    explainer  = _shap.TreeExplainer(model)
    shap_local = explainer.shap_values(X_in)[0]

    r1, r2, r3 = st.columns([1, 1, 2])
    with r1:
        st.plotly_chart(plot_score_gauge(score / 850 * 100, "Score FICO-style"), use_container_width=True)
        st.metric("Score calculado", score, f"Riesgo: {risk}")
    with r2:
        color = RISK_COLORS.get(risk, "#64748b")
        st.markdown(f"""
        <div style='background:{color}18;border:2px solid {color};border-radius:12px;
                    padding:20px;text-align:center;margin-top:20px'>
            <div style='font-size:2rem;font-weight:800;color:{color}'>{decision}</div>
            <div style='font-size:1.1rem;font-weight:600;color:{color};margin-top:6px'>
                Prob. rechazo: {prob:.1%}
            </div>
            <div style='font-size:0.85rem;color:#64748b;margin-top:4px'>Nivel de riesgo: {risk}</div>
        </div>""", unsafe_allow_html=True)
        ratio = annual_income / max(loan_amount, 1)
        st.metric("Ratio ingreso/deuda", f"{ratio:.1f}x", "Mínimo recomendado: 2.5x")
    with r3:
        contrib  = [(features[i], shap_local[i]) for i in range(len(features))]
        contrib.sort(key=lambda x: abs(x[1]), reverse=True)
        labels   = [c[0] for c in contrib[:8]]
        vals     = [c[1] for c in contrib[:8]]
        bar_col  = [COLORS["danger"] if v > 0 else COLORS["success"] for v in vals]
        fig_shap = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h",
            marker_color=bar_col,
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>"
        ))
        fig_shap.update_layout(
            title="Contribución SHAP al rechazo (rojo=sube riesgo, verde=baja riesgo)",
            xaxis_title="Valor SHAP", height=320,
            paper_bgcolor="white", plot_bgcolor="#f8fafc",
            font=dict(size=11), margin=dict(t=40, b=20, l=140, r=20)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Model Performance
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Rendimiento del Modelo en Test Set</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Curva ROC", "🔢 Distribución de Scores", "🧠 Importancia Global SHAP"])

with tab1:
    fig_roc = plot_roc_curve(y_te, y_prob, "Curva ROC — Scoring de Crédito")
    st.plotly_chart(fig_roc, use_container_width=True)
    st.markdown(f'<div class="info-box">AUC = <strong>{auc:.3f}</strong> — Un AUC > 0.80 indica buen poder discriminante. Un modelo aleatorio obtendría 0.50.</div>', unsafe_allow_html=True)

with tab2:
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=y_prob[y_te == 0], name="Aprobados reales",
        opacity=0.7, nbinsx=40, marker_color=COLORS["success"]
    ))
    fig_dist.add_trace(go.Histogram(
        x=y_prob[y_te == 1], name="Rechazados reales",
        opacity=0.7, nbinsx=40, marker_color=COLORS["danger"]
    ))
    fig_dist.add_vline(x=0.45, line_dash="dash", line_color="#1e293b",
                       annotation_text="Umbral 0.45")
    fig_dist.update_layout(
        title="Distribución de Probabilidades por Clase Real",
        barmode="overlay", xaxis_title="Probabilidad de rechazo",
        yaxis_title="Frecuencia",
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        legend=dict(x=0.7, y=0.95)
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    fig_shap_global = plot_shap_bar(shap_values, features, title="Importancia Global SHAP — Crédito")
    st.plotly_chart(fig_shap_global, use_container_width=True)
    st.markdown('<div class="info-box">SHAP (SHapley Additive exPlanations) cuantifica cuánto contribuye cada variable a la predicción. A diferencia de la importancia por Gini, SHAP es consistente y aditivo.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Portfolio Explorer
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Explorador del Portafolio Sintético</div>', unsafe_allow_html=True)

sample = df.sample(min(500, len(df)), random_state=42).copy()
sample_X = scaler.transform(sample[features])
sample["prob_rechazo"] = model.predict_proba(sample_X)[:, 1]
sample["score"]  = sample["prob_rechazo"].apply(prob_to_score)
sample["riesgo"] = sample["prob_rechazo"].apply(
    lambda p: risk_label_from_prob(p, (0.25, 0.45, 0.65, 0.80))
)

filt_cols = st.columns(3)
min_score = filt_cols[0].slider("Score mínimo", 300, 850, 300)
max_dti   = filt_cols[1].slider("DTI máximo", 0.05, 0.70, 0.70, step=0.01)
riesgo_f  = filt_cols[2].multiselect("Nivel de riesgo", ["Muy Bajo","Bajo","Medio","Alto","Muy Alto"],
                                      default=["Muy Bajo","Bajo","Medio","Alto","Muy Alto"])
filtered = sample[(sample["credit_score"] >= min_score) &
                  (sample["dti_ratio"] <= max_dti) &
                  (sample["riesgo"].isin(riesgo_f))]

c1, c2 = st.columns([3, 2])
with c1:
    fig_scatter = go.Figure(go.Scatter(
        x=filtered["credit_score"], y=filtered["annual_income"],
        mode="markers",
        marker=dict(
            color=filtered["prob_rechazo"],
            colorscale="RdYlGn_r", size=7, opacity=0.7,
            colorbar=dict(title="P(rechazo)"),
            line=dict(width=0.5, color="white")
        ),
        text=filtered["riesgo"],
        hovertemplate="Score: %{x}<br>Ingreso: $%{y:,.0f}<br>Riesgo: %{text}<extra></extra>"
    ))
    fig_scatter.update_layout(
        title="Score vs Ingreso (color = probabilidad de rechazo)",
        xaxis_title="Score crediticio", yaxis_title="Ingreso anual (USD)",
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        margin=dict(t=40, b=40, l=60, r=20)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
with c2:
    seg_counts = filtered["riesgo"].value_counts().reindex(
        ["Muy Bajo","Bajo","Medio","Alto","Muy Alto"], fill_value=0
    )
    fig_pie = go.Figure(go.Pie(
        labels=seg_counts.index, values=seg_counts.values,
        marker_colors=[RISK_COLORS[k] for k in seg_counts.index],
        hole=0.4, textinfo="label+percent"
    ))
    fig_pie.update_layout(title="Distribución de Riesgo", height=360, paper_bgcolor="white")
    st.plotly_chart(fig_pie, use_container_width=True)

with st.expander("📋 Ver tabla de solicitantes filtrados"):
    show_cols = ["credit_score","annual_income","loan_amount","dti_ratio","riesgo","score","prob_rechazo"]
    st.dataframe(
        filtered[show_cols].rename(columns={
            "credit_score":"Score hist.","annual_income":"Ingreso","loan_amount":"Préstamo",
            "dti_ratio":"DTI","riesgo":"Riesgo","score":"Score calc.","prob_rechazo":"P(rechazo)"
        }).style.format({
            "Ingreso": "${:,.0f}", "Préstamo": "${:,.0f}", "DTI": "{:.2f}",
            "Score calc.": "{:d}", "P(rechazo)": "{:.1%}"
        }).background_gradient(subset=["P(rechazo)"], cmap="RdYlGn_r"),
        use_container_width=True, height=320
    )
