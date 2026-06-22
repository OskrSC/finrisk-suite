import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.helpers import (
    gen_default_data, build_default_model,
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix,
    plot_shap_bar, plot_score_gauge, prob_to_score,
    risk_label_from_prob, COLORS, RISK_COLORS
)

st.set_page_config(page_title="Predicción de Default", page_icon="⚠️", layout="wide")
st.markdown("""
<style>
[data-testid="stMetric"]{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1rem 1.25rem}
.section-title{font-size:1.1rem;font-weight:700;color:#1e293b;border-left:4px solid #dc2626;padding-left:12px;margin:1.5rem 0 1rem}
.info-box{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:12px 16px;font-size:13px;color:#1e40af;margin:8px 0}
.warn-box{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:12px 16px;font-size:13px;color:#92400e;margin:8px 0}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Entrenando modelo LightGBM con SMOTE…")
def load():
    df = gen_default_data(4000)
    return df, *build_default_model(df)

df, model, scaler, features, X_te, y_te, y_prob, auc, shap_vals, X_te_s = load()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚠️ Predicción de Default")
st.markdown("Probabilidad de incumplimiento a 12 meses usando **LightGBM + SMOTE** para manejo de clases desbalanceadas.")
st.markdown("---")

default_rate = df["default"].mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC-ROC", f"{auc:.3f}", "LightGBM")
c2.metric("Tasa de default real", f"{default_rate:.1%}", "Dataset sintético")
c3.metric("Casos en test set", f"{len(y_te):,}", f"Default: {y_te.sum():,}")
c4.metric("Balanceo con SMOTE", "✓ Activado", "Oversampling clase minoritaria")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Individual Prediction
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Evaluación de Prestatario Individual</div>', unsafe_allow_html=True)

with st.form("default_form"):
    st.markdown("Datos del préstamo y perfil financiero del solicitante.")
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score  = st.slider("Score crediticio", 300, 850, 650, step=5)
        annual_income = st.number_input("Ingreso anual (USD)", 10000, 300000, 50000, step=1000)
        loan_amount   = st.number_input("Monto del préstamo (USD)", 500, 100000, 18000, step=500)
    with c2:
        dti_ratio     = st.slider("Razón deuda-ingreso (DTI)", 0.05, 0.70, 0.35, step=0.01)
        interest_rate = st.slider("Tasa de interés (%)", 5.0, 30.0, 12.0, step=0.5)
        loan_term     = st.selectbox("Plazo (meses)", [12, 24, 36, 48, 60], index=2)
    with c3:
        prev_default   = st.selectbox("¿Default previo?", [0,1], format_func=lambda x:"Sí" if x else "No")
        months_open    = st.slider("Meses con crédito activo", 1, 120, 36)
        revolving_util = st.slider("Utilización de crédito revolvente", 0.0, 1.0, 0.30, step=0.01)
    submitted = st.form_submit_button("🔍 Calcular PD", use_container_width=True, type="primary")

if submitted:
    row = pd.DataFrame([{
        "credit_score":   credit_score,   "annual_income":  annual_income,
        "loan_amount":    loan_amount,     "dti_ratio":      dti_ratio,
        "prev_default":   prev_default,    "loan_term":      loan_term,
        "interest_rate":  interest_rate,   "months_open":    months_open,
        "revolving_util": revolving_util,
    }])
    X_in  = scaler.transform(row[features])
    prob  = model.predict_proba(X_in)[0, 1]
    risk  = risk_label_from_prob(prob, (0.15, 0.30, 0.50, 0.70))
    color = RISK_COLORS.get(risk, "#64748b")

    import shap as _shap
    from utils.helpers import extract_shap_local
    exp       = _shap.TreeExplainer(model)
    sv_raw    = exp.shap_values(X_in)

    r1, r2, r3 = st.columns([1, 1, 2])
    with r1:
        st.plotly_chart(plot_score_gauge(prob * 100, "Probabilidad de Default"), use_container_width=True)
    with r2:
        st.markdown(f"""
        <div style='background:{color}18;border:2px solid {color};border-radius:12px;
                    padding:20px;text-align:center;margin-top:10px'>
            <div style='font-size:2.2rem;font-weight:800;color:{color}'>{prob:.1%}</div>
            <div style='font-size:0.95rem;font-weight:600;color:{color}'>Probabilidad de Default</div>
            <div style='font-size:0.8rem;color:#64748b;margin-top:6px'>Nivel: {risk}</div>
        </div>""", unsafe_allow_html=True)
        ead = loan_amount
        lgd = 0.45
        expected_loss = ead * prob * lgd
        st.metric("Pérdida Esperada (EL)", f"${expected_loss:,.0f}", f"EAD×PD×LGD (LGD=45%)")
    with r3:
        sv_flat = extract_shap_local(sv_raw)
        contrib = [(features[i], float(sv_flat[i])) for i in range(len(features))]
        contrib.sort(key=lambda x: abs(x[1]), reverse=True)
        labels  = [c[0] for c in contrib[:8]]
        vals    = [c[1] for c in contrib[:8]]
        bcolors = [COLORS["danger"] if v > 0 else COLORS["success"] for v in vals]
        fig_shap = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h", marker_color=bcolors,
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>"
        ))
        fig_shap.update_layout(
            title="Drivers del riesgo (SHAP local)",
            xaxis_title="Contribución al PD (rojo=aumenta, verde=reduce)",
            height=300, paper_bgcolor="white", plot_bgcolor="#f8fafc",
            font=dict(size=11), margin=dict(t=40, b=20, l=140, r=20)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Threshold Analysis
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Análisis de Umbral de Decisión</div>', unsafe_allow_html=True)
st.markdown("El umbral óptimo depende del costo relativo de falsos positivos vs falsos negativos.")

threshold = st.slider("Umbral de clasificación (PD)", 0.10, 0.80, 0.35, step=0.01,
                       help="Por encima de este umbral se clasifica como 'Default probable'")

y_pred_t = (y_prob >= threshold).astype(int)
from sklearn.metrics import precision_score, recall_score, f1_score
prec = precision_score(y_te, y_pred_t, zero_division=0)
rec  = recall_score(y_te, y_pred_t, zero_division=0)
f1   = f1_score(y_te, y_pred_t, zero_division=0)
spec = (y_te == 0)[y_pred_t == 0].mean() if (y_pred_t == 0).any() else 0

tc1, tc2, tc3, tc4 = st.columns(4)
tc1.metric("Precisión",    f"{prec:.3f}", "TP/(TP+FP)")
tc2.metric("Recall",       f"{rec:.3f}",  "TP/(TP+FN)")
tc3.metric("F1-Score",     f"{f1:.3f}",   "Balance prec/recall")
tc4.metric("Especificidad",f"{spec:.3f}", "TN/(TN+FP)")

tab1, tab2, tab3, tab4 = st.tabs(["📈 ROC", "📉 Precisión-Recall", "🔢 Conf. Matrix", "🧠 SHAP Global"])
with tab1:
    st.plotly_chart(plot_roc_curve(y_te, y_prob, "Curva ROC — Default Prediction"), use_container_width=True)
with tab2:
    st.plotly_chart(plot_pr_curve(y_te, y_prob, "Curva Precisión-Recall — Default Prediction"), use_container_width=True)
    st.markdown('<div class="warn-box">⚠️ Para datos desbalanceados, la curva Precisión-Recall es más informativa que la ROC. Un modelo perfecto tendría AP=1.0.</div>', unsafe_allow_html=True)
with tab3:
    fig_cm = plot_confusion_matrix(y_te, y_pred_t, labels=["No Default","Default"])
    st.plotly_chart(fig_cm, use_container_width=True)
with tab4:
    fig_shap_g = plot_shap_bar(shap_vals, features, title="Importancia SHAP Global — Default")
    st.plotly_chart(fig_shap_g, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PD Distribution & Expected Loss
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Distribución de PD y Pérdida Esperada en Portafolio</div>', unsafe_allow_html=True)

sample = df.sample(min(500, len(df)), random_state=42).copy()
sample_X = scaler.transform(sample[features])
sample["pd"] = model.predict_proba(sample_X)[:, 1]
sample["el"] = sample["loan_amount"] * sample["pd"] * 0.45

ca, cb = st.columns(2)
with ca:
    fig_pd = go.Figure()
    fig_pd.add_trace(go.Histogram(x=sample["pd"], nbinsx=30, name="PD",
                                  marker_color=COLORS["primary"], opacity=0.75))
    fig_pd.add_vline(x=threshold, line_dash="dash", line_color="#dc2626",
                     annotation_text=f"Umbral={threshold}")
    fig_pd.update_layout(title="Distribución de Probabilidad de Default",
                         xaxis_title="PD estimada", yaxis_title="Préstamos",
                         paper_bgcolor="white", plot_bgcolor="#f8fafc",
                         margin=dict(t=40,b=40,l=50,r=20))
    st.plotly_chart(fig_pd, use_container_width=True)
with cb:
    total_el = sample["el"].sum()
    total_loan = sample["loan_amount"].sum()
    st.metric("Pérdida Esperada Total (portafolio)", f"${total_el:,.0f}",
              f"{total_el/total_loan:.2%} del portafolio")
    fig_el = go.Figure(go.Scatter(
        x=sample["pd"], y=sample["el"],
        mode="markers", marker=dict(color=COLORS["warning"], opacity=0.5, size=5),
        hovertemplate="PD: %{x:.1%}<br>EL: $%{y:,.0f}<extra></extra>"
    ))
    fig_el.update_layout(title="Pérdida Esperada por Préstamo vs PD",
                         xaxis_title="PD estimada", yaxis_title="EL (USD)",
                         paper_bgcolor="white", plot_bgcolor="#f8fafc",
                         margin=dict(t=40,b=40,l=60,r=20))
    st.plotly_chart(fig_el, use_container_width=True)
