import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.helpers import (
    gen_fraud_data, build_fraud_model,
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix,
    plot_shap_bar, COLORS, RISK_COLORS
)

st.set_page_config(page_title="Detección de Fraude", page_icon="🔍", layout="wide")
st.markdown("""
<style>
[data-testid="stMetric"]{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1rem 1.25rem}
.section-title{font-size:1.1rem;font-weight:700;color:#1e293b;border-left:4px solid #d97706;padding-left:12px;margin:1.5rem 0 1rem}
.info-box{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:12px 16px;font-size:13px;color:#1e40af;margin:8px 0}
.warn-box{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:12px 16px;font-size:13px;color:#92400e;margin:8px 0}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Entrenando detectores de fraude…")
def load():
    df = gen_fraud_data(5000, fraud_rate=0.04)
    return df, *build_fraud_model(df)

df, model, scaler, features, X_te, y_te, y_prob, auc, ap, shap_vals, X_te_s = load()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Detección de Fraude en Transacciones")
st.markdown("Modelo **Random Forest + SMOTE** entrenado sobre transacciones financieras sintéticas con ~4% de fraude real.")
st.markdown("---")

fraud_rate = df["is_fraud"].mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC-ROC", f"{auc:.3f}", "Random Forest")
c2.metric("Average Precision", f"{ap:.3f}", "Métrica clave imbalance")
c3.metric("Tasa de fraude", f"{fraud_rate:.1%}", "Dataset sintético")
c4.metric("Transacciones", f"{len(df):,}", f"Fraudes: {df['is_fraud'].sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Evaluate Transaction
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Evaluación de Transacción Individual</div>', unsafe_allow_html=True)

merchant_labels = {0:"Supermercado",1:"Gasolina",2:"Online Retail",3:"Restaurante",4:"Viajes"}

col_f, col_t = st.columns([2, 1])
with col_f:
    with st.form("fraud_form"):
        c1, c2 = st.columns(2)
        with c1:
            amount      = st.number_input("Monto de la transacción (USD)", 1.0, 10000.0, 150.0, step=10.0)
            hour        = st.slider("Hora del día", 0, 23, 14)
            distance_km = st.number_input("Distancia desde ubicación habitual (km)", 0.0, 5000.0, 5.0, step=1.0)
        with c2:
            freq_day    = st.slider("Transacciones hoy (mismo usuario)", 1, 30, 3)
            merchant_cat= st.selectbox("Categoría comercio", list(merchant_labels.keys()),
                                       format_func=lambda x: merchant_labels[x])
            is_online   = st.selectbox("¿Transacción online?", [0,1], format_func=lambda x:"Sí" if x else "No")
        submitted = st.form_submit_button("🔍 Analizar transacción", use_container_width=True, type="primary")

with col_t:
    st.markdown("**Patrones de fraude típicos:**")
    st.markdown("""
    - 🌙 Transacciones en horas nocturnas (0–4am)
    - 💰 Montos inusualmente altos
    - 📍 Distancia geográfica elevada
    - 🔁 Alta frecuencia en pocas horas
    - 🌐 Comercios online + tarjeta presente
    """)

if submitted:
    row = pd.DataFrame([{
        "amount":       amount,  "hour":        hour,
        "distance_km":  distance_km, "freq_day": freq_day,
        "merchant_cat": merchant_cat, "is_online": is_online,
    }])
    X_in = scaler.transform(row[features])
    prob = model.predict_proba(X_in)[0, 1]

    threshold_sel = st.session_state.get("fraud_threshold", 0.40)
    decision_flag = prob >= threshold_sel
    col_a, col_b = st.columns([1, 2])
    with col_a:
        color = COLORS["danger"] if decision_flag else COLORS["success"]
        icon  = "🚨 FRAUDE DETECTADO" if decision_flag else "✅ LEGÍTIMA"
        st.markdown(f"""
        <div style='background:{color}18;border:2px solid {color};border-radius:12px;
                    padding:24px;text-align:center'>
            <div style='font-size:1.6rem;font-weight:800;color:{color}'>{icon}</div>
            <div style='font-size:2rem;font-weight:800;color:{color};margin-top:10px'>{prob:.1%}</div>
            <div style='font-size:0.85rem;color:#64748b'>Probabilidad de fraude</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        import shap as _shap
        import numpy as _np
        exp = _shap.TreeExplainer(model)
        sv  = exp.shap_values(X_in)
        # RandomForest devuelve (1, n_features, n_classes) en sklearn moderno
        # o lista [class0, class1] en versiones anteriores.
        # Siempre extraemos clase positiva (fraude=1) como vector 1D.
        if isinstance(sv, list):
            sv_f = _np.array(sv[1]).reshape(-1)
        else:
            sv = _np.array(sv)
            if sv.ndim == 3:
                sv_f = sv[0, :, 1]
            elif sv.ndim == 2:
                sv_f = sv[0]
            else:
                sv_f = sv.reshape(-1)
        contrib = [(features[i], float(sv_f[i])) for i in range(len(features))]
        contrib.sort(key=lambda x: abs(x[1]), reverse=True)
        labels  = [c[0] for c in contrib]
        vals    = [c[1] for c in contrib]
        bcolors = [COLORS["danger"] if v > 0 else COLORS["success"] for v in vals]
        fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=bcolors,
                               hovertemplate="%{y}: %{x:+.4f}<extra></extra>"))
        fig.update_layout(title="Señales de fraude detectadas (SHAP)",
                          xaxis_title="Contribución al riesgo de fraude",
                          height=280, paper_bgcolor="white", plot_bgcolor="#f8fafc",
                          font=dict(size=11), margin=dict(t=40,b=20,l=130,r=20))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Threshold & Performance
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Configuración de Umbral y Rendimiento</div>', unsafe_allow_html=True)
st.markdown('<div class="warn-box">⚠️ En fraude, un <strong>falso negativo</strong> (fraude no detectado) es mucho más costoso que un falso positivo. Ajustar el umbral según el costo operativo.</div>', unsafe_allow_html=True)

threshold = st.slider("Umbral de detección", 0.05, 0.80, 0.40, step=0.01, key="fraud_threshold")
y_pred_t = (y_prob >= threshold).astype(int)

from sklearn.metrics import precision_score, recall_score, f1_score
prec = precision_score(y_te, y_pred_t, zero_division=0)
rec  = recall_score(y_te, y_pred_t, zero_division=0)
f1   = f1_score(y_te, y_pred_t, zero_division=0)
n_flagged = y_pred_t.sum()

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Precisión",          f"{prec:.3f}", "% fraudes en alertas")
mc2.metric("Recall",             f"{rec:.3f}",  "% fraudes detectados")
mc3.metric("F1-Score",           f"{f1:.3f}")
mc4.metric("Transacciones alertadas", f"{n_flagged:,}", f"de {len(y_pred_t):,} totales")

tab1, tab2, tab3, tab4 = st.tabs(["📈 ROC", "📉 Prec-Recall", "🔢 Matriz", "🧠 SHAP"])
with tab1:
    st.plotly_chart(plot_roc_curve(y_te, y_prob, "ROC — Fraude"), use_container_width=True)
with tab2:
    st.plotly_chart(plot_pr_curve(y_te, y_prob, "Precisión-Recall — Fraude"), use_container_width=True)
with tab3:
    st.plotly_chart(plot_confusion_matrix(y_te, y_pred_t, ["Legítima","Fraude"]), use_container_width=True)
with tab4:
    st.plotly_chart(plot_shap_bar(shap_vals, features, title="SHAP Global — Fraude"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Transaction Explorer
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Explorador de Transacciones</div>', unsafe_allow_html=True)

sample = df.sample(min(800, len(df)), random_state=42).copy()
sample_X = scaler.transform(sample[features])
sample["prob_fraud"] = model.predict_proba(sample_X)[:, 1]
sample["flagged"]    = sample["prob_fraud"] >= threshold
sample["merchant_name"] = sample["merchant_cat"].map(merchant_labels)

ca, cb = st.columns(2)
with ca:
    fig_amnt = go.Figure()
    for label, mask, color in [("Legítima", sample["is_fraud"]==0, COLORS["success"]),
                                ("Fraude",   sample["is_fraud"]==1, COLORS["danger"])]:
        fig_amnt.add_trace(go.Histogram(
            x=sample.loc[mask, "amount"], name=label, opacity=0.65,
            nbinsx=40, marker_color=color
        ))
    fig_amnt.update_layout(barmode="overlay", title="Distribución de Montos por Clase",
                           xaxis_title="Monto (USD)", yaxis_title="Frecuencia",
                           paper_bgcolor="white", plot_bgcolor="#f8fafc",
                           margin=dict(t=40,b=40,l=50,r=20))
    st.plotly_chart(fig_amnt, use_container_width=True)
with cb:
    fig_hour = go.Figure()
    for label, mask, color in [("Legítima", sample["is_fraud"]==0, COLORS["success"]),
                                ("Fraude",   sample["is_fraud"]==1, COLORS["danger"])]:
        hourly = sample.loc[mask, "hour"].value_counts().sort_index()
        fig_hour.add_trace(go.Bar(x=hourly.index, y=hourly.values,
                                  name=label, opacity=0.75, marker_color=color))
    fig_hour.update_layout(barmode="group", title="Fraude por Hora del Día",
                           xaxis_title="Hora", yaxis_title="Transacciones",
                           paper_bgcolor="white", plot_bgcolor="#f8fafc",
                           margin=dict(t=40,b=40,l=50,r=20))
    st.plotly_chart(fig_hour, use_container_width=True)

# Top suspicious transactions
st.markdown("**Transacciones de mayor riesgo en la muestra:**")
top_risky = sample.nlargest(15, "prob_fraud")[
    ["amount","hour","distance_km","freq_day","merchant_name","prob_fraud","is_fraud","flagged"]
].rename(columns={
    "amount":"Monto","hour":"Hora","distance_km":"Distancia km","freq_day":"Freq/día",
    "merchant_name":"Comercio","prob_fraud":"P(Fraude)","is_fraud":"Fraude real","flagged":"Alertada"
})
st.dataframe(
    top_risky.style
    .format({"Monto":"${:,.2f}","P(Fraude)":"{:.1%}"})
    .background_gradient(subset=["P(Fraude)"], cmap="Reds"),
    use_container_width=True, height=350
)
