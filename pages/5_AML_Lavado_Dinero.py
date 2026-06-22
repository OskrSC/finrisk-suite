import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.helpers import (
    gen_aml_data, build_aml_model,
    plot_roc_curve, plot_confusion_matrix,
    plot_shap_bar, COLORS, RISK_COLORS
)

st.set_page_config(page_title="AML — Lavado de Dinero", page_icon="🚨", layout="wide")
st.markdown("""
<style>
[data-testid="stMetric"]{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1rem 1.25rem}
.section-title{font-size:1.1rem;font-weight:700;color:#1e293b;border-left:4px solid #dc2626;padding-left:12px;margin:1.5rem 0 1rem}
.danger-box{background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:12px 16px;font-size:13px;color:#991b1b;margin:8px 0}
.info-box{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:12px 16px;font-size:13px;color:#1e40af;margin:8px 0}
</style>
""", unsafe_allow_html=True)

AML_LEVELS = {
    (0.00, 0.20): ("Bajo",     "#16a34a"),
    (0.20, 0.45): ("Medio",    "#d97706"),
    (0.45, 0.70): ("Alto",     "#ea580c"),
    (0.70, 1.01): ("Crítico",  "#dc2626"),
}

def aml_level(prob):
    for (lo, hi), (label, color) in AML_LEVELS.items():
        if lo <= prob < hi:
            return label, color
    return "Crítico", "#dc2626"

@st.cache_data(show_spinner="Entrenando sistema AML…")
def load():
    df = gen_aml_data(3000)
    return df, *build_aml_model(df)

df, model, scaler, features, X_te, y_te, y_prob, auc, shap_vals, X_te_s = load()

st.markdown("# 🚨 Sistema de Detección AML")
st.markdown("**Anti-Money Laundering** — Modelo XGBoost calibrado para identificar patrones de lavado de dinero en transacciones financieras.")
st.markdown("---")

susp_rate = df["is_suspicious"].mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC-ROC",              f"{auc:.3f}", "XGBoost")
c2.metric("Tasa actividad sospechosa", f"{susp_rate:.1%}", "Dataset sintético")
c3.metric("Señales monitoreadas", "7",          "Variables de riesgo")
c4.metric("Transacciones",        f"{len(df):,}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Transaction Scoring
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Scoring AML de Transacción</div>', unsafe_allow_html=True)

with st.form("aml_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        amount    = st.number_input("Monto de transacción (USD)", 10.0, 500000.0, 4500.0, step=100.0)
        freq      = st.slider("Transacciones por semana", 1, 20, 4)
        is_intl   = st.selectbox("¿Transacción internacional?", [0,1], format_func=lambda x:"Sí" if x else "No")
    with c2:
        prev_susp = st.selectbox("¿Historial sospechoso previo?", [0,1], format_func=lambda x:"Sí" if x else "No")
        cust_age  = st.slider("Edad del cliente", 18, 80, 45)
        acc_age   = st.slider("Antigüedad de cuenta (meses)", 1, 240, 24)
    with c3:
        structuring = st.selectbox(
            "¿Monto cercano al umbral de reporte ($10,000)?",
            [0,1], format_func=lambda x:"Sí (structuring)" if x else "No"
        )
        st.markdown("")
        st.markdown('<div class="danger-box">⚠️ El structuring (fragmentar transacciones cerca de $10K) es una señal regulatoria de alto riesgo AML.</div>', unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Evaluar riesgo AML", use_container_width=True, type="primary")

if submitted:
    row = pd.DataFrame([{
        "transaction_amount":  amount,   "frequency_per_week": freq,
        "is_international":    is_intl,  "prev_suspicious":    prev_susp,
        "customer_age":        cust_age, "account_age_months": acc_age,
        "structuring_flag":    structuring,
    }])
    X_in = scaler.transform(row[features])
    prob = model.predict_proba(X_in)[0, 1]
    level, color = aml_level(prob)

    # AML score 0-100
    aml_score = int(prob * 100)

    r1, r2, r3 = st.columns([1, 1, 2])
    with r1:
        st.markdown(f"""
        <div style='background:{color}18;border:3px solid {color};border-radius:16px;
                    padding:28px;text-align:center'>
            <div style='font-size:1rem;color:#64748b;font-weight:600'>NIVEL DE RIESGO AML</div>
            <div style='font-size:2.5rem;font-weight:900;color:{color};margin:8px 0'>{level.upper()}</div>
            <div style='font-size:1.8rem;font-weight:700;color:{color}'>{aml_score}/100</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        # Risk level bars
        levels_data = [("Bajo", 0.20, "#16a34a"), ("Medio", 0.45, "#d97706"),
                       ("Alto", 0.70, "#ea580c"),  ("Crítico", 1.0, "#dc2626")]
        bar_vals    = []
        bar_colors  = []
        for lname, lmax, lcol in levels_data:
            bar_vals.append(min(prob / lmax if lmax > 0 else 0, 1))
            bar_colors.append(lcol)

        actions = {
            "Bajo":    "✅ Sin acción requerida. Monitoreo rutinario.",
            "Medio":   "⚠️ Revisión manual por oficial de cumplimiento.",
            "Alto":    "🔴 Escalar a Unidad de Inteligencia Financiera (UIF).",
            "Crítico": "🚨 Bloquear transacción y reportar al regulador.",
        }
        st.markdown(f"**Acción recomendada:**")
        action_color = color
        st.markdown(f"""
        <div style='background:{action_color}18;border:1px solid {action_color};
                    border-radius:10px;padding:14px;font-size:13px;color:{action_color};font-weight:600'>
            {actions[level]}
        </div>""", unsafe_allow_html=True)
        st.metric("Probabilidad de actividad sospechosa", f"{prob:.1%}")
    with r3:
        import shap as _shap
        from utils.helpers import extract_shap_local
        exp    = _shap.TreeExplainer(model)
        sv_raw = exp.shap_values(X_in)
        sv_f   = extract_shap_local(sv_raw)
        contrib = [(features[i], float(sv_f[i])) for i in range(len(features))]
        contrib.sort(key=lambda x: abs(x[1]), reverse=True)
        labels  = [c[0].replace("_"," ") for c in contrib]
        vals    = [c[1] for c in contrib]
        bcolors = [COLORS["danger"] if v > 0 else COLORS["success"] for v in vals]
        fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=bcolors))
        fig.update_layout(title="Señales de riesgo AML (SHAP local)",
                          xaxis_title="Contribución (rojo=aumenta riesgo)",
                          height=300, paper_bgcolor="white", plot_bgcolor="#f8fafc",
                          font=dict(size=11), margin=dict(t=40,b=20,l=160,r=20))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: AML Rules Engine + Model
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Motor de Reglas AML + Modelo</div>', unsafe_allow_html=True)
st.markdown("Un sistema AML robusto combina reglas deterministas con el modelo de ML.")

with st.expander("📋 Ver reglas activas del motor"):
    rules_df = pd.DataFrame({
        "Regla": [
            "R01: Structuring",
            "R02: Gran monto internacional",
            "R03: Alta frecuencia + historial sospechoso",
            "R04: Cuenta nueva + gran monto",
            "R05: Cliente joven + internacional"
        ],
        "Condición": [
            "structuring_flag == 1",
            "is_international == 1 AND amount > $50,000",
            "frequency > 10/semana AND prev_suspicious == 1",
            "account_age < 3 meses AND amount > $20,000",
            "customer_age < 25 AND is_international == 1"
        ],
        "Score adicional": ["+30","+25","+35","+20","+15"],
        "Fuente regulatoria": ["FATF R.1","FATF R.6","FATF R.10","CNBV","CNBV"]
    })
    st.dataframe(rules_df, use_container_width=True, hide_index=True)

tab1, tab2, tab3 = st.tabs(["📈 Curva ROC", "🔢 Matriz de Confusión", "🧠 SHAP Global"])
with tab1:
    threshold_aml = st.slider("Umbral AML", 0.10, 0.80, 0.45, step=0.01)
    st.plotly_chart(plot_roc_curve(y_te, y_prob, "ROC — Sistema AML"), use_container_width=True)
with tab2:
    y_pred_t = (y_prob >= threshold_aml).astype(int)
    st.plotly_chart(plot_confusion_matrix(y_te, y_pred_t, ["Normal","Sospechoso"]), use_container_width=True)
with tab3:
    st.plotly_chart(plot_shap_bar(shap_vals, features, title="SHAP Global — AML"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Suspicious Activity Dashboard
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Dashboard de Actividad Sospechosa</div>', unsafe_allow_html=True)

sample = df.sample(min(600, len(df)), random_state=42).copy()
sample_X = scaler.transform(sample[features])
sample["risk_prob"] = model.predict_proba(sample_X)[:, 1]
sample["risk_level"] = sample["risk_prob"].apply(lambda p: aml_level(p)[0])

ca, cb, cc = st.columns(3)
with ca:
    level_counts = sample["risk_level"].value_counts().reindex(["Bajo","Medio","Alto","Crítico"], fill_value=0)
    fig_pie = go.Figure(go.Pie(
        labels=level_counts.index, values=level_counts.values,
        marker_colors=["#16a34a","#d97706","#ea580c","#dc2626"],
        hole=0.45, textinfo="label+percent"
    ))
    fig_pie.update_layout(title="Distribución de Riesgo AML", height=320, paper_bgcolor="white")
    st.plotly_chart(fig_pie, use_container_width=True)
with cb:
    fig_amt = go.Figure()
    for label, color in [("Normal","#16a34a"),("Sospechoso","#dc2626")]:
        mask = sample["is_suspicious"] == (1 if label=="Sospechoso" else 0)
        fig_amt.add_trace(go.Box(
            y=sample.loc[mask,"transaction_amount"],
            name=label, marker_color=color, boxpoints="outliers"
        ))
    fig_amt.update_layout(title="Montos: Normal vs Sospechoso",
                          yaxis_title="Monto (USD)",
                          paper_bgcolor="white", plot_bgcolor="#f8fafc",
                          margin=dict(t=40,b=40,l=60,r=20))
    st.plotly_chart(fig_amt, use_container_width=True)
with cc:
    fig_freq = go.Figure()
    for label, color, flag in [("Normal","#16a34a",0),("Sospechoso","#dc2626",1)]:
        mask = sample["is_suspicious"] == flag
        fig_freq.add_trace(go.Histogram(
            x=sample.loc[mask,"frequency_per_week"],
            name=label, opacity=0.7, marker_color=color, nbinsx=15
        ))
    fig_freq.update_layout(barmode="overlay", title="Frecuencia semanal por clase",
                           xaxis_title="Transacciones/semana",
                           paper_bgcolor="white", plot_bgcolor="#f8fafc",
                           margin=dict(t=40,b=40,l=50,r=20))
    st.plotly_chart(fig_freq, use_container_width=True)

st.markdown("**Top 10 transacciones de mayor riesgo AML:**")
top_risk = sample.nlargest(10, "risk_prob")[[
    "transaction_amount","frequency_per_week","is_international",
    "prev_suspicious","structuring_flag","account_age_months","risk_prob","risk_level"
]].rename(columns={
    "transaction_amount":"Monto","frequency_per_week":"Freq/sem",
    "is_international":"Intl","prev_suspicious":"Prev. Susp.",
    "structuring_flag":"Structuring","account_age_months":"Ant. cuenta (m)",
    "risk_prob":"P(Sospechoso)","risk_level":"Nivel"
})
st.dataframe(
    top_risk.style
    .format({"Monto":"${:,.2f}","P(Sospechoso)":"{:.1%}"})
    .background_gradient(subset=["P(Sospechoso)"], cmap="Reds"),
    use_container_width=True, height=320
)
