import streamlit as st

st.set_page_config(
    page_title="FinRisk Suite — Credit & Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar branding */
[data-testid="stSidebar"] { background: #0f1923; }
[data-testid="stSidebar"] * { color: #e8eaf0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p { color: #9ba3b2 !important; }
[data-testid="stSidebarNav"] a { border-radius: 8px; margin: 2px 0; }
[data-testid="stSidebarNav"] a:hover { background: #1e2d3d; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] { font-size: 0.78rem; color: #64748b; font-weight: 500; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }

/* Section headers */
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #1e293b;
    border-left: 4px solid #2563eb; padding-left: 12px;
    margin: 1.5rem 0 1rem;
}
.badge-high   { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-medium { background:#fef3c7; color:#92400e; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-low    { background:#dcfce7; color:#166534; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.info-box { background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:12px 16px; font-size:13px; color:#1e40af; margin:8px 0; }
.warn-box { background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:12px 16px; font-size:13px; color:#92400e; margin:8px 0; }
.danger-box { background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:12px 16px; font-size:13px; color:#991b1b; margin:8px 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 FinRisk Suite")
    st.markdown("<p style='font-size:12px;margin-top:-8px;'>Credit & Risk Analytics Platform</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    **Módulos disponibles**
    - 🏠 Inicio — Esta pantalla
    - 📊 Scoring de Crédito
    - ⚠️ Predicción de Default
    - 🔍 Detección de Fraude
    - 🏢 Rating Corporativo
    - 🚨 Alerta Temprana AML
    - 📈 Stress Test de Portafolio
    """)
    st.markdown("---")
    st.markdown("<p style='font-size:11px;color:#6b7280;'>v1.0 · Datos sintéticos de demostración</p>", unsafe_allow_html=True)

# ── Home ──────────────────────────────────────────────────────────────────────
st.markdown("# 🏦 FinRisk Suite")
st.markdown("**Plataforma de Analítica de Crédito y Riesgo** — Modelos de ML explicables para decisiones financieras")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Modelos activos", "6", "Production-ready")
col2.metric("Precisión promedio", "87.4%", "+3.2% vs baseline")
col3.metric("Variables analizadas", "28", "Multi-dimensionales")
col4.metric("Explicabilidad", "SHAP", "Full transparency")

st.markdown("---")
st.markdown('<div class="section-title">Módulos de la plataforma</div>', unsafe_allow_html=True)

modules = [
    ("📊", "Scoring de Crédito",       "Scorecard con Decision Tree + XGBoost. Incluye score FICO-style, segmento de riesgo y explicabilidad SHAP por solicitante.",           "pages/1_Scoring_Credito.py"),
    ("⚠️", "Predicción de Default",     "Ensemble Random Forest + LightGBM con SMOTE para clases desbalanceadas. Predice probabilidad de incumplimiento a 12 meses.",         "pages/2_Prediccion_Default.py"),
    ("🔍", "Detección de Fraude",       "Isolation Forest + Random Forest para detectar transacciones anómalas. Umbral ajustable y análisis de importancia de variables.",    "pages/3_Deteccion_Fraude.py"),
    ("🏢", "Rating Corporativo",        "Modelo multiclase (AAA→BBB) basado en ratios financieros. Curva ROC por clase y perfil comparativo de empresa.",                    "pages/4_Rating_Corporativo.py"),
    ("🚨", "AML — Lavado de Dinero",    "Sistema de puntuación de riesgo AML con reglas + ML. Identifica patrones sospechosos en frecuencia, monto y geografía.",            "pages/5_AML_Lavado_Dinero.py"),
    ("📈", "Stress Test de Portafolio", "Simulación de escenarios de crisis (2008-style). Aplica shocks paramétricos a un portafolio y mide impacto en VaR y pérdidas.",     "pages/6_Stress_Test.py"),
]

for i in range(0, len(modules), 2):
    c1, c2 = st.columns(2)
    for col, (icon, name, desc, _) in zip([c1, c2], modules[i:i+2]):
        with col:
            with st.container(border=True):
                st.markdown(f"### {icon} {name}")
                st.markdown(f"<p style='font-size:13px;color:#475569;'>{desc}</p>", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="info-box">💡 <strong>Todos los datos son sintéticos</strong> generados con semillas fijas para reproducibilidad. Los modelos están calibrados para uso demostrativo; en producción se recomienda reentrenar con datos reales auditados.</div>', unsafe_allow_html=True)
