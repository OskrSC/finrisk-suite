# 🏦 FinRisk Suite — Credit & Risk Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

Plataforma de analítica de crédito y riesgo con modelos ML explicables (SHAP).  
**Stack:** Streamlit · XGBoost · LightGBM · scikit-learn · SHAP · Plotly

---

## 📦 Módulos

| # | Módulo | Modelo | Técnica |
|---|--------|--------|---------|
| 1 | Scoring de Crédito | XGBoost | SHAP local + global |
| 2 | Predicción de Default | LightGBM | SMOTE · PD/LGD/EAD |
| 3 | Detección de Fraude | Random Forest | SMOTE · umbral ajustable |
| 4 | Rating Corporativo | XGBoost multiclase | Radar · comparativa |
| 5 | AML — Lavado de Dinero | XGBoost | Reglas FATF + ML |
| 6 | Stress Test Portafolio | Monte Carlo | VaR 95/99 · CVaR |

---

## 🚀 Ejecutar localmente

```bash
git clone https://github.com/TU_USUARIO/finrisk-suite.git
cd finrisk-suite
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Despliegue en Streamlit Community Cloud

1. Fork / push este repo a tu cuenta de GitHub
2. Ir a [share.streamlit.io](https://share.streamlit.io)
3. **New app** → seleccionar repo → `app.py` → Deploy

No se requiere configuración adicional. Las dependencias se instalan automáticamente desde `requirements.txt`.

---

> ⚠️ Todos los datos son **sintéticos** (semillas fijas). Para producción, reemplazar los generadores con conexiones a bases de datos reales.
