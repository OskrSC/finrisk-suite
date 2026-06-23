"""
utils/helpers.py
Shared data generators, model builders and plotting helpers for FinRisk Suite.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings("ignore")

def extract_shap_local(shap_values_raw, sample_idx: int = 0, class_idx: int = 1) -> np.ndarray:
    """
    Extrae un vector 1D de SHAP values para un solo sample y la clase positiva.

    Maneja las tres formas que devuelven distintas versiones de shap + sklearn:
      - list  [class0_arr, class1_arr]   → versiones antiguas de shap/sklearn
      - ndarray (n, features)            → XGBoost, LightGBM (binario)
      - ndarray (n, features, classes)   → RandomForest sklearn moderno
    """
    import numpy as _np
    sv = shap_values_raw
    if isinstance(sv, list):
        arr = _np.array(sv[class_idx])
    else:
        arr = _np.array(sv)

    if arr.ndim == 3:
        return arr[sample_idx, :, class_idx]   # (n, features, classes)
    elif arr.ndim == 2:
        return arr[sample_idx]                  # (n, features)
    else:

        return arr.reshape(-1)                  # ya es 1D


def hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    """
    Convierte un color #rrggbb a rgba(r,g,b,alpha) compatible con Plotly.

    Plotly rechaza hex de 8 dígitos (#rrggbbaa). Esta función produce
    el formato rgba() que Plotly acepta en fillcolor, marker_color, etc.

    Args:
        hex_color: string '#rrggbb'
        alpha:     opacidad 0-1 (default 0.13 ≈ hex "22")
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

SEED = 42
np.random.seed(SEED)

# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#2563eb",
    "danger":    "#dc2626",
    "warning":   "#d97706",
    "success":   "#16a34a",
    "neutral":   "#64748b",
    "light_bg":  "#f8fafc",
    "grid":      "rgba(0,0,0,0.05)",
}

RISK_COLORS = {
    "Muy Alto": "#dc2626",
    "Alto":     "#ea580c",
    "Medio":    "#d97706",
    "Bajo":     "#16a34a",
    "Muy Bajo": "#059669",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def gen_credit_data(n: int = 3000) -> pd.DataFrame:
    """Synthetic credit application dataset with realistic correlations."""
    rng = np.random.default_rng(SEED)
    credit_score   = rng.integers(300, 850, n)
    annual_income  = rng.normal(55000, 18000, n).clip(12000, 250000)
    loan_amount    = rng.normal(22000, 8000, n).clip(1000, 80000)
    dti            = rng.uniform(0.05, 0.60, n)
    prev_default   = rng.choice([0, 1], n, p=[0.82, 0.18])
    emp_years      = rng.integers(0, 30, n)
    num_accounts   = rng.integers(1, 15, n)
    late_payments  = rng.integers(0, 10, n)
    employment     = rng.choice(["Employed", "Self-employed", "Unemployed"], n, p=[0.70, 0.20, 0.10])

    # Realistic approval probability driven by features
    score_norm  = (credit_score - 300) / 550
    dti_norm    = 1 - dti
    income_norm = (annual_income - 12000) / 238000
    prob_approve = (
        0.40 * score_norm +
        0.20 * dti_norm +
        0.15 * income_norm +
        0.10 * (1 - prev_default) +
        0.10 * (emp_years / 30) +
        0.05 * (1 - late_payments / 10)
    )
    prob_approve += rng.normal(0, 0.05, n)
    approved = (prob_approve > 0.48).astype(int)

    emp_map = {"Employed": 2, "Self-employed": 1, "Unemployed": 0}
    return pd.DataFrame({
        "credit_score":      credit_score,
        "annual_income":     annual_income.astype(int),
        "loan_amount":       loan_amount.astype(int),
        "dti_ratio":         dti.round(3),
        "prev_default":      prev_default,
        "emp_years":         emp_years,
        "num_accounts":      num_accounts,
        "late_payments":     late_payments,
        "employment_status": employment,
        "employment_enc":    [emp_map[e] for e in employment],
        "approved":          approved,
    })


def gen_default_data(n: int = 4000) -> pd.DataFrame:
    """Loan portfolio dataset for 12-month PD (probability of default)."""
    rng = np.random.default_rng(SEED + 1)
    credit_score  = rng.integers(300, 850, n)
    income        = rng.normal(58000, 20000, n).clip(10000, 300000)
    loan_amount   = rng.normal(25000, 9000, n).clip(500, 100000)
    dti           = rng.uniform(0.05, 0.65, n)
    prev_default  = rng.choice([0, 1], n, p=[0.80, 0.20])
    loan_term     = rng.choice([12, 24, 36, 48, 60], n)
    interest_rate = (rng.uniform(5, 28, n)).round(2)
    months_open   = rng.integers(1, 120, n)
    revolving_util = rng.uniform(0, 1, n)

    # ~18% base default rate with feature correlations
    base_pd = (
        0.35 * (1 - (credit_score - 300) / 550) +
        0.20 * dti +
        0.15 * prev_default +
        0.10 * revolving_util +
        0.08 * (interest_rate / 28) +
        0.05 * (1 - (income - 10000) / 290000) +
        rng.normal(0, 0.06, n)
    ).clip(0, 1)
    default = (base_pd > 0.52).astype(int)

    return pd.DataFrame({
        "credit_score":     credit_score,
        "annual_income":    income.astype(int),
        "loan_amount":      loan_amount.astype(int),
        "dti_ratio":        dti.round(3),
        "prev_default":     prev_default,
        "loan_term":        loan_term,
        "interest_rate":    interest_rate,
        "months_open":      months_open,
        "revolving_util":   revolving_util.round(3),
        "default":          default,
    })


def gen_fraud_data(n: int = 5000, fraud_rate: float = 0.04) -> pd.DataFrame:
    """Financial transaction dataset with ~4% fraud rate."""
    rng = np.random.default_rng(SEED + 2)
    n_fraud  = int(n * fraud_rate)
    n_legit  = n - n_fraud

    # Probability arrays — normalized to exactly 1.0
    p_legit_hour = np.array([0.01,0.01,0.01,0.01,0.01,0.02,0.04,0.06,
                              0.08,0.08,0.07,0.07,0.07,0.06,0.06,0.06,
                              0.06,0.06,0.05,0.04,0.03,0.02,0.02,0.01], dtype=float)
    p_legit_hour /= p_legit_hour.sum()

    p_fraud_hour = np.array([0.08,0.09,0.09,0.09,0.08,0.07,0.04,0.03,
                              0.03,0.03,0.03,0.03,0.04,0.04,0.04,0.04,
                              0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.05], dtype=float)
    p_fraud_hour /= p_fraud_hour.sum()

    # Legitimate transactions
    legit = pd.DataFrame({
        "amount":       rng.exponential(120, n_legit).clip(1, 2000),
        "hour":         rng.choice(range(24), n_legit, p=p_legit_hour),
        "distance_km":  rng.exponential(15, n_legit).clip(0, 200),
        "freq_day":     rng.integers(1, 6, n_legit),
        "merchant_cat": rng.choice([0,1,2,3,4], n_legit),
        "is_online":    rng.choice([0,1], n_legit, p=[0.6, 0.4]),
        "is_fraud":     0,
    })
    # Fraud transactions — different distribution
    fraud = pd.DataFrame({
        "amount":       rng.exponential(600, n_fraud).clip(50, 9999),
        "hour":         rng.choice(range(24), n_fraud, p=p_fraud_hour),
        "distance_km":  rng.exponential(200, n_fraud).clip(0, 5000),
        "freq_day":     rng.integers(5, 25, n_fraud),
        "merchant_cat": rng.choice([0,1,2,3,4], n_fraud),
        "is_online":    rng.choice([0,1], n_fraud, p=[0.25, 0.75]),
        "is_fraud":     1,
    })
    df = pd.concat([legit, fraud]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["amount"] = df["amount"].round(2)
    return df


def gen_corporate_data(n: int = 2000) -> pd.DataFrame:
    """Corporate financial ratios for credit rating (AAA/AA/A/BBB)."""
    rng = np.random.default_rng(SEED + 3)
    ratings = rng.choice([1,2,3,4], n, p=[0.15, 0.25, 0.35, 0.25])

    rating_means = {1: (0.4, 0.12, 2.0, 5.0),
                    2: (0.9, 0.08, 1.5, 3.5),
                    3: (1.5, 0.05, 1.2, 2.5),
                    4: (2.5, 0.02, 0.9, 1.5)}
    dte, roa, cr, ic = [], [], [], []
    for r in ratings:
        m = rating_means[r]
        dte.append(rng.normal(m[0], 0.2))
        roa.append(rng.normal(m[1], 0.02))
        cr.append(rng.normal(m[2], 0.2))
        ic.append(rng.normal(m[3], 0.5))

    return pd.DataFrame({
        "debt_to_equity":    np.array(dte).round(3).clip(0),
        "return_on_assets":  np.array(roa).round(4),
        "current_ratio":     np.array(cr).round(3).clip(0),
        "interest_coverage": np.array(ic).round(3),
        "revenue_growth":    rng.normal(0.08, 0.06, n).round(4),
        "net_margin":        rng.normal(0.12, 0.05, n).round(4),
        "credit_rating":     ratings,
        "rating_label":      [["AAA","AA","A","BBB"][r-1] for r in ratings],
    })


def gen_aml_data(n: int = 3000) -> pd.DataFrame:
    """Transaction dataset for AML risk scoring."""
    rng = np.random.default_rng(SEED + 4)
    susp_rate = 0.08

    amount        = rng.exponential(3000, n).clip(10, 500000)
    freq          = rng.integers(1, 20, n)
    is_intl       = rng.choice([0,1], n, p=[0.65, 0.35])
    prev_susp     = rng.choice([0,1], n, p=[0.90, 0.10])
    cust_age      = rng.integers(18, 80, n)
    account_age_m = rng.integers(1, 240, n)
    structured    = (amount % 10000 < 500).astype(int)  # structuring signal

    base = (
        0.25 * (amount / 500000) +
        0.20 * (freq / 20) +
        0.15 * is_intl +
        0.20 * prev_susp +
        0.10 * structured +
        0.05 * (1 - account_age_m / 240) +
        rng.normal(0, 0.06, n)
    ).clip(0, 1)
    threshold = float(np.percentile(base, 92))
    suspicious = (base >= threshold).astype(int)

    return pd.DataFrame({
        "transaction_amount":  amount.round(2),
        "frequency_per_week":  freq,
        "is_international":    is_intl,
        "prev_suspicious":     prev_susp,
        "customer_age":        cust_age,
        "account_age_months":  account_age_m,
        "structuring_flag":    structured,
        "is_suspicious":       suspicious,
    })


def gen_portfolio_data(n_loans: int = 500) -> pd.DataFrame:
    """Loan portfolio for stress testing."""
    rng = np.random.default_rng(SEED + 5)
    asset_class = rng.choice(
        ["Hipotecario","Consumo","Corporativo","PYME","Auto"],
        n_loans, p=[0.35, 0.25, 0.20, 0.12, 0.08]
    )
    balance = rng.exponential(80000, n_loans).clip(5000, 2000000)
    pd_base = rng.beta(2, 10, n_loans)
    lgd     = rng.beta(3, 7, n_loans)
    maturity = rng.integers(1, 25, n_loans)

    return pd.DataFrame({
        "asset_class":  asset_class,
        "balance":      balance.round(2),
        "pd_base":      pd_base.round(4),
        "lgd":          lgd.round(4),
        "maturity_yrs": maturity,
        "ead":          (balance * rng.uniform(0.8, 1.0, n_loans)).round(2),
    })


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_credit_model(df: pd.DataFrame):
    features = ["credit_score","annual_income","loan_amount","dti_ratio",
                "prev_default","emp_years","num_accounts","late_payments","employment_enc"]
    X = df[features]
    y = df["approved"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED, n_jobs=-1
    )
    model.fit(X_tr_s, y_tr)

    y_prob = model.predict_proba(X_te_s)[:,1]
    y_pred = model.predict(X_te_s)
    auc    = roc_auc_score(y_te, y_prob)

    explainer    = shap.TreeExplainer(model)
    shap_values  = explainer.shap_values(X_te_s)

    return model, scaler, features, X_te, y_te, y_prob, auc, shap_values, X_te_s


def build_default_model(df: pd.DataFrame):
    features = ["credit_score","annual_income","loan_amount","dti_ratio",
                "prev_default","loan_term","interest_rate","months_open","revolving_util"]
    X = df[features]
    y = df["default"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # SMOTE to handle class imbalance
    smote   = SMOTE(random_state=SEED)
    X_res, y_res = smote.fit_resample(X_tr_s, y_tr)

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.04,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbose=-1
    )
    model.fit(X_res, y_res)

    y_prob = model.predict_proba(X_te_s)[:,1]
    y_pred = model.predict(X_te_s)
    auc    = roc_auc_score(y_te, y_prob)

    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(X_te_s)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    return model, scaler, features, X_te, y_te, y_prob, auc, shap_vals, X_te_s


def build_fraud_model(df: pd.DataFrame):
    features = ["amount","hour","distance_km","freq_day","merchant_cat","is_online"]
    X = df[features]
    y = df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    smote   = SMOTE(random_state=SEED)
    X_res, y_res = smote.fit_resample(X_tr_s, y_tr)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    model.fit(X_res, y_res)

    y_prob = model.predict_proba(X_te_s)[:,1]
    auc    = roc_auc_score(y_te, y_prob)
    ap     = average_precision_score(y_te, y_prob)

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_te_s)
    # RandomForestClassifier con sklearn>=1.4 devuelve shape (n, features, classes)
    # Versiones anteriores devolvían lista [class0_array, class1_array].
    # Normalizamos siempre a (n, features) para la clase positiva (fraude = 1).
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]          # lista → tomar clase 1
    elif shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]    # tensor 3D → slice clase 1

    return model, scaler, features, X_te, y_te, y_prob, auc, ap, shap_vals, X_te_s


def build_rating_model(df: pd.DataFrame):
    features = ["debt_to_equity","return_on_assets","current_ratio",
                "interest_coverage","revenue_growth","net_margin"]
    X = df[features]
    y = df["credit_rating"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="mlogloss",
        random_state=SEED, n_jobs=-1
    )
    model.fit(X_tr_s, y_tr - 1)  # 0-indexed

    y_pred = model.predict(X_te_s) + 1
    acc    = (y_pred == y_te).mean()

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_te_s)

    return model, scaler, features, X_te, y_te, y_pred, acc, shap_vals, X_te_s


def build_aml_model(df: pd.DataFrame):
    features = ["transaction_amount","frequency_per_week","is_international",
                "prev_suspicious","customer_age","account_age_months","structuring_flag"]
    X = df[features]
    y = df["is_suspicious"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=(y_tr==0).sum()/(y_tr==1).sum(),
        eval_metric="logloss",
        random_state=SEED, n_jobs=-1
    )
    model.fit(X_tr_s, y_tr)

    y_prob = model.predict_proba(X_te_s)[:,1]
    auc    = roc_auc_score(y_te, y_prob)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_te_s)

    return model, scaler, features, X_te, y_te, y_prob, auc, shap_vals, X_te_s


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _base_layout(**kwargs) -> dict:
    return dict(
        paper_bgcolor="white", plot_bgcolor=COLORS["light_bg"],
        font=dict(family="Inter, sans-serif", size=12, color="#1e293b"),
        margin=dict(t=40, b=40, l=60, r=20),
        xaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
        yaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
        **kwargs
    )


def plot_roc_curve(y_true, y_prob, title="Curva ROC") -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"Modelo (AUC={auc:.3f})",
        line=dict(color=COLORS["primary"], width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode="lines", name="Aleatorio (AUC=0.5)",
        line=dict(color=COLORS["neutral"], width=1.5, dash="dash")
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Tasa de Falsos Positivos",
        yaxis_title="Tasa de Verdaderos Positivos",
        legend=dict(x=0.55, y=0.15),
        **_base_layout()
    )
    return fig


def plot_pr_curve(y_true, y_prob, title="Curva Precisión-Recall") -> go.Figure:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rec, y=prec, mode="lines", name=f"Modelo (AP={ap:.3f})",
        line=dict(color=COLORS["success"], width=2.5)
    ))
    fig.add_hline(y=baseline, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text=f"Baseline={baseline:.2f}")
    fig.update_layout(
        title=title,
        xaxis_title="Recall", yaxis_title="Precisión",
        legend=dict(x=0.55, y=0.85),
        **_base_layout()
    )
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred)
    labels = labels or [str(i) for i in sorted(set(y_true))]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0,"#eff6ff"],[1,"#1d4ed8"]],
        text=cm, texttemplate="%{text}",
        hovertemplate="Actual: %{y}<br>Predicho: %{x}<br>Conteo: %{z}<extra></extra>"
    ))
    fig.update_layout(
        title="Matriz de Confusión",
        xaxis_title="Predicho", yaxis_title="Actual",
        **_base_layout()
    )
    return fig


def plot_shap_bar(shap_values, feature_names, max_features: int = 10, title="Importancia SHAP") -> go.Figure:
    sv = np.array(shap_values)
    # Normalizar a 2D (n_samples, n_features) independientemente del modelo
    if sv.ndim == 3:
        sv = sv[:, :, 1]          # tensor (n, features, classes) → clase positiva
    elif sv.ndim == 1:
        sv = sv.reshape(1, -1)    # vector plano de un solo sample
    mean_abs = np.abs(sv).mean(axis=0)
    idx = np.argsort(mean_abs)[-max_features:]
    fig = go.Figure(go.Bar(
        x=mean_abs[idx], y=[feature_names[i] for i in idx],
        orientation="h", marker_color=COLORS["primary"],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"
    ))
    fig.update_layout(title=title, xaxis_title="|SHAP value|", **_base_layout())
    return fig


def plot_score_gauge(score: float, label: str = "Score") -> go.Figure:
    color = (COLORS["danger"] if score < 40 else
             COLORS["warning"] if score < 60 else
             COLORS["success"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": label, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0,  40], "color": "#fef2f2"},
                {"range": [40, 60], "color": "#fffbeb"},
                {"range": [60, 80], "color": "#f0fdf4"},
                {"range": [80,100], "color": "#dcfce7"},
            ],
            "threshold": {"line": {"color": "#1e293b","width": 3}, "thickness": 0.8, "value": score}
        }
    ))
    fig.update_layout(height=220, margin=dict(t=30, b=0, l=20, r=20), paper_bgcolor="white")
    return fig


def risk_label_from_prob(prob: float, thresholds=(0.25, 0.45, 0.65, 0.80)) -> str:
    if prob < thresholds[0]: return "Muy Bajo"
    if prob < thresholds[1]: return "Bajo"
    if prob < thresholds[2]: return "Medio"
    if prob < thresholds[3]: return "Alto"
    return "Muy Alto"


def prob_to_score(prob: float, lo: float = 300, hi: float = 850) -> int:
    """Convert a raw probability to a FICO-style integer score (inverse)."""
    return int(hi - prob * (hi - lo))
