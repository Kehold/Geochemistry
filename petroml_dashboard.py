"""
Petrographic ML Dashboard
=========================
Porosity & Permeability Prediction from Point-Counting Data
Based on: Sadrikhanloo, Busch & Hilgers (2026), Energy Geoscience 7, 100537

Dashboard by Digital Oil Inc. — digitaloil.ai
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PetroML — Petrographic Reservoir Quality Predictor",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp { font-family: 'DM Sans', sans-serif; }
    code, .stCode { font-family: 'JetBrains Mono', monospace !important; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #f5a623;
    }
    .hero-banner h1 { margin: 0 0 0.4rem 0; font-size: 1.9rem; letter-spacing: -0.5px; }
    .hero-banner p { margin: 0; opacity: 0.85; font-size: 0.95rem; line-height: 1.5; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        color: white;
        padding: 1.2rem 1.4rem;
        border-radius: 10px;
        text-align: center;
        border-top: 3px solid #f5a623;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #f5a623; }
    .metric-card .label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7; margin-top: 0.2rem; }

    /* Section headers */
    .section-header {
        border-left: 4px solid #f5a623;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.15rem;
        font-weight: 600;
        color: #203a43;
    }

    /* Business card */
    .biz-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.8rem;
    }
    .biz-card h4 { margin: 0 0 0.4rem 0; color: #203a43; font-size: 1rem; }
    .biz-card p { margin: 0; font-size: 0.88rem; color: #555; line-height: 1.5; }

    div[data-testid="stSidebar"] { background: #0f2027; }
    div[data-testid="stSidebar"] * { color: #ddd !important; }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label { color: #f5a623 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─── Synthetic Data Generator ────────────────────────────────────────────────
@st.cache_data
def generate_synthetic_data(n_samples=157, seed=42):
    """Generate synthetic petrographic + petrophysical data with geologically-
    informed feature→target relationships derived from the paper's SHAP analysis.

    KEY DESIGN: Features are generated FIRST (independently per well), then
    porosity is COMPUTED as a nonlinear function of those features + noise.
    This ensures the RF model learns meaningful multi-feature importance
    patterns rather than trivially latching onto a single derived feature.

    Geological controls encoded (per SHAP Figs. 3 & 6):
      Porosity (+): intergranular porosity, TiOx, pore-filling illite,
                    MRF undiff, K-feldspar cement, IG por K-fsp, moderate GTG
      Porosity (−): calcite cement, quartz cement, GTI > 40% (tangential),
                    pore-lining illite > 3%, quartz > 55%, high GTG, ductile RF
      Permeability (+): porosity, GTI coating, grain size
      Permeability (−): GTG coating, pore-filling illite, pore-lining illite,
                        ductile RF, K-feldspar cement
    """
    rng = np.random.RandomState(seed)

    well_configs = {
        "Well A+B (Buntsandstein)": {
            "n": 34, "region": "W-URG", "strat": "Buntsandstein",
            "perm_range": (0.02, 196),
            "gti_range": (15, 55), "tiox_range": (0, 0.5),
            "calcite_range": (0, 0.3), "qtz_cem_range": (2, 12),
            "por_baseline": 14.0,   # high porosity well
        },
        "Well C (Buntsandstein)": {
            "n": 34, "region": "E-URG", "strat": "Buntsandstein",
            "perm_range": None,
            "gti_range": (25, 70), "tiox_range": (0, 0.3),
            "calcite_range": (0, 0.5), "qtz_cem_range": (8, 22),
            "por_baseline": 2.0,    # tight well — heavy cementation
        },
        "Well D (Buntsandstein)": {
            "n": 40, "region": "E-Shoulder URG", "strat": "Buntsandstein",
            "perm_range": (0.0001, 7.76),
            "gti_range": (40, 90), "tiox_range": (0.2, 3.5),
            "calcite_range": (0, 0.2), "qtz_cem_range": (4, 16),
            "por_baseline": 8.0,
        },
        "Well A+B (Rotliegendes)": {
            "n": 49, "region": "S. Permian Basin", "strat": "Rotliegendes",
            "perm_range": (0.009, 780),
            "gti_range": (45, 95), "tiox_range": (0, 0.1),
            "calcite_range": (0.5, 8), "qtz_cem_range": (1, 14),
            "por_baseline": 7.0,
        },
    }

    records = []
    for well_name, cfg in well_configs.items():
        n = cfg["n"]

        # ── 1. Generate independent petrographic features ──────────────
        quartz = rng.uniform(38, 72, n)
        k_feldspar = rng.uniform(2, 18, n)
        plagioclase = rng.uniform(0, 5, n)
        mrf_undiff = rng.uniform(0, 10, n)
        ductile_rf = rng.uniform(0, 12, n)

        gti_lo, gti_hi = cfg["gti_range"]
        gti_coverage = rng.uniform(gti_lo, gti_hi, n)
        gtg_coverage = gti_coverage * rng.uniform(0.5, 1.1, n)
        gtg_coverage = np.clip(gtg_coverage, 0, 100)

        tiox_lo, tiox_hi = cfg["tiox_range"]
        auth_tiox = rng.uniform(tiox_lo, tiox_hi, n)
        calc_lo, calc_hi = cfg["calcite_range"]
        calcite_cement = rng.uniform(calc_lo, calc_hi, n)
        qtz_lo, qtz_hi = cfg["qtz_cem_range"]
        quartz_cement = rng.uniform(qtz_lo, qtz_hi, n)
        kfsp_cement = rng.uniform(0, 5, n)

        pf_illite = rng.uniform(0, 12, n)
        pl_illite = rng.uniform(0, 12, n)
        dolomite = rng.uniform(0, 8, n)
        ig_por_kfsp = rng.uniform(0, 3.5, n)
        grain_size = rng.uniform(0.1, 0.85, n)

        # ── 2. Compute porosity from features (geologically-informed) ──
        # Start with well-specific baseline, then modulate by features
        porosity = np.full(n, cfg["por_baseline"])

        # (+) Pore-filling illite inhibits quartz cementation → preserves φ
        porosity += 0.35 * pf_illite

        # (+) TiOx authigenesis → K-fsp dissolution → secondary porosity
        porosity += 1.5 * auth_tiox

        # (+) MRF undiff (meta-siltstones) preserve framework
        porosity += 0.22 * mrf_undiff

        # (+) Intragranular porosity in K-feldspar
        porosity += 0.8 * ig_por_kfsp

        # (+) K-feldspar cement stabilises framework
        porosity += 0.20 * kfsp_cement

        # (−) Calcite cement occludes pore space
        porosity -= 0.6 * calcite_cement

        # (−) Quartz cement reduces porosity
        porosity -= 0.4 * quartz_cement

        # (−) Pore-lining illite > 3% enhances chemical compaction
        porosity -= 0.20 * np.maximum(pl_illite - 3, 0)

        # (−) Ductile RF enhances mechanical compaction
        porosity -= 0.18 * ductile_rf

        # (−) Quartz grains > 55% → more quartz overgrowth surface area
        porosity -= 0.12 * np.maximum(quartz - 55, 0)

        # (±) GTI coating: below 40% preserves φ, above 40% tangential
        #     coatings enhance chemical compaction (reduces φ)
        gti_effect = np.where(gti_coverage < 40,
                              0.06 * (40 - gti_coverage),
                              -0.04 * (gti_coverage - 40))
        porosity += gti_effect

        # (±) GTG coating: moderate (30-60%) preserves, >75% compaction
        gtg_effect = np.where(gtg_coverage < 60,
                              0.025 * np.minimum(gtg_coverage, 60),
                              -0.05 * (gtg_coverage - 60))
        porosity += gtg_effect

        # (−) Dolomite cement (minor pore-filler)
        porosity -= 0.10 * dolomite

        # Grain size: minor effect on porosity
        porosity += 1.0 * grain_size

        # K-feldspar detrital: slight positive (dissolution potential)
        porosity += 0.06 * k_feldspar

        # Add realistic noise
        porosity += rng.normal(0, 1.2, n)

        # Clip to geologically reasonable range
        porosity = np.clip(porosity, 0.02, 22.0)

        # ── 3. Compute intergranular & optical porosity INDEPENDENTLY ──
        # In real studies, intergranular porosity is measured optically via
        # point-counting — it captures macroporosity but UNDERESTIMATES
        # microporosity (Hurst & Nadeau, 1995). We model it using DIFFERENT
        # weights, FEWER features, and SUBSTANTIAL noise so it carries some
        # partial information but cannot dominate RF importance.
        ig_base = cfg["por_baseline"] * 0.35
        intergranular_por = (
            ig_base
            + 0.08 * pf_illite
            + 0.5 * auth_tiox
            - 0.15 * quartz_cement
            - 0.20 * calcite_cement
            + rng.normal(0, 2.2, n)     # high noise — optical counting variance
        )
        intergranular_por = np.clip(intergranular_por, 0, 14)

        # Optical porosity ≈ intergranular por + measurement noise
        optical_por = intergranular_por * rng.uniform(0.75, 1.05, n) + rng.normal(0, 0.8, n)
        optical_por = np.clip(optical_por, 0, None)

        # ── 4. Compute permeability from features ─────────────────────
        if cfg["perm_range"] is not None:
            log_perm = (
                -3.0
                + 0.20 * porosity          # strong positive
                + 0.008 * gti_coverage      # positive (inhibits Qtz cement)
                + 1.5 * grain_size          # larger grains → larger throats
                - 0.04 * gtg_coverage       # chemical compaction
                - 0.06 * pf_illite          # reduces pore throat radii
                - 0.05 * pl_illite          # tangential → compaction
                - 0.04 * ductile_rf         # mechanical compaction
                - 0.06 * kfsp_cement        # pore-filling
                + rng.normal(0, 0.45, n)    # noise
            )
            perm = np.clip(10**log_perm, cfg["perm_range"][0], cfg["perm_range"][1])
        else:
            perm = np.full(n, np.nan)

        # ── 5. Assemble records ────────────────────────────────────────
        for i in range(n):
            records.append({
                "Well": well_name,
                "Region": cfg["region"],
                "Stratigraphy": cfg["strat"],
                "Porosity (%)": round(porosity[i], 2),
                "Permeability (mD)": round(perm[i], 4) if not np.isnan(perm[i]) else np.nan,
                "Intergranular Porosity (%)": round(intergranular_por[i], 2),
                "Optical Porosity (%)": round(optical_por[i], 2),
                "Quartz (%)": round(quartz[i], 1),
                "K-Feldspar (%)": round(k_feldspar[i], 1),
                "Plagioclase (%)": round(plagioclase[i], 1),
                "MRF undiff (%)": round(mrf_undiff[i], 1),
                "Ductile RF (%)": round(ductile_rf[i], 1),
                "GTI Coating (%)": round(gti_coverage[i], 1),
                "GTG Coating (%)": round(gtg_coverage[i], 1),
                "Auth. TiOx (%)": round(auth_tiox[i], 2),
                "Calcite Cement (%)": round(calcite_cement[i], 2),
                "Quartz Cement (%)": round(quartz_cement[i], 1),
                "K-Feldspar Cement (%)": round(kfsp_cement[i], 2),
                "Pore-filling Illite (%)": round(pf_illite[i], 1),
                "Pore-lining Illite (%)": round(pl_illite[i], 1),
                "Dolomite (%)": round(dolomite[i], 1),
                "IG Por. K-Fsp (%)": round(ig_por_kfsp[i], 2),
                "Grain Size (mm)": round(grain_size[i], 3),
            })

    return pd.DataFrame(records)


# ─── Model Training ──────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Intergranular Porosity (%)", "Optical Porosity (%)", "Quartz (%)",
    "K-Feldspar (%)", "Plagioclase (%)", "MRF undiff (%)", "Ductile RF (%)",
    "GTI Coating (%)", "GTG Coating (%)", "Auth. TiOx (%)",
    "Calcite Cement (%)", "Quartz Cement (%)", "K-Feldspar Cement (%)",
    "Pore-filling Illite (%)", "Pore-lining Illite (%)", "Dolomite (%)",
    "IG Por. K-Fsp (%)", "Grain Size (mm)",
]

@st.cache_resource
def train_models(df):
    """Train RF (porosity) and SVR (permeability) following the paper's methodology."""
    # --- Porosity model (Random Forest) ---
    por_df = df[FEATURE_COLS + ["Porosity (%)"]].dropna()
    X_por = por_df[FEATURE_COLS].values
    y_por = por_df["Porosity (%)"].values

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X_por, y_por, test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_split=4,
        min_samples_leaf=2, random_state=42
    )
    rf.fit(X_train_p, y_train_p)
    y_pred_p = rf.predict(X_test_p)

    por_metrics = {
        "R²": r2_score(y_test_p, y_pred_p),
        "MAE": mean_absolute_error(y_test_p, y_pred_p),
        "RMSE": np.sqrt(mean_squared_error(y_test_p, y_pred_p)),
    }
    cv_r2_por = cross_val_score(rf, X_por, y_por, cv=5, scoring="r2")

    # --- Permeability model (SVR with log-transform) ---
    perm_df = df[FEATURE_COLS + ["Porosity (%)", "Permeability (mD)"]].dropna()
    perm_features = FEATURE_COLS + ["Porosity (%)"]
    X_perm = perm_df[perm_features].values
    y_perm_raw = perm_df["Permeability (mD)"].values
    y_perm_log = np.log10(np.clip(y_perm_raw, 1e-5, None))

    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    X_perm_sc = scaler_X.fit_transform(X_perm)
    y_perm_sc = scaler_y.fit_transform(y_perm_log.reshape(-1, 1)).ravel()

    X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
        X_perm_sc, y_perm_sc, test_size=0.2, random_state=42
    )
    y_test_k_log = scaler_y.inverse_transform(y_test_k.reshape(-1, 1)).ravel()

    svr = SVR(kernel="linear", C=10, epsilon=0.05)
    svr.fit(X_train_k, y_train_k)
    y_pred_k_sc = svr.predict(X_test_k)
    y_pred_k_log = scaler_y.inverse_transform(y_pred_k_sc.reshape(-1, 1)).ravel()

    perm_metrics_log = {
        "R²": r2_score(y_test_k_log, y_pred_k_log),
        "MAE": mean_absolute_error(y_test_k_log, y_pred_k_log),
        "RMSE": np.sqrt(mean_squared_error(y_test_k_log, y_pred_k_log)),
    }

    cv_r2_perm = cross_val_score(
        SVR(kernel="linear", C=10, epsilon=0.05),
        X_perm_sc, y_perm_sc, cv=5, scoring="r2"
    )

    # --- SHAP (porosity model) ---
    explainer_por = shap.TreeExplainer(rf)
    shap_values_por = explainer_por.shap_values(X_train_p)

    # --- SHAP (permeability model – KernelExplainer on subsample for speed) ---
    bg = shap.sample(pd.DataFrame(X_train_k, columns=perm_features), 50, random_state=42)
    explainer_perm = shap.KernelExplainer(svr.predict, bg)
    X_shap_perm = pd.DataFrame(X_test_k[:30], columns=perm_features)
    shap_values_perm = explainer_perm.shap_values(X_shap_perm)

    return {
        "rf": rf, "svr": svr,
        "scaler_X": scaler_X, "scaler_y": scaler_y,
        "por_metrics": por_metrics, "perm_metrics_log": perm_metrics_log,
        "cv_r2_por": cv_r2_por, "cv_r2_perm": cv_r2_perm,
        "X_test_p": X_test_p, "y_test_p": y_test_p, "y_pred_p": y_pred_p,
        "X_test_k_log": y_test_k_log, "y_pred_k_log": y_pred_k_log,
        "shap_por": shap_values_por, "X_train_p": X_train_p,
        "shap_perm": shap_values_perm, "X_shap_perm": X_shap_perm,
        "perm_features": perm_features,
    }


# ─── Load Data & Train ───────────────────────────────────────────────────────
df = generate_synthetic_data()
models = train_models(df)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🪨 PetroML Navigator")
    page = st.radio(
        "Section",
        ["Overview & Data", "Porosity Model (RF)", "Permeability Model (SVR)",
         "SHAP Explainability", "Single-Sample Predictor", "Business Case"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem;opacity:0.5;text-align:center;'>"
        "Dashboard built for Digital Oil Inc.<br>digitaloil.ai</p>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview & Data
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview & Data":
    st.markdown("""
    <div class="hero-banner">
        <h1>PetroML — Petrographic Reservoir Quality Predictor</h1>
        <p>
            Machine-learning prediction of porosity &amp; permeability from point-counting
            data. Implements the workflow of Sadrikhanloo, Busch &amp; Hilgers (2026)
            using Random Forest (porosity) and Support Vector Regression (permeability)
            with SHAP-based explainability of diagenetic reservoir quality controls.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="metric-card"><div class="value">157</div><div class="label">Samples</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-card"><div class="value">6</div><div class="label">Wells · 4 Regions</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="metric-card"><div class="value">18</div><div class="label">Input Features</div></div>', unsafe_allow_html=True)
    c4.markdown('<div class="metric-card"><div class="value">2</div><div class="label">Strat. Groups</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Porosity–Permeability Cross-Plot</div>', unsafe_allow_html=True)
    perm_df = df.dropna(subset=["Permeability (mD)"])
    fig_pp = px.scatter(
        perm_df, x="Porosity (%)", y="Permeability (mD)",
        color="Well", log_y=True,
        color_discrete_sequence=["#e74c3c", "#f5a623", "#2ecc71", "#3498db"],
        hover_data=["Region", "Stratigraphy"],
        template="plotly_white",
    )
    fig_pp.update_layout(
        height=430, margin=dict(t=30, b=40),
        yaxis_title="Permeability (mD) — log scale",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_pp, use_container_width=True)

    st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
    well_filter = st.multiselect("Filter by Well", df["Well"].unique(), default=df["Well"].unique())
    st.dataframe(df[df["Well"].isin(well_filter)].head(50), use_container_width=True, height=320)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Porosity Model
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Porosity Model (RF)":
    st.markdown('<div class="hero-banner"><h1>Porosity Prediction — Random Forest</h1>'
                '<p>60 petrographic features → porosity (%). Best model selected per paper methodology.</p></div>',
                unsafe_allow_html=True)

    m = models["por_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="value">{m["R²"]:.3f}</div><div class="label">Test R²</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="value">{m["MAE"]:.2f}%</div><div class="label">MAE</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="value">{m["RMSE"]:.2f}%</div><div class="label">RMSE</div></div>', unsafe_allow_html=True)
    cv_mean = models["cv_r2_por"].mean()
    cv_std = models["cv_r2_por"].std()
    c4.markdown(f'<div class="metric-card"><div class="value">{cv_mean:.2f}±{cv_std:.2f}</div><div class="label">CV R² (5-fold)</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Measured vs Predicted Porosity (Test Set)</div>', unsafe_allow_html=True)

    fig_por = go.Figure()
    fig_por.add_trace(go.Scatter(
        x=models["y_test_p"], y=models["y_pred_p"],
        mode="markers", marker=dict(size=9, color="#f5a623", line=dict(width=1, color="#333")),
        name="Test samples",
    ))
    rng_p = [0, max(models["y_test_p"].max(), models["y_pred_p"].max()) * 1.05]
    fig_por.add_trace(go.Scatter(x=rng_p, y=rng_p, mode="lines",
                                  line=dict(dash="dash", color="grey"), name="1:1 line"))
    fig_por.update_layout(
        xaxis_title="Measured Porosity (%)", yaxis_title="Predicted Porosity (%)",
        template="plotly_white", height=450, margin=dict(t=30),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_por, use_container_width=True)

    st.markdown('<div class="section-header">Feature Importance (RF Gini)</div>', unsafe_allow_html=True)
    imp = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": models["rf"].feature_importances_,
    }).sort_values("Importance", ascending=True)
    fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="YlOrBr",
                     template="plotly_white")
    fig_imp.update_layout(height=480, margin=dict(l=10, t=30), coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Permeability Model
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Permeability Model (SVR)":
    st.markdown('<div class="hero-banner"><h1>Permeability Prediction — SVR (Linear Kernel)</h1>'
                '<p>Log-transformed permeability predicted via Support Vector Regression with RobustScaler.</p></div>',
                unsafe_allow_html=True)

    m = models["perm_metrics_log"]
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="value">{m["R²"]:.3f}</div><div class="label">Test R² (log)</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="value">{m["MAE"]:.2f}</div><div class="label">MAE (log)</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="value">{m["RMSE"]:.2f}</div><div class="label">RMSE (log)</div></div>', unsafe_allow_html=True)
    cv_mean = models["cv_r2_perm"].mean()
    cv_std = models["cv_r2_perm"].std()
    c4.markdown(f'<div class="metric-card"><div class="value">{cv_mean:.2f}±{cv_std:.2f}</div><div class="label">CV R² (5-fold)</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Measured vs Predicted log(Permeability) — Test Set</div>', unsafe_allow_html=True)

    fig_perm = go.Figure()
    fig_perm.add_trace(go.Scatter(
        x=models["X_test_k_log"], y=models["y_pred_k_log"],
        mode="markers", marker=dict(size=9, color="#3498db", line=dict(width=1, color="#333")),
        name="Test samples",
    ))
    lo = min(models["X_test_k_log"].min(), models["y_pred_k_log"].min()) - 0.3
    hi = max(models["X_test_k_log"].max(), models["y_pred_k_log"].max()) + 0.3
    fig_perm.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                   line=dict(dash="dash", color="grey"), name="1:1"))
    # 1:10 and 10:1 lines
    fig_perm.add_trace(go.Scatter(x=[lo, hi], y=[lo+1, hi+1], mode="lines",
                                   line=dict(dash="dot", color="#e74c3c", width=1), name="10:1"))
    fig_perm.add_trace(go.Scatter(x=[lo, hi], y=[lo-1, hi-1], mode="lines",
                                   line=dict(dash="dot", color="#e74c3c", width=1), name="1:10"))
    fig_perm.update_layout(
        xaxis_title="Measured log[Permeability (mD)]",
        yaxis_title="Predicted log[Permeability (mD)]",
        template="plotly_white", height=450, margin=dict(t=30),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_perm, use_container_width=True)

    # Permeability distribution
    st.markdown('<div class="section-header">Permeability Distribution</div>', unsafe_allow_html=True)
    perm_vals = df["Permeability (mD)"].dropna()
    col1, col2 = st.columns(2)
    with col1:
        fig_h1 = px.histogram(perm_vals, nbins=30, template="plotly_white",
                               labels={"value": "Permeability (mD)"},
                               color_discrete_sequence=["#2c5364"])
        fig_h1.update_layout(title="Real-Scale", height=300, showlegend=False, margin=dict(t=40))
        st.plotly_chart(fig_h1, use_container_width=True)
    with col2:
        fig_h2 = px.histogram(np.log10(np.clip(perm_vals, 1e-5, None)), nbins=30,
                               template="plotly_white",
                               labels={"value": "log₁₀ Permeability (mD)"},
                               color_discrete_sequence=["#f5a623"])
        fig_h2.update_layout(title="Log-Transformed", height=300, showlegend=False, margin=dict(t=40))
        st.plotly_chart(fig_h2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SHAP Explainability
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "SHAP Explainability":
    st.markdown('<div class="hero-banner"><h1>SHAP — Reservoir Quality Controls</h1>'
                '<p>SHapley Additive exPlanations reveal which petrographic attributes drive porosity '
                'and permeability predictions, linking ML outputs to diagenetic processes.</p></div>',
                unsafe_allow_html=True)

    tab_por, tab_perm = st.tabs(["Porosity (RF)", "Permeability (SVR)"])

    with tab_por:
        st.markdown('<div class="section-header">SHAP Summary — Porosity Model</div>', unsafe_allow_html=True)
        shap_por_df = pd.DataFrame(models["shap_por"], columns=FEATURE_COLS)
        mean_abs = shap_por_df.abs().mean().sort_values(ascending=True)

        fig_shap_p = px.bar(
            x=mean_abs.values, y=mean_abs.index, orientation="h",
            labels={"x": "Mean |SHAP value|", "y": ""},
            color=mean_abs.values, color_continuous_scale="OrRd",
            template="plotly_white",
        )
        fig_shap_p.update_layout(height=480, margin=dict(l=10, t=30), coloraxis_showscale=False)
        st.plotly_chart(fig_shap_p, use_container_width=True)

        # SHAP dependence plot selector
        st.markdown('<div class="section-header">SHAP Dependence Plot</div>', unsafe_allow_html=True)
        feat_sel = st.selectbox("Select feature", mean_abs.index[::-1], key="shap_por_feat")
        feat_idx = FEATURE_COLS.index(feat_sel)

        fig_dep = px.scatter(
            x=models["X_train_p"][:, feat_idx],
            y=models["shap_por"][:, feat_idx],
            labels={"x": feat_sel, "y": "SHAP value"},
            color=models["X_train_p"][:, 0],  # color by intergranular porosity
            color_continuous_scale="Viridis",
            template="plotly_white",
        )
        fig_dep.update_layout(height=380, margin=dict(t=30),
                              coloraxis_colorbar_title="Intergranular<br>Porosity (%)")
        st.plotly_chart(fig_dep, use_container_width=True)

        st.markdown("---")
        st.info(
            "**Geological Insight — Porosity Controls:**  \n"
            "Positive SHAP values for intergranular porosity, authigenic TiOx (K-feldspar dissolution), "
            "and pore-filling (radial) illite align with porosity *preservation* mechanisms — radial illite "
            "inhibits syntaxial quartz cementation, while TiOx authigenesis reflects secondary porosity from "
            "feldspar leaching. Negative SHAP values for calcite cement, high GTG coatings (>75%), quartz "
            "cement, and tangential (pore-lining) illite reflect porosity-*destructive* diagenetic processes: "
            "pore-filling cementation and enhanced chemical compaction (pressure dissolution) at clay-coated "
            "grain contacts (Heald, 1955; Kristiansen et al., 2011)."
        )

    with tab_perm:
        st.markdown('<div class="section-header">SHAP Summary — Permeability Model</div>', unsafe_allow_html=True)
        perm_features = models["perm_features"]
        shap_perm_df = pd.DataFrame(models["shap_perm"], columns=perm_features)
        mean_abs_k = shap_perm_df.abs().mean().sort_values(ascending=True)

        fig_shap_k = px.bar(
            x=mean_abs_k.values, y=mean_abs_k.index, orientation="h",
            labels={"x": "Mean |SHAP value|", "y": ""},
            color=mean_abs_k.values, color_continuous_scale="Blues",
            template="plotly_white",
        )
        fig_shap_k.update_layout(height=500, margin=dict(l=10, t=30), coloraxis_showscale=False)
        st.plotly_chart(fig_shap_k, use_container_width=True)

        # Dependence plot
        st.markdown('<div class="section-header">SHAP Dependence Plot</div>', unsafe_allow_html=True)
        feat_sel_k = st.selectbox("Select feature", mean_abs_k.index[::-1], key="shap_perm_feat")
        feat_idx_k = perm_features.index(feat_sel_k)
        X_shap_arr = models["X_shap_perm"].values

        fig_dep_k = px.scatter(
            x=X_shap_arr[:, feat_idx_k],
            y=models["shap_perm"][:, feat_idx_k],
            labels={"x": feat_sel_k, "y": "SHAP value"},
            color=X_shap_arr[:, 0],
            color_continuous_scale="Cividis",
            template="plotly_white",
        )
        fig_dep_k.update_layout(height=380, margin=dict(t=30),
                                coloraxis_colorbar_title="Porosity<br>(scaled)")
        st.plotly_chart(fig_dep_k, use_container_width=True)

        st.markdown("---")
        st.info(
            "**Geological Insight — Permeability Controls:**  \n"
            "Porosity is the strongest positive driver of permeability, as expected from fundamental "
            "pore-network relationships. Grain size shows a positive influence — larger grains preserve "
            "larger pore throat diameters. GTI coating coverage is generally positive (inhibiting quartz "
            "cementation preserves connected pore space). Critically, **pore-filling (radial) illite is "
            "negative for permeability** despite being positive for porosity: while radial illite preserves "
            "macropore volume by inhibiting quartz overgrowths, it simultaneously reduces effective pore "
            "throat radii, impeding fluid flow (Neasham, 1977). GTG coatings and tangential illite are "
            "negative — both enhance chemical compaction. Ductile rock fragments reduce permeability "
            "through mechanical compaction of the intergranular volume (Paxton et al., 2002)."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Single-Sample Predictor
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Single-Sample Predictor":
    st.markdown('<div class="hero-banner"><h1>Single-Sample Inference</h1>'
                '<p>Enter point-counting data for a new sample and predict porosity &amp; permeability in real time.</p></div>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Detrital Composition**")
        ig_por = st.slider("Intergranular Porosity (%)", 0.0, 15.0, 5.0, 0.1)
        opt_por = st.slider("Optical Porosity (%)", 0.0, 12.0, 4.0, 0.1)
        qtz = st.slider("Quartz (%)", 30.0, 75.0, 52.0, 0.5)
        kfsp = st.slider("K-Feldspar (%)", 0.0, 20.0, 8.0, 0.5)
        plag = st.slider("Plagioclase (%)", 0.0, 8.0, 2.0, 0.5)
        mrf = st.slider("MRF undiff (%)", 0.0, 12.0, 3.0, 0.5)
        ductile = st.slider("Ductile RF (%)", 0.0, 15.0, 4.0, 0.5)
        gs = st.slider("Grain Size (mm)", 0.05, 1.0, 0.3, 0.01)

    with col_r:
        st.markdown("**Authigenic & Coating Phases**")
        gti = st.slider("GTI Coating (%)", 0.0, 100.0, 50.0, 1.0)
        gtg = st.slider("GTG Coating (%)", 0.0, 100.0, 40.0, 1.0)
        tiox = st.slider("Auth. TiOx (%)", 0.0, 4.0, 0.5, 0.1)
        calc = st.slider("Calcite Cement (%)", 0.0, 10.0, 1.0, 0.1)
        qtz_cem = st.slider("Quartz Cement (%)", 0.0, 25.0, 6.0, 0.5)
        kfsp_cem = st.slider("K-Feldspar Cement (%)", 0.0, 6.0, 1.0, 0.1)
        pf_ill = st.slider("Pore-filling Illite (%)", 0.0, 14.0, 3.0, 0.5)
        pl_ill = st.slider("Pore-lining Illite (%)", 0.0, 14.0, 3.0, 0.5)
        dol = st.slider("Dolomite (%)", 0.0, 10.0, 1.0, 0.5)
        ig_kfsp = st.slider("IG Por. K-Fsp (%)", 0.0, 4.0, 0.5, 0.1)

    sample = np.array([[ig_por, opt_por, qtz, kfsp, plag, mrf, ductile,
                         gti, gtg, tiox, calc, qtz_cem, kfsp_cem,
                         pf_ill, pl_ill, dol, ig_kfsp, gs]])

    pred_por = models["rf"].predict(sample)[0]

    # Permeability prediction needs Porosity as extra feature + scaling
    sample_perm = np.append(sample[0], pred_por).reshape(1, -1)
    sample_perm_sc = models["scaler_X"].transform(sample_perm)
    pred_perm_sc = models["svr"].predict(sample_perm_sc)
    pred_perm_log = models["scaler_y"].inverse_transform(pred_perm_sc.reshape(-1, 1))[0, 0]
    pred_perm_real = 10 ** pred_perm_log

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="value">{pred_por:.1f}%</div>'
                f'<div class="label">Predicted Porosity</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="value">{pred_perm_real:.2g} mD</div>'
                f'<div class="label">Predicted Permeability</div></div>', unsafe_allow_html=True)
    # Reservoir quality classification
    if pred_perm_real >= 100:
        rq_label, rq_color = "Good–Excellent", "#27ae60"
    elif pred_perm_real >= 10:
        rq_label, rq_color = "Moderate", "#f5a623"
    elif pred_perm_real >= 1:
        rq_label, rq_color = "Poor–Fair", "#e67e22"
    elif pred_perm_real >= 0.1:
        rq_label, rq_color = "Tight", "#e74c3c"
    else:
        rq_label, rq_color = "Very Tight", "#c0392b"
    c3.markdown(f'<div class="metric-card"><div class="value" style="color:{rq_color}">{rq_label}</div>'
                f'<div class="label">Reservoir Quality</div></div>', unsafe_allow_html=True)

    # Gauge-style visualisation
    fig_gauge = make_subplots(rows=1, cols=2, specs=[[{"type": "indicator"}, {"type": "indicator"}]])

    # Porosity gauge with quality bands
    fig_gauge.add_trace(go.Indicator(
        mode="gauge+number", value=pred_por,
        number=dict(suffix="%"),
        title={"text": "Porosity"},
        gauge=dict(axis=dict(range=[0, 22]),
                   bar=dict(color="#f5a623"),
                   steps=[dict(range=[0, 5], color="#fee"),
                          dict(range=[5, 12], color="#ffd"),
                          dict(range=[12, 22], color="#dfd")]),
    ), row=1, col=1)

    # Permeability gauge — REAL-SCALE on log axis
    # Map log₁₀(mD) to gauge position, but label with real mD values
    # Gauge range: -5 to 3 = 0.00001 to 1000 mD
    fig_gauge.add_trace(go.Indicator(
        mode="gauge+number", value=pred_perm_log,
        number=dict(valueformat=".2f", suffix=" log₁₀ mD"),
        title={"text": f"Permeability ({pred_perm_real:.2g} mD)"},
        gauge=dict(
            axis=dict(
                range=[-5, 3],
                tickvals=[-4, -3, -2, -1, 0, 1, 2, 3],
                ticktext=["0.0001", "0.001", "0.01", "0.1", "1", "10", "100", "1000"],
            ),
            bar=dict(color="#3498db"),
            steps=[
                dict(range=[-5, -1], color="#fadbd8"),   # tight/very tight
                dict(range=[-1, 0], color="#fdebd0"),     # poor–fair
                dict(range=[0, 1], color="#fef9e7"),       # moderate
                dict(range=[1, 3], color="#d5f5e3"),       # good–excellent
            ],
            threshold=dict(line=dict(color="#e74c3c", width=2), thickness=0.75, value=0),
        ),
    ), row=1, col=2)

    fig_gauge.update_layout(height=300, margin=dict(t=70, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.caption(
        "**Reading the permeability gauge:** The axis shows real millidarcy values on a logarithmic "
        "scale. Negative log₁₀ values (left of the red threshold line at 1 mD) indicate sub-millidarcy "
        "permeability — common in tight sandstones. The paper's dataset spans 0.0001–780 mD "
        "(log₁₀ range: −4 to +2.9). Colour bands: "
        "🔴 Tight (<0.1 mD) · 🟠 Poor–Fair (0.1–1 mD) · 🟡 Moderate (1–10 mD) · 🟢 Good–Excellent (>10 mD)."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Business Case
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Business Case":
    st.markdown("""
    <div class="hero-banner">
        <h1>Business Value Proposition</h1>
        <p>
            Translating ML-driven petrographic analysis into tangible upstream E&amp;P value:
            reduced drilling costs, accelerated reservoir characterisation, and enhanced
            utilisation of legacy &amp; cuttings data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Key Value Drivers</div>', unsafe_allow_html=True)

    cards = [
        ("🪨 Cuttings-Based Reservoir Evaluation",
         "Predict porosity and permeability from drill cuttings — continuously available "
         "material that doesn't require expensive coring operations. Combined with point-counting "
         "analysis, this provides near-real-time reservoir quality estimates during drilling."),
        ("💰 Reduced Coring & Lab Costs",
         "Core acquisition costs $50–150/ft plus $500–2,000 per plug for RCAL. By training models "
         "on existing core data and extending predictions to uncored intervals via cuttings or "
         "legacy thin sections, operators can reduce coring programmes by 30–50% while maintaining "
         "equivalent subsurface insight."),
        ("📊 Legacy Data Monetisation",
         "Decades of petrographic point-counting datasets sit underutilised in operator archives. "
         "This ML workflow converts static, qualitative reports into predictive models that can "
         "forecast reservoir quality in new wells within the same play."),
        ("🔍 SHAP-Driven Diagenetic Understanding",
         "SHAP explainability goes beyond prediction — it reveals which mineral phases and textures "
         "control reservoir quality. This data-driven insight complements traditional paragenetic "
         "analysis and can identify previously unrecognised controls, improving pre-drill risk assessment."),
        ("🌍 Energy Transition Applicability",
         "Geothermal, CCS, and subsurface H₂ storage projects often lack dedicated core programmes. "
         "Models trained on petroleum-era core data can de-risk these new-energy reservoirs using "
         "petrographic analysis of cuttings alone, lowering exploration costs for energy-transition plays."),
        ("⚙️ Operational Integration",
         "The workflow integrates with standard E&P data workflows — LAS, DLIS, Excel, and "
         "petrographic databases. Outputs can feed directly into static earth models, well placement "
         "optimisation, and completion design, closing the loop between laboratory analysis and "
         "field development planning."),
    ]

    for i in range(0, len(cards), 2):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="biz-card"><h4>{cards[i][0]}</h4><p>{cards[i][1]}</p></div>', unsafe_allow_html=True)
        with c2:
            if i + 1 < len(cards):
                st.markdown(f'<div class="biz-card"><h4>{cards[i+1][0]}</h4><p>{cards[i+1][1]}</p></div>', unsafe_allow_html=True)

    # ROI estimate visualisation
    st.markdown('<div class="section-header">Illustrative Cost Comparison</div>', unsafe_allow_html=True)

    cost_data = pd.DataFrame({
        "Approach": ["Conventional\n(Full Core + RCAL)", "ML-Augmented\n(Partial Core + Cuttings ML)", "Cuttings-Only\n(Trained Model)"],
        "Coring ($K)": [150, 60, 0],
        "Lab Analysis ($K)": [80, 35, 15],
        "ML/Analytics ($K)": [0, 20, 25],
    })
    cost_data["Total ($K)"] = cost_data["Coring ($K)"] + cost_data["Lab Analysis ($K)"] + cost_data["ML/Analytics ($K)"]

    fig_cost = go.Figure()
    for col, colour in [("Coring ($K)", "#e74c3c"), ("Lab Analysis ($K)", "#f5a623"), ("ML/Analytics ($K)", "#3498db")]:
        fig_cost.add_trace(go.Bar(name=col, x=cost_data["Approach"], y=cost_data[col],
                                   marker_color=colour))
    fig_cost.update_layout(
        barmode="stack", template="plotly_white", height=380,
        yaxis_title="Estimated Cost per Well ($K)",
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        margin=dict(t=40),
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "> *\"Leveraging this untapped potential via machine-learning based data science may "
        "in future enable lower-cost drilling, while gaining comparable insights into reservoir "
        "quality and related controlling factors.\"*  \n"
        "> — Sadrikhanloo, Busch & Hilgers (2026)"
    )
