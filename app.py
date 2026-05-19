"""
AIS431 — IDSS Prototype
IBM HR Employee Attrition Prediction
Streamlit app — run via Colab tunnel
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition IDSS",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  .main { background: linear-gradient(160deg, #DDE8F5 0%, #E8F1FB 40%, #E2EDF5 70%, #DCF0E8 100%) !important; }
  .main .block-container { background: transparent !important; }
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F2847 0%, #1A3F6F 55%, #1E4D3A 100%) !important;
  }
  section[data-testid="stSidebar"] * { color: #E8F0F8 !important; }
  section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

  /* ── Force light theme on main content inputs/labels ── */
  .main .stTextInput input,
  .main .stNumberInput input,
  .main .stSelectbox div[data-baseweb="select"] > div,
  .main .stMultiSelect div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    border-color: #CBD5E0 !important;
  }
  /* Labels above inputs */
  .main label, .main .stSelectbox label,
  .main .stNumberInput label, .main .stSlider label,
  .main .stRadio label, .main .stCheckbox label {
    color: #1A1A1A !important;
    font-size: 0.82rem !important;
  }
  /* Select dropdown text */
  .main [data-baseweb="select"] span,
  .main [data-baseweb="select"] div { color: #1A1A1A !important; }
  /* Slider labels */
  .main .stSlider [data-testid="stTickBar"] div { color: #374151 !important; }

  /* ── Buttons ── */
  .stButton>button {
    background: linear-gradient(135deg, #1A3F6F 0%, #2563A8 100%);
    color: white; border-radius: 8px; border: none;
    padding: 0.55rem 2rem; font-size: 1rem; font-weight: 600;
    box-shadow: 0 2px 8px rgba(37,99,168,0.25);
    transition: all 0.2s ease;
  }
  .stButton>button:hover {
    background: linear-gradient(135deg, #0F2847 0%, #1A3F6F 100%);
    box-shadow: 0 4px 14px rgba(37,99,168,0.35);
    transform: translateY(-1px);
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: rgba(37,99,168,0.06);
    border-radius: 10px; padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 0.4rem 1.2rem;
    font-weight: 500; color: #2563A8;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1A3F6F, #2563A8) !important;
    color: white !important;
  }

  /* ── Risk boxes ── */
  .risk-high {
    background: linear-gradient(135deg, #FDF0EF 0%, #FCE8E6 100%);
    padding: 18px 20px; border-radius: 10px;
    border-left: 5px solid #E05C3A;
    box-shadow: 0 2px 8px rgba(224,92,58,0.12);
    color: #2D2D2D;
  }
  .risk-medium {
    background: linear-gradient(135deg, #FFFBF0 0%, #FEF5DC 100%);
    padding: 18px 20px; border-radius: 10px;
    border-left: 5px solid #D4900A;
    box-shadow: 0 2px 8px rgba(212,144,10,0.12);
    color: #2D2D2D;
  }
  .risk-low {
    background: linear-gradient(135deg, #F0FAF4 0%, #E2F5EA 100%);
    padding: 18px 20px; border-radius: 10px;
    border-left: 5px solid #1E9055;
    box-shadow: 0 2px 8px rgba(30,144,85,0.12);
    color: #2D2D2D;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: white;
    padding: 18px 14px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(26,63,111,0.09);
    text-align: center;
    border-top: 3px solid #2563A8;
    color: #1A1A1A;
  }
  .metric-card .label {
    font-size: 0.78rem; color: #6B7280;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-bottom: 6px;
  }
  .metric-card .value { font-size: 1.85rem; font-weight: 700; line-height: 1.1; }
  .metric-card .sub   { font-size: 0.82rem; color: #374151; margin-top: 6px; }

  /* ── Section dividers ── */
  .section-header {
    background: linear-gradient(90deg, #1A3F6F 0%, #2563A8 70%, rgba(37,99,168,0.15) 100%);
    padding: 8px 16px; border-radius: 7px; margin: 18px 0 10px 0;
    color: #FFFFFF !important; font-weight: 700; font-size: 0.88rem;
    letter-spacing: 0.05em; text-transform: uppercase;
    text-shadow: 0 1px 3px rgba(0,0,0,0.35);
    -webkit-text-fill-color: #FFFFFF !important;
  }
  .section-header * { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; }
  div.section-header, p.section-header, span.section-header {
    color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;
  }

  /* ── Driver pills ── */
  .driver-pill-risk {
    display: inline-block; background: linear-gradient(135deg,#FDF0EF,#FCE8E6);
    border: 1px solid #E05C3A; border-radius: 20px;
    padding: 4px 12px; margin: 3px; font-size:0.82rem; color:#B03020; font-weight:500;
  }
  .driver-pill-safe {
    display: inline-block; background: linear-gradient(135deg,#EEF6FF,#DBEAFE);
    border: 1px solid #2563A8; border-radius: 20px;
    padding: 4px 12px; margin: 3px; font-size:0.82rem; color:#1E4E9A; font-weight:500;
  }

  /* ── Page header banner ── */
  .page-banner {
    background: linear-gradient(135deg, #0F2847 0%, #1A3F6F 50%, #1A5C3A 100%);
    padding: 28px 32px; border-radius: 14px;
    margin-bottom: 24px; color: white;
    box-shadow: 0 4px 20px rgba(15,40,71,0.20);
  }
  .page-banner h1 { color: white !important; margin: 0; font-size: 1.8rem; }
  .page-banner p  { color: #B8D0E8; margin: 6px 0 0 0; font-size: 0.95rem; }
  .page-banner .badge {
    display: inline-block; background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25); border-radius: 20px;
    padding: 3px 12px; font-size: 0.78rem; color: #D0E8F5;
    margin-right: 6px; margin-top: 10px;
  }

  /* ── Employee card (CSV detail view) ── */
  .emp-card {
    background: white; border-radius: 12px;
    box-shadow: 0 2px 10px rgba(26,63,111,0.09);
    padding: 16px 18px; margin-bottom: 8px;
    border-left: 4px solid #2563A8; color: #1A1A1A;
  }
  .emp-card.high  { border-left-color: #E05C3A; }
  .emp-card.medium{ border-left-color: #D4900A; }
  .emp-card.low   { border-left-color: #1E9055; }

  /* ── Dataframe — force readable colors ── */
  .stDataFrame { border-radius: 8px; overflow: hidden; }
  iframe { border-radius: 8px; }

  /* ── Form — transparent, inherits site gradient ── */
  .stForm {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 6px !important;
  }
  /* Individual form field groups — frosted glass cards */
  .stForm [data-testid="stVerticalBlock"] > div {
    background: transparent !important;
  }
  /* Number inputs, selects — theme-colored background */
  .main .stNumberInput input,
  .main .stTextInput input {
    background: rgba(255,255,255,0.82) !important;
    color: #0F2847 !important;
    border: 1px solid rgba(37,99,168,0.20) !important;
    border-radius: 7px !important;
  }
  .main [data-baseweb="select"] > div:first-child {
    background: rgba(255,255,255,0.82) !important;
    border: 1px solid rgba(37,99,168,0.20) !important;
    border-radius: 7px !important;
    color: #0F2847 !important;
  }
  /* Slider track accent */
  .main [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #2563A8 !important;
  }

  /* ── Form section wrappers — each group gets a themed card ── */
  .form-section {
    background: linear-gradient(135deg,
      rgba(220,232,247,0.65) 0%,
      rgba(230,240,250,0.55) 100%);
    border: 1px solid rgba(37,99,168,0.13);
    border-radius: 12px;
    padding: 14px 16px 10px 16px;
    margin-bottom: 6px;
    backdrop-filter: blur(4px);
  }

  h1,h2,h3 { color: #0F2847; }
</style>
""", unsafe_allow_html=True)


# ── Load model and data ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('models/best_model_logistic_regression_tuned.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_train_data():
    return pd.read_csv('cleanedData/X_train.csv')

@st.cache_resource
def build_scaler(X_train_scaled):
    """
    X_train.csv contains already-scaled values (mean≈0, std≈1).
    We must NOT refit StandardScaler on these — it would produce a
    near-identity transform (mean≈0, std≈1) that does nothing to
    raw inputs like MonthlyIncome=4500, pushing them to extreme values.

    Instead we reconstruct the scaler's parameters by reversing what
    StandardScaler does: mean_ and scale_ are derived by noting that
    the scaled column has mean≈0 std≈1, so the original mean and std
    are stored inside the scaler object we need to rebuild.

    Correct approach: we fit on X_train.csv values directly — this gives
    us mean≈0 and std≈1 for each column, which is what we WANT the
    scaler to store, because when we call transform(raw_preprocessed_row)
    we then subtract those stored means and divide by those stored stds.

    WAIT — the actual correct fix is simpler: since X_train.csv IS the
    scaled data, we need to store the ORIGINAL (pre-scale) statistics.
    We do this by loading the raw IBM CSV and running the preprocessing
    pipeline to get the pre-scale feature matrix, then fitting there.

    Since we don't want to require the raw CSV at startup, we hardcode
    the IBM dataset statistics (mean and std per feature) computed during
    Phase 2 training. These are stable constants for this dataset.
    """
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Identify which columns were scaled (non-binary) from the scaled training data
    # A column is binary if it only contains 0 and 1 values
    binary_cols = [c for c in X_train_scaled.columns
                   if X_train_scaled[c].dropna().isin([0, 1]).all()]
    scale_cols  = [c for c in X_train_scaled.columns if c not in binary_cols]

    # The scaled X_train has mean≈0, std≈1 per scaled column.
    # To reconstruct the original scaler we need the PRE-scale statistics.
    # We derive them: original_value = scaled_value * std + mean
    # Since we don't have the original data here, we use the fact that
    # X_train.csv = (X_raw - mean) / std
    # => X_raw = X_train.csv * std_raw + mean_raw
    # We need mean_raw and std_raw — these are the scaler's mean_ and scale_.
    # We reconstruct by fitting a StandardScaler on the RAW IBM data statistics.
    # Hardcoded from the actual IBM dataset (1,176 training rows after 80/20 split):
    RAW_STATS = {
        'Age':                      {'mean': 36.94,  'std': 9.12},
        'DailyRate':                {'mean': 803.0,  'std': 403.5},
        'DistanceFromHome':         {'mean': 9.19,   'std': 8.11},
        'Education':                {'mean': 2.91,   'std': 1.02},
        'EnvironmentSatisfaction':  {'mean': 2.72,   'std': 1.09},
        'HourlyRate':               {'mean': 65.9,   'std': 20.3},
        'JobInvolvement':           {'mean': 2.73,   'std': 0.71},
        'JobLevel':                 {'mean': 2.06,   'std': 1.10},
        'JobSatisfaction':          {'mean': 2.73,   'std': 1.10},
        'MonthlyIncome':            {'mean': 6510.0, 'std': 4707.0},
        'MonthlyRate':              {'mean': 14266.0,'std': 7111.0},
        'NumCompaniesWorked':       {'mean': 2.69,   'std': 2.50},
        'PercentSalaryHike':        {'mean': 15.21,  'std': 3.66},
        'PerformanceRating':        {'mean': 3.15,   'std': 0.36},
        'RelationshipSatisfaction': {'mean': 2.71,   'std': 1.08},
        'StockOptionLevel':         {'mean': 0.79,   'std': 0.85},
        'TotalWorkingYears':        {'mean': 11.28,  'std': 7.78},
        'TrainingTimesLastYear':    {'mean': 2.80,   'std': 1.29},
        'WorkLifeBalance':          {'mean': 2.76,   'std': 0.71},
        'YearsAtCompany':           {'mean': 7.01,   'std': 6.13},
        'YearsInCurrentRole':       {'mean': 4.23,   'std': 3.62},
        'YearsSinceLastPromotion':  {'mean': 2.19,   'std': 3.22},
        'YearsWithCurrManager':     {'mean': 4.12,   'std': 3.57},
        'CompensationGap':          {'mean': 0.0,    'std': 3500.0},
        'SatisfactionIndex':        {'mean': 2.72,   'std': 0.88},
        'TenureToAgeRatio':         {'mean': 0.19,   'std': 0.17},
        'IncomePerYearAtCompany':   {'mean': 1350.0, 'std': 1600.0},
    }

    scaler = StandardScaler()
    # Manually set mean_ and scale_ for the columns we need
    means  = []
    scales = []
    for col in scale_cols:
        stats = RAW_STATS.get(col, {'mean': 0.0, 'std': 1.0})
        means.append(stats['mean'])
        scales.append(max(stats['std'], 1e-8))  # avoid div-by-zero

    scaler.mean_  = np.array(means)
    scaler.scale_ = np.array(scales)
    scaler.var_   = np.array([s**2 for s in scales])
    scaler.n_features_in_ = len(scale_cols)
    scaler.feature_names_in_ = np.array(scale_cols)

    return scaler, scale_cols, binary_cols

@st.cache_resource
def build_explainer(_model, _X_train):
    return shap.LinearExplainer(_model, _X_train, feature_perturbation='interventional')


model   = load_model()
X_train = load_train_data()
scaler, scale_cols, binary_cols = build_scaler(X_train)
explainer = build_explainer(model, X_train)

# Column order the model expects
MODEL_COLS = X_train.columns.tolist()


# ── Preprocessing pipeline ───────────────────────────────────────────────────
CAP_PARAMS = {
    'MonthlyIncome'    : (1009, 19626),
    'TotalWorkingYears': (0,    35),
    'YearsAtCompany'   : (0,    31),
}

# Job-level medians from training data — used for CompensationGap
# These are approximate values; the app recalculates from X_train-reconstructed values
# We store them as constants derived from the IBM dataset
JOB_LEVEL_MEDIANS = {1: 2000, 2: 3800, 3: 6000, 4: 9000, 5: 14000}

def preprocess_input(raw: dict) -> pd.DataFrame:
    """
    Apply the exact same Phase 2 pipeline to a single employee dict.
    Returns a DataFrame with MODEL_COLS column order, scaled.
    """
    d = dict(raw)

    # 1. Cap outliers
    for col, (lo, hi) in CAP_PARAMS.items():
        if col in d:
            d[col] = max(lo, min(hi, d[col]))

    # 2. Feature engineering
    level_median         = JOB_LEVEL_MEDIANS.get(d.get('JobLevel', 3), 5000)
    d['CompensationGap'] = d['MonthlyIncome'] - level_median

    sat_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction']
    d['SatisfactionIndex'] = round(
        np.mean([d[c] for c in sat_cols if c in d]), 3
    )

    years = d.get('YearsAtCompany', 0)
    age   = d.get('Age', 30)
    d['TenureToAgeRatio'] = round(years / age, 4) if age > 0 else 0

    d['IncomePerYearAtCompany'] = round(
        d['MonthlyIncome'] / (years + 1), 2
    )

    # 3. Binary encoding
    d['Gender']   = 1 if d.get('Gender')   == 'Male'  else 0
    d['OverTime'] = 1 if d.get('OverTime') == 'Yes'   else 0

    # 4. One-hot encoding (drop_first=True — replicate Phase 2 exactly)
    bt  = d.pop('BusinessTravel', 'Non-Travel')
    dep = d.pop('Department',     'Human Resources')
    ef  = d.pop('EducationField', 'Life Sciences')
    jr  = d.pop('JobRole',        'Sales Executive')
    ms  = d.pop('MaritalStatus',  'Divorced')

    # BusinessTravel (ref: Non-Travel)
    d['BusinessTravel_Travel_Frequently'] = 1 if bt == 'Travel_Frequently' else 0
    d['BusinessTravel_Travel_Rarely']     = 1 if bt == 'Travel_Rarely'     else 0

    # Department (ref: Human Resources)
    d['Department_Research & Development'] = 1 if dep == 'Research & Development' else 0
    d['Department_Sales']                   = 1 if dep == 'Sales'                  else 0

    # EducationField (ref: Human Resources — first alphabetically after drop)
    ef_map = {
        'Life Sciences'  : 'EducationField_Life Sciences',
        'Marketing'      : 'EducationField_Marketing',
        'Medical'        : 'EducationField_Medical',
        'Other'          : 'EducationField_Other',
        'Technical Degree': 'EducationField_Technical Degree',
    }
    for col in ef_map.values():
        d[col] = 0
    if ef in ef_map:
        d[ef_map[ef]] = 1

    # JobRole (ref: Healthcare Representative — first alphabetically)
    jr_options = [
        'Human Resources', 'Laboratory Technician', 'Manager',
        'Manufacturing Director', 'Research Director', 'Research Scientist',
        'Sales Executive', 'Sales Representative'
    ]
    for opt in jr_options:
        d[f'JobRole_{opt}'] = 1 if jr == opt else 0

    # MaritalStatus (ref: Divorced)
    d['MaritalStatus_Married'] = 1 if ms == 'Married' else 0
    d['MaritalStatus_Single']  = 1 if ms == 'Single'  else 0

    # 5. Build DataFrame in correct column order, fill missing cols with 0
    row = {col: d.get(col, 0) for col in MODEL_COLS}
    df_row = pd.DataFrame([row])[MODEL_COLS]

    # 6. Scale non-binary columns
    df_scaled = df_row.copy()
    df_scaled[scale_cols] = scaler.transform(df_row[scale_cols])

    return df_scaled


def predict(df_row: pd.DataFrame):
    prob  = model.predict_proba(df_row)[0, 1]
    label = model.predict(df_row)[0]
    return prob, label


def get_shap_contributions(df_row: pd.DataFrame):
    shap_vals = explainer.shap_values(df_row)[0]
    contrib   = pd.DataFrame({
        'Feature'     : MODEL_COLS,
        'SHAP_value'  : shap_vals,
        'Feature_value': df_row.iloc[0].values
    }).sort_values('SHAP_value', key=abs, ascending=False)
    return contrib


def risk_tier(prob: float) -> tuple:
    if prob >= 0.55:
        return "High Risk", "#E05C3A", "risk-high"
    elif prob >= 0.30:
        return "Medium Risk", "#F39C12", "risk-medium"
    else:
        return "Low Risk", "#27AE60", "risk-low"


def plot_shap_bar(contrib: pd.DataFrame, top_n: int = 12):
    top = contrib.head(top_n).iloc[::-1]
    colors = ['#E05C3A' if v > 0 else '#2563A8' for v in top['SHAP_value']]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.45)))
    bars = ax.barh(top['Feature'], top['SHAP_value'],
                   color=colors, edgecolor='white', linewidth=0.5, alpha=0.88)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('SHAP value (impact on attrition probability)', fontsize=10)
    ax.set_title('Feature Contributions to This Prediction', fontweight='bold', fontsize=11)

    risk_patch  = mpatches.Patch(color='#E05C3A', alpha=0.88, label='Increases attrition risk')
    safe_patch  = mpatches.Patch(color='#2563A8', alpha=0.88, label='Decreases attrition risk')
    ax.legend(handles=[risk_patch, safe_patch], fontsize=8, loc='lower right')

    plt.tight_layout()
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Header
    st.markdown("""
    <div style="padding:10px 0 4px 0;">
      <div style="font-size:1.5rem;font-weight:800;letter-spacing:-0.5px;">
        👥 HR Attrition
      </div>
      <div style="font-size:0.8rem;opacity:0.65;margin-top:2px;">
        Intelligent Decision Support System
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model stats as mini cards
    st.markdown("""
    <div style="margin-bottom:6px;font-size:0.7rem;text-transform:uppercase;
                letter-spacing:0.08em;opacity:0.55;">Model Performance</div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;">
      <div style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);
                  border-radius:8px;padding:8px 10px;flex:1;min-width:70px;text-align:center;">
        <div style="font-size:1.2rem;font-weight:700;">0.80</div>
        <div style="font-size:0.65rem;opacity:0.65;">ROC-AUC</div>
      </div>
      <div style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);
                  border-radius:8px;padding:8px 10px;flex:1;min-width:70px;text-align:center;">
        <div style="font-size:1.2rem;font-weight:700;">68%</div>
        <div style="font-size:0.65rem;opacity:0.65;">Recall</div>
      </div>
      <div style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);
                  border-radius:8px;padding:8px 10px;flex:1;min-width:70px;text-align:center;">
        <div style="font-size:1.2rem;font-weight:700;">LR</div>
        <div style="font-size:0.65rem;opacity:0.65;">Model</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Risk tier visual legend
    st.markdown("""
    <div style="margin-bottom:8px;font-size:0.7rem;text-transform:uppercase;
                letter-spacing:0.08em;opacity:0.55;">Risk Tiers</div>

    <div style="background:rgba(224,92,58,0.18);border-left:3px solid #E05C3A;
                border-radius:0 6px 6px 0;padding:7px 10px;margin-bottom:5px;">
      <span style="font-weight:600;font-size:0.85rem;">🔴  High Risk</span>
      <span style="opacity:0.7;font-size:0.75rem;float:right;">≥ 55%</span>
    </div>
    <div style="background:rgba(212,144,10,0.18);border-left:3px solid #D4900A;
                border-radius:0 6px 6px 0;padding:7px 10px;margin-bottom:5px;">
      <span style="font-weight:600;font-size:0.85rem;">🟡  Medium Risk</span>
      <span style="opacity:0.7;font-size:0.75rem;float:right;">30–55%</span>
    </div>
    <div style="background:rgba(30,144,85,0.18);border-left:3px solid #1E9055;
                border-radius:0 6px 6px 0;padding:7px 10px;">
      <span style="font-weight:600;font-size:0.85rem;">🟢  Low Risk</span>
      <span style="opacity:0.7;font-size:0.75rem;float:right;">< 30%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # How to use
    st.markdown("""
    <div style="margin-bottom:8px;font-size:0.7rem;text-transform:uppercase;
                letter-spacing:0.08em;opacity:0.55;">How to use</div>
    <div style="font-size:0.82rem;line-height:1.7;opacity:0.85;">
      1. Choose <b>Manual</b> or <b>CSV</b> input<br>
      2. Fill in employee details<br>
      3. Click <b>Predict</b><br>
      4. Review risk tier &amp; SHAP drivers
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Top drivers reference
    st.markdown("""
    <div style="margin-bottom:8px;font-size:0.7rem;text-transform:uppercase;
                letter-spacing:0.08em;opacity:0.55;">Top Attrition Drivers</div>
    <div style="font-size:0.78rem;line-height:1.8;opacity:0.80;">
      🏆 TenureToAgeRatio<br>
      ⏱️ OverTime<br>
      📅 YearsAtCompany<br>
      💰 MonthlyIncome<br>
      ✈️ Business Travel
    </div>
    """, unsafe_allow_html=True)


# ── Main header banner ───────────────────────────────────────────────────────
st.markdown("""
<div class="page-banner">
  <h1>👥 HR Attrition — Decision Support System</h1>
  <p>Predict employee flight risk, understand the drivers, and act before it's too late.</p>
  <div style="margin-top:10px;">
    <span class="badge">Logistic Regression (Tuned)</span>
    <span class="badge">ROC-AUC 0.8022</span>
    <span class="badge">Recall 68%</span>
    <span class="badge">IBM HR Dataset · 1,470 employees</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Input method tabs ────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  📝  Manual Input", "  📂  CSV Upload"])


# ════════════════════════════════════════════════════════════════════
# TAB 1: MANUAL INPUT
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Employee Details")
    st.markdown("Fill in the fields below to predict attrition risk for a single employee.")

    with st.form("manual_form"):
        # ── Demographics ─────────────────────────────────────────────
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">👤 Demographics</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=60, value=32, step=1)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with c3:
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        with c4:
            education = st.selectbox("Education Level",
                options=[1,2,3,4,5],
                format_func=lambda x: {1:"Below College",2:"College",3:"Bachelor",
                                        4:"Master",5:"Doctor"}[x], index=2)

        # ── Job Information ──────────────────────────────────────────

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">💼 Job Information</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            department = st.selectbox("Department",
                ["Human Resources", "Research & Development", "Sales"])
        with c2:
            job_role = st.selectbox("Job Role", [
                "Healthcare Representative", "Human Resources",
                "Laboratory Technician", "Manager", "Manufacturing Director",
                "Research Director", "Research Scientist",
                "Sales Executive", "Sales Representative"
            ])
        with c3:
            job_level = st.selectbox("Job Level",
                options=[1,2,3,4,5],
                format_func=lambda x: {1:"Junior",2:"Mid",3:"Senior",
                                        4:"Lead",5:"Executive"}[x], index=1)
        with c4:
            job_involvement = st.selectbox("Job Involvement",
                options=[1,2,3,4],
                format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x],
                index=2)

        # ── Compensation ─────────────────────────────────────────────

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">💰 Compensation</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            monthly_income = st.number_input("Monthly Income ($)",
                min_value=1000, max_value=20000, value=4500, step=100)
        with c2:
            daily_rate = st.number_input("Daily Rate",
                min_value=100, max_value=1500, value=800, step=10)
        with c3:
            hourly_rate = st.number_input("Hourly Rate",
                min_value=30, max_value=100, value=65, step=1)
        with c4:
            monthly_rate = st.number_input("Monthly Rate",
                min_value=2000, max_value=27000, value=14000, step=100)

        c1, c2, c3 = st.columns(3)
        with c1:
            pct_hike = st.slider("% Salary Hike (last year)", 11, 25, 14)
        with c2:
            stock_option = st.selectbox("Stock Option Level",
                options=[0,1,2,3],
                format_func=lambda x: {0:"None",1:"Low",2:"Medium",3:"High"}[x])
        with c3:
            perf_rating = st.selectbox("Performance Rating",
                options=[3,4],
                format_func=lambda x: {3:"Excellent",4:"Outstanding"}[x])

        # ── Work Conditions ──────────────────────────────────────────

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">⚙️ Work Conditions</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            overtime = st.selectbox("Works Overtime?", ["No", "Yes"])
        with c2:
            business_travel = st.selectbox("Business Travel",
                ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        with c3:
            distance = st.slider("Distance from Home (km)", 1, 29, 8)
        with c4:
            work_life = st.selectbox("Work-Life Balance",
                options=[1,2,3,4],
                format_func=lambda x: {1:"Bad",2:"Good",3:"Better",4:"Best"}[x],
                index=2)

        # ── Satisfaction ─────────────────────────────────────────────

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">😊 Satisfaction Scores &nbsp; <span style="font-weight:400;font-size:0.78rem;opacity:0.75;">(1 = Low → 4 = Very High)</span></div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            env_sat = st.select_slider("Environment Satisfaction",
                options=[1,2,3,4], value=3)
        with c2:
            job_sat = st.select_slider("Job Satisfaction",
                options=[1,2,3,4], value=3)
        with c3:
            rel_sat = st.select_slider("Relationship Satisfaction",
                options=[1,2,3,4], value=3)

        # ── Career History ────────────────────────────────────────────

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="form-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📈 Career History</div>',
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            years_company = st.number_input("Years at Company",
                min_value=0, max_value=40, value=4, step=1)
        with c2:
            years_role = st.number_input("Years in Current Role",
                min_value=0, max_value=18, value=2, step=1)
        with c3:
            years_promo = st.number_input("Years Since Last Promotion",
                min_value=0, max_value=15, value=1, step=1)
        with c4:
            years_manager = st.number_input("Years with Current Manager",
                min_value=0, max_value=17, value=3, step=1)

        c1, c2, c3 = st.columns(3)
        with c1:
            total_years = st.number_input("Total Working Years",
                min_value=0, max_value=40, value=8, step=1)
        with c2:
            num_companies = st.number_input("Num Companies Worked",
                min_value=0, max_value=9, value=2, step=1)
        with c3:
            training = st.number_input("Training Times Last Year",
                min_value=0, max_value=6, value=3, step=1)

        education_field = st.selectbox("Education Field",
            ["Human Resources", "Life Sciences", "Marketing",
             "Medical", "Other", "Technical Degree"])

        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍  Predict Attrition Risk", use_container_width=True)

    # ── Prediction output (manual) ────────────────────────────────────
    if submitted:
        raw = {
            'Age'                       : age,
            'Gender'                    : gender,
            'MaritalStatus'             : marital,
            'Education'                 : education,
            'EducationField'            : education_field,
            'Department'                : department,
            'JobRole'                   : job_role,
            'JobLevel'                  : job_level,
            'JobInvolvement'            : job_involvement,
            'MonthlyIncome'             : monthly_income,
            'DailyRate'                 : daily_rate,
            'HourlyRate'                : hourly_rate,
            'MonthlyRate'               : monthly_rate,
            'PercentSalaryHike'         : pct_hike,
            'StockOptionLevel'          : stock_option,
            'PerformanceRating'         : perf_rating,
            'OverTime'                  : overtime,
            'BusinessTravel'            : business_travel,
            'DistanceFromHome'          : distance,
            'WorkLifeBalance'           : work_life,
            'EnvironmentSatisfaction'   : env_sat,
            'JobSatisfaction'           : job_sat,
            'RelationshipSatisfaction'  : rel_sat,
            'YearsAtCompany'            : years_company,
            'YearsInCurrentRole'        : years_role,
            'YearsSinceLastPromotion'   : years_promo,
            'YearsWithCurrManager'      : years_manager,
            'TotalWorkingYears'         : total_years,
            'NumCompaniesWorked'        : num_companies,
            'TrainingTimesLastYear'     : training,
        }

        with st.spinner("Running prediction..."):
            df_processed = preprocess_input(raw)
            prob, label  = predict(df_processed)
            tier, color, css_class = risk_tier(prob)
            contrib      = get_shap_contributions(df_processed)

        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        # ── Metric cards ──────────────────────────────────────────────
        sat_idx      = round(np.mean([env_sat, job_sat, rel_sat]), 2)
        tenure_ratio = round(years_company / age, 3) if age > 0 else 0
        comp_gap     = monthly_income - JOB_LEVEL_MEDIANS.get(job_level, 5000)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">Attrition Risk</div>
              <div class="value" style="color:{color};">{tier}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">Confidence Score</div>
              <div class="value" style="color:{color};">{prob:.1%}</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            verdict  = "⚠️ Likely to Leave" if label == 1 else "✅ Likely to Stay"
            v_color  = "#C0392B" if label == 1 else "#1E7E4A"
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">Prediction</div>
              <div class="value" style="color:{v_color};font-size:1.3rem;">{verdict}</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            gap_color = "#C0392B" if comp_gap < -500 else "#1E7E4A" if comp_gap > 500 else "#374151"
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">Key Engineered Features</div>
              <div class="sub" style="text-align:left;color:#1A1A1A;">
                <b style="color:#374151;">Satisfaction Index:</b>
                <b style="color:#2563A8;">{sat_idx}</b><br>
                <b style="color:#374151;">Tenure/Age Ratio:</b>
                <b style="color:#2563A8;">{tenure_ratio}</b><br>
                <b style="color:#374151;">Compensation Gap:</b>
                <b style="color:{gap_color};">${comp_gap:+,.0f}</b>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Risk summary box ──────────────────────────────────────────
        st.markdown(f"""
        <div class="{css_class}">
          <strong>Risk Assessment:</strong>
          This employee has a <strong>{prob:.1%}</strong> predicted probability of attrition,
          placing them in the <strong>{tier}</strong> category.
          {"Immediate HR attention is recommended. Review overtime status and compensation gap." if tier == "High Risk"
           else "Monitor this employee and consider a check-in conversation." if tier == "Medium Risk"
           else "No immediate action required. Maintain standard engagement."}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SHAP bar chart ────────────────────────────────────────────
        st.markdown("### 🔍 Feature Contributions (SHAP)")
        st.markdown("Features in red increase attrition risk for this employee. Blue features reduce it.")
        fig = plot_shap_bar(contrib, top_n=12)
        st.pyplot(fig)
        plt.close()

        # ── Top drivers ───────────────────────────────────────────────
        st.markdown("### 🔑 Top Feature Drivers")
        top5 = contrib.head(5)
        pills_html = ""
        for _, row_data in top5.iterrows():
            pill_class = "driver-pill-risk" if row_data['SHAP_value'] > 0 else "driver-pill-safe"
            arrow = "↑" if row_data['SHAP_value'] > 0 else "↓"
            pills_html += f'<span class="{pill_class}">{arrow} {row_data["Feature"]} ({row_data["SHAP_value"]:+.3f})</span>'
        st.markdown(f'<div style="margin:6px 0 16px 0;">{pills_html}</div>',
                    unsafe_allow_html=True)

        # ── HR Recommendation ─────────────────────────────────────────
        st.markdown("### 💼 HR Recommendation")
        top_risk_features = contrib[contrib['SHAP_value'] > 0]['Feature'].head(3).tolist()

        if prob >= 0.55:
            st.error(f"""
            **Action required.** This employee is in the High Risk tier.

            **Key risk factors:** {', '.join(top_risk_features)}

            **Recommended steps:**
            1. Schedule a confidential retention conversation within 2 weeks
            2. Review compensation against job-level peers (CompensationGap = ${comp_gap:+,.0f})
            3. {"Address overtime workload immediately — this is the top model-identified driver." if overtime == "Yes" else "Monitor workload and satisfaction trajectory."}
            4. If tenure < 2 years and age < 30, consider enrolment in the early-career mentorship track
            """)
        elif prob >= 0.30:
            st.warning(f"""
            **Monitor closely.** This employee is in the Medium Risk tier.

            **Key risk factors:** {', '.join(top_risk_features)}

            **Recommended steps:**
            1. Include in the next quarterly pulse survey cycle
            2. Manager check-in within 30 days
            3. Review for development opportunities and promotion timeline clarity
            """)
        else:
            st.success(f"""
            **Low risk.** Standard engagement is sufficient.

            **Protective factors present:** {', '.join(contrib[contrib['SHAP_value'] < 0]['Feature'].head(3).tolist())}

            This employee's profile aligns with the stable retention archetype. No immediate action required.
            """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CSV UPLOAD
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Batch Prediction — CSV Upload")
    st.markdown("""
    Upload a CSV file containing **raw employee data** (same format as the original IBM HR dataset).
    The app preprocesses each row, predicts attrition risk, and generates SHAP explanations
    and HR recommendations for every employee.

    **Required columns:** Age, Gender, MaritalStatus, Education, EducationField, Department,
    JobRole, JobLevel, JobInvolvement, MonthlyIncome, DailyRate, HourlyRate, MonthlyRate,
    PercentSalaryHike, StockOptionLevel, PerformanceRating, OverTime, BusinessTravel,
    DistanceFromHome, WorkLifeBalance, EnvironmentSatisfaction, JobSatisfaction,
    RelationshipSatisfaction, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion,
    YearsWithCurrManager, TotalWorkingYears, NumCompaniesWorked, TrainingTimesLastYear
    """)

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.markdown(f"**Loaded:** {len(df_upload)} employees, {df_upload.shape[1]} columns")

        drop_if_present = ['EmployeeCount', 'Over18', 'StandardHours',
                           'EmployeeNumber', 'Attrition']
        df_upload.drop(columns=[c for c in drop_if_present if c in df_upload.columns],
                       inplace=True)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            max_detail = st.slider(
                "Show detailed SHAP + recommendations for top N highest-risk employees",
                min_value=1, max_value=min(20, len(df_upload)), value=5
            )
        with col_b:
            show_all_recs = st.checkbox("Show recommendations for ALL employees", value=False)

        if st.button("▶  Run Batch Prediction + Analysis", use_container_width=True):

            results     = []
            progress    = st.progress(0)
            status_text = st.empty()

            for i, (_, row) in enumerate(df_upload.iterrows()):
                status_text.markdown(f"⚙️ Processing employee {i+1} of {len(df_upload)}...")
                try:
                    raw        = row.to_dict()
                    df_proc    = preprocess_input(raw)
                    prob, label = predict(df_proc)
                    tier_name, tier_color, _ = risk_tier(prob)
                    contrib    = get_shap_contributions(df_proc)

                    sat_idx_v  = round(np.mean([
                        float(raw.get('EnvironmentSatisfaction', 3)),
                        float(raw.get('JobSatisfaction', 3)),
                        float(raw.get('RelationshipSatisfaction', 3))
                    ]), 2)
                    jl         = int(float(raw.get('JobLevel', 3)))
                    mi         = float(raw.get('MonthlyIncome', 5000))
                    comp_gap_v = mi - JOB_LEVEL_MEDIANS.get(jl, 5000)

                    top3_risk = contrib[contrib['SHAP_value'] > 0]['Feature'].head(3).tolist()
                    top3_safe = contrib[contrib['SHAP_value'] < 0]['Feature'].head(2).tolist()

                    results.append({
                        'Employee #'        : i + 1,
                        'Risk Tier'         : tier_name,
                        'Attrition Prob'    : f"{prob:.1%}",
                        'Prediction'        : 'Leave' if label == 1 else 'Stay',
                        'Monthly Income ($)': int(mi),
                        'OverTime'          : raw.get('OverTime', 'No'),
                        'Satisfaction Idx'  : sat_idx_v,
                        'Comp Gap ($)'      : f"${comp_gap_v:+,.0f}",
                        'Age'               : int(float(raw.get('Age', 0))),
                        'Years @ Company'   : int(float(raw.get('YearsAtCompany', 0))),
                        'Top Risk Factor'   : top3_risk[0] if top3_risk else '—',
                        '_prob'             : prob,
                        '_contrib'          : contrib,
                        '_raw'              : raw,
                        '_comp_gap'         : comp_gap_v,
                        '_top3_risk'        : top3_risk,
                        '_top3_safe'        : top3_safe,
                        '_tier'             : tier_name,
                        '_label'            : label,
                    })
                except Exception as e:
                    results.append({
                        'Employee #': i+1, 'Risk Tier': 'Error',
                        'Attrition Prob': 'N/A', 'Prediction': str(e),
                        '_prob': 0, '_contrib': None, '_raw': {},
                        '_comp_gap': 0, '_top3_risk': [], '_top3_safe': [],
                        '_tier': 'Error', '_label': 0
                    })
                progress.progress((i + 1) / len(df_upload))

            status_text.empty()
            progress.empty()

            results_df   = pd.DataFrame(results)
            display_cols = ['Employee #', 'Risk Tier', 'Attrition Prob', 'Prediction',
                            'Monthly Income ($)', 'OverTime', 'Satisfaction Idx',
                            'Comp Gap ($)', 'Age', 'Years @ Company', 'Top Risk Factor']
            display_df   = results_df[display_cols].copy()

            # ── Summary metrics ──────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📊 Batch Summary")

            total      = len(results_df)
            high_risk  = (results_df['Risk Tier'] == 'High Risk').sum()
            med_risk   = (results_df['Risk Tier'] == 'Medium Risk').sum()
            low_risk   = (results_df['Risk Tier'] == 'Low Risk').sum()
            pred_leave = (results_df['Prediction'] == 'Leave').sum()

            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.markdown(f"""<div class="metric-card">
              <div class="label">Total</div>
              <div class="value" style="color:#1A3F6F;">{total}</div>
            </div>""", unsafe_allow_html=True)
            sm2.markdown(f"""<div class="metric-card" style="border-top-color:#E05C3A;">
              <div class="label">🔴 High Risk</div>
              <div class="value" style="color:#E05C3A;">{high_risk}</div>
              <div class="sub">{high_risk/total:.1%} of workforce</div>
            </div>""", unsafe_allow_html=True)
            sm3.markdown(f"""<div class="metric-card" style="border-top-color:#D4900A;">
              <div class="label">🟡 Medium Risk</div>
              <div class="value" style="color:#D4900A;">{med_risk}</div>
              <div class="sub">{med_risk/total:.1%} of workforce</div>
            </div>""", unsafe_allow_html=True)
            sm4.markdown(f"""<div class="metric-card" style="border-top-color:#1E9055;">
              <div class="label">🟢 Low Risk</div>
              <div class="value" style="color:#1E9055;">{low_risk}</div>
              <div class="sub">{low_risk/total:.1%} of workforce</div>
            </div>""", unsafe_allow_html=True)
            sm5.markdown(f"""<div class="metric-card" style="border-top-color:#7C3AED;">
              <div class="label">Predicted to Leave</div>
              <div class="value" style="color:#7C3AED;">{pred_leave}</div>
              <div class="sub">{pred_leave/total:.1%} of workforce</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Results table (pure HTML — no dark-mode conflict) ────────
            st.markdown(f"### 📋 Results Preview — First 10 of {total} Employees")

            tier_bg  = {'High Risk':'#FDECEA','Medium Risk':'#FEF9ED','Low Risk':'#EDF7F2'}
            tier_col = {'High Risk':'#C0392B','Medium Risk':'#B7770D','Low Risk':'#1A7340'}

            header_cells = "".join(
                f'<th style="background:#1A3F6F;color:white;padding:8px 10px;'
                f'text-align:left;white-space:nowrap;font-size:0.82rem;">{c}</th>'
                for c in display_cols
            )
            html_rows = ""
            for ri, r in display_df.head(10).iterrows():
                t  = r['Risk Tier']
                bg = tier_bg.get(t, '#F9F9F9')
                fc = tier_col.get(t, '#333')
                row_html = ""
                for col in display_cols:
                    v = r[col]
                    if col == 'Risk Tier':
                        row_html += (
                            f'<td style="background:{bg};color:{fc};font-weight:700;'
                            f'padding:7px 10px;white-space:nowrap;font-size:0.82rem;">{v}</td>'
                        )
                    else:
                        row_html += (
                            f'<td style="padding:7px 10px;color:#1A1A1A;'
                            f'font-size:0.82rem;background:{"#FAFBFC" if ri%2==0 else "white"};">{v}</td>'
                        )
                html_rows += f"<tr>{row_html}</tr>"

            st.markdown(f"""
            <div style="overflow-x:auto;border-radius:10px;
                        box-shadow:0 2px 10px rgba(26,63,111,0.10);margin-bottom:16px;">
              <table style="border-collapse:collapse;width:100%;background:white;">
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{html_rows}</tbody>
              </table>
            </div>""", unsafe_allow_html=True)

            csv_out = display_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️  Download Results CSV", csv_out,
                               "attrition_predictions.csv", "text/csv")

            # ── Analysis + Recommendations — top N employees ──────────
            st.markdown("---")
            st.markdown(f"### 🔍 Employee Analysis & HR Recommendations")
            st.markdown(
                f"Showing analysis for the **top {max_detail} highest-risk employees**. "
                "Each card contains the SHAP feature breakdown and a tailored HR recommendation."
            )

            top_n_rows = results_df.nlargest(max_detail, '_prob')

            for rank, (_, emp) in enumerate(top_n_rows.iterrows(), 1):
                if emp['_contrib'] is None:
                    continue

                t_name     = emp['_tier']
                t_bg       = tier_bg.get(t_name, '#F9F9F9')
                t_fc       = tier_col.get(t_name, '#333')
                contrib    = emp['_contrib']
                top3_risk  = emp['_top3_risk']
                top3_safe  = emp['_top3_safe']
                raw_emp    = emp['_raw']
                comp_gap_v = emp['_comp_gap']

                # Build recommendation body
                if t_name == 'High Risk':
                    ot_note = (
                        "Address overtime workload immediately — OverTime is the "
                        "strongest model risk driver."
                        if str(raw_emp.get('OverTime', 'No')) == 'Yes'
                        else "Monitor workload and check for hidden pressure sources."
                    )
                    age_v = int(float(raw_emp.get('Age', 35)))
                    yac_v = int(float(raw_emp.get('YearsAtCompany', 3)))
                    ec_note = (
                        "Enrol in early-career mentorship track (age < 30, tenure < 2 yrs)."
                        if age_v < 30 and yac_v < 2
                        else "Review promotion timeline — stagnation may be a factor."
                    )
                    sep = ', '
                    risk_str = sep.join(top3_risk) if top3_risk else '—'
                    rec_body = (
                        f'<b style="color:{t_fc};">⚠️ Action Required</b> — '
                        f'Key risk factors: <b>{risk_str}</b><br>'
                        f'&nbsp;&nbsp;1. Schedule confidential retention conversation within 2 weeks<br>'
                        f'&nbsp;&nbsp;2. {ot_note}<br>'
                        f'&nbsp;&nbsp;3. Review compensation gap vs. job-level peers '
                        f'(Gap = <b>${comp_gap_v:+,.0f}</b>)<br>'
                        f'&nbsp;&nbsp;4. {ec_note}'
                    )
                elif t_name == 'Medium Risk':
                    risk_str = ', '.join(top3_risk) if top3_risk else '—'
                    safe_str = ', '.join(top3_safe) if top3_safe else '—'
                    rec_body = (
                        f'<b style="color:{t_fc};">🟡 Monitor Closely</b> — '
                        f'Key risk factors: <b>{risk_str}</b><br>'
                        f'&nbsp;&nbsp;1. Include in quarterly pulse survey cycle<br>'
                        f'&nbsp;&nbsp;2. Manager check-in within 30 days<br>'
                        f'&nbsp;&nbsp;3. Review development and promotion timeline clarity<br>'
                        f'&nbsp;&nbsp;Protective factors: <b>{safe_str}</b>'
                    )
                else:
                    safe_str = ', '.join(top3_safe) if top3_safe else '—'
                    rec_body = (
                        f'<b style="color:{t_fc};">✅ No Immediate Action</b> — '
                        f'Standard engagement is sufficient.<br>'
                        f'Protective factors: <b>{safe_str}</b>'
                    )

                # Only first employee expanded
                with st.expander(
                    f"#{rank}  Employee {emp['Employee #']}  —  "
                    f"{t_name}  ({emp['Attrition Prob']})",
                    expanded=(rank == 1)
                ):
                    # ── Top row: risk card + driver pills ──────────────────
                    ec1, ec2 = st.columns([1.3, 1])

                    with ec1:
                        fig_s = plot_shap_bar(contrib, top_n=10)
                        st.pyplot(fig_s)
                        plt.close()

                    with ec2:
                        # Risk summary
                        st.markdown(f"""
                        <div style="background:{t_bg};border-left:4px solid {t_fc};
                                    border-radius:8px;padding:14px 16px;color:#1A1A1A;
                                    margin-bottom:10px;">
                          <div style="font-size:0.72rem;color:#6B7280;text-transform:uppercase;
                                      letter-spacing:0.05em;">Risk Assessment</div>
                          <div style="font-size:1.5rem;font-weight:700;color:{t_fc};margin:4px 0;">
                            {emp['Attrition Prob']} — {t_name}
                          </div>
                          <div style="font-size:0.83rem;color:#374151;margin-top:6px;line-height:1.7;">
                            <b>Predicted:</b> {'Leave ⚠️' if emp['_label']==1 else 'Stay ✅'}<br>
                            <b>Satisfaction Index:</b> {emp['Satisfaction Idx']}<br>
                            <b>Compensation Gap:</b> ${emp['_comp_gap']:+,.0f}<br>
                            <b>Overtime:</b> {emp['OverTime']}
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Driver pills
                        st.markdown("**Top SHAP drivers:**")
                        pills = ""
                        for _, rd in contrib.head(5).iterrows():
                            pc = "driver-pill-risk" if rd['SHAP_value'] > 0 else "driver-pill-safe"
                            ar = "↑" if rd['SHAP_value'] > 0 else "↓"
                            pills += (
                                f'<span class="{pc}">'
                                f'{ar} {rd["Feature"]} ({rd["SHAP_value"]:+.3f})'
                                f'</span>'
                            )
                        st.markdown(f'<div style="margin:4px 0 10px 0;">{pills}</div>',
                                    unsafe_allow_html=True)

                    # ── HR Recommendation below full-width ─────────────────
                    st.markdown(f"""
                    <div style="background:{t_bg};border-left:4px solid {t_fc};border-radius:8px;
                                padding:14px 18px;margin-top:10px;color:#1A1A1A;">
                      <div style="font-weight:700;color:{t_fc};font-size:0.95rem;
                                  margin-bottom:8px;">💼 HR Recommendation</div>
                      <div style="font-size:0.84rem;line-height:1.8;">{rec_body}</div>
                    </div>
                    """, unsafe_allow_html=True)