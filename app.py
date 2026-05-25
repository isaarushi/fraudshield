import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Transaction Risk Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13151c !important;
    border-right: 1px solid #1f2230;
}

/* Header hero */
.hero-header {
    background: linear-gradient(135deg, #0f1923 0%, #0d1f3c 50%, #0f1923 100%);
    border: 1px solid #1a3a5c;
    border-radius: 16px;
    padding: 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 60% 40%, rgba(0,180,255,0.06) 0%, transparent 60%);
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 8px 0;
    letter-spacing: -1px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #7a8aa0;
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,180,255,0.12);
    border: 1px solid rgba(0,180,255,0.3);
    color: #00b4ff;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    margin-bottom: 16px;
}

/* Metric cards */
.metric-card {
    background: #13151c;
    border: 1px solid #1f2230;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}
.metric-label {
    font-size: 0.8rem;
    color: #7a8aa0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Risk badges */
.badge-low      { background:#0d2e1a; color:#2ecc71; border:1px solid #2ecc71; padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.85rem; }
.badge-medium   { background:#2e2a0d; color:#f1c40f; border:1px solid #f1c40f; padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.85rem; }
.badge-high     { background:#2e1a0d; color:#e67e22; border:1px solid #e67e22; padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.85rem; }
.badge-critical { background:#2e0d0d; color:#e74c3c; border:1px solid #e74c3c; padding:4px 14px; border-radius:20px; font-weight:700; font-size:0.85rem; }
.badge-fraud    { background:#2e0d0d; color:#ff4d6d; border:1px solid #ff4d6d; padding:4px 14px; border-radius:20px; font-weight:700; }
.badge-legit    { background:#0d2e1a; color:#2ecc71; border:1px solid #2ecc71; padding:4px 14px; border-radius:20px; font-weight:700; }

/* Section heading */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #00b4ff;
    border-left: 3px solid #00b4ff;
    padding-left: 12px;
    margin: 28px 0 16px 0;
}

/* Info box */
.info-box {
    background: #0d1a2e;
    border: 1px solid #1a3a5c;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: #7a9cbf;
    line-height: 1.6;
}

/* Upload zone styling */
[data-testid="stFileUploader"] {
    border: 2px dashed #1f2230 !important;
    border-radius: 12px !important;
    background: #13151c !important;
}

/* Tables */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0066cc, #0099ff);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 0.95rem;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0052a3, #007acc);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0,153,255,0.3);
}

/* Divider */
hr { border-color: #1f2230; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL — train once and cache
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'amount', 'hour_of_day', 'day_of_week', 'distance_from_home_km',
    'previous_txn_amount', 'failed_attempts_today', 'is_international',
    'account_age_days', 'txn_count_last_24h', 'merchant_category_enc',
    'card_type_enc', 'is_night', 'is_weekend', 'is_peak_hour',
    'amount_log', 'amount_to_prev_ratio', 'is_high_value',
    'velocity_flag', 'new_account_flag', 'multi_fail_flag',
    'far_from_home', 'rule_risk_score'
]

CAT_MAP  = {'grocery':0,'retail':1,'restaurant':2,'travel':3,
            'online':4,'entertainment':5,'healthcare':6}
CARD_MAP = {'credit':0,'debit':1}


@st.cache_resource(show_spinner="Training fraud detection model on baseline data…")
def train_model():
    np.random.seed(42)
    n, fr = 40000, 0.05
    nf, nl = int(n*fr), int(n*(1-fr))

    legit = pd.DataFrame({
        'amount':                np.random.lognormal(4.5,1.2,nl).clip(1,5000),
        'hour_of_day':           np.random.choice(range(24),nl),
        'day_of_week':           np.random.randint(0,7,nl),
        'merchant_category':     np.random.choice(list(CAT_MAP.keys()),nl,
                                    p=[0.25,0.20,0.18,0.10,0.15,0.07,0.05]),
        'card_type':             np.random.choice(['credit','debit'],nl,p=[0.55,0.45]),
        'distance_from_home_km': np.abs(np.random.normal(15,20,nl)),
        'previous_txn_amount':   np.random.lognormal(4.4,1.1,nl).clip(1,4000),
        'failed_attempts_today': np.random.choice([0,1,2],nl,p=[0.90,0.08,0.02]),
        'is_international':      np.random.choice([0,1],nl,p=[0.85,0.15]),
        'account_age_days':      np.random.randint(30,3650,nl),
        'txn_count_last_24h':    np.random.randint(1,10,nl),
        'is_fraud':              0
    })
    fraud = pd.DataFrame({
        'amount':                np.random.lognormal(6.0,1.5,nf).clip(50,15000),
        'hour_of_day':           np.random.choice(range(24),nf),
        'day_of_week':           np.random.randint(0,7,nf),
        'merchant_category':     np.random.choice(list(CAT_MAP.keys()),nf,
                                    p=[0.05,0.10,0.05,0.20,0.40,0.15,0.05]),
        'card_type':             np.random.choice(['credit','debit'],nf,p=[0.75,0.25]),
        'distance_from_home_km': np.abs(np.random.normal(120,80,nf)),
        'previous_txn_amount':   np.random.lognormal(4.2,1.3,nf).clip(1,4000),
        'failed_attempts_today': np.random.choice([0,1,2,3,4],nf,p=[0.30,0.25,0.20,0.15,0.10]),
        'is_international':      np.random.choice([0,1],nf,p=[0.40,0.60]),
        'account_age_days':      np.random.randint(1,365,nf),
        'txn_count_last_24h':    np.random.randint(5,30,nf),
        'is_fraud':              1
    })

    df = pd.concat([legit,fraud]).sample(frac=1,random_state=42).reset_index(drop=True)
    df = add_features(df)

    X, y = df[FEATURE_COLS], df['is_fraud']
    Xtr,_,ytr,_ = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    Xtr_b, ytr_b = SMOTE(random_state=42).fit_resample(Xtr,ytr)

    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(Xtr_b)

    model = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
    model.fit(Xtr_s, ytr_b)
    return model, scaler


def add_features(df):
    df = df.copy()
    df['merchant_category_enc'] = df['merchant_category'].map(CAT_MAP).fillna(4).astype(int)
    df['card_type_enc']         = df['card_type'].map(CARD_MAP).fillna(0).astype(int)

    df['is_night']           = df['hour_of_day'].apply(lambda x: 1 if x<=5 or x>=22 else 0)
    df['is_weekend']         = df['day_of_week'].apply(lambda x: 1 if x>=5 else 0)
    df['is_peak_hour']       = df['hour_of_day'].apply(lambda x: 1 if 9<=x<=17 else 0)
    df['amount_log']         = np.log1p(df['amount'])
    df['amount_to_prev_ratio']= df['amount'] / (df['previous_txn_amount'].clip(lower=1) + 1)
    df['is_high_value']      = (df['amount'] > 3000).astype(int)
    df['velocity_flag']      = (df['txn_count_last_24h'] > 10).astype(int)
    df['new_account_flag']   = (df['account_age_days'] < 90).astype(int)
    df['multi_fail_flag']    = (df['failed_attempts_today'] >= 2).astype(int)
    df['far_from_home']      = (df['distance_from_home_km'] > 100).astype(int)
    df['rule_risk_score']    = (
        df['is_night']*2 + df['is_international']*2 +
        df['velocity_flag']*3 + df['new_account_flag']*2 +
        df['multi_fail_flag']*3 + df['far_from_home']*2 +
        df['is_high_value']*1
    )
    return df


def predict_df(df, model, scaler):
    df = add_features(df)
    X  = scaler.transform(df[FEATURE_COLS])
    probs  = model.predict_proba(X)[:, 1]
    labels = model.predict(X)

    df['fraud_probability'] = probs
    df['prediction']        = np.where(labels == 1, 'FRAUD', 'LEGITIMATE')
    df['risk_level']        = pd.cut(
        probs,
        bins=[-0.01, 0.30, 0.60, 0.80, 1.01],
        labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    )
    df['action'] = np.where(probs>0.70, 'BLOCK',
                   np.where(probs>0.40, 'REVIEW', 'APPROVE'))
    return df


def risk_badge(level):
    m = {'LOW':'badge-low','MEDIUM':'badge-medium',
         'HIGH':'badge-high','CRITICAL':'badge-critical'}
    return f'<span class="{m.get(level, "badge-low")}">{level}</span>'

def pred_badge(pred):
    cls = 'badge-fraud' if pred == 'FRAUD' else 'badge-legit'
    return f'<span class="{cls}">{pred}</span>'


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ FraudShield")
    st.markdown("---")
    st.markdown("**Required CSV Columns**")

    required_cols = {
        "amount": "Transaction amount (number)",
        "hour_of_day": "0 – 23",
        "day_of_week": "0=Mon … 6=Sun",
        "merchant_category": "grocery / retail / restaurant / travel / online / entertainment / healthcare",
        "card_type": "credit or debit",
        "distance_from_home_km": "Distance from home (km)",
        "previous_txn_amount": "Last transaction amount",
        "failed_attempts_today": "Failed PIN/auth attempts (0-5)",
        "is_international": "0 or 1",
        "account_age_days": "Account age in days",
        "txn_count_last_24h": "Transactions in last 24 hours",
    }

    for col, desc in required_cols.items():
        st.markdown(f"- **`{col}`** — {desc}")

    st.markdown("---")
    st.markdown("**Risk Levels**")
    st.markdown("🟢 **LOW** — Approve (<30%)")
    st.markdown("🟡 **MEDIUM** — Monitor (30–60%)")
    st.markdown("🟠 **HIGH** — Review (60–80%)")
    st.markdown("🔴 **CRITICAL** — Block (>80%)")
    st.markdown("---")
    st.caption("Model: Random Forest + SMOTE\nTrained on 40,000 synthetic transactions")


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-badge">AI-POWERED · REAL-TIME</div>
  <h1 class="hero-title">🛡️ FraudShield</h1>
  <p class="hero-subtitle">Upload your transaction CSV and instantly detect fraudulent activity using machine learning.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model, scaler = train_model()


# ─────────────────────────────────────────────
# SAMPLE CSV DOWNLOAD
# ─────────────────────────────────────────────
@st.cache_data
def make_sample_csv():
    rows = [
        [350,  14, 2, 'grocery',       'debit',  5,   300, 0, 0, 1200, 3],
        [8500,  2, 6, 'online',        'credit', 250, 200, 3, 1,   45, 18],
        [120,  10, 1, 'restaurant',    'debit',  3,   100, 0, 0,  900, 2],
        [5200,  1, 5, 'travel',        'credit', 180, 150, 2, 1,   30, 12],
        [75,   12, 3, 'grocery',       'debit',  2,    80, 0, 0, 2100, 1],
        [3400,  3, 0, 'entertainment', 'credit', 220, 300, 1, 1,   60, 15],
        [200,   9, 4, 'retail',        'credit', 10,  190, 0, 0,  600, 4],
        [9900,  0, 6, 'online',        'credit', 310, 100, 4, 1,   20, 22],
    ]
    cols = ['amount','hour_of_day','day_of_week','merchant_category','card_type',
            'distance_from_home_km','previous_txn_amount','failed_attempts_today',
            'is_international','account_age_days','txn_count_last_24h']
    return pd.DataFrame(rows, columns=cols).to_csv(index=False)

st.download_button(
    label="📥 Download Sample CSV Template",
    data=make_sample_csv(),
    file_name="sample_transactions.csv",
    mime="text/csv"
)

st.markdown('<p class="section-title">UPLOAD TRANSACTIONS</p>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop your CSV file here",
    type=["csv"],
    help="Must contain all required columns listed in the sidebar"
)

if uploaded is None:
    st.markdown("""
    <div class="info-box">
    💡 <strong>How it works:</strong><br>
    1. Download the sample CSV template above to see the exact format needed.<br>
    2. Fill it with your real transaction data (or paste your own CSV).<br>
    3. Upload and FraudShield will score every row instantly — no account needed.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# PROCESS UPLOADED FILE
# ─────────────────────────────────────────────
try:
    raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

REQUIRED = list({
    'amount','hour_of_day','day_of_week','merchant_category','card_type',
    'distance_from_home_km','previous_txn_amount','failed_attempts_today',
    'is_international','account_age_days','txn_count_last_24h'
})
missing = [c for c in REQUIRED if c not in raw.columns]
if missing:
    st.error(f"Missing columns: **{', '.join(missing)}**\n\nCheck the sidebar for required column names.")
    st.stop()

with st.spinner("Analyzing transactions…"):
    results = predict_df(raw.copy(), model, scaler)

n_total  = len(results)
n_fraud  = (results['prediction'] == 'FRAUD').sum()
n_legit  = n_total - n_fraud
n_block  = (results['action'] == 'BLOCK').sum()
n_review = (results['action'] == 'REVIEW').sum()
avg_risk = results['fraud_probability'].mean() * 100


# ─────────────────────────────────────────────
# SUMMARY METRICS
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">SUMMARY</p>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, str(n_total),         "#00b4ff", "Total Transactions"),
    (c2, str(n_fraud),         "#e74c3c", "Flagged as Fraud"),
    (c3, str(n_legit),         "#2ecc71", "Legitimate"),
    (c4, str(n_block),         "#ff6b35", "Auto-Blocked"),
    (c5, f"{avg_risk:.1f}%",   "#f1c40f", "Avg Risk Score"),
]
for col, val, color, label in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <p class="metric-value" style="color:{color}">{val}</p>
          <p class="metric-label">{label}</p>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">ANALYTICS</p>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    # Fraud probability histogram
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(
        x=results[results['prediction']=='LEGITIMATE']['fraud_probability'],
        name='Legitimate', marker_color='#2ecc71', opacity=0.75, nbinsx=20
    ))
    fig1.add_trace(go.Histogram(
        x=results[results['prediction']=='FRAUD']['fraud_probability'],
        name='Fraud', marker_color='#e74c3c', opacity=0.75, nbinsx=20
    ))
    fig1.update_layout(
        barmode='overlay', template='plotly_dark',
        paper_bgcolor='#13151c', plot_bgcolor='#13151c',
        title=dict(text='Fraud Probability Distribution', font_color='#e8eaf0'),
        xaxis_title='Fraud Probability', yaxis_title='Count',
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=45,b=30,l=30,r=10)
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_r:
    # Risk level donut
    risk_counts = results['risk_level'].value_counts()
    fig2 = go.Figure(go.Pie(
        labels=risk_counts.index.tolist(),
        values=risk_counts.values.tolist(),
        hole=0.55,
        marker_colors=['#2ecc71','#f1c40f','#e67e22','#e74c3c'],
    ))
    fig2.update_layout(
        template='plotly_dark',
        paper_bgcolor='#13151c', plot_bgcolor='#13151c',
        title=dict(text='Risk Level Breakdown', font_color='#e8eaf0'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=45,b=10,l=10,r=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

col_l2, col_r2 = st.columns(2)

with col_l2:
    # Fraud by merchant category
    cat_fraud = results.groupby('merchant_category')['fraud_probability'].mean().sort_values(ascending=True)
    fig3 = go.Figure(go.Bar(
        x=cat_fraud.values * 100,
        y=cat_fraud.index.tolist(),
        orientation='h',
        marker_color='#e74c3c',
        text=[f'{v*100:.1f}%' for v in cat_fraud.values],
        textposition='outside'
    ))
    fig3.update_layout(
        template='plotly_dark',
        paper_bgcolor='#13151c', plot_bgcolor='#13151c',
        title=dict(text='Avg Fraud Risk by Category', font_color='#e8eaf0'),
        xaxis_title='Avg Fraud Probability (%)',
        margin=dict(t=45,b=30,l=100,r=60)
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_r2:
    # Amount vs fraud probability scatter
    fig4 = px.scatter(
        results, x='amount', y='fraud_probability',
        color='risk_level',
        color_discrete_map={'LOW':'#2ecc71','MEDIUM':'#f1c40f','HIGH':'#e67e22','CRITICAL':'#e74c3c'},
        opacity=0.7, size_max=8,
        labels={'fraud_probability':'Fraud Probability','amount':'Amount'},
        title='Amount vs Fraud Probability'
    )
    fig4.update_layout(
        template='plotly_dark',
        paper_bgcolor='#13151c', plot_bgcolor='#13151c',
        title_font_color='#e8eaf0',
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=45,b=30,l=30,r=10)
    )
    st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────
# TRANSACTION TABLE
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">TRANSACTION RESULTS</p>', unsafe_allow_html=True)

filter_col1, filter_col2, _ = st.columns([1, 1, 3])
with filter_col1:
    filter_pred = st.selectbox("Filter by Prediction", ["All", "FRAUD", "LEGITIMATE"])
with filter_col2:
    filter_action = st.selectbox("Filter by Action", ["All", "BLOCK", "REVIEW", "APPROVE"])

display = results.copy()
if filter_pred != "All":
    display = display[display['prediction'] == filter_pred]
if filter_action != "All":
    display = display[display['action'] == filter_action]

show_cols = ['amount','merchant_category','card_type','hour_of_day',
             'is_international','fraud_probability','risk_level','prediction','action']
show_cols = [c for c in show_cols if c in display.columns]

display_out = display[show_cols].copy()
display_out['fraud_probability'] = (display_out['fraud_probability']*100).round(2).astype(str) + '%'

st.dataframe(
    display_out.reset_index(drop=True),
    use_container_width=True,
    height=400
)

# Download results
csv_out = display[show_cols].copy()
csv_out['fraud_probability'] = (csv_out['fraud_probability']*100).round(2)
st.download_button(
    label="📤 Export Results as CSV",
    data=csv_out.to_csv(index=False),
    file_name="fraud_analysis_results.csv",
    mime="text/csv"
)


# ─────────────────────────────────────────────
# HIGH RISK DETAIL
# ─────────────────────────────────────────────
high_risk = results[results['risk_level'].isin(['HIGH','CRITICAL'])].copy()
if len(high_risk) > 0:
    st.markdown('<p class="section-title">⚠️ HIGH RISK TRANSACTIONS — NEEDS ATTENTION</p>',
                unsafe_allow_html=True)
    for i, row in high_risk.iterrows():
        with st.expander(
            f"Transaction #{i+1} — ₹{row['amount']:,.0f} — {row.get('merchant_category','N/A')} "
            f"— {row['action']}  ({row['fraud_probability']*100:.1f}% fraud prob)"
        ):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Amount",        f"₹{row['amount']:,.2f}")
                st.metric("Fraud Prob",    f"{row['fraud_probability']*100:.2f}%")
            with col2:
                st.metric("Category",      row.get('merchant_category','—'))
                st.metric("Card Type",     row.get('card_type','—'))
            with col3:
                st.metric("Hour of Day",   int(row.get('hour_of_day',0)))
                st.metric("International", "Yes" if row.get('is_international',0) else "No")
            st.markdown(
                f"**Recommended Action:** "
                f"{'🔴 BLOCK this transaction immediately' if row['action']=='BLOCK' else '🟠 Flag for manual review'}"
            )
