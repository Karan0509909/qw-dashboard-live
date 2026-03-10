"""
King Living Quality Dashboard — Live (Read-Only)
Lightweight version that reads processed data from the main dashboard.
Run: streamlit run app_live.py --server.port 8502 --server.address 0.0.0.0
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, json, pickle
from datetime import datetime

# ════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════
st.set_page_config(
    page_title="King Living Quality Dashboard",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PKL = os.path.join(_APP_DIR, '_live_data.pkl')
_SALES_PKL = os.path.join(_APP_DIR, '_live_sales.pkl')
_TIMING_PKL = os.path.join(_APP_DIR, '_live_timing.pkl')
_ACTION_PLAN_FILE = os.path.join(_APP_DIR, '_live_action_plan.json')
_CONFIG_FILE = os.path.join(_APP_DIR, 'live_config.json')

# ════════════════════════════════════════════
# CONFIG — credentials from Streamlit Secrets, pages from config file
# ════════════════════════════════════════════
def _load_config():
    """Load page config from file, credentials from Streamlit Secrets."""
    # Pages list from config file
    try:
        with open(_CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
    except Exception:
        cfg = {"pages": ["Executive Summary"]}
    # Users from Streamlit Secrets (secure, not in GitHub)
    try:
        cfg['users'] = dict(st.secrets['users'])
    except Exception:
        # Fallback to config file for local dev
        if 'users' not in cfg:
            cfg['users'] = {}
    return cfg

_CONFIG = _load_config()

# ════════════════════════════════════════════
# COLORS & CONSTANTS
# ════════════════════════════════════════════
NAVY   = '#1B2A4A'
GOLD   = '#C6B5A1'
RED    = '#C0392B'
GREEN  = '#27AE60'
BLUE   = '#2980B9'
ORANGE = '#E67E22'
TEAL   = '#16A085'
PURPLE = '#8E44AD'
GRAY   = '#888888'
WHITE  = '#FFFFFF'
BLACK  = '#000000'
DARK_GRAY = '#323232'
LIGHT_GRAY = '#E7E6E6'

CAT_COLORS = {
    'M - Structural / Mechanical Issues': RED,
    'F - Fabric Issues': ORANGE,
    'D - At Delivery Issues': BLUE,
    'E - Electrical Issues': TEAL,
    'T - Timber Finish Issues': GOLD,
    'L - Leather Issues': PURPLE,
}
CAT_SHORT = {
    'M - Structural / Mechanical Issues': 'Structural/Mechanical',
    'F - Fabric Issues': 'Fabric',
    'D - At Delivery Issues': 'At Delivery',
    'E - Electrical Issues': 'Electrical',
    'T - Timber Finish Issues': 'Timber Issues',
    'L - Leather Issues': 'Leather',
}

MONTH_ORDER = ['Jan-24', 'Feb-24', 'Mar-24', 'Apr-24', 'May-24', 'Jun-24',
               'Jul-24', 'Aug-24', 'Sep-24', 'Oct-24', 'Nov-24', 'Dec-24',
               'Jan-25', 'Feb-25', 'Mar-25', 'Apr-25', 'May-25', 'Jun-25',
               'Jul-25', 'Aug-25', 'Sep-25', 'Oct-25', 'Nov-25', 'Dec-25',
               'Jan-26', 'Feb-26', 'Mar-26', 'Apr-26', 'May-26', 'Jun-26',
               'Jul-26', 'Aug-26', 'Sep-26', 'Oct-26', 'Nov-26', 'Dec-26']

# ════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #FAFAFA;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    div[data-testid="stMetric"] label {
        color: #666666 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1B2A4A !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #F7F5F2;
    }
    .gold-header {
        background: linear-gradient(135deg, #C6B5A1 0%, #D4C5B3 60%, #C6B5A1 100%);
        padding: 22px 28px;
        border-radius: 12px;
        margin-bottom: 18px;
        border-left: 5px solid #1B2A4A;
        box-shadow: 0 2px 8px rgba(27,42,74,0.15);
    }
    .gold-header h2 {
        color: #1B2A4A !important;
        font-family: Garet, Calibri, sans-serif;
        font-size: 1.5rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .gold-header p {
        color: #3A3A3A;
        font-size: 0.85rem;
        margin: 6px 0 0 0;
        letter-spacing: 0.3px;
    }
    .alert-info {
        background-color: #F0F7FF;
        border-left: 4px solid #2980B9;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        margin: 6px 0;
        font-size: 0.9rem;
    }
    .insight-box {
        background-color: #FAFAF7;
        border: 1px solid #E2DAD0;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    .insight-box h4 { color: #1B2A4A; margin: 0 0 8px 0; font-size: 1rem; }
    .insight-box ul { margin: 0; padding-left: 20px; }
    .insight-box li { color: #323232; margin: 4px 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# LOGIN
# ════════════════════════════════════════════
def _check_login():
    if st.session_state.get('_live_authenticated'):
        return True

    st.markdown("""
    <div style="display:flex;justify-content:center;align-items:center;min-height:70vh;">
        <div style="text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:8px;">👑</div>
            <h1 style="color:#1B2A4A;font-family:Garet,Calibri,sans-serif;font-size:1.8rem;margin:0;">
                King Living Quality Dashboard
            </h1>
            <p style="color:#888;font-size:0.9rem;margin:8px 0 24px 0;">Please sign in to continue</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        username = st.text_input("Username", key="_login_user")
        password = st.text_input("Password", type="password", key="_login_pass")
        if st.button("Sign In", type="primary", use_container_width=True):
            users = _CONFIG.get('users', {})
            if username in users and users[username] == password:
                st.session_state['_live_authenticated'] = True
                st.session_state['_live_username'] = username
                st.rerun()
            else:
                st.error("Invalid username or password")
    return False

if not _check_login():
    st.stop()

# ════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════
def month_sort_key(m):
    try: return MONTH_ORDER.index(m)
    except ValueError: return 999

def get_months_present(df_src):
    return [m for m in MONTH_ORDER if m in df_src['Month'].unique()]

def _safe_top(series, truncate=None):
    vc = series.value_counts()
    if len(vc) == 0: return 'N/A'
    val = str(vc.index[0])
    return val[:truncate] if truncate else val

def page_header(title, subtitle=None):
    sub_html = f'<p>{subtitle}</p>' if subtitle else ''
    st.markdown(f'<div class="gold-header"><h2>{title}</h2>{sub_html}</div>', unsafe_allow_html=True)

def section_divider():
    st.markdown(
        '<div style="height:2px;margin:24px 0 20px 0;'
        'background:linear-gradient(90deg,#C6B5A1 0%,#1B2A4A 50%,#C6B5A1 100%);'
        'border-radius:2px;opacity:0.4;"></div>',
        unsafe_allow_html=True)

def plotly_layout(fig, title=None, height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=NAVY, family='Calibri'), x=0.02) if title else None,
        font=dict(family='Calibri', size=12, color=DARK_GRAY),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        height=height,
        margin=dict(l=40, r=20, t=60 if title else 20, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(size=11, color=DARK_GRAY)),
    )
    fig.update_xaxes(showgrid=False, linecolor='#EEEEEE')
    fig.update_yaxes(showgrid=True, gridcolor='#F0F0F0', linecolor='#EEEEEE')
    return fig

def generate_insights(df_src, months_present):
    total = len(df_src)
    if total == 0:
        return {'overview': [], 'dominant': None, 'spikes': [], 'improving': [], 'pareto': None,
                'cat_breakdown': [], 'focus_areas': []}
    result = {'overview': [], 'dominant': None, 'spikes': [], 'improving': [], 'pareto': None}

    warranty_cnt = len(df_src[df_src['CaseType'] == 'Warranty'])
    delivery_cnt = len(df_src[df_src['CaseType'] == 'Delivery'])
    result['overview'].append(f'<b>{total}</b> total cases — {warranty_cnt/total*100:.0f}% Warranty, {delivery_cnt/total*100:.0f}% Delivery')

    if len(months_present) >= 2:
        last_m, prev_m = months_present[-1], months_present[-2]
        last_cnt = len(df_src[df_src['Month'] == last_m])
        prev_cnt = len(df_src[df_src['Month'] == prev_m])
        if prev_cnt > 0:
            chg_pct = (last_cnt - prev_cnt) / prev_cnt * 100
            direction = 'increased' if chg_pct > 0 else 'decreased'
            result['overview'].append(f'{last_m}: <b>{last_cnt}</b> cases ({chg_pct:+.0f}% vs {prev_m}) — volume {direction}')

    top_prod = df_src['ProductFamily'].value_counts()
    top_prod = top_prod[~top_prod.index.isin(['Other', 'Unknown'])]
    if len(top_prod) > 0:
        result['overview'].append(f'Top product: <b>{top_prod.index[0]}</b> ({top_prod.iloc[0]} cases, {top_prod.iloc[0]/total*100:.0f}%)')

    # Dominant issue category
    top_cat = df_src['IssueCategory'].value_counts()
    if len(top_cat) > 0:
        dom_cat_full = top_cat.index[0]
        dom_cat_short = CAT_SHORT.get(dom_cat_full, dom_cat_full)
        dom_df = df_src[df_src['IssueCategory'] == dom_cat_full]
        dom_cnt = top_cat.iloc[0]
        dom_issues = dom_df['IssueDescription'].value_counts().head(3)
        issue_details = []
        for iss, cnt in dom_issues.items():
            label = str(iss).split(' ', 1)[1][:40] if ' ' in str(iss) else str(iss)[:40]
            issue_details.append({'issue': label, 'count': cnt, 'product': '', 'parts': ''})
        root_cause = None
        if 'CaseReason' in dom_df.columns:
            reasons = dom_df['CaseReason'].dropna()
            reasons = reasons[~reasons.isin(['Unknown', ''])]
            if len(reasons) > 0:
                reason_counts = reasons.value_counts()
                top_reason_pct = reason_counts.iloc[0] / dom_cnt * 100
                if top_reason_pct >= 20:
                    top_reasons = reason_counts.head(3)
                    root_cause = [(r, c, c/dom_cnt*100) for r, c in top_reasons.items()]
        fix_info = None
        if 'FixType' in dom_df.columns:
            fixes = dom_df['FixType'].dropna()
            fixes = fixes[~fixes.isin(['Unknown', ''])]
            if len(fixes) > 0:
                fix_counts = fixes.value_counts()
                top_fix_pct = fix_counts.iloc[0] / dom_cnt * 100
                if top_fix_pct >= 20:
                    fix_info = (fix_counts.index[0], fix_counts.iloc[0], top_fix_pct)
        result['dominant'] = {
            'category': dom_cat_short, 'count': dom_cnt, 'pct': dom_cnt / total * 100,
            'issues': issue_details, 'root_cause': root_cause, 'fix_type': fix_info,
        }

    # Spikes
    if len(months_present) >= 2:
        for cat_full, short in CAT_SHORT.items():
            curr = len(df_src[(df_src['Month'] == months_present[-1]) & (df_src['IssueCategory'] == cat_full)])
            prev = len(df_src[(df_src['Month'] == months_present[-2]) & (df_src['IssueCategory'] == cat_full)])
            if prev >= 3 and curr > prev:
                pct = (curr - prev) / prev * 100
                if pct >= 25:
                    result['spikes'].append({'cat': short, 'prev': prev, 'curr': curr, 'pct': pct})
            if prev >= 3 and curr < prev:
                pct = (curr - prev) / prev * 100
                if pct <= -20:
                    result['improving'].append({'cat': short, 'prev': prev, 'curr': curr, 'pct': pct})
        result['spikes'].sort(key=lambda x: -x['pct'])
        result['improving'].sort(key=lambda x: x['pct'])

    # Category breakdown
    cat_breakdown = []
    for cat_full in top_cat.index:
        short = CAT_SHORT.get(cat_full, cat_full)
        color = CAT_COLORS.get(cat_full, GRAY)
        cnt = top_cat[cat_full]
        cat_breakdown.append({'name': short, 'count': cnt, 'pct': cnt / total * 100, 'color': color})
    result['cat_breakdown'] = cat_breakdown

    # Focus areas
    focus_areas = []
    if 'ProductFamily' in df_src.columns and 'IssueDescription' in df_src.columns:
        combo_df = df_src[['ProductFamily', 'IssueDescription', 'IssueCategory']].copy()
        combo_df = combo_df[~combo_df['ProductFamily'].isin(['Other', 'Unknown'])]
        if len(combo_df) > 0:
            combo_counts = combo_df.groupby(['ProductFamily', 'IssueDescription', 'IssueCategory']).size().reset_index(name='cases')
            combo_counts = combo_counts.sort_values('cases', ascending=False).head(8)
            for _, row in combo_counts.iterrows():
                iss_label = str(row['IssueDescription']).split(' ', 1)[1][:35] if ' ' in str(row['IssueDescription']) else str(row['IssueDescription'])[:35]
                cat_short = CAT_SHORT.get(row['IssueCategory'], row['IssueCategory'])
                cat_color = CAT_COLORS.get(row['IssueCategory'], GRAY)
                part_str = ''
                if 'PartModule' in df_src.columns:
                    sub = df_src[(df_src['ProductFamily'] == row['ProductFamily']) &
                                 (df_src['IssueDescription'] == row['IssueDescription'])]
                    parts = sub['PartModule'].value_counts()
                    parts = parts[~parts.index.isin(['Other', 'Unknown'])]
                    if len(parts) > 0:
                        part_str = parts.index[0]
                trend = ''
                if len(months_present) >= 2:
                    last_m, prev_m = months_present[-1], months_present[-2]
                    curr_cnt = len(df_src[(df_src['Month'] == last_m) &
                                          (df_src['ProductFamily'] == row['ProductFamily']) &
                                          (df_src['IssueDescription'] == row['IssueDescription'])])
                    prev_cnt = len(df_src[(df_src['Month'] == prev_m) &
                                          (df_src['ProductFamily'] == row['ProductFamily']) &
                                          (df_src['IssueDescription'] == row['IssueDescription'])])
                    if prev_cnt > 0:
                        chg = (curr_cnt - prev_cnt) / prev_cnt * 100
                        trend = f'+{chg:.0f}%' if chg > 0 else f'{chg:.0f}%'
                    elif curr_cnt > 0:
                        trend = 'NEW'
                focus_areas.append({
                    'product': row['ProductFamily'], 'issue': iss_label,
                    'cases': row['cases'], 'pct': row['cases'] / total * 100,
                    'category': cat_short, 'cat_color': cat_color,
                    'part': part_str, 'trend': trend,
                })
    result['focus_areas'] = focus_areas

    # Pareto
    issue_counts = df_src['IssueDescription'].value_counts()
    if len(issue_counts) > 0:
        cumsum = issue_counts.cumsum()
        threshold_80 = total * 0.8
        n_80 = (cumsum <= threshold_80).sum() + 1
        result['pareto'] = {'top_n': n_80, 'total_types': len(issue_counts)}

    return result

# ════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════
@st.cache_data(ttl=60)
def _load_live_data():
    if not os.path.exists(_DATA_PKL):
        return None
    try:
        df = pd.read_pickle(_DATA_PKL)
        mtime = datetime.fromtimestamp(os.path.getmtime(_DATA_PKL))
        return df, mtime
    except Exception:
        return None

@st.cache_data(ttl=60)
def _load_live_sales():
    if not os.path.exists(_SALES_PKL):
        return {}, {}, {}
    try:
        with open(_SALES_PKL, 'rb') as f:
            data = pickle.load(f)
        return data.get('total', {}), data.get('product', {}), data.get('model', {})
    except Exception:
        return {}, {}, {}

# ════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════
result = _load_live_data()
if result is None:
    st.title("King Living Quality Dashboard")
    st.markdown("---")
    st.warning("No data available yet. Please open your main dashboard first to generate the data snapshot.\n\n"
               "The main dashboard at `localhost:8501` needs to load data at least once — "
               "it will automatically export a snapshot for this live view.")
    st.stop()

df, data_mtime = result
exec_sales, exec_prod_sales, exec_model_sales = _load_live_sales()

# Apply default filters (Sep-25 onwards, all case types, all categories)
_oct25_idx = MONTH_ORDER.index('Sep-25') if 'Sep-25' in MONTH_ORDER else 0
available_months = get_months_present(df)
default_months = [m for m in available_months if m in MONTH_ORDER and MONTH_ORDER.index(m) >= _oct25_idx]
if not default_months:
    default_months = available_months

all_case_types = sorted(df['CaseType'].unique())
all_categories = sorted([c for c in df['IssueCategory'].dropna().unique() if c != 'Not Specified'])

# Sidebar — minimal for live view
with st.sidebar:
    st.markdown("### King Living Quality Dashboard")
    st.markdown(f'<div style="font-size:0.78rem;color:#888;">Logged in as <b>{st.session_state.get("_live_username", "")}</b></div>',
                unsafe_allow_html=True)
    if st.button("Sign Out"):
        st.session_state['_live_authenticated'] = False
        st.session_state['_live_username'] = ''
        st.rerun()
    st.markdown("---")

    # Data freshness
    age_mins = (datetime.now() - data_mtime).total_seconds() / 60
    if age_mins < 60:
        _fresh_color = GREEN
        _fresh_text = f"Updated {int(age_mins)} min ago"
    elif age_mins < 1440:
        _fresh_color = ORANGE
        _fresh_text = f"Updated {int(age_mins / 60)} hours ago"
    else:
        _fresh_color = RED
        _fresh_text = f"Updated {int(age_mins / 1440)} days ago"
    st.markdown(f'<div style="font-size:0.78rem;color:{_fresh_color};font-weight:600;">{_fresh_text}</div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Filters")
    sel_months = st.multiselect("Months", available_months, default=default_months)
    sel_case_types = st.multiselect("Case Type", all_case_types, default=all_case_types)
    sel_categories = st.multiselect("Issue Category", all_categories, default=all_categories)

# Apply filters
mask = (df['Month'].isin(sel_months) &
        df['CaseType'].isin(sel_case_types) &
        (df['IssueCategory'].isin(sel_categories) | df['IssueCategory'].isna() | (df['IssueCategory'] == 'Not Specified')))
df_filtered = df[mask].copy()

st.sidebar.caption(f"Showing **{len(df_filtered)}** of {len(df)} cases")

# ════════════════════════════════════════════
# EXECUTIVE SUMMARY
# ════════════════════════════════════════════
_months_all = sorted(df_filtered['Month'].unique(), key=month_sort_key)
_period = f"{_months_all[0]} — {_months_all[-1]}" if len(_months_all) >= 2 else (_months_all[0] if _months_all else '')
page_header("Quality Review", _period)

# ── Filters row ──
fc1, fc2, fc3 = st.columns([2, 2, 3])
with fc1:
    _all_prods = df_filtered['ProductFamily'].value_counts()
    _all_prods = sorted(_all_prods[~_all_prods.index.isin(['Other', 'Unknown'])].index.tolist())
    sel_top10_prod = st.multiselect("Product", _all_prods, default=[], key="exec_qf_prods")
with fc2:
    _all_issues_exec = sorted(df_filtered['IssueDescription'].dropna().unique().tolist())
    _iss_labels = {x: (str(x).split(' ', 1)[1][:40] if ' ' in str(x) else str(x)[:40]) for x in _all_issues_exec}
    sel_top10_iss = st.multiselect("Issue", options=_all_issues_exec,
                                    format_func=lambda x: _iss_labels.get(x, x),
                                    default=[], key="exec_qf_issues")
with fc3:
    all_months_exec = sorted(df_filtered['Month'].unique(), key=month_sort_key)
    if len(all_months_exec) >= 2:
        range_start, range_end = st.select_slider("Month Range", options=all_months_exec,
                                                   value=(all_months_exec[0], all_months_exec[-1]),
                                                   key="exec_month_range")
    else:
        range_start = all_months_exec[0] if all_months_exec else None
        range_end = range_start

# Apply quick filters
if sel_top10_prod:
    df_filtered = df_filtered[df_filtered['ProductFamily'].isin(sel_top10_prod)].copy()
if sel_top10_iss:
    df_filtered = df_filtered[df_filtered['IssueDescription'].isin(sel_top10_iss)].copy()
if len(all_months_exec) >= 2:
    idx_start = all_months_exec.index(range_start)
    idx_end = all_months_exec.index(range_end)
    sel_range = all_months_exec[idx_start:idx_end + 1]
    df_filtered = df_filtered[df_filtered['Month'].isin(sel_range)].copy()

section_divider()

total = len(df_filtered)
warranty_cnt = len(df_filtered[df_filtered['CaseType'] == 'Warranty'])
delivery_cnt = len(df_filtered[df_filtered['CaseType'] == 'Delivery'])
months_present = sorted(df_filtered['Month'].unique(), key=month_sort_key)

top_prod_vc = df_filtered['ProductFamily'].value_counts()
top_prod_vc = top_prod_vc[~top_prod_vc.index.isin(['Other', 'Unknown'])]

if len(months_present) >= 2:
    last_m, prev_m = months_present[-1], months_present[-2]
    last_cnt = len(df_filtered[df_filtered['Month'] == last_m])
    prev_cnt = len(df_filtered[df_filtered['Month'] == prev_m])
    mom_chg = (last_cnt - prev_cnt) / prev_cnt * 100 if prev_cnt else 0
else:
    last_m = months_present[-1] if months_present else ''
    last_cnt = total
    mom_chg = 0

# Sales data
has_sales = any(exec_sales.get(m, 0) > 0 for m in months_present)
if sel_top10_prod and exec_prod_sales:
    effective_exec_sales = {}
    for m in months_present:
        m_data = exec_prod_sales.get(m, {})
        effective_exec_sales[m] = sum(m_data.get(p, 0) for p in sel_top10_prod)
    has_sales = any(effective_exec_sales.get(m, 0) > 0 for m in months_present)
else:
    effective_exec_sales = exec_sales

# ═══════════════════════════════════════
# ROW 1: KPI strip
# ═══════════════════════════════════════
total_sales_sum = sum(effective_exec_sales.get(m, 0) for m in months_present) if has_sales else 0

_w_mom_delta = None
_d_mom_delta = None
if len(months_present) >= 2:
    _w_last = len(df_filtered[(df_filtered['Month'] == months_present[-1]) & (df_filtered['CaseType'] == 'Warranty')])
    _w_prev = len(df_filtered[(df_filtered['Month'] == months_present[-2]) & (df_filtered['CaseType'] == 'Warranty')])
    _d_last = len(df_filtered[(df_filtered['Month'] == months_present[-1]) & (df_filtered['CaseType'] == 'Delivery')])
    _d_prev = len(df_filtered[(df_filtered['Month'] == months_present[-2]) & (df_filtered['CaseType'] == 'Delivery')])
    _w_mom_delta = f"{(_w_last - _w_prev) / _w_prev * 100:+.0f}% vs {months_present[-2]}" if _w_prev > 0 else None
    _d_mom_delta = f"{(_d_last - _d_prev) / _d_prev * 100:+.0f}% vs {months_present[-2]}" if _d_prev > 0 else None

k1, k2, k3, k4, k5, k6 = st.columns(6)
if has_sales:
    case_rate = total / total_sales_sum * 100 if total_sales_sum > 0 else 0
    sales_label = "Product Units" if sel_top10_prod and exec_prod_sales else "Delivered Units"
    k1.metric(sales_label, f"{total_sales_sum:,.0f}")
else:
    k1.metric("Delivered Units", "No data")
k2.metric("Total Cases", f"{total:,}", f"{mom_chg:+.0f}% vs last month" if len(months_present) >= 2 else None,
          delta_color="inverse" if len(months_present) >= 2 else "off")
if has_sales and total_sales_sum > 0:
    _case_pct_delta = None
    if len(months_present) >= 2:
        _cp_last_sales = effective_exec_sales.get(months_present[-1], 0)
        _cp_last_cases = len(df_filtered[df_filtered['Month'] == months_present[-1]])
        _cp_prev_sales = effective_exec_sales.get(months_present[-2], 0)
        _cp_prev_cases = len(df_filtered[df_filtered['Month'] == months_present[-2]])
        _cp_last_pct = _cp_last_cases / _cp_last_sales * 100 if _cp_last_sales > 0 else 0
        _cp_prev_pct = _cp_prev_cases / _cp_prev_sales * 100 if _cp_prev_sales > 0 else 0
        if _cp_prev_pct > 0:
            _case_pct_delta = f"{_cp_last_pct - _cp_prev_pct:+.2f}pp vs {months_present[-2]}"
    k3.metric("Total Case %", f"{case_rate:.2f}%", _case_pct_delta, delta_color="inverse")
else:
    k3.metric("Total Case %", "—")
k4.metric("Warranty Cases", f"{warranty_cnt:,}", _w_mom_delta, delta_color="inverse")
k5.metric("At Delivery Cases", f"{delivery_cnt:,}", _d_mom_delta, delta_color="inverse")
if len(months_present) >= 2:
    k6.metric(last_m, f"{last_cnt}", f"{mom_chg:+.0f}% MoM", delta_color="inverse")
else:
    k6.metric("Period", f"{len(months_present)} months")

# ── Monthly sparkline strip ──
if len(months_present) >= 3:
    _spark_monthly = df_filtered.groupby('Month').size().reindex(months_present, fill_value=0)
    _spark_vals = _spark_monthly.values.tolist()
    _spark_max = max(_spark_vals) if _spark_vals else 1
    _spark_bars = ''
    for i, (m, v) in enumerate(zip(months_present, _spark_vals)):
        h = max(int(v / _spark_max * 40), 2)
        if i == 0:
            color = NAVY
        elif v < _spark_vals[i - 1]:
            color = GREEN
        elif v > _spark_vals[i - 1]:
            color = RED
        else:
            color = NAVY
        _spark_bars += (f'<div style="display:inline-block;text-align:center;margin:0 3px;">'
                       f'<div style="font-size:0.65rem;color:#888;">{v}</div>'
                       f'<div style="width:28px;height:{h}px;background:{color};border-radius:3px 3px 0 0;"></div>'
                       f'<div style="font-size:0.6rem;color:#AAA;margin-top:2px;">{m}</div>'
                       f'</div>')
    _trend_word = "improving" if len(_spark_vals) >= 2 and _spark_vals[-1] < _spark_vals[-2] else "worsening" if len(_spark_vals) >= 2 and _spark_vals[-1] > _spark_vals[-2] else "stable"
    _trend_color = GREEN if _trend_word == "improving" else RED if _trend_word == "worsening" else '#888'
    st.markdown(f'''<div style="background:#FAFAFA;border:1px solid #E8E8E8;border-radius:8px;
        padding:10px 16px;margin:6px 0 12px 0;display:flex;align-items:flex-end;gap:16px;">
        <div style="display:flex;align-items:flex-end;">{_spark_bars}</div>
        <div style="margin-left:auto;text-align:right;">
            <div style="font-size:0.72rem;color:#888;text-transform:uppercase;">Monthly Trend</div>
            <div style="font-size:0.95rem;font-weight:600;color:{_trend_color};">{_trend_word.title()}</div>
        </div>
    </div>''', unsafe_allow_html=True)

# ═══════════════════════════════════════
# ALERT BANNER
# ═══════════════════════════════════════
if len(months_present) >= 2:
    _alert_last = len(df_filtered[df_filtered['Month'] == months_present[-1]])
    _alert_prev = len(df_filtered[df_filtered['Month'] == months_present[-2]])
    if _alert_prev > 0:
        _alert_chg = (_alert_last - _alert_prev) / _alert_prev * 100
        if _alert_chg >= 30:
            _alert_last_cats = df_filtered[df_filtered['Month'] == months_present[-1]]['IssueCategory'].value_counts()
            _alert_prev_cats = df_filtered[df_filtered['Month'] == months_present[-2]]['IssueCategory'].value_counts()
            _alert_drivers = []
            for _ac in _alert_last_cats.index[:5]:
                _ac_curr = _alert_last_cats.get(_ac, 0)
                _ac_prev = _alert_prev_cats.get(_ac, 0)
                if _ac_prev > 0 and (_ac_curr - _ac_prev) / _ac_prev * 100 > 20:
                    _alert_drivers.append(f'{CAT_SHORT.get(_ac, _ac)} (+{(_ac_curr - _ac_prev) / _ac_prev * 100:.0f}%)')
            _driver_text = f' — driven by {", ".join(_alert_drivers[:3])}' if _alert_drivers else ''
            st.markdown(f'''<div style="background:#FFF3F3;border:1px solid #FFBABA;border-left:4px solid {RED};
                border-radius:8px;padding:12px 18px;margin:6px 0 12px 0;display:flex;align-items:center;gap:12px;">
                <span style="font-size:1.4rem;">&#9888;</span>
                <div>
                    <div style="font-size:0.85rem;font-weight:700;color:{RED};">
                        Case Spike Alert: {months_present[-1]} is up {_alert_chg:.0f}% vs {months_present[-2]}
                        ({_alert_prev} &rarr; {_alert_last} cases)
                    </div>
                    <div style="font-size:0.78rem;color:#666;margin-top:2px;">{_driver_text.lstrip(" — ")}</div>
                </div>
            </div>''', unsafe_allow_html=True)

# ═══════════════════════════════════════
# Category badges
# ═══════════════════════════════════════
insights = generate_insights(df_filtered, months_present)
spikes = insights.get('spikes', [])
improving = insights.get('improving', [])
cat_breakdown = insights.get('cat_breakdown', [])
focus_areas = insights.get('focus_areas', [])

if cat_breakdown:
    badge_html = ''
    for cb in cat_breakdown:
        badge_html += (f'<span style="display:inline-block;background:{cb["color"]};color:white;'
                       f'padding:6px 14px;border-radius:20px;margin:3px 4px;font-size:0.82rem;'
                       f'font-weight:600;white-space:nowrap;min-width:120px;text-align:center;">'
                       f'{cb["name"]} <span style="opacity:0.85;">{cb["count"]} ({cb["pct"]:.0f}%)</span></span>')
    st.markdown(f'<div style="margin:4px 0 10px 0;">{badge_html}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════
# Customer Voice
# ═══════════════════════════════════════
if 'WebDescribesIssue' in df_filtered.columns:
    _web_desc_vc = df_filtered['WebDescribesIssue'].dropna().replace('', np.nan).dropna().value_counts()
    _web_help_vc = df_filtered['WebNeedHelpWith'].dropna().replace('', np.nan).dropna().value_counts() if 'WebNeedHelpWith' in df_filtered.columns else pd.Series(dtype=int)
    if len(_web_desc_vc) > 0 or len(_web_help_vc) > 0:
        _cv_html = f'''<div style="background:#F9F7F4;border:1px solid #E8E4DE;border-left:4px solid {GOLD};
            border-radius:8px;padding:12px 16px;margin:6px 0 12px 0;">
            <div style="font-size:0.72rem;color:{GOLD};font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">
                Customer Voice — What customers are reporting</div>
            <div style="display:flex;gap:24px;flex-wrap:wrap;">'''
        if len(_web_desc_vc) > 0:
            _cv_html += '<div><div style="font-size:0.7rem;color:#888;margin-bottom:4px;">What describes the issue?</div>'
            for _wd, _wc in _web_desc_vc.head(5).items():
                _cv_html += (f'<span style="display:inline-block;background:white;border:1px solid #D6CFC5;'
                             f'padding:4px 10px;border-radius:12px;margin:2px 3px;font-size:0.78rem;color:#323232;">'
                             f'{str(_wd)[:40]} <b style="color:{NAVY};">({_wc})</b></span>')
            _cv_html += '</div>'
        if len(_web_help_vc) > 0:
            _cv_html += '<div><div style="font-size:0.7rem;color:#888;margin-bottom:4px;">What do you need help with?</div>'
            for _wh, _whc in _web_help_vc.head(5).items():
                _cv_html += (f'<span style="display:inline-block;background:white;border:1px solid #D6CFC5;'
                             f'padding:4px 10px;border-radius:12px;margin:2px 3px;font-size:0.78rem;color:#323232;">'
                             f'{str(_wh)[:40]} <b style="color:{NAVY};">({_whc})</b></span>')
            _cv_html += '</div>'
        _cv_html += '</div></div>'
        st.markdown(_cv_html, unsafe_allow_html=True)

# ═══════════════════════════════════════
# Product Case Rates
# ═══════════════════════════════════════
if has_sales and exec_prod_sales:
    _exec_prod_rates = []
    _MIN_DELIVERIES = 30
    _exec_all_pf = df_filtered[~df_filtered['ProductFamily'].isin(['Other', 'Unknown'])]['ProductFamily'].value_counts()
    for _pf in _exec_all_pf.index[:20]:
        _p_del_tot = sum(exec_prod_sales.get(m, {}).get(_pf, 0) for m in months_present)
        _p_cases_tot = len(df_filtered[df_filtered['ProductFamily'] == _pf])
        if _p_del_tot >= _MIN_DELIVERIES:
            _p_rate = _p_cases_tot / _p_del_tot * 100
            _exec_prod_rates.append({'product': _pf, 'cases': _p_cases_tot, 'deliveries': _p_del_tot, 'rate': _p_rate})
    _exec_prod_rates.sort(key=lambda x: x['rate'], reverse=True)

    if _exec_prod_rates:
        st.markdown(f'''<div style="margin:8px 0 4px 0;">
            <div style="font-size:0.78rem;color:{NAVY};font-weight:600;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">
                Product Case Rates — cases as % of delivered units (min {_MIN_DELIVERIES}+ units)
            </div></div>''', unsafe_allow_html=True)
        _epc_html = ''
        for _pr in _exec_prod_rates[:10]:
            _sev = RED if _pr['rate'] >= 2.0 else (ORANGE if _pr['rate'] >= 1.0 else GREEN)
            _epc_html += f'''<div style="display:inline-block;vertical-align:top;background:white;
                border:1px solid #E0E0E0;border-top:3px solid {_sev};border-radius:8px;
                padding:10px 14px;margin:4px 6px;min-width:140px;max-width:180px;">
                <div style="font-size:0.78rem;font-weight:700;color:{NAVY};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{_pr['product']}</div>
                <div style="font-size:1.6rem;font-weight:700;color:{_sev};margin:2px 0;">{_pr['rate']:.2f}%</div>
                <div style="font-size:0.7rem;color:#888;">
                    {_pr['cases']} cases / {_pr['deliveries']:,} units
                </div>
            </div>'''
        st.markdown(f'<div style="overflow-x:auto;white-space:nowrap;margin:4px 0 16px 0;">{_epc_html}</div>',
                    unsafe_allow_html=True)

# ═══════════════════════════════════════
# Main chart + Issue category
# ═══════════════════════════════════════
ch1, ch2 = st.columns([3, 2])

with ch1:
    monthly_warranty = df_filtered[df_filtered['CaseType'] == 'Warranty'].groupby('Month').size().reindex(
        get_months_present(df_filtered), fill_value=0)
    monthly_delivery = df_filtered[df_filtered['CaseType'] == 'Delivery'].groupby('Month').size().reindex(
        get_months_present(df_filtered), fill_value=0)
    _exec_months = get_months_present(df_filtered)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if has_sales:
        sales_vals = [effective_exec_sales.get(m, 0) for m in _exec_months]
        w_vals = [monthly_warranty.get(m, 0) for m in _exec_months]
        d_vals = [monthly_delivery.get(m, 0) for m in _exec_months]
        total_cases = [w + d for w, d in zip(w_vals, d_vals)]
        t_pct = [(t / s * 100) if s > 0 else 0 for t, s in zip(total_cases, sales_vals)]
        w_pct = [(w / s * 100) if s > 0 else 0 for w, s in zip(w_vals, sales_vals)]
        d_pct = [(d / s * 100) if s > 0 else 0 for d, s in zip(d_vals, sales_vals)]

        fig.add_trace(go.Bar(x=list(_exec_months), y=sales_vals,
                             name='Delivered Units', marker_color='rgba(39,174,96,0.12)',
                             marker_line=dict(color=GREEN, width=1.2),
                             text=[f'<b>{v:,.0f}</b>' for v in sales_vals],
                             textposition='outside',
                             textfont=dict(size=11, color=GREEN, family='Calibri'),
                             width=0.55), secondary_y=False)
        fig.add_trace(go.Bar(x=list(_exec_months), y=w_vals,
                             name='Warranty Cases', marker_color=NAVY, opacity=0.9, width=0.35),
                      secondary_y=False)
        fig.add_trace(go.Bar(x=list(_exec_months), y=d_vals, base=w_vals,
                             name='Delivery Cases', marker_color=GOLD, opacity=0.9, width=0.35),
                      secondary_y=False)
        for i, m in enumerate(list(_exec_months)):
            fig.add_annotation(x=m, y=total_cases[i], text=f'<b>{total_cases[i]}</b>',
                               showarrow=False, yshift=12, font=dict(size=11, color=NAVY),
                               secondary_y=False, yref='y')
        fig.add_trace(go.Scatter(x=list(_exec_months), y=t_pct,
                                 name='Total Case Rate %', mode='lines+markers+text',
                                 line=dict(color=RED, width=3.5),
                                 marker=dict(size=11, color=RED, symbol='circle',
                                             line=dict(color='white', width=2)),
                                 text=[f'<b>{v:.1f}%</b>' for v in t_pct],
                                 textposition='bottom center',
                                 textfont=dict(size=12, color=RED, family='Calibri')),
                      secondary_y=True)
        fig.add_trace(go.Scatter(x=list(_exec_months), y=w_pct,
                                 name='Warranty %', mode='lines+markers',
                                 line=dict(color=NAVY, width=2, dash='dot'),
                                 marker=dict(size=6, color=NAVY)),
                      secondary_y=True)
        fig.add_trace(go.Scatter(x=list(_exec_months), y=d_pct,
                                 name='Delivery %', mode='lines+markers',
                                 line=dict(color='#8B7355', width=2, dash='dot'),
                                 marker=dict(size=6, color='#8B7355')),
                      secondary_y=True)
        fig.update_yaxes(title_text='Count', secondary_y=False)
        _max_pct = max(t_pct + w_pct + d_pct) if t_pct else 3
        _y_top = max(4, (int(_max_pct / 0.5) + 3) * 0.5)
        _tvals = [i * 0.5 for i in range(int(_y_top / 0.5) + 1)]
        fig.update_yaxes(title_text='Case Rate %', secondary_y=True,
                         showgrid=False, range=[0, _y_top],
                         tickvals=_tvals,
                         ticktext=[f'{v:.1f}%' if v % 1 else f'{int(v)}%' for v in _tvals])
        fig.update_layout(barmode='overlay', bargap=0.15,
                          legend=dict(orientation='h', yanchor='top', y=-0.18,
                                      xanchor='center', x=0.5, font=dict(size=10)))
    else:
        fig.add_trace(go.Bar(x=monthly_warranty.index, y=monthly_warranty.values,
                             name='Warranty', marker_color=NAVY, opacity=0.85,
                             text=monthly_warranty.values, textposition='outside',
                             textfont=dict(size=11, color=NAVY)), secondary_y=False)
        fig.add_trace(go.Bar(x=monthly_delivery.index, y=monthly_delivery.values,
                             name='Delivery', marker_color=GOLD, opacity=0.85,
                             text=monthly_delivery.values, textposition='outside',
                             textfont=dict(size=11, color='#8B7355')), secondary_y=False)
        fig.update_layout(barmode='group')
    plotly_layout(fig, 'Delivered Units vs Case Rate', 440)
    st.plotly_chart(fig, width="stretch")

with ch2:
    st.markdown("**Issue Category Trends**")
    cat_monthly_exec = df_filtered.groupby(['Month', 'IssueCategory']).size().unstack(fill_value=0)
    cat_monthly_exec = cat_monthly_exec.reindex(get_months_present(df_filtered), fill_value=0)

    _exec_use_sec = effective_exec_sales and any(effective_exec_sales.get(m, 0) > 0 for m in get_months_present(df_filtered))
    fig = make_subplots(specs=[[{"secondary_y": True}]]) if _exec_use_sec else go.Figure()

    for cat_full, color in CAT_COLORS.items():
        if cat_full in cat_monthly_exec.columns:
            vals = cat_monthly_exec[cat_full]
            _bar = go.Bar(x=vals.index, y=vals.values,
                          name=CAT_SHORT.get(cat_full, cat_full),
                          marker_color=color, opacity=0.9,
                          text=vals.values, textposition='outside',
                          textfont=dict(size=9))
            if _exec_use_sec:
                fig.add_trace(_bar, secondary_y=False)
            else:
                fig.add_trace(_bar)

    if _exec_use_sec:
        _exec_total_pm = cat_monthly_exec.sum(axis=1)
        _exec_wpct_vals = []
        for m in get_months_present(df_filtered):
            cases = _exec_total_pm.get(m, 0)
            deliveries = effective_exec_sales.get(m, 0)
            _exec_wpct_vals.append(round((cases / deliveries) * 100, 2) if deliveries > 0 else None)
        fig.add_trace(go.Scatter(
            x=list(get_months_present(df_filtered)), y=_exec_wpct_vals,
            name='Total % of Sales', mode='lines+markers+text',
            line=dict(color=RED, width=3),
            marker=dict(size=8, color=RED, line=dict(color='white', width=1.5)),
            text=[f'{v:.1f}%' if v is not None else '' for v in _exec_wpct_vals],
            textposition='top center',
            textfont=dict(size=11, color=RED, family='Calibri'),
        ), secondary_y=True)
        fig.update_yaxes(title_text="Cases", secondary_y=False)
        _exec_max_wpct = max((v for v in _exec_wpct_vals if v is not None), default=3)
        _exec_ytop = max(3, (int(_exec_max_wpct / 0.5) + 2) * 0.5)
        _exec_tvals = [i * 0.5 for i in range(int(_exec_ytop / 0.5) + 1)]
        fig.update_yaxes(title_text="% of Deliveries", secondary_y=True,
                         showgrid=False, range=[0, _exec_ytop],
                         tickvals=_exec_tvals,
                         ticktext=[f'{v:.1f}%' if v % 1 else f'{int(v)}%' for v in _exec_tvals])

    fig.update_layout(barmode='group', bargap=0.15, bargroupgap=0.05)
    plotly_layout(fig, None, 380)
    fig.update_layout(legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5, font=dict(size=9)))
    st.plotly_chart(fig, width="stretch")

# ═══════════════════════════════════════
# Where to Focus + Spike/Improving
# ═══════════════════════════════════════
fa_col, si_col = st.columns([3, 1])

with fa_col:
    if focus_areas:
        fa_rows = ''
        for idx, fa in enumerate(focus_areas[:6]):
            bg = '#FAFAFA' if idx % 2 == 0 else '#FFFFFF'
            part_html = f'<span style="color:#888;font-size:0.78rem;">{fa["part"]}</span>' if fa['part'] else ''
            cat_badge = (f'<span style="background:{fa["cat_color"]};color:white;padding:2px 8px;'
                         f'border-radius:10px;font-size:0.7rem;white-space:nowrap;">{fa["category"]}</span>')
            _tr = fa.get('trend', '')
            if _tr == 'NEW':
                trend_html = '<span style="color:#2980B9;font-weight:700;font-size:0.75rem;">NEW</span>'
            elif _tr.startswith('+'):
                trend_html = f'<span style="color:#C0392B;font-weight:700;font-size:0.8rem;">&#9650; {_tr}</span>'
            elif _tr.startswith('-'):
                trend_html = f'<span style="color:#27AE60;font-weight:700;font-size:0.8rem;">&#9660; {_tr}</span>'
            else:
                trend_html = '<span style="color:#999;font-size:0.75rem;">—</span>'
            fa_rows += (f'<tr style="background:{bg};">'
                        f'<td style="padding:6px 10px;font-weight:600;color:#1B2A4A;font-size:0.85rem;white-space:nowrap;">{fa["product"]}</td>'
                        f'<td style="padding:6px 10px;font-size:0.83rem;">{fa["issue"]}</td>'
                        f'<td style="padding:6px 10px;text-align:center;white-space:nowrap;">{part_html}</td>'
                        f'<td style="padding:6px 10px;text-align:center;white-space:nowrap;">{cat_badge}</td>'
                        f'<td style="padding:6px 10px;text-align:center;font-weight:700;color:#1B2A4A;width:1%;white-space:nowrap;">{fa["cases"]}</td>'
                        f'<td style="padding:6px 10px;text-align:center;font-size:0.85rem;width:1%;white-space:nowrap;">{fa["pct"]:.1f}%</td>'
                        f'<td style="padding:6px 10px;text-align:center;width:1%;white-space:nowrap;">{trend_html}</td>'
                        f'</tr>')

        st.markdown(f'''<div style="background:white;border:1px solid #E0E0E0;border-radius:8px;overflow:hidden;">
            <div style="background:#1B2A4A;color:white;padding:10px 14px;font-size:0.9rem;font-weight:600;">
                Where to Focus — Top Product-Issue Combinations
            </div>
            <table style="width:100%;border-collapse:collapse;table-layout:auto;">
                <tr style="background:#F0EDE8;border-bottom:2px solid #C6B5A1;">
                    <td style="padding:6px 10px;font-size:0.75rem;color:#666;font-weight:600;white-space:nowrap;">PRODUCT</td>
                    <td style="padding:6px 10px;font-size:0.75rem;color:#666;font-weight:600;">ISSUE</td>
                    <td style="padding:6px 10px;font-size:0.75rem;color:#666;font-weight:600;text-align:center;width:1%;white-space:nowrap;">PART</td>
                    <td style="padding:6px 10px;font-size:0.75rem;color:#666;font-weight:600;text-align:center;width:1%;white-space:nowrap;">CATEGORY</td>
                    <td style="padding:6px 10px;font-size:0.75rem;color:#666;font-weight:600;text-align:center;width:1%;white-space:nowrap;">CASES</td>
                    <td style="padding:6px 10px;font-size:0.75rem;color:#666;font-weight:600;text-align:center;width:1%;white-space:nowrap;">%</td>
                    <td style="padding:6px 10px;font-size:0.75rem;color:#666;font-weight:600;text-align:center;width:1%;white-space:nowrap;">TREND</td>
                </tr>
                {fa_rows}
            </table>
        </div>''', unsafe_allow_html=True)

with si_col:
    if spikes:
        spike_html = ''.join(f'<div style="padding:3px 0;font-size:0.82rem;">'
                             f'{s["cat"]} <span style="color:#C0392B;font-weight:600;">+{s["pct"]:.0f}%</span> '
                             f'<span style="color:#999;">({s["prev"]}&rarr;{s["curr"]})</span></div>' for s in spikes[:3])
        st.markdown(f'''<div style="background:#FFF3F3;border:1px solid #FFD0D0;border-radius:8px;padding:12px;margin-bottom:8px;">
            <div style="font-size:0.72rem;color:#888;text-transform:uppercase;letter-spacing:0.5px;">&#9650; Spiking</div>
            {spike_html}
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'''<div style="background:#F0FFF4;border:1px solid #C6F0D0;border-radius:8px;padding:12px;margin-bottom:8px;">
            <div style="font-size:0.72rem;color:#888;text-transform:uppercase;letter-spacing:0.5px;">&#9650; Spiking</div>
            <div style="font-size:0.85rem;color:#27AE60;margin-top:2px;">No spikes</div>
        </div>''', unsafe_allow_html=True)

    if improving:
        imp_html = ''.join(f'<div style="padding:3px 0;font-size:0.82rem;">'
                           f'{d["cat"]} <span style="color:#27AE60;font-weight:600;">{d["pct"]:.0f}%</span> '
                           f'<span style="color:#999;">({d["prev"]}&rarr;{d["curr"]})</span></div>' for d in improving[:3])
        st.markdown(f'''<div style="background:#F0FFF4;border:1px solid #C6F0D0;border-radius:8px;padding:12px;">
            <div style="font-size:0.72rem;color:#888;text-transform:uppercase;letter-spacing:0.5px;">&#9660; Improving</div>
            {imp_html}
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'''<div style="background:#FAFAFA;border:1px solid #E0E0E0;border-radius:8px;padding:12px;">
            <div style="font-size:0.72rem;color:#888;text-transform:uppercase;letter-spacing:0.5px;">&#9660; Improving</div>
            <div style="font-size:0.85rem;color:#888;margin-top:2px;">No declining categories</div>
        </div>''', unsafe_allow_html=True)

st.markdown("")

# ═══════════════════════════════════════
# Pareto Analysis
# ═══════════════════════════════════════
section_divider()
with st.expander("Pareto Analysis — Top Issues", expanded=True):
    issue_counts_all = df_filtered['IssueDescription'].value_counts()
    issue_counts = issue_counts_all.head(15)
    total_issues = max(issue_counts_all.sum(), 1)
    issue_labels = [str(x).split(' ', 1)[1][:40] if ' ' in str(x) else str(x)[:40] for x in issue_counts.index]
    cumulative_pct = issue_counts.cumsum() / total_issues * 100
    individual_pct = issue_counts / total_issues * 100

    cutoff_idx = 0
    for i, cp in enumerate(cumulative_pct.values):
        if cp >= 80:
            cutoff_idx = i; break
    else:
        cutoff_idx = len(cumulative_pct) - 1

    n_80 = cutoff_idx + 1
    pct_of_types = n_80 / len(issue_counts_all) * 100

    _top3_labels = issue_labels[:min(3, len(issue_labels))]
    _top3_cases = issue_counts.values[:min(3, len(issue_counts))]
    _top3_pcts = individual_pct.values[:min(3, len(individual_pct))]
    _top3_html = ''.join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
        f'<span style="background:rgba(198,181,161,0.3);color:#C6B5A1;font-weight:700;font-size:1.1rem;'
        f'width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;">#{i+1}</span>'
        f'<span style="color:#fff;font-size:0.85rem;flex:1;">{lbl}</span>'
        f'<span style="color:#C6B5A1;font-weight:700;font-size:0.9rem;">{cnt} ({pct:.1f}%)</span></div>'
        for i, (lbl, cnt, pct) in enumerate(zip(_top3_labels, _top3_cases, _top3_pcts))
    )
    st.markdown(f'''<div style="background:linear-gradient(135deg,#1B2A4A 0%,#2C3E6A 100%);
        border-radius:10px;padding:18px 28px;margin:0 0 16px 0;display:flex;align-items:stretch;gap:30px;">
        <div style="text-align:center;min-width:80px;">
            <div style="color:#C6B5A1;font-size:2.8rem;font-weight:700;line-height:1;">{n_80}</div>
            <div style="color:#FFFFFF;font-size:0.78rem;opacity:0.8;">issue types</div>
        </div>
        <div style="color:#C6B5A1;font-size:2rem;font-weight:300;display:flex;align-items:center;">=</div>
        <div style="text-align:center;min-width:80px;">
            <div style="color:#FFFFFF;font-size:2.8rem;font-weight:700;line-height:1;">80%</div>
            <div style="color:#FFFFFF;font-size:0.78rem;opacity:0.8;">of all cases</div>
        </div>
        <div style="flex:0 0 1px;background:rgba(255,255,255,0.15);margin:0 8px;"></div>
        <div style="flex:1;padding-left:10px;">
            <div style="color:#C6B5A1;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">Top Contributors</div>
            {_top3_html}
        </div>
        <div style="flex:0 0 1px;background:rgba(255,255,255,0.15);margin:0 8px;"></div>
        <div style="display:flex;align-items:center;min-width:180px;">
            <div style="color:#FFFFFF;font-size:0.85rem;opacity:0.85;line-height:1.5;">
                Just <b style="color:#C6B5A1;">{pct_of_types:.0f}%</b> of issue types
                ({n_80} of {len(issue_counts_all)}) drive <b style="color:#C6B5A1;">80%</b> of volume.
            </div>
        </div>
    </div>''', unsafe_allow_html=True)

    # Pareto chart
    _bar_colors = []
    for i in range(len(issue_counts)):
        if i <= cutoff_idx:
            _fade = i / max(cutoff_idx, 1) * 0.4
            _r = int(27 + 60 * _fade); _g = int(42 + 70 * _fade); _b = int(74 + 80 * _fade)
            _bar_colors.append(f'rgb({_r},{_g},{_b})')
        else:
            _bar_colors.append('#D0D0D0')

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=issue_labels, y=issue_counts.values,
        marker=dict(color=_bar_colors, line=dict(width=0)),
        text=[f'{v}' for v in issue_counts.values],
        textposition='outside', textfont=dict(size=11, color=NAVY, family='Calibri'),
        customdata=individual_pct.values, name='Cases', showlegend=False,
    ), secondary_y=False)

    _cum_text = []
    for i, cp in enumerate(cumulative_pct.values):
        if i == 0 or i == cutoff_idx or i == len(cumulative_pct) - 1 or abs(cp - 50) < 5:
            _cum_text.append(f'{cp:.0f}%')
        else:
            _cum_text.append('')
    fig.add_trace(go.Scatter(
        x=issue_labels, y=cumulative_pct.values,
        mode='lines+markers+text', fill='tozeroy',
        fillcolor='rgba(192, 57, 43, 0.06)',
        line=dict(color=RED, width=2.5), marker=dict(size=6, color=RED),
        text=_cum_text, textposition='top center',
        textfont=dict(size=11, color=RED, family='Calibri'),
        name='Cumulative %', showlegend=False,
    ), secondary_y=True)

    fig.add_vrect(x0=-0.5, x1=cutoff_idx + 0.5, fillcolor='rgba(27,42,74,0.04)', line_width=0, layer='below')
    fig.add_hline(y=80, line_dash='dash', line_color=GOLD, line_width=2,
                   annotation_text=f'80% threshold  (top {n_80} issues)',
                   annotation_font=dict(size=11, color=GOLD, family='Calibri'),
                   annotation_position='top right', secondary_y=True)
    fig.add_vline(x=cutoff_idx + 0.5, line_dash='dot', line_color=GOLD, line_width=1.5, opacity=0.5)

    plotly_layout(fig, None, 500)
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=9, family='Calibri'))
    fig.update_yaxes(title_text='Cases', title_font=dict(size=11), secondary_y=False, gridcolor='rgba(0,0,0,0.06)')
    fig.update_yaxes(title_text='Cumulative %', range=[0, 108], secondary_y=True, showgrid=False, title_font=dict(size=11))
    fig.update_layout(margin=dict(l=50, r=50, t=30, b=110),
                     hoverlabel=dict(bgcolor=NAVY, font_color='white', font_size=12))
    st.plotly_chart(fig, width="stretch")

# ═══════════════════════════════════════
# Category Breakdown
# ═══════════════════════════════════════
section_divider()
st.markdown("### Category Breakdown — Top Issues & Products")
_exec_cat_cols = st.columns(3)
_cat_items = [(k, v) for k, v in CAT_SHORT.items() if len(df_filtered[df_filtered['IssueCategory'] == k]) > 0]
_cat_items.sort(key=lambda x: len(df_filtered[df_filtered['IssueCategory'] == x[0]]), reverse=True)
for _ci, (_cat_key, _cat_short) in enumerate(_cat_items):
    _df_cat_vis = df_filtered[df_filtered['IssueCategory'] == _cat_key]
    _cat_cnt = len(_df_cat_vis)
    _cat_pct = _cat_cnt / total * 100 if total > 0 else 0
    _cat_color = CAT_COLORS.get(_cat_key, GRAY)
    _top_iss = _df_cat_vis['IssueShort'].value_counts().head(3) if 'IssueShort' in _df_cat_vis.columns else _df_cat_vis['IssueDescription'].value_counts().head(3)
    _top_iss = _top_iss[~_top_iss.index.isin(['Unknown', 'nan', ''])]
    _iss_html = ''
    for _iss_name, _iss_cnt in _top_iss.items():
        _bar_w = min(int(_iss_cnt / max(_top_iss.iloc[0], 1) * 100), 100)
        _iss_html += (f'<div style="margin:3px 0;display:flex;align-items:center;gap:6px;">'
                      f'<div style="flex:1;font-size:0.75rem;color:#444;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{str(_iss_name)[:30]}</div>'
                      f'<div style="width:60px;background:#EEE;border-radius:3px;height:10px;">'
                      f'<div style="width:{_bar_w}%;background:{_cat_color};height:10px;border-radius:3px;"></div></div>'
                      f'<div style="font-size:0.72rem;font-weight:600;color:{NAVY};min-width:20px;text-align:right;">{_iss_cnt}</div>'
                      f'</div>')
    _top_prod = _df_cat_vis['ProductFamily'].value_counts()
    _top_prod = _top_prod[~_top_prod.index.isin(['Other', 'Unknown'])].head(3)
    _prod_html = ''
    for _pn, _pc in _top_prod.items():
        _prod_html += f'<span style="display:inline-block;background:#F0EDE8;border-radius:12px;padding:2px 8px;margin:2px 3px;font-size:0.7rem;color:{NAVY};font-weight:600;">{_pn} ({_pc})</span>'
    with _exec_cat_cols[_ci % 3]:
        st.markdown(f'''<div style="background:white;border:1px solid #E0E0E0;border-top:3px solid {_cat_color};
            border-radius:8px;padding:12px 14px;margin-bottom:10px;min-height:180px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="font-weight:700;color:{NAVY};font-size:0.88rem;">{_cat_short}</span>
                <span style="background:{_cat_color};color:white;padding:2px 10px;border-radius:12px;font-size:0.78rem;font-weight:600;">{_cat_cnt} ({_cat_pct:.0f}%)</span>
            </div>
            <div style="font-size:0.68rem;color:#888;text-transform:uppercase;margin-bottom:4px;">Top Issues</div>
            {_iss_html}
            <div style="font-size:0.68rem;color:#888;text-transform:uppercase;margin-top:8px;margin-bottom:4px;">Top Products</div>
            <div>{_prod_html if _prod_html else '<span style="font-size:0.75rem;color:#CCC;">—</span>'}</div>
        </div>''', unsafe_allow_html=True)

# ═══════════════════════════════════════
# Top Products Heatmap
# ═══════════════════════════════════════
section_divider()
st.markdown("### Top Products Heatmap")
_exec_top_prods = df_filtered[~df_filtered['ProductFamily'].isin(['Other', 'Unknown'])]
if len(_exec_top_prods) > 0:
    _exec_prod_items = _exec_top_prods['ProductFamily'].value_counts().head(12).index.tolist()
    _exec_hm_prod_rows = []
    for _ep in _exec_prod_items:
        _emonthly = [len(_exec_top_prods[(_exec_top_prods['Month'] == m) & (_exec_top_prods['ProductFamily'] == _ep)]) for m in months_present]
        _exec_hm_prod_rows.append({'label': _ep[:40], 'monthly': _emonthly, 'total': sum(_emonthly)})
    if _exec_hm_prod_rows:
        _exec_max_val_p = max(v for r in _exec_hm_prod_rows for v in r['monthly']) if _exec_hm_prod_rows else 1
        _ephdr = (f'<th style="background:{NAVY};color:#fff;padding:8px 12px;font-size:0.82rem;'
                  f'text-align:left;border-bottom:2px solid {GOLD};min-width:200px;">Product</th>')
        for _em in months_present:
            _ephdr += (f'<th style="background:{NAVY};color:#fff;padding:8px 10px;font-size:0.82rem;'
                       f'text-align:center;border-bottom:2px solid {GOLD};">{_em}</th>')
        _ephdr += (f'<th style="background:{NAVY};color:#fff;padding:8px 10px;font-size:0.82rem;'
                   f'text-align:center;border-bottom:2px solid {GOLD};">Total</th>')
        _ephdr += (f'<th style="background:{NAVY};color:#fff;padding:8px 10px;font-size:0.82rem;'
                   f'text-align:center;border-bottom:2px solid {GOLD};">Trend</th>')
        _epbody = ''
        for _epi, _epr in enumerate(_exec_hm_prod_rows):
            _epbg = '#FFFFFF' if _epi % 2 == 0 else '#F7F6F4'
            _epcells = (f'<td style="padding:7px 12px;font-weight:600;color:{NAVY};font-size:0.82rem;'
                        f'border-bottom:1px solid #EBE8E4;white-space:nowrap;">{_epr["label"]}</td>')
            for _epv in _epr['monthly']:
                if _epv == 0:
                    _epcell_bg = _epbg
                    _eptxt_color = '#CCC'
                else:
                    _epintensity = min(_epv / max(_exec_max_val_p, 1), 1.0)
                    _epred = int(255 - 63 * _epintensity)
                    _epgreen = int(235 - 195 * _epintensity)
                    _epblue = int(200 - 160 * _epintensity)
                    _epcell_bg = f'rgb({_epred},{_epgreen},{_epblue})'
                    _eptxt_color = '#1B2A4A' if _epintensity < 0.7 else '#FFFFFF'
                _epcells += (f'<td style="padding:7px 8px;text-align:center;font-size:0.85rem;font-weight:600;'
                             f'background:{_epcell_bg};color:{_eptxt_color};border-bottom:1px solid #EBE8E4;">'
                             f'{_epv if _epv > 0 else "·"}</td>')
            _epcells += (f'<td style="padding:7px 10px;text-align:center;font-weight:700;font-size:0.85rem;'
                         f'color:{NAVY};border-bottom:1px solid #EBE8E4;border-left:2px solid {GOLD};">{_epr["total"]}</td>')
            _epvals = _epr['monthly']
            if len(_epvals) >= 2 and _epvals[-2] > 0:
                _epchg = _epvals[-1] - _epvals[-2]
                if _epchg > 0:
                    _eptrend = f'<span style="color:#C0392B;font-size:1rem;">&#9650;</span> <span style="font-size:0.75rem;color:#C0392B;">+{_epchg}</span>'
                elif _epchg < 0:
                    _eptrend = f'<span style="color:#27AE60;font-size:1rem;">&#9660;</span> <span style="font-size:0.75rem;color:#27AE60;">{_epchg}</span>'
                else:
                    _eptrend = '<span style="color:#888;">&#8212;</span>'
            else:
                _eptrend = '<span style="color:#CCC;">—</span>'
            _epcells += f'<td style="padding:7px 8px;text-align:center;border-bottom:1px solid #EBE8E4;">{_eptrend}</td>'
            _epbody += f'<tr style="background:{_epbg};">{_epcells}</tr>'
        st.markdown(f'''<div style="overflow-x:auto;border-radius:8px;border:1px solid #E2DAD0;margin:4px 0;">
            <table style="width:100%;border-collapse:collapse;">
            <thead><tr>{_ephdr}</tr></thead>
            <tbody>{_epbody}</tbody>
            </table></div>''', unsafe_allow_html=True)
        st.caption("Color intensity = case count. Arrows show change from previous month.")

# ═══════════════════════════════════════
# Top Issues Heatmap
# ═══════════════════════════════════════
section_divider()
st.markdown("### Top Issues Heatmap")
_exec_iss_vc = df_filtered['IssueShort'].value_counts() if 'IssueShort' in df_filtered.columns else df_filtered['IssueDescription'].value_counts()
_exec_iss_vc = _exec_iss_vc[~_exec_iss_vc.index.isin(['Unknown', 'nan', ''])]
_exec_top_iss = _exec_iss_vc.head(15)
_field = 'IssueShort' if 'IssueShort' in df_filtered.columns else 'IssueDescription'
_exec_hm_rows = []
for _iss in _exec_top_iss.index:
    _monthly = [len(df_filtered[(df_filtered['Month'] == m) & (df_filtered[_field] == _iss)]) for m in months_present]
    _exec_hm_rows.append({'issue': str(_iss)[:40], 'monthly': _monthly, 'total': sum(_monthly)})
if _exec_hm_rows:
    _exec_max_val = max(v for r in _exec_hm_rows for v in r['monthly']) if _exec_hm_rows else 1
    _ehdr = (f'<th style="background:{NAVY};color:#fff;padding:8px 12px;font-size:0.82rem;'
             f'text-align:left;border-bottom:2px solid {GOLD};min-width:200px;">Issue</th>')
    for _em in months_present:
        _ehdr += (f'<th style="background:{NAVY};color:#fff;padding:8px 10px;font-size:0.82rem;'
                  f'text-align:center;border-bottom:2px solid {GOLD};">{_em}</th>')
    _ehdr += (f'<th style="background:{NAVY};color:#fff;padding:8px 10px;font-size:0.82rem;'
              f'text-align:center;border-bottom:2px solid {GOLD};">Total</th>')
    _ehdr += (f'<th style="background:{NAVY};color:#fff;padding:8px 10px;font-size:0.82rem;'
              f'text-align:center;border-bottom:2px solid {GOLD};">Trend</th>')
    _ebody = ''
    for _ei, _er in enumerate(_exec_hm_rows):
        _ebg = '#FFFFFF' if _ei % 2 == 0 else '#F7F6F4'
        _ecells = (f'<td style="padding:7px 12px;font-weight:600;color:{NAVY};font-size:0.82rem;'
                   f'border-bottom:1px solid #EBE8E4;white-space:nowrap;">{_er["issue"]}</td>')
        for _ev in _er['monthly']:
            if _ev == 0:
                _ecell_bg = _ebg
                _etxt_color = '#CCC'
            else:
                _eintensity = min(_ev / max(_exec_max_val, 1), 1.0)
                _ered = int(255 - 63 * _eintensity)
                _egreen = int(235 - 195 * _eintensity)
                _eblue = int(200 - 160 * _eintensity)
                _ecell_bg = f'rgb({_ered},{_egreen},{_eblue})'
                _etxt_color = '#1B2A4A' if _eintensity < 0.7 else '#FFFFFF'
            _ecells += (f'<td style="padding:7px 8px;text-align:center;font-size:0.85rem;font-weight:600;'
                        f'background:{_ecell_bg};color:{_etxt_color};border-bottom:1px solid #EBE8E4;">'
                        f'{_ev if _ev > 0 else "·"}</td>')
        _ecells += (f'<td style="padding:7px 10px;text-align:center;font-weight:700;font-size:0.85rem;'
                    f'color:{NAVY};border-bottom:1px solid #EBE8E4;border-left:2px solid {GOLD};">{_er["total"]}</td>')
        _evals = _er['monthly']
        if len(_evals) >= 2 and _evals[-2] > 0:
            _echg = _evals[-1] - _evals[-2]
            if _echg > 0:
                _etrend = f'<span style="color:#C0392B;font-size:1rem;">&#9650;</span> <span style="font-size:0.75rem;color:#C0392B;">+{_echg}</span>'
            elif _echg < 0:
                _etrend = f'<span style="color:#27AE60;font-size:1rem;">&#9660;</span> <span style="font-size:0.75rem;color:#27AE60;">{_echg}</span>'
            else:
                _etrend = '<span style="color:#888;">&#8212;</span>'
        else:
            _etrend = '<span style="color:#CCC;">—</span>'
        _ecells += f'<td style="padding:7px 8px;text-align:center;border-bottom:1px solid #EBE8E4;">{_etrend}</td>'
        _ebody += f'<tr style="background:{_ebg};">{_ecells}</tr>'
    st.markdown(f'''<div style="overflow-x:auto;border-radius:8px;border:1px solid #E2DAD0;margin:4px 0;">
        <table style="width:100%;border-collapse:collapse;">
        <thead><tr>{_ehdr}</tr></thead>
        <tbody>{_ebody}</tbody>
        </table></div>''', unsafe_allow_html=True)
    st.caption("Color intensity = case count. Arrows show change from previous month.")

# ═══════════════════════════════════════
# Case Timing vs Delivery Analysis
# ═══════════════════════════════════════
section_divider()

_timing_data = None
if os.path.exists(_TIMING_PKL):
    try:
        with open(_TIMING_PKL, 'rb') as _tf:
            _timing_data = pickle.load(_tf)
    except Exception:
        pass

if _timing_data:
    _t = _timing_data
    _match_pct = _t['matched'] / max(_t['total'], 1) * 100

    _cls_colors = {'At Delivery': BLUE, 'Within 6 Months': GREEN, '6-12 Months': ORANGE, 'Over 12 Months': RED, 'No Match': '#CCCCCC'}
    _cls_order = ['At Delivery', 'Within 6 Months', '6-12 Months', 'Over 12 Months', 'No Match']

    # Header with badge
    st.markdown(f'''<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:1.15rem;font-weight:700;color:{NAVY};">Case Timing vs Delivery Date</span>
            <span style="background:{GREEN};color:#fff;padding:3px 12px;border-radius:14px;font-size:0.72rem;font-weight:600;">
                {_match_pct:.0f}% matched</span>
        </div>
        <span style="font-size:0.72rem;color:#999;">{_t['total']:,} cases analysed</span>
    </div>''', unsafe_allow_html=True)

    # Donut charts + comparison table side by side
    _tc_chart, _tc_detail = st.columns([2, 3])

    with _tc_chart:
        _donut_fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}]],
                                   subplot_titles=[f'Warranty ({_t["warranty_total"]})', f'Delivery ({_t["delivery_total"]})'])

        for _di, (_d_cls, _d_tot, _d_mean, _d_med) in enumerate([
            (_t['warranty_cls'], _t['warranty_total'], _t['w_mean'], _t['w_med']),
            (_t['delivery_cls'], _t['delivery_total'], _t['d_mean'], _t['d_med']),
        ]):
            _d_labels = [c for c in _cls_order if _d_cls.get(c, 0) > 0]
            _d_values = [_d_cls.get(c, 0) for c in _d_labels]
            _d_colors = [_cls_colors[c] for c in _d_labels]
            _donut_fig.add_trace(go.Pie(
                labels=_d_labels, values=_d_values,
                marker=dict(colors=_d_colors),
                hole=0.55, textinfo='percent', textfont=dict(size=11, color='white'),
                hovertemplate='%{label}<br>%{value} cases (%{percent})<extra></extra>',
                sort=False,
            ), row=1, col=_di + 1)

        _donut_fig.add_annotation(text=f'<b>{_t["w_med"]}d</b><br><span style="font-size:9px">median</span>',
            x=0.19, y=0.5, font=dict(size=14, color=NAVY), showarrow=False, xref='paper', yref='paper')
        _donut_fig.add_annotation(text=f'<b>{_t["d_med"]}d</b><br><span style="font-size:9px">median</span>',
            x=0.81, y=0.5, font=dict(size=14, color=NAVY), showarrow=False, xref='paper', yref='paper')

        _donut_fig.update_layout(
            height=260, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Calibri', color='#323232'),
            showlegend=False,
        )
        st.plotly_chart(_donut_fig, width="stretch")

        # Shared legend
        _legend_html = '<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:6px 16px;margin-top:2px;">'
        for _cl in _cls_order:
            if _t['overall'].get(_cl, 0) > 0:
                _legend_html += (f'<span style="display:inline-flex;align-items:center;gap:4px;">'
                                 f'<span style="width:10px;height:10px;border-radius:50%;background:{_cls_colors[_cl]};"></span>'
                                 f'<span style="font-size:0.72rem;color:#555;">{_cl}</span></span>')
        _legend_html += '</div>'
        st.markdown(_legend_html, unsafe_allow_html=True)

    with _tc_detail:
        # Comparison table
        _tbl_rows = ''
        for _cl in _cls_order:
            _w_cnt = _t['warranty_cls'].get(_cl, 0)
            _d_cnt = _t['delivery_cls'].get(_cl, 0)
            _w_pct = _w_cnt / max(_t['warranty_total'], 1) * 100
            _d_pct = _d_cnt / max(_t['delivery_total'], 1) * 100
            _tot_cnt = _t['overall'].get(_cl, 0)
            _tot_pct = _tot_cnt / max(_t['total'], 1) * 100
            _clr = _cls_colors[_cl]
            _tbl_rows += f'''<tr style="border-bottom:1px solid #F0EDE8;">
                <td style="padding:7px 10px;white-space:nowrap;">
                    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{_clr};margin-right:6px;vertical-align:middle;"></span>
                    <span style="font-size:0.82rem;font-weight:600;color:{NAVY};">{_cl}</span>
                </td>
                <td style="padding:7px 10px;text-align:center;font-size:0.85rem;font-weight:700;color:{NAVY};">{_w_cnt}</td>
                <td style="padding:7px 10px;text-align:center;font-size:0.8rem;color:#888;">{_w_pct:.1f}%</td>
                <td style="padding:7px 10px;text-align:center;font-size:0.85rem;font-weight:700;color:{NAVY};">{_d_cnt}</td>
                <td style="padding:7px 10px;text-align:center;font-size:0.8rem;color:#888;">{_d_pct:.1f}%</td>
                <td style="padding:7px 10px;text-align:center;font-size:0.85rem;font-weight:700;color:{NAVY};">{_tot_cnt}</td>
                <td style="padding:7px 10px;text-align:center;font-size:0.8rem;color:#888;">{_tot_pct:.1f}%</td>
            </tr>'''

        st.markdown(f'''<div style="background:white;border:1px solid #E2DAD0;border-radius:10px;overflow:hidden;">
            <table style="width:100%;border-collapse:collapse;">
                <tr style="background:{NAVY};">
                    <td style="padding:8px 10px;font-size:0.72rem;color:{GOLD};font-weight:600;text-transform:uppercase;letter-spacing:0.3px;">Classification</td>
                    <td style="padding:8px 10px;text-align:center;font-size:0.72rem;color:{GOLD};font-weight:600;" colspan="2">Warranty ({_t['warranty_total']})</td>
                    <td style="padding:8px 10px;text-align:center;font-size:0.72rem;color:{GOLD};font-weight:600;" colspan="2">Delivery ({_t['delivery_total']})</td>
                    <td style="padding:8px 10px;text-align:center;font-size:0.72rem;color:{GOLD};font-weight:600;" colspan="2">Total ({_t['total']})</td>
                </tr>
                {_tbl_rows}
                <tr style="background:#F8F6F3;border-top:2px solid {GOLD};">
                    <td style="padding:8px 10px;font-size:0.78rem;font-weight:700;color:{NAVY};">Avg. Days to Case</td>
                    <td style="padding:8px 10px;text-align:center;font-size:0.92rem;font-weight:700;color:{RED};" colspan="2">{_t['w_mean']}d <span style="font-size:0.7rem;font-weight:400;color:#888;">(med: {_t['w_med']}d)</span></td>
                    <td style="padding:8px 10px;text-align:center;font-size:0.92rem;font-weight:700;color:{GREEN};" colspan="2">{_t['d_mean']}d <span style="font-size:0.7rem;font-weight:400;color:#888;">(med: {_t['d_med']}d)</span></td>
                    <td style="padding:8px 10px;" colspan="2"></td>
                </tr>
            </table>
        </div>''', unsafe_allow_html=True)

    # Products with latent issues
    if _t.get('prod_stats'):
        st.markdown(f'''<div style="margin-top:14px;">
            <div style="font-size:0.72rem;color:{GOLD};font-weight:600;text-transform:uppercase;
                letter-spacing:0.5px;margin-bottom:6px;">Products with Highest Latent Issues (Over 12 Months)</div>
        </div>''', unsafe_allow_html=True)
        _lat_html = ''
        for _ps in _t['prod_stats'][:6]:
            _sev_clr = RED if _ps['pct'] >= 40 else (ORANGE if _ps['pct'] >= 25 else NAVY)
            _bar_w = min(int(_ps['pct']), 100)
            _lat_html += f'''<div style="display:inline-block;vertical-align:top;background:white;
                border:1px solid #E0E0E0;border-left:4px solid {_sev_clr};border-radius:8px;
                padding:10px 14px;margin:4px 6px;min-width:170px;max-width:200px;">
                <div style="font-size:0.82rem;font-weight:700;color:{NAVY};">{_ps['product']}</div>
                <div style="display:flex;align-items:baseline;gap:6px;margin:4px 0;">
                    <span style="font-size:1.3rem;font-weight:700;color:{_sev_clr};">{_ps['over12']}</span>
                    <span style="font-size:0.72rem;color:#888;">of {_ps['total']} ({_ps['pct']:.0f}%)</span>
                </div>
                <div style="width:100%;height:5px;background:#EEE;border-radius:3px;">
                    <div style="width:{_bar_w}%;height:5px;background:{_sev_clr};border-radius:3px;"></div>
                </div>
                <div style="font-size:0.68rem;color:#AAA;margin-top:3px;">median {_ps['median_days']}d after delivery</div>
            </div>'''
        st.markdown(f'<div style="overflow-x:auto;white-space:nowrap;">{_lat_html}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════
# Corrective Action Plan (Read-Only)
# ═══════════════════════════════════════
section_divider()

_action_plan = []
if os.path.exists(_ACTION_PLAN_FILE):
    try:
        with open(_ACTION_PLAN_FILE, 'r') as _apf:
            _action_plan = json.load(_apf)
    except Exception:
        pass

if _action_plan:
    st.markdown(f'''<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
        <span style="font-size:1.15rem;font-weight:700;color:{NAVY};">Corrective Action Plan</span>
        <span style="background:{GOLD};color:{NAVY};padding:3px 12px;border-radius:14px;font-size:0.72rem;font-weight:600;">
            {len(_action_plan)} items</span>
    </div>''', unsafe_allow_html=True)

    _STATUS_COLORS = {
        'Implemented': ('#27AE60', '#FFFFFF'), 'In Process': ('#E67E22', '#FFFFFF'),
        'Under Investigation': ('#F1948A', '#FFFFFF'),
    }
    _CAT_CELL_COLORS = {
        'Structural': '#C0392B', 'Fabric': '#E67E22', 'Delivery': '#2980B9',
        'Electrical': '#16A085', 'Timber': '#C6B5A1', 'Leather': '#8E44AD',
    }
    _STATUS_ORDER = {'Implemented': 0, 'In Process': 1, 'Under Investigation': 2}
    _display_cols = ['Date Added', 'Product', 'Issue', 'Category', 'Root Cause',
                     'Corrective Action', 'Date Implemented', 'Owner', 'Due Date', 'Notes', 'Status']
    _col_w = {'Date Added': '7%', 'Product': '8%', 'Issue': '8%', 'Category': '6%',
              'Root Cause': '14%', 'Corrective Action': '16%', 'Date Implemented': '7%',
              'Owner': '6%', 'Due Date': '6%', 'Notes': '12%', 'Status': '8%'}
    _cell_base = 'padding:6px 6px;font-size:0.78rem;color:#323232;border-bottom:1px solid #EBE8E4;overflow:hidden;word-wrap:break-word;'

    # Sort by status, then by Date Implemented (most recent first)
    def _parse_impl_date_ordinal(row):
        val = str(row.get('Date Implemented', '') or '').strip()
        if not val:
            return 0
        for fmt in ('%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d %b %Y', '%d %B %Y'):
            try:
                return datetime.strptime(val, fmt).toordinal()
            except ValueError:
                continue
        return 0
    _sorted_plan = sorted(range(len(_action_plan)),
                          key=lambda i: (_STATUS_ORDER.get(_action_plan[i].get('Status') or '', 3),
                                         -_parse_impl_date_ordinal(_action_plan[i])))

    # Header row
    _hdr_cells = ''
    for _col in _display_cols:
        _hdr_cells += (f'<div style="width:{_col_w[_col]};padding:8px 6px;color:#FFF;font-size:0.76rem;'
                       f'font-weight:600;white-space:nowrap;">{_col}</div>')
    st.markdown(f'<div style="display:flex;background:#1B2A4A;border-bottom:2px solid #C6B5A1;'
                f'border-radius:6px 6px 0 0;">{_hdr_cells}</div>', unsafe_allow_html=True)

    # Data rows
    _even_bg, _odd_bg = '#FFFFFF', '#F7F6F4'
    for _ri, _si in enumerate(_sorted_plan):
        _r = _action_plan[_si]
        _bg = _even_bg if _ri % 2 == 0 else _odd_bg
        _cells_html = ''
        for _col in _display_cols:
            _val = str(_r.get(_col, '') or '').strip()
            if _col == 'Status':
                _sbg, _sfg = _STATUS_COLORS.get(_val, ('#C0392B', '#FFFFFF'))
                _slbl = _val if _val else 'Not Started'
                _cells_html += (f'<div style="width:{_col_w[_col]};padding:6px 4px;text-align:center;'
                                f'font-size:0.76rem;font-weight:600;background:{_sbg};color:{_sfg};'
                                f'border-bottom:1px solid #EBE8E4;border-radius:3px;white-space:nowrap;">{_slbl}</div>')
            elif _col == 'Category':
                _cc = _CAT_CELL_COLORS.get(_val, '#1B2A4A')
                _cells_html += (f'<div style="width:{_col_w[_col]};padding:6px 4px;text-align:center;'
                                f'font-size:0.76rem;font-weight:600;background:{_cc};color:#FFF;'
                                f'border-bottom:1px solid #EBE8E4;border-radius:3px;white-space:nowrap;">{_val or "—"}</div>')
            elif _col == 'Due Date' and _val:
                _dd_overdue = False
                try:
                    for _fmt in ('%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y'):
                        try:
                            _dd_overdue = datetime.strptime(_val, _fmt).date() < datetime.now().date()
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass
                _dd_c = '#C0392B' if _dd_overdue else '#323232'
                _dd_w = 'font-weight:700;' if _dd_overdue else ''
                _cells_html += (f'<div style="width:{_col_w[_col]};{_cell_base}color:{_dd_c};{_dd_w}">{_val}</div>')
            else:
                _cells_html += (f'<div style="width:{_col_w[_col]};{_cell_base}">{_val or "—"}</div>')
        st.markdown(f'<div style="display:flex;background:{_bg};align-items:stretch;min-height:40px;">'
                    f'{_cells_html}</div>', unsafe_allow_html=True)
