import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AutoML Decision Engine", page_icon="🤖", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .success-box { padding:15px; background-color:#d4edda; color:#155724; border-radius:10px; margin-bottom: 20px;}
    .azure-box { padding:15px; background-color:#cce5ff; color:#004085; border-radius:10px; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_data(file):
    """Caches the dataset so it doesn't reload on every button click."""
    return pd.read_csv(file)

def check_seasonality(series, frequency='Daily'):
    """
    Intelligently checks for seasonality using Autocorrelation (ACF) based on data frequency.
    """
    try:
        clean_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(clean_series) < 36: # Ensure enough data points exist
            return False 
        
        acf_values = acf(clean_series, nlags=55, fft=True)
        
        if frequency == 'Daily':
            return (acf_values[7] > 0.3) or (any(acf_values[28:32] > 0.3))
        elif frequency == 'Weekly':
            return acf_values[52] > 0.3 if len(acf_values) > 52 else False
        elif frequency == 'Monthly':
            return acf_values[12] > 0.3 if len(acf_values) > 12 else False
            
        return False
    except Exception:
        return False

def analyze_dataset(df, target_col, date_col, entity_col, horizon_days, frequency):
    """
    The Core Decision Logic Gates.
    """
    reasons = []
    score_azure = 0
    
    # --- GATE 1: MULTIVARIATE COMPLEXITY ---
    feature_cols = [c for c in df.columns if c not in [target_col, date_col, entity_col]]
    if len(feature_cols) > 2:
        score_azure += 1
        reasons.append(f"📊 **Multivariate Data Detected:** Found {len(feature_cols)} extra features (e.g., {', '.join(feature_cols[:2])}). Power BI works best with simple trends; Azure handles complex correlations better.")
    
    # --- GATE 2: GRANULARITY (MANY MODELS) ---
    if entity_col != "None" and entity_col in df.columns:
        unique_count = df[entity_col].nunique()
        if unique_count > 10:
            score_azure += 1
            reasons.append(f"🏪 **High Granularity:** You are forecasting {unique_count} distinct entities in '{entity_col}'. Training separate models simultaneously requires Azure's 'Many Models' accelerator.")

    # --- GATE 3: DATA VOLUME ---
    if len(df) > 500000:
        score_azure += 1
        reasons.append(f"💾 **High Volume:** Dataset has {len(df):,} rows. Power BI may hit timeout limits during training.")

    # --- GATE 4: HISTORY vs HORIZON RATIO ---
    try:
        df_dates = pd.to_datetime(df[date_col])
        history_days = (df_dates.max() - df_dates.min()).days
        if history_days > 0:
            ratio = horizon_days / history_days
            if ratio > 0.25: 
                score_azure += 1
                reasons.append(f"🔭 **Long Horizon:** Predicting {horizon_days} days ahead with only {history_days} days of history requires Azure's Deep Learning (Prophet/TCN) for stability.")
    except Exception as e:
        reasons.append(f"⚠️ Could not calculate Date logic: Please ensure your date column is formatted correctly.")

    # --- GATE 5: SEASONALITY ---
    is_seasonal = check_seasonality(df[target_col], frequency)
    if is_seasonal and score_azure > 0:
        reasons.append(f"🌊 **Complex Seasonality:** Strong recurring {frequency.lower()} patterns detected alongside other complexities.")

    return score_azure, reasons

# --- MAIN APP LAYOUT ---

st.title("🤖 Intelligent AutoML Decision Engine")
st.markdown("Upload your time-series dataset to find the best tool: **Power BI** or **Azure ML**.")
st.divider()

# 1. FILE UPLOADER
uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file:
    try:
        df = load_data(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Configuration")
            all_cols = df.columns.tolist()
            
            # Selectors
            date_col = st.selectbox("Select Date Column", all_cols)
            target_col = st.selectbox("Select Target (Prediction) Column", [c for c in all_cols if c != date_col])
            entity_col = st.selectbox("Group By (Optional - e.g., Store ID)", ["None"] + [c for c in all_cols if c not in [date_col, target_col]])
            
            # Forecasting Parameters
            frequency = st.selectbox("Data Frequency", ["Daily", "Weekly", "Monthly"])
            horizon = st.number_input("Forecast Horizon (Days/Periods)", min_value=1, value=30)
            
            run_btn = st.button("Analyze Dataset", type="primary", use_container_width=True)

        with col2:
            st.subheader("📋 Data Preview & Trend")
            st.dataframe(df.head(4), use_container_width=True)
            
            # Simple EDA Line Chart
            try:
                chart_data = df.groupby(date_col)[target_col].sum().reset_index()
                chart_data = chart_data.set_index(date_col)
                st.line_chart(chart_data)
            except Exception:
                st.caption("Could not render time-series chart. Ensure Target column is numeric.")
                
            st.caption(f"Total Rows: {len(df):,} | Total Columns: {len(df.columns)}")

        # 2. RUN LOGIC
        if run_btn:
            st.divider()
            with st.spinner('Running logic gates...'):
                score, reasons = analyze_dataset(df, target_col, date_col, entity_col, horizon, frequency)
                
                st.header("🎯 Recommendation")
                
                if score >= 1:
                    st.markdown(f"""
                        <div class="azure-box">
                            <h2>🔵 Recommended Tool: Azure Machine Learning</h2>
                            <p><strong>Why?</strong> Your dataset triggers <strong>{score} complexity flags</strong> that exceed Power BI's standard capabilities.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="success-box">
                            <h2>📊 Recommended Tool: Power BI AutoML</h2>
                            <p><strong>Why?</strong> Your dataset is clean, univariate, and fits well within the low-code, budget-friendly capabilities of Power BI.</p>
                        </div>
                    """, unsafe_allow_html=True)

                st.subheader("📝 Technical Analysis")
                if not reasons:
                    st.info("✅ Simple Univariate Data detected. No complex external factors or extreme granularity found.")
                else:
                    for r in reasons:
                        st.write(r)

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("👋 Waiting for file upload...")