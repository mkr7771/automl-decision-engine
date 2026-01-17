import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AutoML Decision Engine", page_icon="🤖", layout="wide")

# --- CSS STYLING (For visual appeal) ---
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .success-box { padding:15px; background-color:#d4edda; color:#155724; border-radius:10px; }
    .azure-box { padding:15px; background-color:#cce5ff; color:#004085; border-radius:10px; }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def check_seasonality(series):
    """
    Intelligently checks for seasonality using Autocorrelation (ACF).
    Returns True if a strong pattern (like weekly/monthly) is detected.
    """
    try:
        # Drop NaNs and ensure numeric
        clean_series = pd.to_numeric(series, errors='coerce').dropna()
        if len(clean_series) < 50: return False # Not enough data
        
        # Calculate ACF (correlation with past versions of itself)
        # We look at lag 7 (Weekly) or 30 (Monthly) for spikes
        acf_values = acf(clean_series, nlags=40, fft=True)
        
        # If correlation at lag 7 or 30 is high (> 0.3), seasonality exists
        if (acf_values[7] > 0.3) or (any(acf_values[28:32] > 0.3)):
            return True
        return False
    except:
        return False

def analyze_dataset(df, target_col, date_col, horizon_days):
    """
    The Core Decision Logic Gates.
    """
    reasons = []
    score_azure = 0
    
    # --- GATE 1: MULTIVARIATE COMPLEXITY ---
    # We count columns that are NOT the date or the target
    feature_cols = [c for c in df.columns if c not in [target_col, date_col]]
    
    # If there are more than 2 extra features (e.g., Price, Promo, Holiday) -> Azure
    if len(feature_cols) > 2:
        score_azure += 1
        reasons.append(f"📊 **Multivariate Data Detected:** Found {len(feature_cols)} extra features ({', '.join(feature_cols[:3])}...). Power BI works best with simple trends; Azure handles complex correlations better.")
    
    # --- GATE 2: GRANULARITY (MANY MODELS) ---
    # Check for categorical columns that might define different products/stores
    potential_ids = [c for c in feature_cols if df[c].nunique() > 1 and df[c].dtype == 'object']
    
    if len(potential_ids) > 0:
        # If any column has more than 10 unique items (e.g., 50 Stores), PBI struggles
        for col in potential_ids:
            if df[col].nunique() > 10:
                score_azure += 1
                reasons.append(f"🏪 **High Granularity:** The column '{col}' has {df[col].nunique()} unique items. Training {df[col].nunique()} separate models requires Azure's 'Many Models' accelerator.")
                break

    # --- GATE 3: DATA VOLUME ---
    if len(df) > 500000:
        score_azure += 1
        reasons.append(f"💾 **High Volume:** Dataset has {len(df):,} rows. Power BI may hit timeout limits during training.")

    # --- GATE 4: HISTORY vs HORIZON RATIO ---
    # Ensure date column is datetime
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        history_days = (df[date_col].max() - df[date_col].min()).days
        if history_days > 0:
            ratio = horizon_days / history_days
            if ratio > 0.25: # Predicting too far into the future
                score_azure += 1
                reasons.append(f"🔭 **Long Horizon:** You want to predict {horizon_days} days ahead, but only have {history_days} days of history. This requires Azure's Deep Learning (Prophet/TCN) for stability.")
    except Exception as e:
        reasons.append(f"⚠️ Could not calculate Date logic: {e}")

    # --- GATE 5: SEASONALITY ---
    is_seasonal = check_seasonality(df[target_col])
    if is_seasonal and score_azure > 0:
        reasons.append("🌊 **Complex Seasonality:** Strong recurring patterns detected alongside other complexities.")

    return score_azure, reasons

# --- MAIN APP LAYOUT ---

st.title("🤖 Intelligent AutoML Decision Engine")
st.markdown("Upload your time-series dataset to find the best tool: **Power BI** or **Azure ML**.")
st.divider()

# 1. FILE UPLOADER
uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file:
    # Load Data
    try:
        df = pd.read_csv(uploaded_file)
        
        # Split Layout for Inputs
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Configuration")
            # Select Columns
            all_cols = df.columns.tolist()
            date_col = st.selectbox("Select Date Column", all_cols)
            target_col = st.selectbox("Select Target (Prediction) Column", [c for c in all_cols if c != date_col])
            
            # Select Horizon
            horizon = st.number_input("Forecast Horizon (Days)", min_value=1, value=30)
            
            run_btn = st.button("Analyze Dataset", type="primary")

        with col2:
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(5), use_container_width=True)
            st.caption(f"Total Rows: {len(df):,} | Total Columns: {len(df.columns)}")

        # 2. RUN LOGIC ON BUTTON CLICK
        if run_btn:
            st.divider()
            with st.spinner('Running logic gates... checking multivariate complexity, seasonality, and volume...'):
                
                # Execute the "Brain" Function
                score, reasons = analyze_dataset(df, target_col, date_col, horizon)
                
                # 3. DISPLAY RESULTS
                st.header("🎯 Recommendation")
                
                if score >= 1:
                    # AZURE RECOMMENDATION
                    st.markdown(f"""
                        <div class="azure-box">
                            <h2>🔵 Recommended Tool: Azure Machine Learning</h2>
                            <p><strong>Why?</strong> Your dataset triggers <strong>{score} complexity flags</strong> that exceed Power BI's standard capabilities.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # POWER BI RECOMMENDATION
                    st.markdown("""
                        <div class="success-box">
                            <h2>📊 Recommended Tool: Power BI AutoML</h2>
                            <p><strong>Why?</strong> Your dataset is clean, univariate, and fits well within the low-code, budget-friendly capabilities of Power BI.</p>
                        </div>
                    """, unsafe_allow_html=True)

                # 4. SHOW DETAILED REASONS
                st.subheader("📝 Technical Analysis")
                if not reasons:
                    st.info("✅ Simple Univariate Data detected. No complex external factors found.")
                else:
                    for r in reasons:
                        st.write(r)

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("👋 Waiting for file upload...")