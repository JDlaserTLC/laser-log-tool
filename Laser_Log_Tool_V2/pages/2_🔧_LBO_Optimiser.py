import streamlit as st
import pandas as pd
import zipfile
import os
import glob
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="LBO Optimiser", layout="wide")
st.title("🔧 LBO Optimisation Analyzer")

# --- 1. HELPER: Load and Process Data ---
@st.cache_data(show_spinner=False)
def load_and_process_lbo(uploaded_file):
    def parse_lbo(file_path):
        try:
            df = pd.read_csv(file_path, sep='\t', engine='python', index_col=False)
            if df.shape[1] < 2:
                 df = pd.read_csv(file_path, sep=r'\s{2,}', engine='python', index_col=False)

            if 'Date and Time' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Date and Time'], dayfirst=True, errors='coerce')
            elif 'Timestamp' in df.columns:
                 df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
            else:
                df['Timestamp'] = pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors='coerce')
            return df
        except:
            return None

    # Extract and Parse
    dfs = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
            
        all_files = glob.glob(os.path.join(tmp_dir, "**", "*lbo*.txt"), recursive=True)
        if not all_files:
             all_files = glob.glob(os.path.join(tmp_dir, "**", "*.txt"), recursive=True)
        
        all_files = sorted(all_files)

        for f in all_files:
            d = parse_lbo(f)
            if d is not None: dfs.append(d)
            
    if not dfs:
        return None, None

    full_df = pd.concat(dfs, ignore_index=True).dropna(how='all')

    # Identify Columns
    x_col = 'LBO Temperature'
    y_col = 'Maximum Green PD'
    
    if x_col not in full_df.columns or y_col not in full_df.columns:
        return None, "Missing Columns"

    # Identify Sessions
    if 'Status' not in full_df.columns:
        full_df['Session_ID'] = 1
    else:
        full_df = full_df.sort_values(by='Timestamp')
        full_df['Session_ID'] = (full_df['Status'] == 'Starting').cumsum()

    # Build Session Dictionary
    sessions = {}
    history_records = [] # Store summary stats for trend analysis

    for uid in full_df['Session_ID'].unique():
        run_data = full_df[full_df['Session_ID'] == uid].copy()
        
        # Filter noise (need enough points for a curve)
        if len(run_data) > 5:
            # Find peaks for this specific run
            best_idx = run_data[y_col].idxmax()
            best_temp = run_data.loc[best_idx, x_col]
            best_pwr = run_data.loc[best_idx, y_col]
            start_time = run_data['Timestamp'].iloc[0]
            
            # Save for Dropdown View
            time_lbl = start_time.strftime('%Y-%m-%d %H:%M')
            sessions[f"Run #{uid} | {time_lbl}"] = run_data
            
            # Save for History View
            history_records.append({
                'Date': start_time,
                'Optimal Temp': best_temp,
                'Max Power': best_pwr,
                'Duration': len(run_data)
            })
            
    history_df = pd.DataFrame(history_records).sort_values(by='Date')
    
    return sessions, history_df

# --- 2. SESSION STATE LOGIC ---

# Initialize Keys
if 'lbo_sessions' not in st.session_state:
    st.session_state['lbo_sessions'] = None
if 'lbo_history' not in st.session_state:
    st.session_state['lbo_history'] = None

# Clear Data Callback
def clear_lbo_data():
    st.session_state['lbo_sessions'] = None
    st.session_state['lbo_history'] = None

# Retrieve Data
sessions = st.session_state['lbo_sessions']
history_df = st.session_state['lbo_history']

# --- 3. MAIN APP UI ---

if sessions is None:
    # SHOW UPLOADER
    uploaded_file = st.file_uploader("Upload LBO Zip File", type="zip")

    if uploaded_file is not None:
        with st.spinner("Analyzing LBO History..."):
            sess, hist = load_and_process_lbo(uploaded_file)
            
            if sess == "Missing Columns":
                st.error("❌ Could not find 'LBO Temperature' or 'Maximum Green PD' columns.")
            elif sess and hist is not None:
                # Save to Session State
                st.session_state['lbo_sessions'] = sess
                st.session_state['lbo_history'] = hist
                st.rerun()
            else:
                st.error("No valid LBO runs found in zip.")

else:
    # SHOW DASHBOARD (Data Loaded)
    
    # Header with Clear Button
    c_head1, c_head2 = st.columns([0.85, 0.15])
    with c_head1:
        st.success("✅ LBO Data Loaded")
    with c_head2:
        if st.button("🗑️ Clear Data"):
            clear_lbo_data()
            st.rerun()

    # --- TAB LAYOUT ---
    tab1, tab2 = st.tabs(["🔍 Detailed Analysis (Single Run)", "📈 Historical Trends (Drift Analysis)"])
    
    # --- TAB 1: THE TUNING TOOL (Interactive Dropdown) ---
    with tab1:
        st.sidebar.header("Select Run")
        # Ensure we have keys to select
        if sessions:
            choice = st.sidebar.selectbox("Choose Date/Run:", list(sessions.keys()))
            run_data = sessions[choice]
            
            x_col = 'LBO Temperature'
            y_col = 'Maximum Green PD'
            
            best_row = run_data.loc[run_data[y_col].idxmax()]
            max_pwr = best_row[y_col]
            optimal_temp = best_row[x_col]
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("🔥 Optimal Temp", f"{optimal_temp:.2f} °C")
            c2.metric("💡 Max Power", f"{max_pwr:.4f} V")
            c3.metric("⏱ Samples", f"{len(run_data)}")
            st.markdown("---")
            
            # Plot
            run_data['Point Type'] = 'Data Point'
            run_data.loc[best_row.name, 'Point Type'] = 'Optimal Peak'
            
            fig = px.scatter(
                run_data, x=x_col, y=y_col,
                color='Point Type',
                color_discrete_map={'Data Point': 'green', 'Optimal Peak': 'red'},
                title=f"Tuning Curve: {choice}",
                labels={x_col: "LBO Temperature (°C)", y_col: "Green Power (V)"},
                hover_data=['Timestamp']
            )
            fig.update_traces(marker=dict(size=12), selector=dict(name='Optimal Peak'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No runs found in the processed data.")

    # --- TAB 2: THE HISTORY TOOL (Trend Analysis) ---
    with tab2:
        st.subheader("⚠️ Long-Term Health & Drift Analysis")
        st.info("This view tracks the 'Optimal Peak' of every run found in the zip file to show aging.")
        
        if not history_df.empty:
            col1, col2 = st.columns(2)
            
            # GRAPH 1: Is the temperature drifting?
            with col1:
                fig_temp = px.line(history_df, x='Date', y='Optimal Temp', markers=True,
                                    title="Drift: Optimal Temperature over Time")
                fig_temp.update_traces(line_color='orange')
                st.plotly_chart(fig_temp, use_container_width=True)
                st.caption("If this line goes UP, the crystal may be aging or the sensor drifting.")
            
            # GRAPH 2: Is the power dropping?
            with col2:
                fig_pwr = px.line(history_df, x='Date', y='Max Power', markers=True,
                                    title="Health: Peak Power over Time")
                fig_pwr.update_traces(line_color='green')
                st.plotly_chart(fig_pwr, use_container_width=True)
                st.caption("If this line goes DOWN, the laser system is losing efficiency.")
            
            with st.expander("View Historical Summary Data"):
                st.dataframe(history_df)
        else:
            st.warning("Not enough data points to plot trends.")