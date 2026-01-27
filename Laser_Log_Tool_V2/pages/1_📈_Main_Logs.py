import streamlit as st
import pandas as pd
import zipfile
import os
import glob
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import timedelta
import re

st.set_page_config(page_title="Main Laser Logs", layout="wide")
st.title("📈 Main Laser Log Stitcher")

# --- GLOBAL SIDEBAR SETTINGS ---
st.sidebar.header("Global Settings")

target_voltage = st.sidebar.number_input(
    "Failure Threshold (W):", value=18.0, step=0.5
)

st.sidebar.markdown("---")

# Graph Settings
st.sidebar.header("Graph Options")
show_gaps_as_zero = st.sidebar.checkbox(
    "Show 'Off' Gaps as Zero", 
    value=True,
    help="Drops the line to 0 if the laser was off (removes diagonal stitch lines)."
)
gap_threshold_hours = st.sidebar.number_input(
    "Gap Threshold (Hours)", 
    value=10.0, 
    min_value=0.1
)

# --- HELPER FUNCTIONS ---

def parse_log_file(file_path, filename):
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
        
        df['SourceFile'] = filename
        return df
    except Exception:
        return None

def insert_zeros_for_gaps(df, threshold_hours=10.0):
    if df.empty: return df
    df = df.sort_values('Timestamp')
    threshold_sec = threshold_hours * 3600
    df['delta'] = df['Timestamp'].diff().dt.total_seconds()
    
    gap_mask = df['delta'] > threshold_sec
    if not gap_mask.any(): return df
        
    gap_ends = df[gap_mask]
    gap_starts = df.shift(1)[gap_mask]
    
    drop_timestamps = gap_starts['Timestamp'] + pd.Timedelta(seconds=1)
    rise_timestamps = gap_ends['Timestamp'] - pd.Timedelta(seconds=1)
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    cols = df.columns
    
    def create_fill_df(timestamps):
        new_df = pd.DataFrame(index=timestamps.index, columns=cols)
        new_df['Timestamp'] = timestamps
        new_df[numeric_cols] = 0
        new_df['SourceFile'] = "GAP_FILL"
        return new_df

    drop_df = create_fill_df(drop_timestamps)
    rise_df = create_fill_df(rise_timestamps)
    
    augmented_df = pd.concat([df, drop_df, rise_df], ignore_index=True)
    augmented_df = augmented_df.sort_values('Timestamp').reset_index(drop=True)
    augmented_df['DateTime'] = augmented_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return augmented_df

def predict_failure_range(df, file_range_indices, target_val=18.0):
    if 'SourceFile' not in df.columns: return "N/A", None, "No SourceFile", None, None
    
    file_stats = df.groupby('SourceFile')['Timestamp'].max().sort_values()
    sorted_files = file_stats.index.tolist()
    total_files = len(sorted_files)
    
    start_idx = max(0, file_range_indices[0] - 1)
    end_idx = min(total_files, file_range_indices[1])
    
    selected_files = sorted_files[start_idx:end_idx]
    
    if not selected_files:
        return "N/A", None, "No files selected", None, None

    window_df = df[df['SourceFile'].isin(selected_files)].copy()
    window_df = window_df[window_df['SourceFile'] != "GAP_FILL"] 
    
    window_start_time = window_df['Timestamp'].min()
    window_end_time = window_df['Timestamp'].max()
    
    cav_col = next((c for c in window_df.columns if 'cavity' in c.lower() and 'pd' in c.lower()), None)
    if not cav_col: return "N/A", None, "No Cavity PD", None, None

    window_df = window_df.set_index('Timestamp').sort_index()
    resampled_df = window_df[[cav_col]].resample('1H').mean().dropna()
    
    if len(resampled_df) < 2: return "Insufficient Data", None, "Not enough points", None, None

    calc_base_time = resampled_df.index.min()
    x_hours = (resampled_df.index - calc_base_time).total_seconds().values / 3600.0
    y_volts = resampled_df[cav_col].values
    
    if len(np.unique(x_hours)) < 2: return "Constant", None, "Flat data", None, None
    m, c = np.polyfit(x_hours, y_volts, 1) 

    trend_type = "Rising" if m > 0 else "Falling"
    debug_msg = f"{trend_type} ({m:.5f} V/hr) using files {file_range_indices[0]}-{file_range_indices[1]}"

    if m >= 0: return "Stable/Rising", None, debug_msg, window_start_time, window_end_time
    
    hours_to_target = (target_val - c) / m
    predicted_date = calc_base_time + timedelta(hours=hours_to_target)
    
    if hours_to_target < x_hours[-1]: return "Failed", None, debug_msg, window_start_time, window_end_time
    if predicted_date.year > 2050: return "> 2050", None, debug_msg, window_start_time, window_end_time

    plot_x_start = calc_base_time
    plot_y_start = c
    plot_x_end = predicted_date
    plot_y_end = target_val
    
    plot_data = {
        "x": [plot_x_start, plot_x_end],
        "y": [plot_y_start, plot_y_end]
    }

    return predicted_date.strftime('%d/%m/%Y'), plot_data, debug_msg, window_start_time, window_end_time


# --- MAIN LOGIC WITH SESSION STATE ---

# 1. Initialize Session State key if not present
if 'main_logs_df' not in st.session_state:
    st.session_state['main_logs_df'] = None

# 2. Define Clear Data Callback
def clear_data():
    st.session_state['main_logs_df'] = None
    st.session_state['file_list_len'] = 0

# 3. Logic: If no data in session, show uploader. If data exists, show Dashboard.
full_df = st.session_state['main_logs_df']

if full_df is None:
    uploaded_file = st.file_uploader("Upload Zip File containing `log_*.txt`", type="zip")
    
    if uploaded_file is not None:
        st.info("Processing logs...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
                
            all_files = sorted(glob.glob(os.path.join(tmp_dir, "**", "log_*.txt"), recursive=True))
            
            if not all_files:
                st.error("No 'log_*.txt' files found.")
            else:
                data_frames = []
                progress = st.progress(0)
                for i, file in enumerate(all_files):
                    df = parse_log_file(file, os.path.basename(file))
                    if df is not None: data_frames.append(df)
                    progress.progress((i + 1) / len(all_files))
                
                if data_frames:
                    stitched_df = pd.concat(data_frames, ignore_index=True)
                    stitched_df = stitched_df.sort_values(by="Timestamp").dropna(subset=['Timestamp'])
                    
                    if stitched_df.empty:
                        st.error("❌ No valid data found.")
                    else:
                        # Calculations to run ONCE
                        global_start = stitched_df['Timestamp'].min()
                        stitched_df['Time (h)'] = (stitched_df['Timestamp'] - global_start).dt.total_seconds() / 3600
                        stitched_df['DateTime'] = stitched_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # SAVE TO SESSION STATE
                        st.session_state['main_logs_df'] = stitched_df
                        st.session_state['file_list_len'] = len(all_files)
                        st.rerun() # Refresh to hide uploader and show dashboard

else:
    # --- DATA IS LOADED: SHOW DASHBOARD ---
    
    # Header with "Clear Data" button
    col_head1, col_head2 = st.columns([0.85, 0.15])
    with col_head1:
        st.success("✅ Log Data Loaded")
    with col_head2:
        if st.button("🗑️ Clear Data"):
            clear_data()
            st.rerun()

    # --- DYNAMIC SIDEBAR ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Prediction Options")
    show_prediction = st.sidebar.checkbox("Show Prediction & Trend Line", value=True)
    
    count = st.session_state.get('file_list_len', 10)
    file_range_selection = (1, count)
    
    if show_prediction:
        start_def = max(1, count - 9)
        file_range_selection = st.sidebar.slider(
            "Select File Range for Trend:",
            min_value=1,
            max_value=count,
            value=(start_def, count),
            step=1,
            help="Select the start and end file index to define the trend."
        )

    # --- METRICS CALCULATION ---
    # Metrics run on full_df (which is pulled from session_state)
    global_start = full_df['Timestamp'].min()
    
    # Diode Hours
    diode_hours = 0.0
    d_col = next((c for c in full_df.columns if 'current' in c.lower()), None)
    if d_col:
        full_df[d_col] = pd.to_numeric(full_df[d_col], errors='coerce')
        on_df = full_df[full_df[d_col] > 0.05].copy()
        if len(on_df) > 1:
            on_df['delta'] = on_df['Timestamp'].diff().dt.total_seconds().fillna(0)
            diode_hours = on_df[on_df['delta'] < 7200]['delta'].sum() / 3600.0

    # Events
    all_events_list = []
    e_col = next((c for c in full_df.columns if 'event' in c.lower()), None)
    if e_col:
        valid_events_df = full_df[~full_df[e_col].astype(str).isin(['0', '0.0', 'nan', ''])]
        for i, row in valid_events_df.iterrows():
            ts_str = row['Timestamp'].strftime('%d/%m/%Y %H:%M:%S')
            clean_code = str(row[e_col])
            clean_code = re.sub(r'\d{2}/\d{2}/\d{4}', '', clean_code)
            clean_code = re.sub(r'\d{2}:\d{2}:\d{2}', '', clean_code).strip()
            all_events_list.append(f"{ts_str} | Code: {clean_code}")

    # Temp & Warmup
    max_temp = "N/A"
    t_col = next((c for c in full_df.columns if 'laser diode' in c.lower() or ('diode' in c.lower() and 'temp' in c.lower())), None)
    if t_col: max_temp = f"{full_df[t_col].max():.1f} °C"

    warmup_hrs = "0.00"
    c_col = next((c for c in full_df.columns if 'cavity' in c.lower()), None)
    if c_col:
        full_df[c_col] = pd.to_numeric(full_df[c_col], errors='coerce')
        mx = full_df[c_col].max()
        if mx > 0:
            hp = full_df[full_df[c_col] > mx*0.8].copy()
            if len(hp)>1:
                hp['d'] = hp['Timestamp'].diff().dt.total_seconds().fillna(0)
                warmup_hrs = f"{hp[hp['d'] < 7200]['d'].sum()/3600:.2f} hrs"

    # Prediction
    pred_date = "Disabled"
    pred_plot = None
    debug_msg = "Off"
    hl_start = None
    hl_end = None
    
    if show_prediction:
        pred_date, pred_plot, debug_msg, hl_start, hl_end = predict_failure_range(
            full_df, 
            file_range_indices=file_range_selection, 
            target_val=target_voltage
        )
        
        if hl_start and hl_end:
            st.sidebar.info(f"**Analysis Window:**\n{hl_start.strftime('%d/%m/%y')} to {hl_end.strftime('%d/%m/%y')}")

    # --- DASHBOARD UI ---
    st.markdown("### 📊 Engineering Health Check")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("⏱ Diode Hours", f"{diode_hours:.2f} hrs")
    c2.metric("⚠️ Total Events", f"{len(all_events_list)}")
    c3.metric("🌡 Max Temp", max_temp)
    c4.metric("🔥 Hrs > 80%", warmup_hrs)
    c5.metric("🔮 Predicted Fail", pred_date, help=debug_msg)
    st.markdown("---")

    if all_events_list:
        with st.expander("⚠️ View All Logged Events", expanded=False):
            for e in reversed(all_events_list): st.text(e)
    else:
        st.success("No system events found.")
    st.markdown("---")

    # --- PLOTTING ---
    tab1, tab2 = st.tabs(["📈 Graphical View", "📄 Data Table"])
    
    with tab1:
        if show_gaps_as_zero:
            plot_df = insert_zeros_for_gaps(full_df.copy(), gap_threshold_hours)
        else:
            plot_df = full_df
        
        st.subheader("Time Series Analysis")
        num_cols = plot_df.select_dtypes(include=['number']).columns.tolist()
        opts = [c for c in num_cols if c not in ['Time (h)', 'delta']]
        
        if opts:
            cl, cr = st.columns(2)
            with cl:
                defs = []
                if c_col and c_col in opts: defs.append(c_col)
                out_col = next((c for c in opts if 'output' in c.lower()), None)
                if out_col: defs.append(out_col)
                left_ax = st.multiselect("Left Axis", opts, default=defs)
            with cr:
                right_ax = st.selectbox("Right Axis", ["None"] + opts)

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # 1. HIGHLIGHT BOX (Fixed Date Logic)
            if show_prediction and hl_start and hl_end:
                s_str = hl_start.strftime("%Y-%m-%d %H:%M:%S")
                e_str = hl_end.strftime("%Y-%m-%d %H:%M:%S")

                fig.add_vrect(
                    x0=s_str, x1=e_str,
                    fillcolor="yellow", opacity=0.3, layer="below", line_width=0,
                )
                fig.add_shape(
                    type="line", x0=s_str, x1=s_str, y0=0, y1=1, xref="x", yref="paper",
                    line=dict(color="orange", width=2, dash="dash")
                )
                fig.add_shape(
                    type="line", x0=e_str, x1=e_str, y0=0, y1=1, xref="x", yref="paper",
                    line=dict(color="orange", width=2, dash="dash")
                )

            # 2. Data Traces
            if left_ax:
                for col in left_ax:
                    fig.add_trace(
                        go.Scatter(x=plot_df['Timestamp'], y=plot_df[col], name=col),
                        secondary_y=False,
                    )
                
                if show_prediction and c_col and c_col in left_ax and pred_plot:
                    fig.add_trace(
                        go.Scatter(
                            x=pred_plot['x'], y=pred_plot['y'], 
                            name=f"Trend ({pred_date})",
                            line=dict(color='red', width=2, dash='dash')
                        ), secondary_y=False
                    )

            if right_ax != "None":
                fig.add_trace(
                    go.Scatter(
                        x=plot_df['Timestamp'], y=plot_df[right_ax], 
                        name=right_ax,
                        line=dict(color='green', dash='dot')
                    ), secondary_y=True
                )

            fig.update_layout(
                title="Laser Parameters vs Time", hovermode="x unified",
                xaxis_title="Date/Time",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text="Primary Axis", secondary_y=False)
            if right_ax != "None":
                fig.update_yaxes(title_text=right_ax, secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric data to plot.")

    with tab2:
        st.dataframe(full_df, use_container_width=True)
        d_start = full_df['Timestamp'].min().strftime('%d-%m-%Y')
        d_end = full_df['Timestamp'].max().strftime('%d-%m-%Y')
        csv = full_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, f"log from {d_start} to {d_end}.csv", "text/csv")