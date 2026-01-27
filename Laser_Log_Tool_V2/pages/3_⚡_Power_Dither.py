import streamlit as st
import pandas as pd
import zipfile
import os
import glob
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Power Dither Analysis", layout="wide")
st.title("⚡ Power Dither Stitcher (Optimized)")

# --- 1. HELPER: Load and Stitch Data ---
@st.cache_data(show_spinner=False)
def load_and_stitch_data(uploaded_file):
    data_frames = []
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
            
        all_files = glob.glob(os.path.join(tmp_dir, "**", "*dither*.txt"), recursive=True)
        if not all_files:
             all_files = glob.glob(os.path.join(tmp_dir, "**", "*.txt"), recursive=True)
        
        all_files = sorted(all_files) # Sort chronologically

        for file in all_files:
            try:
                # Fast read
                df = pd.read_csv(file, sep='\t', engine='python', index_col=False)
                if df.shape[1] < 2:
                    df = pd.read_csv(file, sep=r'\s{2,}', engine='python', index_col=False)
                
                # Fast Date Parse
                if 'Date and Time' in df.columns:
                    df['Timestamp'] = pd.to_datetime(df['Date and Time'], dayfirst=True, errors='coerce')
                elif 'Timestamp' in df.columns:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
                else:
                    df['Timestamp'] = pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors='coerce')

                # Only keep necessary columns to save memory
                cols_to_keep = ['Timestamp', 'Diode Current', 'Green PD', 'Max Green PD', 'Status']
                df = df[[c for c in cols_to_keep if c in df.columns]]
                
                data_frames.append(df)
            except:
                continue
                
    if data_frames:
        full_df = pd.concat(data_frames, ignore_index=True)
        full_df = full_df.sort_values(by="Timestamp").dropna(subset=['Timestamp'])
        
        # Create a string date column for plotting
        full_df['DateTime'] = full_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        return full_df
    return None

# --- 2. SESSION STATE LOGIC ---

# Initialize key if missing
if 'power_dither_df' not in st.session_state:
    st.session_state['power_dither_df'] = None

# Callback to clear data
def clear_dither_data():
    st.session_state['power_dither_df'] = None

# Get data from state
full_df = st.session_state['power_dither_df']

# --- 3. MAIN APP LOGIC ---

if full_df is None:
    # SHOW UPLOADER
    uploaded_file = st.file_uploader("Upload Power Dither Zip", type="zip")

    if uploaded_file is not None:
        with st.spinner("Stitching files..."):
            processed_df = load_and_stitch_data(uploaded_file)
            
            if processed_df is not None:
                # Save to session state and reload
                st.session_state['power_dither_df'] = processed_df
                st.rerun()
            else:
                st.error("No valid text files found in zip.")

else:
    # SHOW DASHBOARD (Data exists in memory)
    
    # Header with Clear Button
    c_head1, c_head2 = st.columns([0.85, 0.15])
    with c_head1:
        st.success("✅ Power Dither Data Loaded")
    with c_head2:
        if st.button("🗑️ Clear Data"):
            clear_dither_data()
            st.rerun()

    # --- METRICS ---
    total_runs = full_df[full_df['Status'] == 'Starting'].shape[0] if 'Status' in full_df.columns else 0
    
    avg_max_pwr = 0
    if 'Max Green PD' in full_df.columns:
        avg_max_pwr = full_df[full_df['Max Green PD'] > 0]['Max Green PD'].mean()
    elif 'Green PD' in full_df.columns:
        avg_max_pwr = full_df['Green PD'].max()

    # --- DISPLAY METRICS ---
    st.markdown("### 📊 Historical Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("🔢 Total Runs", f"{total_runs}")
    c2.metric("💡 Avg Max Power", f"{avg_max_pwr:.3f} V")
    c3.metric("📅 Data Points", f"{len(full_df):,}")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Time Series", "⚡ XY Curves (Averaged)", "📄 Data Table"])
    
    # TAB 1: History (PLOTLY)
    with tab1:
        st.subheader("Time Series Analysis")
        
        plot_df = full_df
        numeric_cols = plot_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if numeric_cols:
            # Axis Controls
            c_left, c_right = st.columns(2)
            with c_left:
                defaults = []
                if 'Green PD' in numeric_cols: defaults.append('Green PD')
                left_axis = st.multiselect("Left Axis", numeric_cols, default=defaults)
            with c_right:
                right_axis = st.selectbox("Right Axis", ["None"] + numeric_cols)
            
            # Plot Logic
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            if left_axis:
                for col in left_axis:
                    fig.add_trace(
                        go.Scatter(x=plot_df['Timestamp'], y=plot_df[col], name=col, mode='lines'),
                        secondary_y=False,
                    )

            if right_axis != "None":
                fig.add_trace(
                    go.Scatter(
                        x=plot_df['Timestamp'], y=plot_df[right_axis], name=right_axis,
                        mode='lines', line=dict(dash='dot')
                    ),
                    secondary_y=True,
                )

            fig.update_layout(
                title="Power Dither Parameters vs Time",
                hovermode="x unified",
                xaxis_title="Date/Time",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=600
            )
            
            fig.update_yaxes(title_text="Primary Signals", secondary_y=False)
            if right_axis != "None":
                fig.update_yaxes(title_text=right_axis, secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric data found to plot.")

    # TAB 2: OPTIMIZED XY CURVE
    with tab2:
        st.subheader("Linearity Checks (Current vs Power)")
        
        if 'Diode Current' in full_df.columns and 'Green PD' in full_df.columns:
            
            # 1. Calculate Average Curve
            full_df['Current_Bin'] = full_df['Diode Current'].round(2)
            avg_curve = full_df.groupby('Current_Bin')['Green PD'].mean().reset_index().sort_values('Current_Bin')
            
            # 2. Downsample for Cloud
            cloud_df = full_df
            if len(cloud_df) > 2000:
                cloud_df = cloud_df.sample(n=2000, random_state=42)
            
            # 3. Plot
            fig = go.Figure()

            # Cloud
            fig.add_trace(go.Scatter(
                x=cloud_df['Diode Current'], y=cloud_df['Green PD'],
                mode='markers', name='Raw Data (Sample)',
                marker=dict(color='rgba(0, 0, 255, 0.1)', size=5),
                hovertext=cloud_df['DateTime']
            ))

            # Average Line
            fig.add_trace(go.Scatter(
                x=avg_curve['Current_Bin'], y=avg_curve['Green PD'],
                mode='lines+markers', name='Average Performance',
                line=dict(color='red', width=3), marker=dict(color='red', size=8)
            ))

            fig.update_layout(
                title="Average Tuning Curve (Red) vs Historical Spread (Blue)",
                xaxis_title="Diode Current (A)",
                yaxis_title="Green Power (V)",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Note: Blue dots show a random sample of history. Red line is the average.")

        else:
            st.warning("Columns 'Diode Current' and 'Green PD' required.")

    # TAB 3: DATA
    with tab3:
        st.dataframe(full_df.head(1000), use_container_width=True)
        d_start = full_df['Timestamp'].min().strftime('%d-%m-%Y')
        d_end = full_df['Timestamp'].max().strftime('%d-%m-%Y')
        st.download_button("📥 Download Stitched CSV", full_df.to_csv(index=False).encode('utf-8'), f"dither_{d_start}_to_{d_end}.csv", "text/csv")