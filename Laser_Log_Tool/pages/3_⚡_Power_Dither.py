import streamlit as st
import pandas as pd
import zipfile
import os
import glob
import tempfile
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Power Dither Analysis", layout="wide")
st.title("⚡ Power Dither Stitcher (Optimized)")

# --- 1. OPTIMIZATION: Caching the Loader ---
# This prevents the app from re-processing the ZIP file every time you change a tab.
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

uploaded_file = st.file_uploader("Upload Power Dither Zip", type="zip")

if uploaded_file is not None:
    # Show a spinner while the cached function runs
    with st.spinner("Stitching files..."):
        full_df = load_and_stitch_data(uploaded_file)

    if full_df is not None:
        # --- METRICS ---
        total_runs = full_df[full_df['Status'] == 'Starting'].shape[0] if 'Status' in full_df.columns else 0
        
        avg_max_pwr = 0
        if 'Max Green PD' in full_df.columns:
            avg_max_pwr = full_df[full_df['Max Green PD'] > 0]['Max Green PD'].mean()
        elif 'Green PD' in full_df.columns:
            avg_max_pwr = full_df['Green PD'].max()

        # --- DASHBOARD ---
        st.markdown("### 📊 Historical Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("🔢 Total Runs", f"{total_runs}")
        c2.metric("💡 Avg Max Power", f"{avg_max_pwr:.3f} V")
        c3.metric("📅 Data Points", f"{len(full_df):,}")
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["📈 Time Series", "⚡ XY Curves (Averaged)", "📄 Data Table"])
        
        # TAB 1: History
        with tab1:
            st.subheader("History over Time")
            # Downsample for matplotlib if huge (take 1 point every 10 rows)
            # This makes the static plot render much faster
            plot_df = full_df
            if len(full_df) > 10000:
                plot_df = full_df.iloc[::10, :] 
            
            numeric_cols = plot_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if numeric_cols:
                c_left, c_right = st.columns(2)
                with c_left:
                    defaults = []
                    if 'Green PD' in numeric_cols: defaults.append('Green PD')
                    left_axis = st.multiselect("Left Axis", numeric_cols, default=defaults)
                with c_right:
                    right_axis = st.selectbox("Right Axis", ["None"] + numeric_cols)
                
                fig, ax1 = plt.subplots(figsize=(12, 6))
                plot_dates = pd.to_datetime(plot_df['DateTime'])
                
                if left_axis:
                    for col in left_axis:
                        ax1.plot(plot_dates, plot_df[col], label=col, linewidth=1)
                    ax1.legend(loc='upper left')
                    ax1.grid(True)
                
                if right_axis != "None":
                    ax2 = ax1.twinx()
                    ax2.plot(plot_dates, plot_df[right_axis], color='red', linestyle='--', alpha=0.5, label=right_axis)
                    ax2.set_ylabel(right_axis)
                    ax2.legend(loc='upper right')
                
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                st.pyplot(fig)

        # TAB 2: OPTIMIZED XY CURVE
        with tab2:
            st.subheader("Linearity Checks (Current vs Power)")
            
            if 'Diode Current' in full_df.columns and 'Green PD' in full_df.columns:
                
                # 1. CALCULATE AVERAGE CURVE
                # Round current to nearest 0.05A (or 0.01A) to group similar steps
                full_df['Current_Bin'] = full_df['Diode Current'].round(2)
                avg_curve = full_df.groupby('Current_Bin')['Green PD'].mean().reset_index().sort_values('Current_Bin')
                
                # 2. DOWNSAMPLE RAW DATA (The Cloud)
                # Instead of plotting 100k points, we plot a random sample of 2,000.
                # This shows the "spread" without killing the browser.
                cloud_df = full_df
                if len(cloud_df) > 2000:
                    cloud_df = cloud_df.sample(n=2000, random_state=42)
                
                # 3. BUILD PLOTLY CHART
                fig = go.Figure()

                # Add the "Cloud" (Raw Data)
                fig.add_trace(go.Scatter(
                    x=cloud_df['Diode Current'],
                    y=cloud_df['Green PD'],
                    mode='markers',
                    name='Raw Data (Sample)',
                    marker=dict(color='rgba(0, 0, 255, 0.1)', size=5), # Transparent blue
                    hovertext=cloud_df['DateTime']
                ))

                # Add the "Average Curve" (Solid Line)
                fig.add_trace(go.Scatter(
                    x=avg_curve['Current_Bin'],
                    y=avg_curve['Green PD'],
                    mode='lines+markers',
                    name='Average Performance',
                    line=dict(color='red', width=3),
                    marker=dict(color='red', size=8)
                ))

                fig.update_layout(
                    title="Average Tuning Curve (Red) vs Historical Spread (Blue)",
                    xaxis_title="Diode Current (A)",
                    yaxis_title="Green Power (V)",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Note: Blue dots show a random sample of history to indicate spread. Red line is the mathematical average of all runs.")

            else:
                st.warning("Columns 'Diode Current' and 'Green PD' required.")

        with tab3:
            st.dataframe(full_df.head(1000), use_container_width=True) # Limit display for speed
            d_start = pd.to_datetime(full_df['DateTime']).min().strftime('%Y-%m-%d')
            d_end = pd.to_datetime(full_df['DateTime']).max().strftime('%Y-%m-%d')
            st.download_button("📥 Download Stitched CSV", full_df.to_csv(index=False).encode('utf-8'), f"dither_{d_start}_to_{d_end}.csv", "text/csv")
    else:
         st.error("No text files found in zip.")