import streamlit as st
import pandas as pd
import zipfile
import os
import glob
import tempfile
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(page_title="Main Laser Logs", layout="wide")
st.title("📈 Main Laser Log Stitcher")

def parse_log_file(file_path, filename):
    try:
        # PRIORITY 1: Try Tab Separation
        df = pd.read_csv(file_path, sep='\t', engine='python', index_col=False)
        
        # PRIORITY 2: Fallback to multi-space
        if df.shape[1] < 2:
             df = pd.read_csv(file_path, sep=r'\s{2,}', engine='python', index_col=False)

        # Standardize Date/Time Parsing
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

uploaded_file = st.file_uploader("Upload Zip File containing `log_*.txt`", type="zip")

if uploaded_file is not None:
    st.info("Processing logs...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
            
        all_files = glob.glob(os.path.join(tmp_dir, "**", "log_*.txt"), recursive=True)
        all_files = sorted(all_files)
        
        if not all_files:
            st.error("No 'log_*.txt' files found.")
        else:
            data_frames = []
            progress = st.progress(0)
            
            for i, file in enumerate(all_files):
                filename_only = os.path.basename(file)
                df = parse_log_file(file, filename_only)
                if df is not None:
                    data_frames.append(df)
                progress.progress((i + 1) / len(all_files))
            
            if data_frames:
                full_df = pd.concat(data_frames, ignore_index=True)
                full_df = full_df.sort_values(by="Timestamp").dropna(subset=['Timestamp'])
                
                if full_df.empty:
                    st.error("❌ Error: Files were found, but no valid dates could be read.")
                else:
                    # --- 1. BASIC CALCULATIONS ---
                    start_time = full_df['Timestamp'].min()
                    duration_hours = (full_df['Timestamp'].max() - start_time).total_seconds() / 3600
                    full_df['Time (h)'] = (full_df['Timestamp'] - start_time).dt.total_seconds() / 3600
                    full_df['DateTime'] = full_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

                    # --- 2. REORDER COLUMNS ---
                    cols = list(full_df.columns)
                    special_cols = ['Date and Time', 'Timestamp', 'SourceFile', 'DateTime', 'Time (h)']
                    for special in special_cols:
                        if special in cols: cols.remove(special)
                    final_order = ['Date and Time'] + ['Time (h)'] + cols + ['SourceFile', 'DateTime']
                    final_order = [c for c in final_order if c in full_df.columns]
                    full_df = full_df[final_order]

                    st.success(f"Stitched {len(all_files)} files! ({len(full_df)} rows)")

                    # --- 📊 ENGINEERING DASHBOARD (With Tooltips) ---
                    st.markdown("### 📊 Engineering Health Check")
                    
                    # Row 1
                    c1, c2, c3, c4 = st.columns(4)
                    
                    c1.metric("⏱ Total Duration", f"{duration_hours:.1f} Hours", 
                              help="Total time elapsed between the first and last log entry.")
                    
                    if '532 Output PD' in full_df.columns:
                        avg_pwr = full_df['532 Output PD'].mean()
                        c2.metric("💡 Avg Output", f"{avg_pwr:.2f} V", 
                                  help="Average voltage of the 532 Output Photodiode across the entire session.")
                    else:
                        c2.metric("💡 Avg Output", "N/A")

                    if '532 Output PD' in full_df.columns and 'Diode Current' in full_df.columns:
                        valid_rows = full_df[full_df['Diode Current'] > 0]
                        if not valid_rows.empty:
                            eff = (valid_rows['532 Output PD'] / valid_rows['Diode Current']).mean()
                            c3.metric("⚡ Efficiency (Out/In)", f"{eff:.2f}",
                                      help="Ratio of Light Output to Current Input. If this drops over time, the pump diode may be degrading.")
                        else:
                            c3.metric("⚡ Efficiency", "N/A")
                    else:
                        c3.metric("⚡ Efficiency", "N/A")

                    if 'Events' in full_df.columns:
                        event_count = full_df['Events'].dropna().astype(str).str.strip().replace('', pd.NA).dropna().count()
                        c4.metric("⚠️ Events Logged", f"{event_count}",
                                  help="Total count of system warnings, errors, or status changes recorded in the 'Events' column.")
                    else:
                        c4.metric("⚠️ Events", "N/A")

                    # Row 2
                    c5, c6, c7, c8 = st.columns(4)

                    if '532 Output PD' in full_df.columns:
                        std_dev = full_df['532 Output PD'].std()
                        c5.metric("📉 Instability (RMS)", f"{std_dev:.4f}",
                                  help="Root Mean Square deviation. Measures how much the power 'wobbles'. Ideally close to 0.0.")
                    else:
                        c5.metric("📉 Instability", "N/A")

                    if 'PID Error' in full_df.columns:
                        max_pid = full_df['PID Error'].abs().max()
                        c6.metric("🔥 Max PID Error", f"{max_pid:.2f}",
                                  help="Maximum struggle of the temperature controller. A value of 0.00 is perfect. High values indicate thermal stress.")
                    else:
                        c6.metric("🔥 PID Error", "N/A")

                    if 'Laser Diode' in full_df.columns:
                        max_temp = full_df['Laser Diode'].max()
                        c7.metric("🌡 Max Diode Temp", f"{max_temp:.1f} °C",
                                  help="The highest temperature reached by the Laser Diode during this run.")
                    else:
                        c7.metric("🌡 Diode Temp", "N/A")
                        
                    if '532 Output PD' in full_df.columns and 'Time (h)' in full_df.columns:
                        target = full_df['532 Output PD'].mean() * 0.95
                        warmup_rows = full_df[full_df['532 Output PD'] > target]
                        if not warmup_rows.empty:
                            warmup_time = warmup_rows.iloc[0]['Time (h)'] * 60 
                            c8.metric("⏳ Warm-up Time", f"{warmup_time:.1f} min",
                                      help="Estimated time taken for the laser to reach (and stay above) 95% of its average power.")
                        else:
                            c8.metric("⏳ Warm-up", "Never Stabilized")
                    else:
                        c8.metric("⏳ Warm-up", "N/A")

                    st.markdown("---")

                    # --- TABS ---
                    tab1, tab2 = st.tabs(["📈 Graphical View", "📄 Data Table"])
                    
                    with tab1:
                        st.subheader("Time Series Analysis")
                        numeric_df = full_df.select_dtypes(include=['float64', 'int64'])
                        numeric_cols = numeric_df.columns.tolist()
                        col_opts = [c for c in numeric_cols if c not in ['Time (h)']]
                        
                        if not col_opts:
                            st.warning("No numeric data found to plot.")
                        else:
                            c1, c2 = st.columns(2)
                            with c1:
                                defaults = []
                                if '532 Cavity PD' in col_opts: defaults.append('532 Cavity PD')
                                if '532 Output PD' in col_opts: defaults.append('532 Output PD')
                                left_axis_cols = st.multiselect("Left Axis (Primary)", col_opts, default=defaults)
                            with c2:
                                right_axis_col = st.selectbox("Right Axis (Optional)", ["None"] + col_opts, index=0)

                            fig, ax1 = plt.subplots(figsize=(12, 6))
                            plot_dates = pd.to_datetime(full_df['DateTime'])

                            if left_axis_cols:
                                for col in left_axis_cols:
                                    ax1.plot(plot_dates, full_df[col], label=col)
                                ax1.set_ylabel("PD Signal / Value")
                                ax1.legend(loc='upper left')
                                ax1.grid(True)
                            
                            if right_axis_col != "None":
                                ax2 = ax1.twinx()
                                ax2.plot(plot_dates, full_df[right_axis_col], color='red', linestyle='--', alpha=0.5, label=right_axis_col)
                                ax2.set_ylabel(right_axis_col)
                                ax2.legend(loc='upper right')

                            ax1.set_title("Laser Parameters vs Time")
                            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                            plt.xticks(rotation=45)
                            st.pyplot(fig)

                    with tab2:
                        st.subheader("Stitched Log Data")
                        st.dataframe(full_df, use_container_width=True)

                    d_start = pd.to_datetime(full_df['DateTime']).min().strftime('%Y-%m-%d')
                    d_end = pd.to_datetime(full_df['DateTime']).max().strftime('%Y-%m-%d')
                    st.download_button("📥 Download Stitched CSV", full_df.to_csv(index=False).encode('utf-8'), f"log from {d_start} to {d_end}.csv", "text/csv")