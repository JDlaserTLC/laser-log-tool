import streamlit as st

st.set_page_config(
    page_title="Laser Log Analysis Suite",
    page_icon="🔬",
    layout="wide"
)

# Sidebar Info
st.sidebar.success("Select a tool above to begin.")

# Main Page Design
st.title("🔬 Laser Log Analysis Suite")
st.markdown("### Welcome to the Engineering Dashboard")
st.markdown("---")

# Columns for the 3 Tools
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📈 1. Main Logs")
    st.info("**General Health Check**")
    st.markdown("""
    * **Stitch** weeks of log files instantly.
    * **Analyze** vital signs (Stability, Efficiency).
    * **Export** clean CSVs for reports.
    """)

with col2:
    st.markdown("### 🔧 2. LBO Optimiser")
    st.info("**Crystal Tuning**")
    st.markdown("""
    * **Visualize** bell curves for tuning.
    * **Track** historical drift (Aging Analysis).
    * **Find** optimal setpoints automatically.
    """)

with col3:
    st.markdown("### ⚡ 3. Power Dither")
    st.info("**Linearity & Sensitivity**")
    st.markdown("""
    * **Plot** Power vs Current curves.
    * **Check** laser linearity.
    * **Detect** performance degradation over time.
    """)

st.markdown("---")
st.markdown("#### 🚀 How to use:")
st.markdown("1. Select a tool from the **left sidebar**.")
st.markdown("2. Upload the **Zip file** containing your raw logs.")
st.markdown("3. The dashboard will automatically process and visualize the data.")

# Footer
st.markdown("---")
st.caption("v1.0 | Developed for TLC")