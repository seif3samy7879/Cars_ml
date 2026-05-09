import streamlit as st

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="USA Cars Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3, h4 {
    color: white;
}

.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

.sidebar .sidebar-content {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HOME PAGE CONTENT
# =========================

st.title("🚗 USA Cars Dashboard & Prediction App")

st.markdown("""
Welcome to the **USA Cars Analysis System**.

### Features:
- 📊 Interactive Dashboard
- 📈 KPIs & Charts
- 🔍 Dynamic Filters
- 🤖 Car Price Prediction Model

Use the sidebar to navigate between pages.
""")

# =========================
# QUICK OVERVIEW SECTION
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Machine Learning Model",
        value="Random Forest"
    )

with col2:
    st.metric(
        label="Prediction Type",
        value="Car Price"
    )

with col3:
    st.metric(
        label="Dashboard",
        value="Interactive"
    )

# =========================
# INFORMATION SECTION
# =========================

st.markdown("---")

st.header("📌 Project Overview")

st.write("""
This application is built using:

- Streamlit
- Pandas
- Plotly
- Scikit-Learn

The app allows users to:
1. Explore the dataset
2. Analyze insights
3. Predict car prices using Machine Learning
""")

# =========================
# HOW TO USE
# =========================

st.markdown("---")

st.header("⚙️ How To Use")

st.write("""
### Step 1
Go to the **Insights Page** to explore the dashboard.

### Step 2
Use filters to analyze specific brands or states.

### Step 3
Open the **Prediction Page** and enter car information.

### Step 4
Click Predict to get the estimated price.
""")

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption("Developed with ❤️ using Streamlit")
