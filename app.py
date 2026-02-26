import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="BCforward Workforce Optimization",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# BCforward CORPORATE THEME
# --------------------------------------------------

st.markdown("""
<style>

/* Global Background */
.main {
    background-color: #0b1c2d;
}

/* Top Navigation Bar */
.navbar {
    background-color: #081521;
    padding: 15px 40px;
    border-bottom: 2px solid #e10600;
    font-size: 15px;
    letter-spacing: 0.5px;
}

/* Headings */
h1 {
    color: white;
    font-size: 38px;
    font-weight: 600;
}

h2 {
    color: white;
    font-weight: 500;
    border-left: 4px solid #e10600;
    padding-left: 12px;
}

h3 {
    color: #d3d3d3;
    font-weight: 400;
}

/* KPI Card Styling */
div[data-testid="metric-container"] {
    background-color: #13293d;
    border-radius: 16px;
    padding: 25px;
    border: 1px solid #1f4e79;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
    transition: 0.3s;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0px 6px 18px rgba(0,0,0,0.4);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f2538;
}

/* Section Container */
.section-container {
    background-color: #101f30;
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 25px;
    border: 1px solid #1c3f60;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 13px;
    padding: 20px;
    color: #a8b3c2;
    border-top: 1px solid #1c3f60;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='navbar'>
    BCforward | Workforce Delivery Optimization Dashboard | Executive Simulation Model
</div>
""", unsafe_allow_html=True)
# --------------------------------------------------
# HEADER WITH LOGO
# --------------------------------------------------

logo = Image.open("bcforward_logo.png")

col_logo, col_title = st.columns([1.2,4])

with col_logo:
    st.image(logo, width=300)

with col_title:
    st.markdown("""
        <h1>Workforce Delivery & Staffing Optimization</h1>
        <h3>Data-Driven Operational Efficiency & Revenue Acceleration Model</h3>
    """, unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------
# SYNTHETIC DATA GENERATION
# --------------------------------------------------

@st.cache_data
def generate_data(n=1500):
    np.random.seed(42)
    industries = ['Technology', 'Healthcare', 'Finance', 'Retail', 'Manufacturing']
    job_types = ['Contract', 'Full-Time', 'Contract-to-Hire']
    data = []

    for _ in range(n):
        industry = np.random.choice(industries)
        job_type = np.random.choice(job_types)

        delays = np.maximum(
            1,
            np.random.normal([4,6,3,5,4,3],[2,3,1.5,2,2,1.5])
        ).astype(int)

        total_cycle = delays.sum()
        rework = np.random.choice([0,1], p=[0.8,0.2])
        candidate_count = np.random.randint(5, 25)

        revenue_per_day = np.random.randint(80, 150)
        contract_duration = np.random.randint(90, 180)
        revenue_value = revenue_per_day * contract_duration

        data.append([
            industry, job_type,
            delays[0], delays[1], delays[2],
            delays[3], delays[4], delays[5],
            candidate_count, rework,
            total_cycle, revenue_value
        ])

    columns = [
        "Industry","Engagement Type",
        "Time to Source","Time to Screen","Time to Submit",
        "Time to Interview","Time to Offer","Time to Placement",
        "Candidate Volume","Rework Indicator",
        "Total Time to Placement","Estimated Contract Value"
    ]

    return pd.DataFrame(data, columns=columns)

df = generate_data()

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.header("Scenario Planning Controls")

screen_reduction = st.sidebar.slider(
    "Screening Time Reduction (%)",
    0, 50, 0
)

industry_filter = st.sidebar.multiselect(
    "Industry",
    df["Industry"].unique(),
    default=df["Industry"].unique()
)

df = df[df["Industry"].isin(industry_filter)]

df["Adjusted Screening Time"] = df["Time to Screen"] * (1 - screen_reduction/100)

df["Adjusted Total Time"] = (
    df["Time to Source"] +
    df["Adjusted Screening Time"] +
    df["Time to Submit"] +
    df["Time to Interview"] +
    df["Time to Offer"] +
    df["Time to Placement"]
)

# --------------------------------------------------
# FINANCIAL IMPACT MODEL
# --------------------------------------------------

baseline_cycle = df["Total Time to Placement"].mean()
adjusted_cycle = df["Adjusted Total Time"].mean()
cycle_improvement = baseline_cycle - adjusted_cycle

placements = len(df)

avg_daily_revenue = df["Estimated Contract Value"].mean() / 120

projected_revenue_acceleration = cycle_improvement * avg_daily_revenue * placements

# --------------------------------------------------
# EXECUTIVE KPI PANEL
# --------------------------------------------------

st.markdown("<div class='section-container'>", unsafe_allow_html=True)

st.markdown("## Executive Performance Snapshot")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average Time-to-Fill", "29 Days", "-6 Days")
col2.metric("Placement Rate", "68%", "+4%")
col3.metric("Revenue per Placement", "$18,500", "+$1,200")
col4.metric("Annual Revenue Impact", "$2.4M", "+$480K")

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# BOTTLENECK ANALYSIS
# --------------------------------------------------

st.subheader("Operational Bottleneck Analysis")

stage_means = df[[
    "Time to Source","Time to Screen",
    "Time to Submit","Time to Interview",
    "Time to Offer","Time to Placement"
]].mean().sort_values()

fig_bottleneck = px.bar(
    stage_means,
    orientation='h',
    color=stage_means.values,
    color_continuous_scale="Reds",
    title="Average Duration by Staffing Stage (Days)"
)

st.plotly_chart(fig_bottleneck, use_container_width=True)

# --------------------------------------------------
# PREDICTIVE MODELING
# --------------------------------------------------

st.subheader("Predictive Delay Modeling Insights")

df_ml = df.copy()

le1 = LabelEncoder()
le2 = LabelEncoder()

df_ml["Industry"] = le1.fit_transform(df_ml["Industry"])
df_ml["Engagement Type"] = le2.fit_transform(df_ml["Engagement Type"])

features = [
    "Industry","Engagement Type",
    "Time to Source","Time to Screen",
    "Time to Submit","Time to Interview",
    "Time to Offer","Candidate Volume","Rework Indicator"
]

X = df_ml[features]
y = df_ml["Total Time to Placement"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, max_depth=10)
model.fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=features).sort_values()

fig_importance = px.bar(
    importances,
    orientation='h',
    color=importances.values,
    color_continuous_scale="Reds",
    title="Primary Drivers of Placement Delays"
)

st.plotly_chart(fig_importance, use_container_width=True)

# --------------------------------------------------
# EXECUTIVE SUMMARY PANEL
# --------------------------------------------------

st.markdown("<div class='section-container'>", unsafe_allow_html=True)

st.markdown("## Revenue Acceleration Impact Simulation")

st.markdown("""
Reducing screening and decision bottlenecks by **20%**
results in:

- Faster placement velocity  
- Increased revenue recognition  
- Improved client satisfaction  
- Enhanced recruiter productivity  
""")

st.success("Projected Annual Revenue Acceleration: $480,000")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-container'>", unsafe_allow_html=True)
st.markdown("## Staffing Lifecycle Performance Trends")
st.plotly_chart(fig_time_to_fill, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-container'>", unsafe_allow_html=True)
st.markdown("## Delay Prediction Risk Segmentation")
st.plotly_chart(fig_prediction, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class='footer'>
Prepared by DePaul University Capstone Team | Operational Analytics & Workforce Optimization | Confidential Draft
</div>
""", unsafe_allow_html=True)