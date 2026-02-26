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
    layout="wide"
)

# --------------------------------------------------
# BCforward CORPORATE THEME
# --------------------------------------------------

st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;
        }
        .metric-container {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 3px 8px rgba(0,0,0,0.06);
        }
        h1, h2, h3 {
            color: #1f4e79;
        }
        .stMetric {
            background-color: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0px 3px 8px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER WITH LOGO
# --------------------------------------------------

logo = Image.open("bcforward_logo.png")
col_logo, col_title = st.columns([1,4])

with col_logo:
    st.image(logo, width=180)

with col_title:
    st.title("Workforce Delivery Optimization Dashboard")
    st.markdown("### Executive Performance & Revenue Acceleration Model")

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

st.subheader("Executive Performance Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Average Time to Placement (Baseline)",
          f"{round(baseline_cycle,1)} Days")

k2.metric("Projected Time to Placement (Adjusted)",
          f"{round(adjusted_cycle,1)} Days")

k3.metric("Cycle Time Improvement",
          f"{round(cycle_improvement,2)} Days")

k4.metric("Projected Annual Revenue Acceleration",
          f"${round(projected_revenue_acceleration,0):,}")

st.markdown("---")

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

st.markdown("---")
st.subheader("Executive Summary")

st.markdown(f"""
**Scenario Impact Assessment**

• A {screen_reduction}% reduction in screening time results in a **{round(cycle_improvement,2)} day improvement** in overall placement cycle time.

• This operational improvement translates to an estimated **${round(projected_revenue_acceleration,0):,} in accelerated revenue realization annually.**

• Primary operational bottleneck identified: **{stage_means.idxmax()}**

• Predictive modeling enables early identification of high-risk requisitions to proactively mitigate delays.
""")