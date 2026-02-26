import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------
# PAGE CONFIG (Executive Layout)
# ------------------------------------------------
st.set_page_config(
    page_title="Workforce Optimization Executive Dashboard",
    layout="wide"
)

# Corporate styling
st.markdown("""
    <style>
        .main {background-color: #f4f6f9;}
        .metric-container {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
        }
        h1 {color: #1f4e79;}
    </style>
""", unsafe_allow_html=True)

st.title("Workforce Delivery & Staffing Optimization")
st.markdown("### Executive Performance & Revenue Acceleration Model")

# ------------------------------------------------
# SYNTHETIC DATA GENERATION
# ------------------------------------------------

@st.cache_data
def generate_data(n=1500):
    np.random.seed(42)
    industries = ['Technology', 'Healthcare', 'Finance', 'Retail', 'Manufacturing']
    job_types = ['Contract', 'Full-Time', 'Contract-to-Hire']
    start_date = datetime(2024, 1, 1)
    data = []

    for i in range(n):
        req_date = start_date + timedelta(days=np.random.randint(0, 365))
        industry = np.random.choice(industries)
        job_type = np.random.choice(job_types)

        delays = np.maximum(
            1,
            np.random.normal([4,6,3,5,4,3],[2,3,1.5,2,2,1.5])
        ).astype(int)

        total_cycle = delays.sum()
        rework = np.random.choice([0,1], p=[0.8,0.2])
        candidate_count = np.random.randint(5, 25)

        # Revenue per placement (simulate realistic contract economics)
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
        "Industry","Job_Type",
        "Time_to_Source","Time_to_Screen","Time_to_Submit",
        "Time_to_Interview","Time_to_Offer","Time_to_Place",
        "Candidate_Count","Rework_Flag",
        "Total_Cycle_Time","Revenue_Value"
    ]

    return pd.DataFrame(data, columns=columns)

df = generate_data()

# ------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------

st.sidebar.header("Scenario Controls")

screen_reduction = st.sidebar.slider(
    "Reduce Screening Time (%)",
    0, 50, 0
)

industry_filter = st.sidebar.multiselect(
    "Industry",
    df["Industry"].unique(),
    default=df["Industry"].unique()
)

df = df[df["Industry"].isin(industry_filter)]

# Apply scenario adjustment
df["Adjusted_Screen_Time"] = df["Time_to_Screen"] * (1 - screen_reduction/100)

df["Adjusted_Total_Cycle"] = (
    df["Time_to_Source"] +
    df["Adjusted_Screen_Time"] +
    df["Time_to_Submit"] +
    df["Time_to_Interview"] +
    df["Time_to_Offer"] +
    df["Time_to_Place"]
)

# ------------------------------------------------
# FINANCIAL IMPACT MODEL
# ------------------------------------------------

avg_cycle = df["Total_Cycle_Time"].mean()
adjusted_cycle = df["Adjusted_Total_Cycle"].mean()

cycle_reduction = avg_cycle - adjusted_cycle

placements_per_year = len(df)

# Revenue acceleration logic:
# Faster placement -> revenue realized earlier
daily_revenue_avg = df["Revenue_Value"].mean() / 120

revenue_acceleration = cycle_reduction * daily_revenue_avg * placements_per_year

# ------------------------------------------------
# KPI ROW
# ------------------------------------------------

st.subheader("Operational Performance Snapshot")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Time-to-Placement",
            f"{round(avg_cycle,1)} days")

col2.metric("Adjusted Cycle Time",
            f"{round(adjusted_cycle,1)} days")

col3.metric("Cycle Time Reduction",
            f"{round(cycle_reduction,2)} days")

col4.metric("Projected Revenue Acceleration",
            f"${round(revenue_acceleration,0):,}")

# ------------------------------------------------
# BOTTLENECK VISUAL
# ------------------------------------------------

st.subheader("Stage Bottleneck Analysis")

stage_means = df[[
    "Time_to_Source","Time_to_Screen",
    "Time_to_Submit","Time_to_Interview",
    "Time_to_Offer","Time_to_Place"
]].mean().sort_values()

fig_bottleneck = px.bar(
    stage_means,
    orientation='h',
    title="Average Duration by Stage (Days)"
)

st.plotly_chart(fig_bottleneck, use_container_width=True)

# ------------------------------------------------
# ML DELAY PREDICTION
# ------------------------------------------------

st.subheader("Predictive Delay Modeling")

df_ml = df.copy()

le1 = LabelEncoder()
le2 = LabelEncoder()

df_ml["Industry"] = le1.fit_transform(df_ml["Industry"])
df_ml["Job_Type"] = le2.fit_transform(df_ml["Job_Type"])

features = [
    "Industry","Job_Type",
    "Time_to_Source","Time_to_Screen",
    "Time_to_Submit","Time_to_Interview",
    "Time_to_Offer","Candidate_Count","Rework_Flag"
]

X = df_ml[features]
y = df_ml["Total_Cycle_Time"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, max_depth=10)
model.fit(X_train, y_train)

df_ml["Predicted_Cycle_Time"] = model.predict(X)

importances = pd.Series(model.feature_importances_, index=features).sort_values()

fig_importance = px.bar(
    importances,
    orientation='h',
    title="Key Drivers of Placement Delays"
)

st.plotly_chart(fig_importance, use_container_width=True)

# ------------------------------------------------
# EXECUTIVE SUMMARY
# ------------------------------------------------

st.subheader("Executive Impact Summary")

st.markdown(f"""
- Reducing screening time by **{screen_reduction}%** decreases average cycle time by **{round(cycle_reduction,2)} days**
- Annual revenue acceleration potential: **${round(revenue_acceleration,0):,}**
- Most significant operational bottleneck: **{stage_means.idxmax()}**
- Predictive modeling enables early identification of high-risk requisitions
""")


