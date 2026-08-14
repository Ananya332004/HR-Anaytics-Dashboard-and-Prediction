"""
HR Attrition Prediction — Streamlit App
---------------------------------------
Loads artifacts produced by hr_attrition_pipeline.py:
    HR_Analytics_Prediction/best_model_hr.pkl
    HR_Analytics_Prediction/scaler_hr.pkl
    HR_Analytics_Prediction/columns_hr.pkl
    HR_Analytics_Prediction/continuous_cols_hr.pkl

Run with:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HR Attrition Intelligence",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ARTIFACT_DIR = "HR_Analytics_Prediction"

# ---------------------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg,#0e1117 0%, #141a26 100%); color: #ffffff !important; }
      .hero {
        padding: 2.2rem 2.5rem; border-radius: 18px; margin-bottom: 1.6rem;
        background: linear-gradient(120deg,#1f6feb 0%, #7b3fe4 55%, #d6336c 100%);
        box-shadow: 0 18px 40px rgba(0,0,0,.45);
      }
      .hero h1 { color:#fff; margin:0; font-size:2.1rem; font-weight:800; letter-spacing:-.5px; }
      .hero p  { color:rgba(255,255,255,.85); margin:.45rem 0 0; font-size:1rem; }
      .section {
        background: rgba(255,255,255,.05);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 16px; padding: 1.3rem 1.6rem 0.6rem; margin-bottom: 1.4rem;
        box-shadow: 0 8px 22px rgba(0,0,0,.25);
      }
      .section-title {
        font-size:1.12rem; font-weight:700; color:#fff; margin:0 0 .25rem;
        display:flex; align-items:center; gap:.55rem;
      }
      .section-sub { color:#b0c0d8; font-size:.86rem; margin:0 0 1rem; }
      .badge {
        display:inline-block; padding:.18rem .6rem; border-radius:999px;
        background:rgba(31,111,235,.22); color:#b8d4ff; font-size:.72rem;
        font-weight:700; letter-spacing:.06em; text-transform:uppercase;
      }
      .metric-card {
        background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.10);
        border-radius:14px; padding:1.1rem 1.3rem; text-align:center;
        box-shadow: 0 4px 16px rgba(0,0,0,.2);
      }
      .metric-card .v { font-size:2.1rem; font-weight:800; color:#fff; }
      .metric-card .l { font-size:.78rem; color:#b0c0d8; text-transform:uppercase; letter-spacing:.08em; }
      div.stButton > button {
        width:100%; border:0; border-radius:12px; padding:.85rem 1rem;
        font-weight:700; font-size:1rem; color:#fff;
        background:linear-gradient(90deg,#1f6feb,#7b3fe4);
        transition: transform .15s ease, box-shadow .15s ease;
      }
      div.stButton > button:hover { filter:brightness(1.15); transform: translateY(-2px); box-shadow: 0 8px 20px rgba(123,63,228,.35); color:#fff; }
      div.stSlider label, div.stSelectbox label, div.stNumberInput label, div.stSlider div, div.stSelectbox div, div.stNumberInput div {
        color: #ffffff !important;
      }
      div.stSlider [data-testid="stThumbValue"] { color: #ffffff !important; font-weight: 600; }
      .stAlert { color: #ffffff !important; }
      .stAlert [data-testid="stAlertContent"] { color: #ffffff !important; }
      .stDataFrame { color: #ffffff !important; }
      .st-bf, .st-bh, .st-bk, .st-bl, .st-bs, .st-bt, .st-bv, .st-bw, .st-bx, .st-by {
        color: #ffffff !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>👥 HR Attrition Intelligence</h1>
      <p>Calibrated SVM model · predicts the probability that an employee leaves the organisation</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# ARTIFACT LOADING
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts(base_dir: str):
    model = joblib.load(os.path.join(base_dir, "best_model_hr.pkl"))
    scaler = joblib.load(os.path.join(base_dir, "scaler_hr.pkl"))
    columns = joblib.load(os.path.join(base_dir, "columns_hr.pkl"))
    continuous_cols = joblib.load(os.path.join(base_dir, "continuous_cols_hr.pkl"))
    return model, scaler, list(columns), list(continuous_cols)


try:
    model, scaler, EXPECTED_COLUMNS, CONTINUOUS_COLS = load_artifacts(ARTIFACT_DIR)
except FileNotFoundError as exc:
    st.error(
        f"Could not load artifacts from `{ARTIFACT_DIR}/`. "
        f"Run `hr_attrition_pipeline.py` first.\n\n**Details:** {exc}"
    )
    st.stop()


def section(title: str, subtitle: str, tag: str):
    st.markdown(
        f"""<div class="section">
              <div class="section-title">{title} <span class="badge">{tag}</span></div>
              <p class="section-sub">{subtitle}</p>
            </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# OPTION SETS (must match the raw dataset's category spellings)
# ---------------------------------------------------------------------------
GENDERS = ["Female", "Male"]
MARITAL_STATUS = ["Single", "Married", "Divorced"]
TRAVEL = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}  # ordinal, as in training
DEPARTMENTS = ["Human Resources", "Research & Development", "Sales"]
JOB_ROLES = [
    "Healthcare Representative", "Human Resources", "Laboratory Technician",
    "Manager", "Manufacturing Director", "Research Director",
    "Research Scientist", "Sales Executive", "Sales Representative",
]

# ---------------------------------------------------------------------------
# SECTION A — DEMOGRAPHICS & PROFILE
# ---------------------------------------------------------------------------
with st.container():
    section("🧑‍💼 Demographics &amp; Profile", "Who the employee is.", "Section A")
    a1, a2 = st.columns(2)
    with a1:
        age = st.slider("Age", 18, 60, 35)
        gender = st.selectbox("Gender", GENDERS, index=1)
        distance_from_home = st.slider("Distance From Home (km)", 1, 30, 8)
    with a2:
        marital_status = st.selectbox("Marital Status", MARITAL_STATUS, index=1)
        num_companies_worked = st.slider("Number of Companies Worked", 0, 10, 2)

# ---------------------------------------------------------------------------
# SECTION B — ROLE DETAILS & TENURE
# ---------------------------------------------------------------------------
with st.container():
    section("🏢 Role Details &amp; Tenure", "Where they sit and how long they've stayed.", "Section B")
    b1, b2 = st.columns(2)
    with b1:
        business_travel = st.selectbox("Business Travel Frequency", list(TRAVEL.keys()), index=1)
        department = st.selectbox("Department", DEPARTMENTS, index=1)
        job_role = st.selectbox("Job Role", JOB_ROLES, index=7)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
    with b2:
        total_working_years = st.slider("Total Working Years", 0, 40, 10)
        years_in_current_role = st.slider("Years in Current Role", 0, 20, 3)
        years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        years_with_curr_manager = st.slider("Years with Current Manager", 0, 20, 3)

# ---------------------------------------------------------------------------
# SECTION C — OPERATIONAL & FINANCIAL CONTEXT
# ---------------------------------------------------------------------------
with st.container():
    section("💰 Operational &amp; Financial Context", "Pay, effort and development signals.", "Section C")
    c1, c2 = st.columns(2)
    with c1:
        monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000, step=100)
        percent_salary_hike = st.slider("Percent Salary Hike (%)", 10, 25, 14)
        daily_rate = st.slider("Daily Rate", 100, 1500, 800)
    with c2:
        hourly_rate = st.slider("Hourly Rate", 30, 100, 65)
        overtime = st.selectbox("Works Overtime", ["No", "Yes"], index=0)
        training_times_last_year = st.slider("Trainings Last Year", 0, 6, 3)

# ---------------------------------------------------------------------------
# FEATURE VECTOR CONSTRUCTION
# ---------------------------------------------------------------------------
def build_input_frame() -> pd.DataFrame:
    """Build a single-row frame aligned exactly with the trained column order."""
    # 1. Numeric / ordinal features (lowercase schema used at training time)
    values = {
        "age": age,
        "dailyrate": daily_rate,
        "distancefromhome": distance_from_home,
        "distancefromhome(km)": distance_from_home,   # alt spelling, harmless if unused
        "hourlyrate": hourly_rate,
        "monthlyincome": monthly_income,
        "numcompaniesworked": num_companies_worked,
        "percentsalaryhike": percent_salary_hike,
        "trainingtimeslastyear": training_times_last_year,
        "totalworkingyears": total_working_years,
        "yearsatcompany": years_at_company,
        "yearsincurrentrole": years_in_current_role,
        "yearssincelastpromotion": years_since_last_promotion,
        "yearswithcurrmanager": years_with_curr_manager,
        "businesstravel": TRAVEL[business_travel],   # ordinal encoded at training time
    }

    # 2. Dynamically constructed dummy columns (drop_first=True at training time,
    #    so the baseline level simply stays absent and every flag defaults to 0.0)
    dummies = [
        f"gender_{gender}",
        f"maritalstatus_{marital_status}",
        f"department_{department}",
        f"jobrole_{job_role}",
        f"overtime_{overtime}",
        f"businesstravel_{business_travel}",   # in case travel was one-hot encoded instead
    ]
    for col in dummies:
        values[col] = 1.0

    # 3. Derived flag created during cleaning
    values["income_outlier"] = 0.0

    # 4. Backfill every missing expected column with 0.0 and align the order
    row = {col: float(values.get(col, 0.0)) for col in EXPECTED_COLUMNS}
    frame = pd.DataFrame([row], columns=EXPECTED_COLUMNS)

    # 5. Scale ONLY the continuous features the scaler was fitted on
    cols_to_scale = [c for c in CONTINUOUS_COLS if c in frame.columns]
    if cols_to_scale:
        frame[cols_to_scale] = scaler.transform(frame[cols_to_scale])

    return frame


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
st.markdown("###")

# Predict button
predict = st.button("🔍  Predict Attrition Risk", type="primary")

# Helper to map risk band to color
RISK_COLOR = {
    "Low": "#27c686",
    "Moderate": "#f4d06f",
    "Elevated": "#ff9f43",
    "High": "#ff6b6b",
}

if predict:
    input_df = build_input_frame()
    probability = float(model.predict_proba(input_df)[0][1])
    risk_pct = probability * 100
    prediction = int(probability >= 0.5)

    band = "High" if risk_pct >= 70 else "Elevated" if risk_pct >= 50 else "Moderate" if risk_pct >= 30 else "Low"
    band_color = RISK_COLOR[band]

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.markdown(
        f'<div class="metric-card"><div class="v" style="color:{band_color};">{risk_pct:.1f}%</div>'
        f'<div class="l">Attrition Probability</div></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="metric-card"><div class="v" style="color:#27c686;">{100 - risk_pct:.1f}%</div>'
        f'<div class="l">Retention Probability</div></div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        f'<div class="metric-card"><div class="v" style="color:{band_color};">{band}</div>'
        f'<div class="l">Risk Band</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("####")
    st.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.10);
                    border-radius: 12px; padding: 0.4rem 0.8rem; margin-bottom: 1rem;">
            <div style="height: 10px; border-radius: 6px;
                        background: linear-gradient(90deg, #27c686 0%, #f4d06f 33%, #ff9f43 66%, #ff6b6b 100%);">
                <div style="width: {risk_pct:.1f}%; height: 100%; border-radius: 6px;
                            background: {band_color}; box-shadow: 0 0 12px {band_color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if prediction == 1:
        st.error(
            f"🚨 **FLIGHT RISK DETECTED** — this employee has a **{risk_pct:.1f}%** "
            "probability of leaving the organisation."
        )
        st.markdown(
            """
            <div style="background: rgba(255,107,107,0.08); border: 1px solid rgba(255,107,107,0.25);
                        border-radius: 14px; padding: 1rem 1.25rem;">
                <p style="margin:0 0 .5rem; color:#fff; font-weight:700;">Recommended retention actions</p>
                <ul style="margin:0; color:#e0e8f8;">
                    <li>Schedule a structured 1:1 stay-interview with the direct manager.</li>
                    <li>Review work-life balance: overtime load, travel frequency and workload distribution.</li>
                    <li>Benchmark compensation against the internal band and current market rate.</li>
                    <li>Map a concrete promotion / career-progression path if the last promotion is overdue.</li>
                    <li>Offer targeted upskilling or certification if training exposure has been low.</li>
                    <li>Consider a role, team or manager change where engagement signals are weak.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.success(
            f"✅ **LIKELY TO STAY** — attrition probability is only **{risk_pct:.1f}%**."
        )
        st.markdown(
            """
            <div style="background: rgba(39,198,134,0.08); border: 1px solid rgba(39,198,134,0.25);
                        border-radius: 14px; padding: 1rem 1.25rem;">
                <p style="margin:0 0 .5rem; color:#fff; font-weight:700;">Suggested reinforcement actions</p>
                <ul style="margin:0; color:#e0e8f8;">
                    <li>Keep recognition and feedback cadence consistent.</li>
                    <li>Continue regular development conversations to sustain engagement.</li>
                    <li>Re-score after any major change in role, manager or compensation.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("🔬 Inspect the aligned model input vector"):
        st.dataframe(
            input_df.T.rename(columns={0: "value"}),
            use_container_width=True,
            column_config={"value": st.column_config.NumberColumn(format="%.4f")},
        )

    st.caption(
        f"Model: {type(model).__name__} · {len(EXPECTED_COLUMNS)} expected features · "
        f"{len([c for c in CONTINUOUS_COLS if c in EXPECTED_COLUMNS])} scaled continuous features"
    )
else:
    st.info("Fill in the employee profile above, then run the prediction.")

