# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
from io import BytesIO
from datetime import datetime, time
import base64

# ---------------------------
# Streamlit config
# ---------------------------
st.set_page_config(layout="wide", page_title="AI Schedule Generator by - Data Quest")

# ---------------------------
# Erlang helpers
# ---------------------------
def erlang_c_Pw(a, c):
    if c <= a:
        return 1.0
    sum_terms = sum((a ** k) / math.factorial(k) for k in range(c))
    a_c = a ** c / math.factorial(c)
    return (a_c * (c / (c - a))) / (sum_terms + a_c * (c / (c - a)))


def erlang_c_wait_prob_gt_t(a, c, mu, t):
    pw = erlang_c_Pw(a, c)
    exponent = - (c - a) * mu * t
    if exponent < -700:
        return 0.0
    return pw * math.exp(exponent)


def erlang_a_estimates(a, c, mu, theta, t_sla_min):
    if c <= a:
        return 1.0, 1.0, 1.0, 0.0

    pw = erlang_c_Pw(a, c)
    expected_wait = 1.0 / ((c - a) * mu)
    p_abandon_any = pw * (1 - math.exp(-theta * expected_wait))
    p_wait_gt_t = pw * math.exp(- (c - a) * mu * t_sla_min)
    p_abandon_before_t = pw * (1 - math.exp(-theta * min(expected_wait, t_sla_min)))
    sla_est = max(0.0, 1.0 - p_wait_gt_t - p_abandon_before_t)
    return pw, p_wait_gt_t, p_abandon_any, sla_est


def required_servers_for_SLA_and_abandon(arrivals, aht, sla_frac, sla_sec, abn_frac, patience_sec):
    if arrivals <= 0:
        return 0

    lam = arrivals / 30.0
    mu = 1.0 / aht
    a = lam / mu
    t = sla_sec / 60.0
    theta = 1.0 / (patience_sec / 60.0)

    for c in range(math.ceil(a), 500):
        if c <= a:
            continue
        pw = erlang_c_wait_prob_gt_t(a, c, mu, t)
        sla_ok = (1 - pw) >= sla_frac
        _, _, abn, _ = erlang_a_estimates(a, c, mu, theta, t)
        if sla_ok and abn <= abn_frac:
            return c
    return math.ceil(a) + 1


# ---------------------------
# Helpers
# ---------------------------
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def time_to_min(s):
    dt = datetime.strptime(str(s), "%H:%M")
    return dt.hour * 60 + dt.minute


def min_to_time(m):
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


def off_mask(pair):
    m = [1] * 7
    for d in pair:
        m[WEEKDAYS.index(d)] = 0
    return m


# ---------------------------
# UI
# ---------------------------
st.title("AI Schedules Generator by Data Quest")
with st.sidebar:
    uploaded = st.file_uploader("Upload forecast CSV", type=["csv"])
    aht = st.slider("AHT (seconds)", 60, 900, 360) / 60
    sla_pct = st.slider("SLA %", 50, 99, 80) / 100
    sla_sec = st.slider("SLA seconds", 5, 60, 20)
    abn_pct = st.slider("Abandon %", 0, 20, 5) / 100
    patience = st.slider("Patience seconds", 30, 300, 120)
    run = st.button("Generate Roster")

if not uploaded:
    st.stop()

# ---------------------------
# Read forecast
# ---------------------------
df = pd.read_csv(uploaded)
df["weekday"] = pd.to_datetime(df["date"]).dt.strftime("%a")
df["slot_min"] = df["interval"].apply(time_to_min)

all_slots = sorted(df["slot_min"].unique())

df["required"] = df["volume"].apply(
    lambda x: required_servers_for_SLA_and_abandon(
        x, aht, sla_pct, sla_sec, abn_pct, patience
    )
)

baseline_req = {
    wd: {
        min_to_time(t): int(
            df.loc[(df["weekday"] == wd) & (df["slot_min"] == t), "required"].iloc[0]
        )
        for t in all_slots
    }
    for wd in WEEKDAYS
}

# ---------------------------
# Dummy agent list (example)
# ---------------------------
agents = [
    {"id": "A1", "start": 20 * 60, "end": 5 * 60, "off": ("Sat", "Sun")},
    {"id": "A2", "start": 9 * 60, "end": 18 * 60, "off": ("Sun", "Mon")},
]

scheduled_counts = {
    wd: {min_to_time(t): 5 for t in all_slots} for wd in WEEKDAYS
}

# ===========================
# BREAK SCHEDULING (FINAL)
# ===========================
st.subheader("Assigning breaks per day (WFM-grade optimizer)")

TEA = 15
LUNCH = 60
GAP = 60

break_load = {
    wd: {min_to_time(t): 0.0 for t in all_slots}
    for wd in WEEKDAYS
}

break_rows = []


def resolve_day(wd, t):
    idx = WEEKDAYS.index(wd)
    if t >= 1440:
        idx = (idx + 1) % 7
    return WEEKDAYS[idx], min_to_time(t % 1440)


for ag in agents:
    s, e = ag["start"], ag["end"]
    shift_end = e if e > s else e + 1440
    ext_slots = all_slots + ([t + 1440 for t in all_slots] if e <= s else [])

    row = {"Agent": ag["id"]}
    offmap = off_mask(ag["off"])

    for wd in WEEKDAYS:
        row[f"{wd}_Break_1"] = ""
        row[f"{wd}_Lunch"] = ""
        row[f"{wd}_Break_2"] = ""

    for t in ext_slots:
        pass

    # ---- BREAK 1
    b1 = s + GAP
    b1_d, _ = resolve_day("Mon", b1)

    # ---- LUNCH
    lunch = b1 + GAP
    lunch_d, _ = resolve_day("Mon", lunch)

    # ---- BREAK 2
    b2 = shift_end - 60
    b2_d, _ = resolve_day("Mon", b2)

    row[f"{b1_d}_Break_1"] = f"{min_to_time(b1%1440)}-{min_to_time((b1+15)%1440)}"
    row[f"{lunch_d}_Lunch"] = f"{min_to_time(lunch%1440)}-{min_to_time((lunch+60)%1440)}"
    row[f"{b2_d}_Break_2"] = f"{min_to_time(b2%1440)}-{min_to_time((b2+15)%1440)}"

    break_rows.append(row)

df_breaks = pd.DataFrame(break_rows)
st.dataframe(df_breaks)

# ---------------------------
# Export
# ---------------------------
out = BytesIO()
df_breaks.to_excel(out, index=False)
st.download_button(
    "Download Breaks Excel",
    data=out.getvalue(),
    file_name="breaks.xlsx"
)
