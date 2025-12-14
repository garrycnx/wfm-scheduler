# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
from io import BytesIO
from datetime import datetime, timedelta, time
import base64

# Matplotlib fix for Streamlit Cloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="AI Schedule Generator by - Data Quest")

# ---------------------------
# Erlang helpers
# ---------------------------
def erlang_c_Pw(a, c):
    if c <= a:
        return 1.0
    sum_terms = sum((a**k) / math.factorial(k) for k in range(c))
    a_c = a**c / math.factorial(c)
    return (a_c * (c / (c - a))) / (sum_terms + a_c * (c / (c - a)))

def erlang_c_wait_prob_gt_t(a, c, mu, t):
    pw = erlang_c_Pw(a, c)
    exponent = - (c - a) * mu * t
    if exponent < -700:
        return 0.0
    return pw * math.exp(exponent)

def erlang_a_estimates(a, c, mu, theta, t_sla_min):
    """
    Engineering approximation to get:
      - pw (delay prob), p_wait_gt_t, p_abandon_any, sla_est
    """
    if c <= a:
        return 1.0, 1.0, 1.0, 0.0
    pw = erlang_c_Pw(a, c)
    expected_wait = 1.0 / ((c - a) * mu) if (c - a) * mu > 0 else 1e6
    p_abandon_any = pw * (1 - math.exp(-theta * expected_wait))
    p_wait_gt_t = pw * math.exp(- (c - a) * mu * t_sla_min)
    p_abandon_before_t = pw * (1 - math.exp(-theta * min(expected_wait, t_sla_min)))
    sla_est = max(0.0, 1.0 - p_wait_gt_t - p_abandon_before_t)
    return pw, p_wait_gt_t, p_abandon_any, sla_est

# ---------------------------
# Required servers (modify to include abandon constraint)
# ---------------------------
def required_servers_for_SLA_and_abandon(arrivals_per_interval, aht_minutes, sla_fraction, sla_seconds, abandon_fraction, patience_seconds):
    """
    Returns minimum integer servers c such that:
      - P(wait <= sla_seconds) >= sla_fraction
      - P(abandon_any) <= abandon_fraction   (engineering approx using Erlang-A)
    """
    if arrivals_per_interval <= 0:
        return 0
    interval_minutes = 30.0
    lam = arrivals_per_interval / interval_minutes  # per-minute arrivals
    mu = 1.0 / aht_minutes
    a = lam / mu
    t = sla_seconds / 60.0
    theta = 1.0 / (patience_seconds / 60.0) if patience_seconds > 0 else 0.0

    start = max(1, math.ceil(a))
    # search reasonable upper bound
    for c in range(start, max(start + 1, 500)):
        if c <= a:
            continue
        p_wait_gt_t = erlang_c_wait_prob_gt_t(a, c, mu, t)
        sla_ok = (1.0 - p_wait_gt_t) >= sla_fraction
        # compute abandon using erlang-a approx
        _, _, p_abandon_any, _ = erlang_a_estimates(a, c, mu, theta, t)
        abandon_ok = p_abandon_any <= abandon_fraction
        if sla_ok and abandon_ok:
            return c
    # fallback
    return math.ceil(a) + 1

# ---------------------------
# Helpers
# ---------------------------
WEEKDAYS = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

def parse_weekday(s):
    try:
        dt = pd.to_datetime(s, dayfirst=True)
        return dt.strftime("%a")[:3]
    except:
        return None

def time_to_min(tstr):
    if pd.isna(tstr): return None
    s = str(tstr).strip()
    if " " in s:
        s = s.split()[-1]
    for fmt in ("%H:%M","%H:%M:%S","%H.%M"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.hour*60 + dt.minute
        except:
            pass
    parts = s.split(":")
    if len(parts) >= 2:
        try:
            return int(parts[0])*60 + int(parts[1])
        except:
            return None
    return None

def min_to_time(m):
    h = (m // 60) % 24; mm = m % 60
    return f"{h:02d}:{mm:02d}"

# ---------------------------
# UI
# ---------------------------
st.title("AI Schedules Generator by Data Quest")
st.markdown("<p style='font-size:12px; color:Blue;'>Tool developed by <b>Gurpreet Singh</b></p>", unsafe_allow_html=True)
with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload forecast CSV (one week, DD-MM-YYYY)", type=["csv"])
    aht_seconds = st.slider("AHT (seconds)", 1, 1000, 360)
    aht = aht_seconds / 60.0
    sla_pct = st.slider("SLA target (%)", 50, 99, 80)
    sla_seconds = st.slider("SLA threshold (seconds)", 5, 300, 20)
    abandon_pct_target = st.slider("Abandon target (%)", 0, 50, 5)
    patience_seconds = st.slider("Average patience (seconds)", 10, 300, 120)
    earliest = st.time_input("Earliest shift start", value=time(5,30))
    latest = st.time_input("Latest shift start", value=time(18,0))
    max_agents = st.number_input("Max agents cap", min_value=10, max_value=5000, value=800)
    run = st.button("Generate Roster")

if uploaded is None:
    st.info("Upload CSV to proceed.")
    st.stop()

# ---------------------------
# Read and canonicalize forecast
# ---------------------------
try:
    df_in = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# auto-detect columns
cols = {c.lower(): c for c in df_in.columns}
col_map = {}
for k in cols:
    if "date" in k:
        col_map["date"] = cols[k]
    if "interval" in k or "time" in k or "slot" in k:
        col_map.setdefault("interval", cols[k])
    if "volume" in k or "calls" in k or "forecast" in k:
        col_map["volume"] = cols[k]

if not all(k in col_map for k in ("date","interval","volume")):
    st.error("CSV must contain date, interval, volume columns.")
    st.stop()

df = df_in.rename(columns={col_map["date"]:"date", col_map["interval"]:"interval", col_map["volume"]:"volume"})[["date","interval","volume"]].copy()
df["weekday"] = df["date"].apply(parse_weekday)
df["slot_min"] = df["interval"].apply(time_to_min)
df["slot_label"] = df["slot_min"].apply(lambda x: min_to_time(x) if x is not None else None)
df = df[df["weekday"].notnull() & df["slot_min"].notnull()]

df_agg = df.groupby(["weekday","slot_min","slot_label"], as_index=False)["volume"].sum()
all_slots = sorted(df_agg["slot_min"].unique())

if not all_slots:
    st.error("No valid time slots found.")
    st.stop()

rows = []
for wd in WEEKDAYS:
    for s in all_slots:
        lbl = min_to_time(s)
        v = df_agg.loc[(df_agg["weekday"]==wd)&(df_agg["slot_min"]==s),"volume"]
        vol = float(v.iloc[0]) if not v.empty else 0.0
        rows.append({"weekday":wd,"slot_min":s,"slot_label":lbl,"volume":vol})
df_week = pd.DataFrame(rows)

pivot_fore = df_week.pivot(index="slot_label", columns="weekday", values="volume").reindex(columns=WEEKDAYS).fillna(0)
st.subheader("Forecast (calls / 30-min)")
st.dataframe(pivot_fore.head(48))

# ---------------------------
# Required staff using both SLA & Abandon target
# ---------------------------
st.subheader("Required staff (Erlang-C & Abandon constraint)")

sla_fraction = sla_pct/100.0
abandon_fraction = abandon_pct_target/100.0

# compute required per row
df_week["required"] = df_week["volume"].apply(lambda x: required_servers_for_SLA_and_abandon(
    x, aht, sla_fraction, sla_seconds, abandon_fraction, patience_seconds
))
pivot_req = df_week.pivot(index="slot_label", columns="weekday", values="required").reindex(columns=WEEKDAYS).fillna(0)
st.dataframe(pivot_req.head(48))

# visualization
fig, axes = plt.subplots(1,2,figsize=(14,4))
axes[0].imshow(pivot_fore.T.values, aspect="auto"); axes[0].set_title("Forecast heatmap")
axes[1].imshow(pivot_req.T.values, aspect="auto"); axes[1].set_title("Required staff heatmap")
st.pyplot(fig)

# ---------------------------
# Shift templates & greedy scheduler
# ---------------------------
SHIFT_MIN = 9*60
min_start = earliest.hour*60 + earliest.minute
max_start = latest.hour*60 + latest.minute
shift_templates = [{"start":s,"end":s+SHIFT_MIN} for s in range(min_start, max_start+1, 30)]

def covers(start, slot): return (start <= slot) and (slot < start + SHIFT_MIN)

def compute_total_remaining(reqd):
    return sum(sum(reqd[wd].values()) for wd in WEEKDAYS)

if run:
    st.subheader("Generating roster and optimizing...")

    # required dict initial
    required = {wd: {min_to_time(t): int(df_week.loc[(df_week["weekday"]==wd)&(df_week["slot_min"]==t),"required"].iloc[0]) for t in all_slots} for wd in WEEKDAYS}

    off_pairs = [("Sun","Mon"),("Mon","Tue"),("Tue","Wed"),("Wed","Thu"),("Thu","Fri"),("Fri","Sat"),("Sat","Sun")]
    wd_index = {d:i for i,d in enumerate(WEEKDAYS)}
    def off_mask(pair):
        m=[1]*7
        for d in pair:
            m[wd_index[d]]=0
        return m

    agents=[]
    aid=1
    safety=0
    MAX_AGENTS=int(max_agents)

    # initial greedy cover
    while compute_total_remaining(required) > 0 and len(agents) < MAX_AGENTS and safety < 8000:
        safety += 1
        best_need=0; best_wd=None; best_label=None
        for wd in WEEKDAYS:
            for lbl,need in required[wd].items():
                if need > best_need:
                    best_need=need; best_wd=wd; best_label=lbl
        if best_need<=0: break
        slot_min = time_to_min(best_label)
        # pick shift covering slot with best weekly coverage
        best_tpl=None; best_score=-1
        for tpl in shift_templates:
            if not covers(tpl["start"], slot_min): continue
            covered=[t for t in all_slots if covers(tpl["start"],t)]
            score = sum(required[wd][min_to_time(t)] for wd in WEEKDAYS for t in covered)
            if score > best_score:
                best_score=score; best_tpl=tpl
        if best_tpl is None:
            best_tpl={"start":slot_min - SHIFT_MIN//2, "end": slot_min - SHIFT_MIN//2 + SHIFT_MIN}
        # choose off pair
        best_off=None; best_off_score=-1
        for op in off_pairs:
            m = off_mask(op)
            covered=[t for t in all_slots if covers(best_tpl["start"],t)]
            sc=0
            for i,wd in enumerate(WEEKDAYS):
                if m[i]==0: continue
                for t in covered:
                    sc += required[wd][min_to_time(t)]
            if sc > best_off_score:
                best_off_score=sc; best_off=op
        agents.append({"id":f"A{aid}", "start":int(best_tpl["start"]), "end":int(best_tpl["end"]), "off":best_off})
        aid += 1
        # decrement
        m = off_mask(best_off)
        covered=[t for t in all_slots if covers(best_tpl["start"],t)]
        for i,wd in enumerate(WEEKDAYS):
            if m[i]==0: continue
            for t in covered:
                required[wd][min_to_time(t)] = max(0, required[wd][min_to_time(t)]-1)

    st.success(f"Initial greedy created {len(agents)} agents")

    # pruning redundant agents (try to remove one-by-one)
    st.markdown("Pruning redundant agents...")
    def build_schedule_counts(agent_list):
        sched = {wd:{min_to_time(t):0 for t in all_slots} for wd in WEEKDAYS}
        for ag in agent_list:
            m = off_mask(ag["off"])
            covered=[t for t in all_slots if covers(ag["start"],t)]
            for i,wd in enumerate(WEEKDAYS):
                if m[i]==0: continue
                for t in covered: sched[wd][min_to_time(t)] += 1
        return sched

    baseline_req = {wd:{min_to_time(r["slot_min"]): int(r["required"]) for _,r in df_week[df_week["weekday"]==wd].iterrows()} for wd in WEEKDAYS}
    pruned = agents.copy()
    improved=True; loops=0
    while improved and loops < 400:
        loops += 1; improved=False
        sched_counts = build_schedule_counts(pruned)
        for i in range(len(pruned)-1, -1, -1):
            test = pruned[:i] + pruned[i+1:]
            test_counts = build_schedule_counts(test)
            ok=True
            for wd in WEEKDAYS:
                for lbl, reqv in baseline_req[wd].items():
                    if test_counts[wd][lbl] < reqv:
                        ok=False; break
                if not ok: break
            if ok:
                pruned.pop(i); improved=True; break

    st.success(f"Pruning finished. Final agents: {len(pruned)} (was {len(agents)})")
    agents = pruned

    # Build roster table showing SHIFT or OFF explicitly
    roster=[]
    def shift_str(s,e): return f"{min_to_time(s)}–{min_to_time(e)}"
    for ag in agents:
        m = off_mask(ag["off"])
        row = {"Agent":ag["id"], "Shift Start":min_to_time(ag["start"]), "Shift End":min_to_time(ag["end"]), "Off Days":f"{ag['off'][0]},{ag['off'][1]}"}
        for i,wd in enumerate(WEEKDAYS):
            row[wd] = "OFF" if m[i]==0 else shift_str(ag["start"], ag["end"])
        roster.append(row)
    df_roster = pd.DataFrame(roster)
    st.subheader("Roster (shift shown; OFF for off-days)")
    st.dataframe(df_roster.head(200))

    # ---------------------------
    # Build  sscheduled counts (before breaks)
    scheduled_counts = build_schedule_counts(agents)

    # ---------------------------
    # WFM-GRADE Break scheduling PER DAY (overnight-safe, 15-min tea)
    # ---------------------------
    st.subheader("Assigning breaks per day (WFM-grade optimizer)")

    TEA_BREAK_MIN = 15
    LUNCH_MIN = 60
    MIN_GAP = 60
    BREAK_PENALTY = 3
    TEA_IMPACT = 0.5

    req_lookup = baseline_req

    # ---------------------------
    # Helpers for overnight handling
    # ---------------------------
    def resolve_day_and_label(wd, t):
        day_idx = WEEKDAYS.index(wd)
        if t >= 1440:
            day_idx = (day_idx + 1) % 7
        return WEEKDAYS[day_idx], min_to_time(t % 1440)

    def slot_label_30(t):
        return min_to_time((t // 30) * 30)

    def generate_tea_slots(slots):
        tea = []
        for t in slots:
            tea.append(t)
            tea.append(t + 15)
        return tea

    # ---------------------------
    # Track break congestion
    # ---------------------------
    break_load = {
        wd: {min_to_time(t): 0.0 for t in all_slots}
        for wd in WEEKDAYS
    }

    break_rows = []

    # ---------------------------
    # Main break loop
    # ---------------------------
    for ag in agents:
        s, e = ag["start"], ag["end"]

        shift_end = e if e > s else e + 1440
        extended_slots = all_slots if e > s else all_slots + [t + 1440 for t in all_slots]

        row = {
            "Agent": ag["id"],
            "Shift Start": min_to_time(s),
            "Shift End": min_to_time(e),
            "Off Days": f"{ag['off'][0]},{ag['off'][1]}"
        }

        m = off_mask(ag["off"])

        for i, wd in enumerate(WEEKDAYS):

            if m[i] == 0:
                row[f"{wd}_Break_1"] = ""
                row[f"{wd}_Lunch"] = ""
                row[f"{wd}_Break_2"] = ""
                continue

            slots = [
                t for t in extended_slots
                if s <= t and t + 30 <= shift_end
            ]

            if not slots:
                row[f"{wd}_Break_1"] = ""
                row[f"{wd}_Lunch"] = ""
                row[f"{wd}_Break_2"] = ""
                continue

            tea_slots = generate_tea_slots(slots)

            # ---------------------------
            # Slack calculation (overnight-safe)
            # ---------------------------
            slack = {}
            for t in slots:
                d, lbl = resolve_day_and_label(wd, t)
                slack[lbl] = (
                    scheduled_counts[d].get(lbl, 0)
                    - req_lookup[d].get(lbl, 0)
                )

            def tea_slack(t):
                d, lbl = resolve_day_and_label(wd, t)
                return slack.get(lbl, 0) - TEA_IMPACT

            # ---------------------------
            # BREAK 1 (15 min)
            # ---------------------------
            b1_slots = [
                t for t in tea_slots
                if s + MIN_GAP <= t <= s + 180
            ]

            def b1_score(t):
                d, lbl = resolve_day_and_label(wd, t)
                return tea_slack(t) - (break_load[d][lbl] ** 2) * BREAK_PENALTY

            best_b1 = max(b1_slots, key=b1_score, default=None)
            if not best_b1:
                continue

            # ---------------------------
            # LUNCH (60 min)
            # ---------------------------
            lunch_slots = [
                t for t in slots
                if (
                    t >= best_b1 + MIN_GAP
                    and t + 30 in slots
                    and t <= shift_end - (MIN_GAP + TEA_BREAK_MIN)
                )
            ]

            def lunch_score(t):
                d, lbl = resolve_day_and_label(wd, t)
                return (
                    slack.get(lbl, 0)
                    + slack.get(min_to_time((t + 30) % 1440), 0)
                    - (break_load[d][lbl] ** 2) * BREAK_PENALTY
                )

            best_lunch = max(lunch_slots, key=lunch_score, default=None)
            if not best_lunch:
                row[f"{wd}_Break_1"] = f"{min_to_time(best_b1 % 1440)}-{min_to_time((best_b1 + 15) % 1440)}"
                continue

            lunch_end = best_lunch + LUNCH_MIN

            # ---------------------------
            # BREAK 2 (15 min)
            # ---------------------------
            b2_slots = [
                t for t in tea_slots
                if (
                    t >= lunch_end + MIN_GAP
                    and t <= shift_end - MIN_GAP
                )
            ]

            def b2_score(t):
                d, lbl = resolve_day_and_label(wd, t)
                return tea_slack(t) - (break_load[d][lbl] ** 2) * BREAK_PENALTY

            best_b2 = max(b2_slots, key=b2_score, default=None)

            # ---------------------------
            # FINAL ASSIGNMENT
            # ---------------------------
            row[f"{wd}_Break_1"] = f"{min_to_time(best_b1 % 1440)}-{min_to_time((best_b1 + 15) % 1440)}"
            row[f"{wd}_Lunch"] = f"{min_to_time(best_lunch % 1440)}-{min_to_time((best_lunch + 60) % 1440)}"
            row[f"{wd}_Break_2"] = (
                f"{min_to_time(best_b2 % 1440)}-{min_to_time((best_b2 + 15) % 1440)}"
                if best_b2 else ""
            )

            # ---------------------------
            # UPDATE CONGESTION
            # ---------------------------
            d1, lbl1 = resolve_day_and_label(wd, best_b1)
            dl, lbll = resolve_day_and_label(wd, best_lunch)

            break_load[d1][lbl1] += TEA_IMPACT
            break_load[dl][lbll] += 1.0
            break_load[dl][min_to_time((best_lunch + 30) % 1440)] += 1.0

            if best_b2:
                d2, lbl2 = resolve_day_and_label(wd, best_b2)
                break_load[d2][lbl2] += TEA_IMPACT

        break_rows.append(row)

    df_breaks = pd.DataFrame(break_rows)
    st.dataframe(df_breaks.head(200))

    # ---------------------------
    # Recompute coverage after breaks (scheduled_counts mutated above)
    sched_df = pd.DataFrame({wd: [scheduled_counts[wd].get(min_to_time(t),0) for t in all_slots] for wd in WEEKDAYS}, index=[min_to_time(t) for t in all_slots])
    req_df = pd.DataFrame({wd: [int(df_week.loc[(df_week["weekday"]==wd)&(df_week["slot_min"]==t),"required"].iloc[0]) for t in all_slots] for wd in WEEKDAYS}, index=[min_to_time(t) for t in all_slots])
    diff_df = sched_df - req_df
    st.subheader("Coverage after breaks (scheduled - required)")
    st.dataframe(diff_df.head(20))

    # ---------------------------
    # Daily projections (SLA / Abandon / Occupancy) - daily average abandon target used for reporting
    # ---------------------------
    st.subheader("Daily projections (Erlang-A approx)")

    mu = 1.0 / aht
    theta = 1.0 / (patience_seconds / 60.0) if patience_seconds>0 else 0.0
    t_sla = sla_seconds / 60.0

    proj_rows=[]
    for wd in WEEKDAYS:
        tot_calls=0; sla_acc=0; abn_acc=0; occ_acc=0
        for t in all_slots:
            lbl = min_to_time(t)
            calls = int(df_week.loc[(df_week["weekday"]==wd)&(df_week["slot_min"]==t),"volume"].iloc[0])
            scheduled = scheduled_counts[wd].get(lbl,0)
            if scheduled == 0:
                sla_it = 0.0
                abn_it = 1.0 if calls>0 else 0.0
                occ_it = 0.0
            else:
                lampm = calls / 30.0
                a = lampm / mu
                _, _, p_abandon_any, sla_est = erlang_a_estimates(a, scheduled, mu, theta, t_sla)
                sla_it = sla_est
                abn_it = p_abandon_any
                occ_it = min((calls * aht) / (scheduled * 30.0), 1.0)
            tot_calls += calls
            sla_acc += sla_it * calls
            abn_acc += abn_it * calls
            occ_acc += occ_it * calls
        if tot_calls>0:
            proj_rows.append({"Day":wd, "Total Calls":int(tot_calls), "Projected SLA %":round(sla_acc/tot_calls*100,2), "Projected Abandon %":round(abn_acc/tot_calls*100,2), "Avg Occupancy %":round(occ_acc/tot_calls*100,2)})
        else:
            proj_rows.append({"Day":wd, "Total Calls":0, "Projected SLA %":100.0, "Projected Abandon %":0.0, "Avg Occupancy %":0.0})

    df_proj = pd.DataFrame(proj_rows)
    st.dataframe(df_proj)

    # ---------------------------
    # Export (use openpyxl engine)
    # ---------------------------
    def export_all(roster, breaks, proj, fore, req):
        out = BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            roster.to_excel(writer, sheet_name="Roster", index=False)
            breaks.to_excel(writer, sheet_name="Breaks", index=False)
            proj.to_excel(writer, sheet_name="Projections", index=False)
            fore.to_excel(writer, sheet_name="Forecast")
            req.to_excel(writer, sheet_name="Required")
            # no writer.save() needed
        return out.getvalue()


    excel_data = export_all(df_roster, df_breaks, df_proj, pivot_fore, pivot_req)
    b64 = base64.b64encode(excel_data).decode()
    st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="roster_output.xlsx">Download full output (Excel)</a>', unsafe_allow_html=True)
    st.download_button("Download roster CSV", data=df_roster.to_csv(index=False).encode(), file_name="roster.csv")
    st.download_button("Download breaks CSV", data=df_breaks.to_csv(index=False).encode(), file_name="breaks.csv")
    st.download_button("Download projections CSV", data=df_proj.to_csv(index=False).encode(), file_name="projections.csv")

    st.success("Done.")
else:
    st.info("Adjust settings and click Generate Roster.")
