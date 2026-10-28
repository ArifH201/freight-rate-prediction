"""Streamlit UI: predict a load's rate, see the December chart and validation results.
Run:  streamlit run app.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from freight_model import FreightModel, haversine_miles

st.set_page_config(page_title="Freight Rate Predictor", page_icon="🚚", layout="wide")


@st.cache_resource(show_spinner="Training model on 48,000 loads (about a minute, once)...")
def load():
    tr = pd.read_csv(ROOT / "data" / "train_test.csv", parse_dates=["date"])
    va = pd.read_csv(ROOT / "data" / "validation.csv", parse_dates=["date"])
    model = FreightModel().fit(tr)
    cities = {}
    for side in ("pickup", "delivery"):
        for _, r in tr.drop_duplicates(side).iterrows():
            cities[r[side]] = (r[f"{side}_lat"], r[f"{side}_lon"])
    for side in ("pickup", "delivery"):                      # include the 8 validation-only cities
        for _, r in va.drop_duplicates(side).iterrows():
            cities.setdefault(r[side], (r[f"{side}_lat"], r[f"{side}_lon"]))
    mi = pd.concat([tr.groupby("date").market_index.median(), va.groupby("date").market_index.median()])
    return tr, va, model, cities, mi


tr, va, model, cities, mi_by_date = load()
st.title("🚚 Freight Rate Predictor")
st.caption("LightGBM lane model + market adjustment | trained on Jan–Oct 2025 loads | validated with forward-chaining")

tab1, tab2, tab3, tab4 = st.tabs(["Predict a load", "December chart", "Validation results", "Data findings"])

with tab1:
    c1, c2 = st.columns(2)
    names = sorted(cities)
    pickup = c1.selectbox("Pickup city", names, index=names.index("Lexington"))
    delivery = c2.selectbox("Delivery city", names, index=names.index("Fort Wayne"))
    c3, c4, c5 = st.columns(3)
    equipment = c3.selectbox("Equipment", ["Dry Van", "Flatbed", "Reefer"])
    weight = c4.number_input("Weight (lb)", 1000, 47500, 32000, 500)
    date = c5.date_input("Ship date", pd.Timestamp("2025-12-01"))
    (la1, lo1), (la2, lo2) = cities[pickup], cities[delivery]
    est = float(max(70.0, 1.18 * haversine_miles(la1, lo1, la2, lo2)))
    distance = st.number_input("Distance (miles) – estimated from the cities, editable", 70.0, 3500.0, round(est, 0), 10.0)
    if pickup == delivery:
        st.warning("Pickup and delivery are the same city.")
    row = pd.DataFrame({"distance": [distance], "equipment": [equipment], "weight": [float(weight)],
                        "pickup_lat": [la1], "pickup_lon": [lo1], "delivery_lat": [la2], "delivery_lon": [lo2],
                        "date": [pd.Timestamp(date)]})
    rate = float(model.predict(row, mi_by_date=mi_by_date)[0])
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted rate", f"${rate:,.0f}")
    m2.metric("Per mile", f"${rate / distance:,.2f}")
    known = pd.Timestamp(date) in mi_by_date.index
    m3.metric("Market index used", f"{mi_by_date.get(pd.Timestamp(date), model.recent_mi):.2f}",
              help="Same-day market index from the data" if known else "No data for that date – using the recent average")

with tab2:
    dec = pd.read_csv(ROOT / "outputs" / "december_chart_inputs_completed.csv")
    st.subheader("Fixed December 2025 prediction: Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb")
    st.image(str(ROOT / "scorer_results" / "candidate_december.png"), caption="Chart produced by the provided score.py")
    st.dataframe(dec[["date", "predicted_rate"]], width="stretch", height=250)

with tab3:
    st.subheader("Forward-chaining validation (train on earlier months → test on the next)")
    st.image(str(ROOT / "outputs" / "fig_split.png"))
    s = pd.read_csv(ROOT / "outputs" / "cv_summary.csv")
    st.dataframe(s.round(2), width="stretch")
    st.subheader("Final model, per test month")
    st.dataframe(pd.read_csv(ROOT / "outputs" / "cv_final_per_fold.csv").round(2), width="stretch")
    st.subheader("Why quote_signal was excluded")
    st.dataframe(pd.read_csv(ROOT / "outputs" / "cv_quote_signal_ablation.csv").round(1), width="stretch")
    st.caption("MAE with quote_signal is worse in August, the month that looks most like the validation period.")

with tab4:
    st.subheader("quote_signal is unreliable")
    st.image(str(ROOT / "outputs" / "fig_quote_signal.png"))
    a, b, c = st.columns(3)
    a.metric("Negative weights (train)", int((tr.weight < 0).sum()))
    b.metric("Missing weights (train)", int(tr.weight.isna().sum()))
    c.metric("Missing market_index (train)", int(tr.market_index.isna().sum()))
    seen = set(tr.pickup) | set(tr.delivery)
    new = sorted((set(va.pickup) | set(va.delivery)) - seen)
    st.write(f"**Cities only in validation ({len(new)}):** {', '.join(new)} – handled by using coordinates, not city names.")
    monthly = tr.assign(rpm=tr.posted_rate / tr.distance).groupby(tr.date.dt.month).rpm.median()
    st.subheader("Median rate per mile by month (train)")
    st.line_chart(monthly)