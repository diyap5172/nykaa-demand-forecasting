import json

import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

st.set_page_config(page_title="Nykaa Demand Forecast", page_icon="💄", layout="wide")


@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model("nykaa_xgb.json")
    with open("nykaa_meta.json") as f:
        meta = json.load(f)
    return model, meta


@st.cache_data
def load_data():
    return pd.read_csv("Nykaa_Demand.csv", parse_dates=["date"])


def forecast(df_raw, model, meta, city, product, days, festival_days, sale_discount):
    """Predict day by day. Each prediction is added to history and used for the next day."""
    hist = df_raw[(df_raw["city"] == city) & (df_raw["product"] == product)].sort_values("date")
    values = list(hist["units_sold"].values)
    last_date = hist["date"].max()
    rows = []
    for i in range(1, days + 1):
        d = last_date + pd.Timedelta(days=i)
        is_fest = 1 if i <= festival_days else 0
        disc = sale_discount if is_fest else 0
        month, weekday = d.month, d.weekday()
        x = {
            "city_code": meta["city_map"][city],
            "product_code": meta["product_map"][product],
            "discount_pct": disc,
            "is_festival": is_fest,
            "month": month,
            "day": d.day,
            "weekday": weekday,
            "weekend": int(weekday >= 5),
            "m_sin": np.sin(month * 2 * np.pi / 12),
            "m_cos": np.cos(month * 2 * np.pi / 12),
            "lag_7": values[-7],
            "lag_14": values[-14],
            "lag_28": values[-28],
            "roll_mean_7": np.mean(values[-7:]),
            "roll_mean_28": np.mean(values[-28:]),
        }
        pred = max(0.0, float(model.predict(pd.DataFrame([x])[meta["features"]])[0]))
        values.append(pred)
        rows.append({"date": d, "predicted_units": int(round(pred))})
    return pd.DataFrame(rows)


model, meta = load_model()
df = load_data()

st.title("💄 Nykaa Demand Forecasting")
st.caption(
    "Predict how many units will sell and how much stock to reorder. "
    "Demo project using synthetic data modelled on Indian retail."
)

# ---------- Sidebar inputs ----------
st.sidebar.header("Choose inputs")
city = st.sidebar.selectbox("City", sorted(df["city"].unique()))
product = st.sidebar.selectbox("Product", sorted(df["product"].unique()))
days = st.sidebar.slider("Forecast days", 7, 30, 14)
festival_days = st.sidebar.slider("Festival / sale days at the start", 0, 7, 0)
sale_discount = st.sidebar.select_slider("Discount % on sale days", options=[0, 10, 15, 20, 25], value=10)
current_stock = st.sidebar.number_input("Current stock (units)", min_value=0, value=500, step=50)

# ---------- Forecast ----------
fc = forecast(df, model, meta, city, product, days, festival_days, sale_discount)
total = int(fc["predicted_units"].sum())
safety = int(0.10 * total)
reorder = max(0, total + safety - int(current_stock))

c1, c2, c3 = st.columns(3)
c1.metric(f"Predicted demand ({days} days)", f"{total} units")
c2.metric("Average per day", f"{total / days:.0f} units")
c3.metric("Suggested reorder", f"{reorder} units")

if reorder > 0:
    st.warning(
        f"Stock may run out. Reorder about {reorder} units "
        f"(demand {total} + 10% safety stock {safety} - current stock {int(current_stock)})."
    )
else:
    st.success("Current stock is enough for this period.")

# ---------- Chart ----------
st.subheader(f"{product} in {city}: last 60 days and forecast")
hist = df[(df["city"] == city) & (df["product"] == product)].set_index("date")["units_sold"].tail(60).rename("Actual")
pred = fc.set_index("date")["predicted_units"].rename("Forecast")
st.line_chart(pd.concat([hist, pred], axis=1))

with st.expander("See forecast table"):
    st.dataframe(fc, use_container_width=True)

st.caption(f"History in this demo ends on {df['date'].max().date()}. Forecast starts the next day.")
