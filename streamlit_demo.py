import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---------------------------------------------------------------------------
# Config — must match xgboost_cross.py
# ---------------------------------------------------------------------------
HNEI_DATA_PATH = "Battery_RUL.csv"
NASA_DATA_PATH = "nasa_battery_cycles.csv"
CKPT_PATH      = "checkpoint_xgboost_cross_standard.joblib"
SCALER_TYPE    = "standard"   # matches the checkpoint above

FEATURE_COLS = [
    "Discharge Time (s)",
    "Decrement 3.6-3.4V (s)",
    "Max. Voltage Dischar. (V)",
    "Min. Voltage Charg. (V)",
    "Time at 4.15V (s)",
    "Time constant current (s)",
    "Charging time (s)",
]


# ---------------------------------------------------------------------------
# Data helpers (cached so they only run once per session)
# ---------------------------------------------------------------------------

def split_into_battery_segments(dataframe: pd.DataFrame) -> list[pd.DataFrame]:
    diffs      = np.diff(dataframe["RUL"].values)
    boundaries = np.where(diffs > 0)[0] + 1
    starts     = np.concatenate([[0], boundaries])
    ends       = np.concatenate([boundaries, [len(dataframe)]])
    return [dataframe.iloc[s:e].reset_index(drop=True) for s, e in zip(starts, ends)]


def fit_domain_scaler(
    train_segs: list[pd.DataFrame],
) -> tuple[MinMaxScaler | StandardScaler, np.ndarray, np.ndarray]:
    vals  = pd.concat(train_segs)[FEATURE_COLS].values
    lower = np.percentile(vals, 1, axis=0)
    upper = np.percentile(vals, 99, axis=0)
    scaler = MinMaxScaler() if SCALER_TYPE == "minmax" else StandardScaler()
    scaler.fit(np.clip(vals, lower, upper))
    return scaler, lower, upper


def apply_scaler(
    seg: pd.DataFrame,
    scaler: MinMaxScaler | StandardScaler,
    lower: np.ndarray,
    upper: np.ndarray,
) -> pd.DataFrame:
    s = seg.copy()
    s[FEATURE_COLS] = scaler.transform(np.clip(s[FEATURE_COLS].values, lower, upper))
    return s


@st.cache_resource
def load_model():
    return joblib.load(CKPT_PATH)


@st.cache_data
def prepare_demo_data():
    hnei_df = pd.read_csv(HNEI_DATA_PATH)
    nasa_df = pd.read_csv(NASA_DATA_PATH)
    hnei_df["Is_NASA"] = 0
    nasa_df["Is_NASA"] = 1

    hnei_segments = split_into_battery_segments(hnei_df)
    nasa_segments = split_into_battery_segments(nasa_df)

    # reserve last battery of each domain (same as xgboost_cross.py)
    demo_hnei = hnei_segments[-1]
    demo_nasa = nasa_segments[-1]

    # refit scalers on training data (everything except the last battery)
    hnei_scaler, hnei_lower, hnei_upper = fit_domain_scaler(hnei_segments[:-1])
    nasa_scaler, nasa_lower, nasa_upper = fit_domain_scaler(nasa_segments[:-1])

    demo_hnei_scaled = apply_scaler(demo_hnei, hnei_scaler, hnei_lower, hnei_upper)
    demo_nasa_scaled = apply_scaler(demo_nasa, nasa_scaler, nasa_lower, nasa_upper)

    return demo_hnei, demo_nasa, demo_hnei_scaled, demo_nasa_scaled


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Battery RUL Demo", layout="centered")
st.title("Battery RUL Prediction — XGBoost Demo")

model = load_model()
demo_hnei, demo_nasa, demo_hnei_scaled, demo_nasa_scaled = prepare_demo_data()

domain = st.radio("Select demo battery", ["HNEI", "NASA"], horizontal=True)

if domain == "HNEI":
    raw_seg    = demo_hnei
    scaled_seg = demo_hnei_scaled
    color      = "steelblue"
else:
    raw_seg    = demo_nasa
    scaled_seg = demo_nasa_scaled
    color      = "darkorange"

st.markdown(f"**Battery cycles:** {len(raw_seg)}")

if not st.button("Predict"):
    st.stop()

# build feature matrix with Is_NASA flag
X_demo = scaled_seg[FEATURE_COLS + ["Is_NASA"]].values.astype(np.float32)
y_true = raw_seg["RUL"].values

y_pred = model.predict(X_demo)

# metrics
mae  = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2   = 1 - ss_res / ss_tot

col1, col2, col3 = st.columns(3)
col1.metric("MAE",  f"{mae:.1f}")
col2.metric("RMSE", f"{rmse:.1f}")
col3.metric("R²",   f"{r2:.4f}")

# RUL over cycles plot
fig, ax = plt.subplots(figsize=(9, 4))
cycles = np.arange(len(y_true))
ax.plot(cycles, y_true, label="True RUL",      color="gray",  linewidth=1.5)
ax.plot(cycles, y_pred, label="Predicted RUL", color=color,   linewidth=1.5, linestyle="--")
ax.set_xlabel("Cycle index")
ax.set_ylabel("RUL (cycles)")
ax.set_title(f"RUL over cycles — {domain} demo battery")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# scatter plot
fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.scatter(y_true, y_pred, alpha=0.4, s=15, color=color)
lo, hi = y_true.min(), y_true.max()
ax2.plot([lo, hi], [lo, hi], "--", color="red")
ax2.set_xlabel("True RUL")
ax2.set_ylabel("Predicted RUL")
ax2.set_title("Predicted vs True RUL")
ax2.grid(True)
st.pyplot(fig2)
