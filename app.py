import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import pickle

# ── Feature metadata ──────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "Discharge Time (s)",
    "Decrement 3.6-3.4V (s)",
    "Max. Voltage Dischar. (V)",
    "Min. Voltage Charg. (V)",
    "Time at 4.15V (s)",
    "Time constant current (s)",
    "Charging time (s)",
]
FEATURE_KEYS = [f"F{i+1}" for i in range(7)]
DATA_PATH = "combined_scaled_battery_data.csv"

# ── Model definitions (must match training architecture) ──────────────────────

class CNNModel(nn.Module):
    def __init__(self, input_size=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 7)
        return self.net(x).squeeze(-1)


class TFTModel(nn.Module):
    """Simplified TFT-style model using transformer encoder."""
    def __init__(self, input_size=7, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)          # (B, 7, 1)
        x = self.input_proj(x)       # (B, 7, d_model)
        x = self.transformer(x)
        return self.head(x).squeeze(-1)


# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_cnn(path="checkpoints/cnn_model.pt"):
    model = CNNModel()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_tft(path="checkpoints/tft_model.pt"):
    model = TFTModel()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_xgb(path="checkpoints/xgb_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_cols = FEATURE_NAMES
    df = df[feature_cols + ["RUL"]].dropna()
    return df


def predict_all(features: np.ndarray, cnn, tft, xgb):
    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        cnn_pred = cnn(tensor).item()
        tft_pred = tft(tensor).item()
    xgb_pred = xgb.predict(features.reshape(1, -1))[0]
    return {"CNN": cnn_pred, "TFT": tft_pred, "XGBoost": xgb_pred}


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Battery RUL Predictor", layout="wide")
st.title("Battery Remaining Useful Life (RUL) Predictor")

# Check which checkpoints exist
checkpoint_dir = "checkpoints"
cnn_ready = os.path.exists(f"{checkpoint_dir}/cnn_model.pt")
tft_ready = os.path.exists(f"{checkpoint_dir}/tft_model.pt")
xgb_ready = os.path.exists(f"{checkpoint_dir}/xgb_model.pkl")
models_ready = cnn_ready and tft_ready and xgb_ready

if not models_ready:
    missing = [
        name for name, ready in [("CNN (cnn_model.pt)", cnn_ready),
                                   ("TFT (tft_model.pt)", tft_ready),
                                   ("XGBoost (xgb_model.pkl)", xgb_ready)]
        if not ready
    ]
    st.warning(f"Missing checkpoints: {', '.join(missing)}. Place them in `{checkpoint_dir}/` to enable inference.")

# ── Sidebar: feature sliders ──────────────────────────────────────────────────
st.sidebar.header("Cycle Feature Inputs")

# Compute per-feature ranges from data for sensible slider bounds
if os.path.exists(DATA_PATH):
    df = load_data()
    feature_stats = {col: (df[col].min(), df[col].max(), df[col].mean())
                     for col in FEATURE_NAMES}
else:
    df = None
    feature_stats = {col: (-5.0, 5.0, 0.0) for col in FEATURE_NAMES}

input_values = {}
for key, name in zip(FEATURE_KEYS, FEATURE_NAMES):
    lo, hi, default = feature_stats[name]
    input_values[key] = st.sidebar.slider(
        label=f"{key}: {name}",
        min_value=float(lo),
        max_value=float(hi),
        value=float(default),
        step=float((hi - lo) / 200),
        format="%.4f",
    )

features_array = np.array([input_values[k] for k in FEATURE_KEYS], dtype=np.float32)

predict_btn = st.sidebar.button("Predict RUL", type="primary", disabled=not models_ready)

# ── Main area ─────────────────────────────────────────────────────────────────
col_metrics, col_scatter = st.columns([1, 2])

if predict_btn and models_ready:
    cnn_model = load_cnn()
    tft_model = load_tft()
    xgb_model = load_xgb()

    preds = predict_all(features_array, cnn_model, tft_model, xgb_model)

    with col_metrics:
        st.subheader("Predicted RUL")
        colors = {"CNN": "#4C9BE8", "TFT": "#F4A261", "XGBoost": "#2A9D8F"}
        for model_name, pred in preds.items():
            st.metric(label=model_name, value=f"{pred:.1f} cycles")

        # Bar chart comparison
        fig_bar = go.Figure(go.Bar(
            x=list(preds.keys()),
            y=list(preds.values()),
            marker_color=list(colors.values()),
            text=[f"{v:.1f}" for v in preds.values()],
            textposition="outside",
        ))
        fig_bar.update_layout(
            yaxis_title="Predicted RUL (cycles)",
            height=300,
            margin=dict(t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_scatter:
        if df is not None:
            st.subheader("Predicted vs. True RUL — Dataset Overview")

            # Sample for performance
            sample = df.sample(min(2000, len(df)), random_state=42)
            X_sample = sample[FEATURE_NAMES].values.astype(np.float32)
            y_true = sample["RUL"].values

            cnn_model = load_cnn()
            tft_model = load_tft()
            xgb_model = load_xgb()

            with torch.no_grad():
                t = torch.tensor(X_sample)
                cnn_preds = cnn_model(t).numpy()
                tft_preds = tft_model(t).numpy()
            xgb_preds = xgb_model.predict(X_sample)

            model_preds_map = {"CNN": cnn_preds, "TFT": tft_preds, "XGBoost": xgb_preds}
            scatter_model = st.selectbox("Show scatter for:", list(model_preds_map.keys()))
            y_pred_scatter = model_preds_map[scatter_model]

            current_pred = preds[scatter_model]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_true, y=y_pred_scatter, mode="markers",
                marker=dict(color=colors[scatter_model], opacity=0.4, size=5),
                name="Dataset samples",
            ))
            # Perfect prediction line
            rng = [min(y_true.min(), y_pred_scatter.min()),
                   max(y_true.max(), y_pred_scatter.max())]
            fig.add_trace(go.Scatter(
                x=rng, y=rng, mode="lines",
                line=dict(color="gray", dash="dash"), name="Perfect prediction",
            ))
            # Highlight current input's prediction (no true label available)
            fig.add_trace(go.Scatter(
                x=[current_pred], y=[current_pred], mode="markers",
                marker=dict(color="red", size=14, symbol="star"),
                name=f"Your input ({current_pred:.1f})",
            ))
            fig.update_layout(
                xaxis_title="True RUL (cycles)",
                yaxis_title="Predicted RUL (cycles)",
                height=420,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    with col_metrics:
        st.info("Adjust features in the sidebar and click **Predict RUL**.")
    with col_scatter:
        if df is not None:
            st.subheader("Dataset RUL Distribution")
            fig = go.Figure(go.Histogram(x=df["RUL"], nbinsx=60,
                                         marker_color="#4C9BE8", opacity=0.8))
            fig.update_layout(xaxis_title="RUL (cycles)", yaxis_title="Count",
                               height=380, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
