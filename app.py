import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"       # prevent XGBoost from spawning OMP workers
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import plotly.graph_objects as go

# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "Discharge Time (s)",
    "Decrement 3.6-3.4V (s)",
    "Max. Voltage Dischar. (V)",
    "Min. Voltage Charg. (V)",
    "Time at 4.15V (s)",
    "Time constant current (s)",
    "Charging time (s)",
]

CKPT_CNN  = "checkpoints/checkpoint_cnn_rul.pt"
CKPT_XGB  = "checkpoints/xgboost_baseline_best.joblib"
CKPT_TFT  = "checkpoints/tft_nasa_weight_2p0_checkpoint.pt"

DATASETS = {
    "HNEI": ("HNEI_STANDARD_SCALED.csv", 0),
    "NASA": ("NASA_STANDARD_SCALED.csv", 1),
}

WINDOW_SIZE = 10  # cycles passed to each model

# ── Model definitions ─────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=k // 2)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CNNWithStatic(nn.Module):
    """Matches checkpoint_cnn_rul.pt: ConvEncoder + Is_NASA static branch."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(7, 64),    # encoder.0
            ConvBlock(64, 128),  # encoder.1
            nn.MaxPool1d(2),     # encoder.2
            ConvBlock(128, 128), # encoder.3
            ConvBlock(128, 64),  # encoder.4
        )
        self.pool           = nn.AdaptiveAvgPool1d(1)
        self.static_encoder = nn.Sequential(nn.Linear(1, 8), nn.ReLU())
        self.regressor      = nn.Sequential(
            nn.Linear(72, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x, is_nasa):
        h = self.pool(self.encoder(x)).squeeze(-1)          # (B, 64)
        s = self.static_encoder(is_nasa.view(-1, 1)).view(h.shape[0], -1)  # (B, 8)
        return self.regressor(torch.cat([h, s], dim=1))                  # (B, 1)

# ── TFT Hybrid (CNN + simplified Temporal Fusion) ─────────────────────────────

class _TFTConvBlock(nn.Module):
    """ConvBlock matching state dict layout: .block.0 = Conv1d, .block.1 = BN."""
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=k // 2),
            nn.BatchNorm1d(out_ch),
        )
    def forward(self, x):
        return F.relu(self.block(x))

class _GRN(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim)
        self.fc2  = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        return self.norm(g * h + (1 - g) * x)

class _GatedAddNorm(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, residual):
        g = torch.sigmoid(self.gate(x))
        return self.norm(g * x + residual)

class _TemporalFusion(nn.Module):
    def __init__(self, dim=64, n_heads=4):
        super().__init__()
        self.input_grn                = _GRN(dim)
        self.lstm                     = nn.LSTM(dim, dim, batch_first=False)
        self.post_lstm_gate_norm      = _GatedAddNorm(dim)
        self.attention                = nn.MultiheadAttention(dim, n_heads, batch_first=False)
        self.post_attention_gate_norm = _GatedAddNorm(dim)
        self.positionwise_grn         = _GRN(dim)

class TFTHybrid(nn.Module):
    """Matches tft_nasa_weight_2p0_checkpoint.pt: 3×ConvBlock → TemporalFusion → static → head."""
    def __init__(self):
        super().__init__()
        self.conv_feature_extractor = nn.ModuleList([
            _TFTConvBlock(7, 64),
            _TFTConvBlock(64, 64),
            _TFTConvBlock(64, 64),
        ])
        self.temporal_fusion = _TemporalFusion(64, 4)
        self.static_branch   = nn.Sequential(nn.Linear(1, 8), nn.ReLU())
        self.regression_head = nn.Sequential(
            nn.Linear(72, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x, is_nasa):
        for blk in self.conv_feature_extractor:
            x = blk(x)                                           # (B, 64, W)
        x = x.permute(2, 0, 1)                                  # (W, B, 64)

        tf = self.temporal_fusion
        x_grn = tf.input_grn(x)                                 # (W, B, 64)
        lstm_out, _ = tf.lstm(x_grn)                            # (W, B, 64)
        x_lstm = tf.post_lstm_gate_norm(lstm_out, x_grn)        # (W, B, 64)
        attn_out, _ = tf.attention(x_lstm, x_lstm, x_lstm)      # (W, B, 64)
        x_attn = tf.post_attention_gate_norm(attn_out, x_lstm)  # (W, B, 64)
        x_out  = tf.positionwise_grn(x_attn)                    # (W, B, 64)

        h = x_out[-1]                                           # (B, 64) last step
        s = self.static_branch(is_nasa.view(-1, 1))             # (B, 8)
        return self.regression_head(torch.cat([h, s], dim=1))   # (B, 1)

# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_cnn():
    model = CNNWithStatic()
    model.load_state_dict(torch.load(CKPT_CNN, map_location="cpu", weights_only=True))
    model.eval()
    return model

@st.cache_resource
def load_xgb():
    return joblib.load(CKPT_XGB)

@st.cache_resource
def load_tft():
    model = TFTHybrid()
    ckpt  = torch.load(CKPT_TFT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

@st.cache_data
def load_dataset(name):
    path, _ = DATASETS[name]
    return pd.read_csv(path)

# ── Inference helpers ─────────────────────────────────────────────────────────

def predict_cnn(model, window_np, is_nasa_val):
    """window_np: (WINDOW_SIZE, 7)"""
    x = torch.tensor(window_np, dtype=torch.float32).T.unsqueeze(0)  # (1, 7, W)
    isn = torch.tensor([[is_nasa_val]], dtype=torch.float32)          # (1, 1)
    with torch.no_grad():
        return max(0.0, model(x, isn).item())

def predict_xgb(bundle, last_row_np):
    """last_row_np: (7,) — XGBoost is a per-cycle model, use last cycle of window."""
    X = last_row_np.reshape(1, -1).astype(np.float32)
    # Replace any NaN with column median as fallback (imputer has sklearn version mismatch)
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
    return max(0.0, float(bundle["model"].predict(X)[0]))

def predict_tft(model, window_np, is_nasa_val):
    """window_np: (WINDOW_SIZE, 7)"""
    x   = torch.tensor(window_np, dtype=torch.float32).T.unsqueeze(0)  # (1, 7, W)
    isn = torch.tensor([[is_nasa_val]], dtype=torch.float32)
    with torch.no_grad():
        return max(0.0, model(x, isn).item())

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Battery RUL Prediction", layout="wide")
st.title("Battery RUL Prediction")

# Check all checkpoints
missing = [n for n, p in [("CNN", CKPT_CNN), ("XGBoost", CKPT_XGB), ("TFT", CKPT_TFT)]
           if not os.path.exists(p)]
if missing:
    st.error(f"Missing checkpoints: {', '.join(missing)}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")

dataset_name = st.sidebar.radio("Dataset", list(DATASETS.keys()))
model_choice = st.sidebar.radio("Model", ["CNN", "XGBoost", "TFT", "All"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Window size:** {WINDOW_SIZE} cycles")
st.sidebar.markdown(f"**Predicts:** RUL at cycle {WINDOW_SIZE + 1}")

# ── Session state for sampled window ──────────────────────────────────────────
state_key = f"start_{dataset_name}"
if state_key not in st.session_state:
    st.session_state[state_key] = None

if st.sidebar.button("Random Sample", type="primary"):
    df_full = load_dataset(dataset_name)
    max_start = len(df_full) - WINDOW_SIZE - 1
    if max_start < 0:
        st.warning(f"Dataset too short (need > {WINDOW_SIZE + 1} rows).")
        st.stop()
    st.session_state[state_key] = int(np.random.randint(0, max_start + 1))

start = st.session_state[state_key]

if start is None:
    st.info("Select a dataset and model, then click **Random Sample**.")
    st.stop()

# ── Extract window ────────────────────────────────────────────────────────────
df_full   = load_dataset(dataset_name)
_, is_nasa = DATASETS[dataset_name]

window_df = df_full.iloc[start : start + WINDOW_SIZE].copy()
true_rul  = float(df_full.iloc[start + WINDOW_SIZE]["RUL"])
window_np = window_df[FEATURE_COLS].values.astype(np.float32)

st.markdown(
    f"**Dataset:** {dataset_name} &nbsp;|&nbsp; "
    f"**Sampled rows:** {start}–{start + WINDOW_SIZE - 1} &nbsp;|&nbsp; "
    f"**Predicting RUL at row:** {start + WINDOW_SIZE}"
)

# ── Run inference ─────────────────────────────────────────────────────────────
cnn_model = load_cnn()
xgb_bundle = load_xgb()
tft_model = load_tft()

run_cnn = model_choice in ("CNN", "All")
run_xgb = model_choice in ("XGBoost", "All")
run_tft = model_choice in ("TFT", "All")

results = {}
errors  = {}

if run_cnn:
    results["CNN"] = predict_cnn(cnn_model, window_np, is_nasa)

if run_xgb:
    results["XGBoost"] = predict_xgb(xgb_bundle, window_np[-1])

if run_tft:
    results["TFT"] = predict_tft(tft_model, window_np, is_nasa)

# ── Metric cards ──────────────────────────────────────────────────────────────
model_colors = {"CNN": "#4C9BE8", "XGBoost": "#2A9D8F", "TFT": "#F4A261"}

num_cols = 1 + len(results) + len(errors)
metric_cols = st.columns(num_cols)
metric_cols[0].metric("True RUL", f"{true_rul:.1f} cycles")

for i, (name, pred) in enumerate(results.items(), 1):
    err = pred - true_rul
    metric_cols[i].metric(name, f"{pred:.1f} cycles", delta=f"{err:+.1f}", delta_color="inverse")

for name, msg in errors.items():
    st.warning(f"**{name} error:** {msg}")

# ── Bar chart ─────────────────────────────────────────────────────────────────
if results:
    st.subheader("Predicted vs. True RUL")

    names  = ["True RUL"] + list(results.keys())
    values = [true_rul]   + list(results.values())
    colors = ["#888888"]  + [model_colors[n] for n in results]

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        width=0.45,
    ))
    fig.add_hline(y=true_rul, line_dash="dot", line_color="gray",
                  annotation_text="True RUL", annotation_position="top right")
    fig.update_layout(
        yaxis_title="RUL (cycles)",
        height=380,
        margin=dict(t=40, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Input window table ────────────────────────────────────────────────────────
st.subheader("Input Window — Feature Values")
display_df = window_df[FEATURE_COLS].copy()
display_df.index = [f"Cycle {start + i}" for i in range(WINDOW_SIZE)]
st.dataframe(
    display_df.style.background_gradient(axis=0, cmap="RdYlGn"),
    use_container_width=True,
)
