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

CKPT_CNN         = "checkpoints/checkpoint_cnn_rul.pt"
CKPT_XGB         = "checkpoints/checkpoint_xgboost_cross_standard.joblib"
CKPT_TFT         = "checkpoints/tft_nasa_weight_2p0_checkpoint.pt"
CKPT_TL_BASELINE = "checkpoints/ckpt_fewshot_baseline.pt"
CKPT_TL_TRANSFER = "checkpoints/ckpt_fewshot_transfer.pt"
TL_DATASET_PATH  = "best_demo_battery_features.csv"

# Demo batteries are the ones excluded from all training/val/test splits in tft.ipynb
# (segment_ids 1 and 40, picked with seed 42). Using these ensures fair evaluation.
DATASETS = {
    "HNEI": ("HNEI_TFT_SCALED.csv", 0),
    "NASA": ("NASA_TFT_SCALED.csv", 1),
}


# ── Scaling constants ─────────────────────────────────────────────────────────
# combined_scaled_battery_data.csv stats (used by CNN/TFT — display unscaling)
_MEANS = {
    "HNEI": np.array([1584.997, 470.916,   3.904, 3.581,  2963.343, 3811.899,  8299.621]),
    "NASA": np.array([3225.380, 820.715,   4.152, 3.336,  8854.451, 1793.840, 10427.877]),
}
_STDS = {
    "HNEI": np.array([1111.032, 302.688,   0.083, 0.119,  1244.820, 1576.372, 1078.826]),
    "NASA": np.array([1385.771, 639.193,   0.357, 0.452,  1768.100, 1064.731, 1039.556]),
}

# XGBoost-specific scaler stats (StandardScaler fit on training batteries only,
# with 1st–99th percentile clipping — reproduced from xgboost_cross.py with seed 42)
_XGB_MEANS = {
    "HNEI": np.array([3706.960902,  475.815189,    3.905020,    3.577858,
                      3531.167047, 4571.447794, 9038.368170]),
    "NASA": np.array([3233.128842,  805.012754,    4.167581,    3.319760,
                      8834.630621, 1806.651505, 10455.114178]),
}
_XGB_STDS = {
    "HNEI": np.array([19568.378821,  195.665861,    0.083549,    0.118819,
                       5283.927426, 6906.258161,  6366.051059]),
    "NASA": np.array([ 1335.023931,  642.048276,    0.165694,    0.456822,
                       1810.528347, 1064.012524,   916.599969]),
}
_XGB_LOWER = {
    "HNEI": np.array([  838.934400,  192.333333,    3.748000,    3.313960,
                        964.940094, 1520.558800, 7031.225600]),
    "NASA": np.array([  586.719000,    0.000000,    2.571725,    0.338425,
                          7.907980,    0.000000, 3962.583420]),
}
_XGB_UPPER = {
    "HNEI": np.array([187300.791000, 1295.323035,    4.238010,    3.933010,
                       52114.484160, 66707.294300, 66887.683500]),
    "NASA": np.array([  6533.499660, 2908.261700,    4.207383,    4.143350,
                       10803.216640, 3514.551840, 10817.635560]),
}

def unscale_window(scaled_df, dataset_name):
    """Return a copy with FEATURE_COLS converted back to physical units for display."""
    means, stds = _MEANS[dataset_name], _STDS[dataset_name]
    out = scaled_df.copy()
    out[FEATURE_COLS] = np.clip(scaled_df[FEATURE_COLS].values * stds + means, 0, None)
    return out

def rescale_for_xgb(last_row_np, dataset_name):
    """Convert one row from combined_scaled space to XGBoost-scaler space."""
    raw     = last_row_np * _STDS[dataset_name] + _MEANS[dataset_name]
    clipped = np.clip(raw, _XGB_LOWER[dataset_name], _XGB_UPPER[dataset_name])
    return (clipped - _XGB_MEANS[dataset_name]) / _XGB_STDS[dataset_name]

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
        residual = x
        h = self.fc2(F.elu(self.fc1(x)))       # fc1 → elu → fc2
        gated = torch.sigmoid(self.gate(h)) * h # gate on fc2 output
        return self.norm(residual + gated)       # residual + gated (matches GatedResidualNetwork)

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
        self.lstm                     = nn.LSTM(dim, dim, batch_first=True)
        self.post_lstm_gate_norm      = _GatedAddNorm(dim)
        self.attention                = nn.MultiheadAttention(dim, n_heads, batch_first=True)
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
        x = x.transpose(1, 2)                                   # (B, W, 64) — batch_first

        tf = self.temporal_fusion
        x_grn = tf.input_grn(x)                                 # (B, W, 64)
        lstm_out, _ = tf.lstm(x_grn)                            # (B, W, 64)
        x_lstm = tf.post_lstm_gate_norm(lstm_out, x_grn)        # (B, W, 64)
        attn_out, _ = tf.attention(x_lstm, x_lstm, x_lstm,
                                   need_weights=False)           # (B, W, 64)
        x_attn = tf.post_attention_gate_norm(attn_out, x_lstm)  # (B, W, 64)
        x_out  = tf.positionwise_grn(x_attn)                    # (B, W, 64)

        h = x_out[:, -1, :]                                     # (B, 64) last time step
        s = self.static_branch(is_nasa.view(-1, 1))             # (B, 8)
        return self.regression_head(torch.cat([h, s], dim=1))   # (B, 1)

# ── Transfer Learning model definitions ───────────────────────────────────────
# ConvBlock with .block submodule (matches ckpt_fewshot_*.pt state dict keys)

class TL_ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=k // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.block(x)

class TL_ConvEncoder(nn.Module):
    def __init__(self, n_features=7):
        super().__init__()
        self.net = nn.Sequential(
            TL_ConvBlock(n_features, 64),
            TL_ConvBlock(64, 128),
            nn.MaxPool1d(2),
            TL_ConvBlock(128, 128),
            TL_ConvBlock(128, 64),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return self.pool(self.net(x)).squeeze(-1)  # (B, 64)

class TL_BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = TL_ConvEncoder(7)
        self.regressor = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.regressor(self.encoder(x))  # (B, 1)

class TL_TransferRULNet(nn.Module):
    def __init__(self, encoder, fc_mu, latent_dim=32):
        super().__init__()
        self.encoder   = encoder
        self.fc_mu     = fc_mu
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        h = self.encoder(x)   # (B, 64)
        z = self.fc_mu(h)     # (B, latent_dim)
        return self.regressor(z)  # (B, 1)

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

@st.cache_resource
def load_tl_baseline():
    model = TL_BaselineCNN()
    ckpt  = torch.load(CKPT_TL_BASELINE, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    return model

@st.cache_resource
def load_tl_transfer():
    encoder = TL_ConvEncoder(7)
    fc_mu   = nn.Linear(64, 32)
    model   = TL_TransferRULNet(encoder, fc_mu, latent_dim=32)
    ckpt    = torch.load(CKPT_TL_TRANSFER, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    return model

@st.cache_data
def load_tl_dataset():
    return pd.read_csv(TL_DATASET_PATH)

# ── Inference helpers ─────────────────────────────────────────────────────────

def predict_cnn(model, window_np, is_nasa_val):
    """window_np: (WINDOW_SIZE, 7)"""
    x = torch.tensor(window_np, dtype=torch.float32).T.unsqueeze(0)  # (1, 7, W)
    isn = torch.tensor([[is_nasa_val]], dtype=torch.float32)          # (1, 1)
    with torch.no_grad():
        return max(0.0, model(x, isn).item())

def predict_xgb(bundle, last_row_np, is_nasa_val, dataset_name):
    """last_row_np: (7,) in combined_scaled space — converts to XGB-scaler space first."""
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    xgb_row = rescale_for_xgb(last_row_np, dataset_name).astype(np.float32)
    X = xgb_row.reshape(1, -1)
    if model.n_features_in_ == 8:
        X = np.hstack([X, np.array([[is_nasa_val]], dtype=np.float32)])
    return max(0.0, float(model.predict(X)[0]))

def predict_tft(model, window_np, is_nasa_val):
    """window_np: (WINDOW_SIZE, 7) in combined_scaled space."""
    x   = torch.tensor(window_np, dtype=torch.float32).T.unsqueeze(0)  # (1, 7, W)
    isn = torch.tensor([[is_nasa_val]], dtype=torch.float32)
    with torch.no_grad():
        return max(0.0, model(x, isn).item())

def tl_inverse_rul(raw):
    """Invert MinMax RUL scaling: raw ∈ [0,1] → cycles. rul_min=1, rul_max=1055."""
    return float(raw) * 1054 + 1

def predict_tl_baseline(model, window_np):
    """window_np: (W, 7) MinMax-scaled."""
    x = torch.tensor(window_np, dtype=torch.float32).T.unsqueeze(0)  # (1, 7, W)
    with torch.no_grad():
        return max(0.0, tl_inverse_rul(model(x).item()))

def predict_tl_transfer(model, window_np):
    """window_np: (W, 7) MinMax-scaled."""
    x = torch.tensor(window_np, dtype=torch.float32).T.unsqueeze(0)  # (1, 7, W)
    with torch.no_grad():
        return max(0.0, tl_inverse_rul(model(x).item()))

# ── Batch inference helpers (used by RUL Trajectory tab) ──────────────────────

def predict_cnn_batch(model, windows_np, is_nasa_val):
    """windows_np: (N, W, 7) → predictions (N,)"""
    x   = torch.tensor(windows_np, dtype=torch.float32).permute(0, 2, 1)  # (N, 7, W)
    isn = torch.full((len(windows_np), 1), is_nasa_val, dtype=torch.float32)
    with torch.no_grad():
        return np.clip(model(x, isn).squeeze(-1).numpy(), 0, None)

def predict_tft_batch(model, windows_np, is_nasa_val):
    """windows_np: (N, W, 7) → predictions (N,)"""
    x   = torch.tensor(windows_np, dtype=torch.float32).permute(0, 2, 1)  # (N, 7, W)
    isn = torch.full((len(windows_np), 1), is_nasa_val, dtype=torch.float32)
    with torch.no_grad():
        return np.clip(model(x, isn).squeeze(-1).numpy(), 0, None)

def predict_xgb_batch(bundle, last_rows_np, is_nasa_val, dataset_name):
    """last_rows_np: (N, 7) in combined_scaled space → predictions (N,)"""
    model    = bundle["model"] if isinstance(bundle, dict) else bundle
    xgb_rows = np.vstack([rescale_for_xgb(r, dataset_name) for r in last_rows_np]).astype(np.float32)
    if model.n_features_in_ == 8:
        xgb_rows = np.hstack([xgb_rows, np.full((len(xgb_rows), 1), is_nasa_val, dtype=np.float32)])
    return np.clip(model.predict(xgb_rows), 0, None)

def get_tft_attention(model, window_np, is_nasa_val):
    """Returns (W, W) averaged self-attention weight matrix for one window."""
    x   = torch.tensor(window_np, dtype=torch.float32).T.unsqueeze(0)  # (1, 7, W)
    isn = torch.tensor([[is_nasa_val]], dtype=torch.float32)
    with torch.no_grad():
        for blk in model.conv_feature_extractor:
            x = blk(x)
        x = x.transpose(1, 2)
        tf = model.temporal_fusion
        x_grn  = tf.input_grn(x)
        lstm_out, _ = tf.lstm(x_grn)
        x_lstm = tf.post_lstm_gate_norm(lstm_out, x_grn)
        _, attn_w = tf.attention(x_lstm, x_lstm, x_lstm, need_weights=True)
    return attn_w[0].numpy()  # (W, W)

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Battery RUL Prediction", layout="wide")

# ── Page selector (top of sidebar) ───────────────────────────────────────────
page = st.sidebar.radio("Navigation", ["RUL Prediction", "Transfer Learning"],
                        label_visibility="collapsed")
st.sidebar.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — RUL Prediction (existing demo)
# ══════════════════════════════════════════════════════════════════════════════
if page == "RUL Prediction":

    st.title("Battery RUL Prediction")

    # Check all checkpoints
    missing = [n for n, p in [("CNN", CKPT_CNN), ("XGBoost", CKPT_XGB), ("TFT", CKPT_TFT)]
               if not os.path.exists(p)]
    if missing:
        st.error(f"Missing checkpoints: {', '.join(missing)}")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("Settings")

    dataset_name = st.sidebar.radio("Dataset", list(DATASETS.keys()))
    model_choice = st.sidebar.radio("Model", ["CNN", "XGBoost", "TFT", "All"])

    st.sidebar.markdown("---")

    # Load dataset early (cached) so slider bounds are known
    df_full = load_dataset(dataset_name)
    _, is_nasa = DATASETS[dataset_name]

    window_size = st.sidebar.slider("Window size (cycles)", min_value=5, max_value=50, value=10, step=1)
    max_start = max(0, len(df_full) - window_size - 1)

    # Clamp stored position if it now exceeds max_start (e.g. after window_size increase or dataset switch)
    pos_key = f"pos_{dataset_name}"
    if pos_key in st.session_state and st.session_state[pos_key] > max_start:
        st.session_state[pos_key] = max_start

    if st.sidebar.button("Random Sample", type="primary"):
        st.session_state[pos_key] = int(np.random.randint(0, max_start + 1))

    start = st.sidebar.slider("Window start (cycle index)", 0, max(1, max_start), 0, key=pos_key)

    st.sidebar.markdown(f"**Predicts:** RUL at cycle index {start + window_size}")

    # ── Shared colour map ─────────────────────────────────────────────────────
    model_colors = {"CNN": "#4C9BE8", "XGBoost": "#2A9D8F", "TFT": "#F4A261"}

    # ── Extract demo window (shared across tabs) ───────────────────────────────
    window_df = df_full.iloc[start : start + window_size].copy()
    true_rul  = float(df_full.iloc[start + window_size]["RUL"])
    window_np = window_df[FEATURE_COLS].values.astype(np.float32)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["Demo", "Manual Input", "RUL Trajectory", "TFT Attention"])

    # ══ TAB 1 — Demo ══════════════════════════════════════════════════════════
    with tab1:

        st.markdown(
            f"**Dataset:** {dataset_name} &nbsp;|&nbsp; "
            f"**Window:** cycles {start}–{start + window_size - 1} ({window_size} cycles) &nbsp;|&nbsp; "
            f"**Predicting RUL at cycle:** {start + window_size}"
        )

        # ── Run inference ─────────────────────────────────────────────────────
        cnn_model  = load_cnn()
        xgb_bundle = load_xgb()
        tft_model  = load_tft()

        run_cnn = model_choice in ("CNN", "All")
        run_xgb = model_choice in ("XGBoost", "All")
        run_tft = model_choice in ("TFT", "All")

        results = {}
        errors  = {}

        if run_cnn:
            results["CNN"] = predict_cnn(cnn_model, window_np, is_nasa)
        if run_xgb:
            results["XGBoost"] = predict_xgb(xgb_bundle, window_np[-1], is_nasa, dataset_name)
        if run_tft:
            results["TFT"] = predict_tft(tft_model, window_np, is_nasa)

        # ── Metric cards ──────────────────────────────────────────────────────
        num_cols = 1 + len(results) + len(errors)
        metric_cols = st.columns(num_cols)
        metric_cols[0].metric("True RUL", f"{true_rul:.1f} cycles")
        for i, (name, pred) in enumerate(results.items(), 1):
            err = pred - true_rul
            metric_cols[i].metric(name, f"{pred:.1f} cycles", delta=f"{err:+.1f}", delta_color="inverse")
        for name, msg in errors.items():
            st.warning(f"**{name} error:** {msg}")

        # ── Bar chart ─────────────────────────────────────────────────────────
        if results:
            st.subheader("Predicted vs. True RUL")
            names  = ["True RUL"] + list(results.keys())
            values = [true_rul]   + list(results.values())
            colors = ["#888888"]  + [model_colors[n] for n in results]
            fig = go.Figure(go.Bar(
                x=names, y=values, marker_color=colors,
                text=[f"{v:.1f}" for v in values], textposition="outside", width=0.45,
            ))
            fig.add_hline(y=true_rul, line_dash="dot", line_color="gray",
                          annotation_text="True RUL", annotation_position="top right")
            fig.update_layout(yaxis_title="RUL (cycles)", height=380,
                              margin=dict(t=40, b=40), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # ── Input window table ────────────────────────────────────────────────
        st.subheader("Input Window — Feature Values (Physical Units)")
        display_df = unscale_window(window_df[FEATURE_COLS], dataset_name)
        display_df.index = [f"Cycle {start + i}" for i in range(window_size)]
        st.dataframe(display_df.style.background_gradient(axis=0, cmap="RdYlGn"),
                     use_container_width=True)

    # ══ TAB 2 — Manual Input ══════════════════════════════════════════════════
    with tab2:
        st.markdown(
            "Paste rows copied from the dataset — one cycle per line, comma-separated, "
            "**already scaled** (same scale as `combined_scaled_battery_data.csv`).  \n"
            "Accepted formats per line: `feat1..7` (7 values), `feat1..7, RUL` (8), "
            "or `cycle_idx, feat1..7, RUL` (9).  \n"
            "CNN / TFT use all pasted rows as the window; XGBoost uses only the last row."
        )

        mcol1, mcol2 = st.columns(2)
        with mcol1:
            manual_dataset = st.radio("Dataset", list(DATASETS.keys()),
                                      key="manual_dataset", horizontal=True)
        with mcol2:
            manual_model = st.radio("Model", ["CNN", "XGBoost", "TFT", "All"],
                                    key="manual_model", horizontal=True)

        _, manual_is_nasa = DATASETS[manual_dataset]

        paste_text = st.text_area(
            "Paste rows here:",
            height=220,
            placeholder=(
                "Example (9-value format):\n"
                "1.0,0.9171988257767602,2.364163027648995,-2.856,-3.099,1.978,1.836,1.000,1107\n"
                "2.0,0.8912345678901234,2.201234567890123,-2.712,-2.987,1.856,1.723,0.987,1106"
            ),
            key="manual_paste",
        )

        if st.button("Run Prediction", type="primary", key="manual_run"):
            lines = [l.strip() for l in paste_text.strip().splitlines() if l.strip()]
            if not lines:
                st.warning("No data pasted. Paste at least one row.")
            else:
                rows: list = []
                rul_val = None
                parse_error = None

                for i, line in enumerate(lines):
                    try:
                        floats = [float(v.strip()) for v in line.split(",")]
                    except ValueError:
                        parse_error = f"Line {i + 1}: non-numeric value — `{line}`"
                        break

                    n = len(floats)
                    if n == 9:          # cycle_idx, 7 features, RUL
                        rows.append(floats[1:8])
                        rul_val = floats[8]
                    elif n == 8:        # 7 features, RUL
                        rows.append(floats[:7])
                        rul_val = floats[7]
                    elif n == 7:        # 7 features only
                        rows.append(floats[:7])
                    else:
                        parse_error = (f"Line {i + 1}: expected 7, 8, or 9 values, "
                                       f"got {n} — `{line}`")
                        break

                if parse_error:
                    st.error(parse_error)
                else:
                    manual_np = np.array(rows, dtype=np.float32)  # (N, 7)

                    m_results: dict = {}
                    m_errors:  dict = {}

                    run_cnn_m = manual_model in ("CNN", "All")
                    run_xgb_m = manual_model in ("XGBoost", "All")
                    run_tft_m = manual_model in ("TFT", "All")

                    if run_cnn_m:
                        if manual_np.shape[0] < 2:
                            m_errors["CNN"] = "Need ≥ 2 rows for CNN."
                        else:
                            m_results["CNN"] = predict_cnn(load_cnn(), manual_np, manual_is_nasa)

                    if run_xgb_m:
                        m_results["XGBoost"] = predict_xgb(
                            load_xgb(), manual_np[-1], manual_is_nasa, manual_dataset)

                    if run_tft_m:
                        if manual_np.shape[0] < 2:
                            m_errors["TFT"] = "Need ≥ 2 rows for TFT."
                        else:
                            m_results["TFT"] = predict_tft(load_tft(), manual_np, manual_is_nasa)

                    # ── Metric cards ──────────────────────────────────────────
                    n_mcols = (1 if rul_val is not None else 0) + len(m_results)
                    if n_mcols > 0:
                        m_cols = st.columns(n_mcols)
                        col_idx = 0
                        if rul_val is not None:
                            m_cols[col_idx].metric("True RUL", f"{rul_val:.1f} cycles")
                            col_idx += 1
                        for name, pred in m_results.items():
                            delta_str = f"{pred - rul_val:+.1f}" if rul_val is not None else None
                            m_cols[col_idx].metric(name, f"{pred:.1f} cycles",
                                                   delta=delta_str, delta_color="inverse")
                            col_idx += 1

                    for name, msg in m_errors.items():
                        st.warning(f"**{name}:** {msg}")

                    # ── Bar chart ─────────────────────────────────────────────
                    if m_results:
                        st.subheader("Predicted vs. True RUL")
                        m_names  = (["True RUL"] if rul_val is not None else []) + list(m_results.keys())
                        m_values = ([rul_val]    if rul_val is not None else []) + list(m_results.values())
                        m_colors = (["#888888"]  if rul_val is not None else []) + [model_colors[n] for n in m_results]
                        fig_m = go.Figure(go.Bar(
                            x=m_names, y=m_values, marker_color=m_colors,
                            text=[f"{v:.1f}" for v in m_values], textposition="outside", width=0.45,
                        ))
                        if rul_val is not None:
                            fig_m.add_hline(y=rul_val, line_dash="dot", line_color="gray",
                                            annotation_text="True RUL", annotation_position="top right")
                        fig_m.update_layout(yaxis_title="RUL (cycles)", height=380,
                                            margin=dict(t=40, b=40), showlegend=False)
                        st.plotly_chart(fig_m, use_container_width=True)

                    # ── Feature table ─────────────────────────────────────────
                    st.subheader("Pasted Rows — Feature Values (Physical Units)")
                    display_m = pd.DataFrame(manual_np, columns=FEATURE_COLS)
                    display_m = unscale_window(display_m, manual_dataset)
                    display_m.index = [f"Row {i}" for i in range(len(rows))]
                    st.dataframe(display_m.style.background_gradient(axis=0, cmap="RdYlGn"),
                                 use_container_width=True)

    # ══ TAB 3 — RUL Trajectory ════════════════════════════════════════════════
    with tab3:
        st.markdown(
            f"Slide the window across every cycle in **{dataset_name}** (window size = {window_size}). "
            "Each point is the predicted RUL for the cycle immediately after that window."
        )

        traj_key = f"traj_{dataset_name}_{window_size}_{model_choice}"

        if st.button("Generate Trajectory", type="primary", key="traj_btn"):
            n_full   = len(df_full)
            max_s    = n_full - window_size - 1
            if max_s < 1:
                st.warning("Dataset too short for a trajectory with this window size.")
            else:
                all_windows = np.stack(
                    [df_full[FEATURE_COLS].values[s : s + window_size] for s in range(max_s + 1)]
                ).astype(np.float32)  # (N, W, 7)
                true_ruls_t = [float(df_full.iloc[s + window_size]["RUL"]) for s in range(max_s + 1)]
                x_axis      = list(range(max_s + 1))

                traj = {"x": x_axis, "true": true_ruls_t}
                with st.spinner("Running inference across all windows…"):
                    if model_choice in ("CNN", "All"):
                        traj["CNN"] = predict_cnn_batch(load_cnn(), all_windows, is_nasa)
                    if model_choice in ("XGBoost", "All"):
                        traj["XGBoost"] = predict_xgb_batch(
                            load_xgb(), all_windows[:, -1, :], is_nasa, dataset_name)
                    if model_choice in ("TFT", "All"):
                        traj["TFT"] = predict_tft_batch(load_tft(), all_windows, is_nasa)
                st.session_state[traj_key] = traj

        if traj_key in st.session_state:
            traj = st.session_state[traj_key]
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=traj["x"], y=traj["true"], mode="lines", name="True RUL",
                line=dict(color="#888888", width=2),
            ))
            for m_name, m_color in model_colors.items():
                if m_name in traj:
                    fig_t.add_trace(go.Scatter(
                        x=traj["x"], y=list(traj[m_name]), mode="lines", name=m_name,
                        line=dict(color=m_color, width=1.5),
                    ))
            fig_t.update_layout(
                xaxis_title="Window start (cycle index)",
                yaxis_title="RUL (cycles)",
                height=460,
                margin=dict(t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("Click **Generate Trajectory** to run sliding-window inference across the full battery.")

    # ══ TAB 4 — TFT Attention ═════════════════════════════════════════════════
    with tab4:
        st.markdown(
            "Self-attention weights computed by the TFT's attention layer for the **current demo window** "
            "(set via the sidebar sliders). "
            "Rows = query time step (attending *from*), Columns = key time step (attended *to*)."
        )

        cycle_labels = [f"Cycle {start + i}" for i in range(window_size)]
        attn_w = get_tft_attention(load_tft(), window_np, is_nasa)  # (W, W)

        fig_heat = go.Figure(go.Heatmap(
            z=attn_w,
            x=cycle_labels,
            y=cycle_labels,
            colorscale="Blues",
            text=[[f"{v:.3f}" for v in row] for row in attn_w],
            texttemplate="%{text}",
            showscale=True,
        ))
        fig_heat.update_layout(
            title="Self-Attention Weight Matrix",
            xaxis_title="Key (attended to)",
            yaxis_title="Query (attending from)",
            height=520,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Last-Step Attention — Prediction Focus")
        st.caption("Attention weights from the final time step's perspective: which earlier cycles mattered most for the RUL prediction.")
        fig_bar_a = go.Figure(go.Bar(
            x=cycle_labels,
            y=attn_w[-1],
            marker_color=model_colors["TFT"],
            text=[f"{v:.3f}" for v in attn_w[-1]],
            textposition="outside",
        ))
        fig_bar_a.update_layout(
            xaxis_title="Cycle",
            yaxis_title="Attention weight",
            height=320,
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_bar_a, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Transfer Learning
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Transfer Learning":

    st.title("Transfer Learning — Battery RUL")
    st.markdown(
        "Comparing a **Baseline CNN** (trained from scratch on the target domain) "
        "against **Transfer RULNet** (VAE-pretrained encoder frozen, only the regressor fine-tuned). "
        "Both models use MinMax-scaled features from the held-out NASA demo battery."
    )

    # Check checkpoints and dataset
    tl_missing = [(n, p) for n, p in [
        ("Baseline CNN", CKPT_TL_BASELINE), ("Transfer RULNet", CKPT_TL_TRANSFER)
    ] if not os.path.exists(p)]
    if tl_missing:
        st.error(f"Missing checkpoints: {', '.join(n for n, _ in tl_missing)}")
        st.stop()
    if not os.path.exists(TL_DATASET_PATH):
        st.error(f"Dataset not found: {TL_DATASET_PATH}")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("Settings")

    tl_df = load_tl_dataset()
    tl_window_size = st.sidebar.slider("Window size (cycles)", min_value=5, max_value=50,
                                       value=10, step=1, key="tl_window_size")
    tl_max_start = max(0, len(tl_df) - tl_window_size)

    tl_pos_key = "tl_pos"
    if tl_pos_key in st.session_state and st.session_state[tl_pos_key] > tl_max_start:
        st.session_state[tl_pos_key] = tl_max_start

    if st.sidebar.button("Random Sample", type="primary", key="tl_random"):
        st.session_state[tl_pos_key] = int(np.random.randint(0, tl_max_start + 1))

    tl_start = st.sidebar.slider("Window start (cycle index)", 0, max(1, tl_max_start), 0,
                                  key=tl_pos_key)
    st.sidebar.markdown(f"**Predicts:** RUL at cycle index {tl_start + tl_window_size - 1}")

    tl_colors = {"Baseline CNN": "#E07B39", "Transfer RULNet": "#6C63FF"}

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tl_tab1, tl_tab2 = st.tabs(["Sample Window", "Manual Input"])

    # ══ TL TAB 1 — Sample Window ══════════════════════════════════════════════
    with tl_tab1:
        tl_window_df = tl_df.iloc[tl_start : tl_start + tl_window_size].copy()
        tl_true_rul  = float(tl_df.iloc[tl_start + tl_window_size - 1]["RUL"])
        tl_window_np = tl_window_df[FEATURE_COLS].values.astype(np.float32)

        tl_baseline_model  = load_tl_baseline()
        tl_transfer_model  = load_tl_transfer()
        tl_pred_base = predict_tl_baseline(tl_baseline_model, tl_window_np)
        tl_pred_tran = predict_tl_transfer(tl_transfer_model, tl_window_np)

        st.markdown(
            f"**Battery:** NASA demo &nbsp;|&nbsp; "
            f"**Window:** cycles {tl_start}–{tl_start + tl_window_size - 1} ({tl_window_size} cycles) &nbsp;|&nbsp; "
            f"**True RUL at cycle {tl_start + tl_window_size - 1}:** {tl_true_rul:.0f}"
        )

        # Metric cards
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("True RUL", f"{tl_true_rul:.1f} cycles")
        mc2.metric("Baseline CNN", f"{tl_pred_base:.1f} cycles",
                   delta=f"{tl_pred_base - tl_true_rul:+.1f}", delta_color="inverse")
        mc3.metric("Transfer RULNet", f"{tl_pred_tran:.1f} cycles",
                   delta=f"{tl_pred_tran - tl_true_rul:+.1f}", delta_color="inverse")

        # Bar chart
        st.subheader("Predicted vs. True RUL")
        tl_names  = ["True RUL", "Baseline CNN", "Transfer RULNet"]
        tl_values = [tl_true_rul, tl_pred_base, tl_pred_tran]
        tl_bar_colors = ["#888888", tl_colors["Baseline CNN"], tl_colors["Transfer RULNet"]]
        fig_tl = go.Figure(go.Bar(
            x=tl_names, y=tl_values, marker_color=tl_bar_colors,
            text=[f"{v:.1f}" for v in tl_values], textposition="outside", width=0.45,
        ))
        fig_tl.add_hline(y=tl_true_rul, line_dash="dot", line_color="gray",
                         annotation_text="True RUL", annotation_position="top right")
        fig_tl.update_layout(yaxis_title="RUL (cycles)", height=380,
                              margin=dict(t=40, b=40), showlegend=False)
        st.plotly_chart(fig_tl, use_container_width=True)

        # Feature table (MinMax-scaled, shown as-is)
        st.subheader("Input Window — Feature Values (MinMax-Scaled, 0–1)")
        tl_display_df = tl_window_df[FEATURE_COLS].copy()
        tl_display_df.index = [f"Cycle {tl_start + i}" for i in range(tl_window_size)]
        st.dataframe(tl_display_df.style.background_gradient(axis=0, cmap="RdYlGn"),
                     use_container_width=True)

    # ══ TL TAB 2 — Manual Input ═══════════════════════════════════════════════
    with tl_tab2:
        st.markdown(
            "Paste **MinMax-scaled** feature rows (values in 0–1 range) — one cycle per line.  \n"
            "Accepted formats per line: `feat1..7` (7 values), `feat1..7, RUL` (8), "
            "or `cycle, feat1..7, RUL` (9).  \n"
            "Both models use all pasted rows as the window (min 2 rows required)."
        )

        tl_paste = st.text_area(
            "Paste rows here:",
            height=220,
            placeholder=(
                "Example (7-value format):\n"
                "1.0,1.0,0.0,0.0,1.0,1.0,1.0\n"
                "1.0,1.0,1.0,0.0,1.0,1.0,1.0"
            ),
            key="tl_manual_paste",
        )

        if st.button("Run Prediction", type="primary", key="tl_manual_run"):
            lines = [l.strip() for l in tl_paste.strip().splitlines() if l.strip()]
            if not lines:
                st.warning("No data pasted. Paste at least one row.")
            else:
                tl_rows: list = []
                tl_rul_val = None
                tl_parse_err = None

                for i, line in enumerate(lines):
                    try:
                        floats = [float(v.strip()) for v in line.split(",")]
                    except ValueError:
                        tl_parse_err = f"Line {i + 1}: non-numeric value — `{line}`"
                        break

                    n = len(floats)
                    if n == 9:
                        tl_rows.append(floats[1:8])
                        tl_rul_val = floats[8]
                    elif n == 8:
                        tl_rows.append(floats[:7])
                        tl_rul_val = floats[7]
                    elif n == 7:
                        tl_rows.append(floats[:7])
                    else:
                        tl_parse_err = (f"Line {i + 1}: expected 7, 8, or 9 values, "
                                        f"got {n} — `{line}`")
                        break

                if tl_parse_err:
                    st.error(tl_parse_err)
                elif len(tl_rows) < 2:
                    st.warning("Need at least 2 rows for CNN-based models.")
                else:
                    tl_manual_np = np.array(tl_rows, dtype=np.float32)

                    tl_m_base = predict_tl_baseline(load_tl_baseline(), tl_manual_np)
                    tl_m_tran = predict_tl_transfer(load_tl_transfer(), tl_manual_np)

                    # Metric cards
                    n_tl_cols = (1 if tl_rul_val is not None else 0) + 2
                    tl_m_cols = st.columns(n_tl_cols)
                    col_i = 0
                    if tl_rul_val is not None:
                        tl_m_cols[col_i].metric("True RUL", f"{tl_rul_val:.1f} cycles")
                        col_i += 1
                    tl_m_cols[col_i].metric(
                        "Baseline CNN", f"{tl_m_base:.1f} cycles",
                        delta=f"{tl_m_base - tl_rul_val:+.1f}" if tl_rul_val is not None else None,
                        delta_color="inverse",
                    )
                    tl_m_cols[col_i + 1].metric(
                        "Transfer RULNet", f"{tl_m_tran:.1f} cycles",
                        delta=f"{tl_m_tran - tl_rul_val:+.1f}" if tl_rul_val is not None else None,
                        delta_color="inverse",
                    )

                    # Bar chart
                    st.subheader("Predicted vs. True RUL")
                    tl_m_names  = (["True RUL"] if tl_rul_val is not None else []) + ["Baseline CNN", "Transfer RULNet"]
                    tl_m_values = ([tl_rul_val] if tl_rul_val is not None else []) + [tl_m_base, tl_m_tran]
                    tl_m_clrs   = (["#888888"]  if tl_rul_val is not None else []) + [tl_colors["Baseline CNN"], tl_colors["Transfer RULNet"]]
                    fig_tl_m = go.Figure(go.Bar(
                        x=tl_m_names, y=tl_m_values, marker_color=tl_m_clrs,
                        text=[f"{v:.1f}" for v in tl_m_values], textposition="outside", width=0.45,
                    ))
                    if tl_rul_val is not None:
                        fig_tl_m.add_hline(y=tl_rul_val, line_dash="dot", line_color="gray",
                                           annotation_text="True RUL", annotation_position="top right")
                    fig_tl_m.update_layout(yaxis_title="RUL (cycles)", height=380,
                                           margin=dict(t=40, b=40), showlegend=False)
                    st.plotly_chart(fig_tl_m, use_container_width=True)

                    # Feature table
                    st.subheader("Pasted Rows — Feature Values (MinMax-Scaled, 0–1)")
                    tl_m_display = pd.DataFrame(tl_manual_np, columns=FEATURE_COLS)
                    tl_m_display.index = [f"Row {i}" for i in range(len(tl_rows))]
                    st.dataframe(tl_m_display.style.background_gradient(axis=0, cmap="RdYlGn"),
                                 use_container_width=True)
