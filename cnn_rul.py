import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CALCE_DATA_PATH   = "Battery_RUL_Cleaned.csv"
NEW_DATA_PATH     = "features.csv"
CKPT_PATH         = "checkpoint_cnn_rul.pt"
WINDOW_SIZE       = 10
NUM_TEST_CALCE    = 4
NUM_TEST_NEW      = 4
BATCH_SIZE        = 256
NUM_EPOCHS        = 50
LR                = 1e-3

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
# Data utilities
# ---------------------------------------------------------------------------

def split_into_battery_segments(dataframe: pd.DataFrame) -> list[pd.DataFrame]:
    diffs      = np.diff(dataframe["RUL"].values)
    boundaries = np.where(diffs > 0)[0] + 1
    starts     = np.concatenate([[0], boundaries])
    ends       = np.concatenate([boundaries, [len(dataframe)]])
    return [dataframe.iloc[s:e].reset_index(drop=True) for s, e in zip(starts, ends)]


def fit_domain_scaler(
    train_segs: list[pd.DataFrame],
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    vals  = pd.concat(train_segs)[FEATURE_COLS].values
    lower = np.percentile(vals, 1, axis=0)
    upper = np.percentile(vals, 99, axis=0)
    scaler = StandardScaler()
    scaler.fit(np.clip(vals, lower, upper))
    return scaler, lower, upper


def scale_segments(
    segs: list[pd.DataFrame],
    scaler: StandardScaler,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[pd.DataFrame]:
    result = []
    for seg in segs:
        s = seg.copy()
        s[FEATURE_COLS] = scaler.transform(np.clip(s[FEATURE_COLS].values, lower, upper))
        result.append(s)
    return result


def build_sliding_windows(
    segments: list[pd.DataFrame], window_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows, domain_flags, labels = [], [], []
    for seg in segments:
        feature_array = seg[FEATURE_COLS].values.astype(np.float32)
        rul_array     = seg["RUL"].values.astype(np.float32)
        domain_flag   = float(seg["Is_New"].iloc[0])
        for i in range(len(seg) - window_size):
            windows.append(feature_array[i : i + window_size])
            domain_flags.append(domain_flag)
            labels.append(rul_array[i + window_size - 1])
    return (
        np.array(windows),
        np.array(domain_flags, dtype=np.float32),
        np.array(labels, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BatteryWindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, domain_flags: np.ndarray, labels: np.ndarray):
        self.X_seq    = torch.tensor(windows).permute(0, 2, 1)
        self.X_static = torch.tensor(domain_flags).unsqueeze(1)
        self.y        = torch.tensor(labels).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_seq[idx], self.X_static[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn   = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class RULConvNet(nn.Module):
    """Dual-branch 1D CNN for battery RUL regression.

    Branch A (sequential): Conv1D layers over the 7 cycle features.
    Branch B (static):     Small Dense layer for the domain flag (Is_New).
    Both branches are concatenated before the final regression head.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(num_features, 64,  kernel_size=3),
            ConvBlock(64,           128, kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
            ConvBlock(128,          128, kernel_size=3),
            ConvBlock(128,          64,  kernel_size=3),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.static_encoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(64 + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        temporal_features = self.global_avg_pool(self.encoder(seq)).squeeze(-1)
        static_features   = self.static_encoder(static)
        combined          = torch.cat([temporal_features, static_features], dim=1)
        return self.regressor(combined)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch_seq, batch_static, batch_y in loader:
        batch_seq, batch_static, batch_y = (
            batch_seq.to(device), batch_static.to(device), batch_y.to(device)
        )
        optimizer.zero_grad()
        loss = criterion(model(batch_seq, batch_static), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_y)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_flags = 0.0, [], [], []
    with torch.no_grad():
        for batch_seq, batch_static, batch_y in loader:
            batch_seq, batch_static, batch_y = (
                batch_seq.to(device), batch_static.to(device), batch_y.to(device)
            )
            preds = model(batch_seq, batch_static)
            total_loss += criterion(preds, batch_y).item() * len(batch_y)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            all_flags.append(batch_static.cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(all_preds).flatten(),
        np.concatenate(all_labels).flatten(),
        np.concatenate(all_flags).flatten(),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- load & tag ---
    calce_df = pd.read_csv(CALCE_DATA_PATH)
    calce_df["Is_New"] = 0
    calce_segments = split_into_battery_segments(calce_df)

    new_df = pd.read_csv(NEW_DATA_PATH).dropna(subset=FEATURE_COLS)
    new_df["Is_New"] = 1
    new_segments = split_into_battery_segments(new_df)
    print(f"CALCE: {len(calce_segments)} batteries, New: {len(new_segments)} batteries")

    rng = np.random.default_rng(42)

    # --- reserve 1 demo battery from each domain ---
    demo_calce = calce_segments[rng.integers(len(calce_segments))]
    demo_new   = new_segments[rng.integers(len(new_segments))]
    demo_df    = pd.concat([demo_calce, demo_new], ignore_index=True)
    demo_df.to_csv("demo_batteries_cnn.csv", index=False)
    print(f"Demo saved (CALCE: {len(demo_calce)} cycles, New: {len(demo_new)} cycles)")

    calce_remaining = [s for s in calce_segments if not s.equals(demo_calce)]
    new_remaining   = [s for s in new_segments   if not s.equals(demo_new)]

    # --- stratified train/test split ---
    calce_test_idx = set(rng.choice(len(calce_remaining), size=NUM_TEST_CALCE, replace=False).tolist())
    new_test_idx   = set(rng.choice(len(new_remaining),   size=NUM_TEST_NEW,   replace=False).tolist())

    calce_train = [s for i, s in enumerate(calce_remaining) if i not in calce_test_idx]
    calce_test  = [s for i, s in enumerate(calce_remaining) if i in calce_test_idx]
    new_train   = [s for i, s in enumerate(new_remaining)   if i not in new_test_idx]
    new_test    = [s for i, s in enumerate(new_remaining)   if i in new_test_idx]

    print(f"CALCE — train: {len(calce_train)}, test: {len(calce_test)}")
    print(f"New   — train: {len(new_train)},   test: {len(new_test)}")

    # --- per-domain feature normalization ---
    calce_scaler, calce_lower, calce_upper = fit_domain_scaler(calce_train)
    new_scaler,   new_lower,   new_upper   = fit_domain_scaler(new_train)

    calce_train = scale_segments(calce_train, calce_scaler, calce_lower, calce_upper)
    calce_test  = scale_segments(calce_test,  calce_scaler, calce_lower, calce_upper)
    new_train   = scale_segments(new_train,   new_scaler,   new_lower,   new_upper)
    new_test    = scale_segments(new_test,    new_scaler,   new_lower,   new_upper)

    train_segments = calce_train + new_train
    test_segments  = calce_test  + new_test

    X_train, flags_train, y_train = build_sliding_windows(train_segments, WINDOW_SIZE)
    X_test,  flags_test,  y_test  = build_sliding_windows(test_segments,  WINDOW_SIZE)
    print(f"Train windows: {X_train.shape}, Test windows: {X_test.shape}")

    train_loader = DataLoader(
        BatteryWindowDataset(X_train, flags_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True,  num_workers=0,
    )
    test_loader = DataLoader(
        BatteryWindowDataset(X_test, flags_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # --- model ---
    model     = RULConvNet(num_features=len(FEATURE_COLS)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- train or load checkpoint ---
    if os.path.exists(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print(f"Loaded checkpoint from {CKPT_PATH}, skipping training.")
    else:
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, _, _, _ = evaluate(model, test_loader, criterion, device)
            scheduler.step()
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  train={train_loss:.2f}  val={val_loss:.2f}")
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"Checkpoint saved to {CKPT_PATH}")

    # --- evaluation ---
    _, preds, labels, flags = evaluate(model, test_loader, criterion, device)
    print(f"\nCNN  MAE:  {mean_absolute_error(labels, preds):.4f}")
    print(f"CNN  RMSE: {np.sqrt(mean_squared_error(labels, preds)):.4f}")
    print(f"CNN  R²:   {r2_score(labels, preds):.4f}")

    # --- scatter plot color-coded by domain ---
    _, ax = plt.subplots(figsize=(6, 6))
    calce_mask = flags == 0
    new_mask   = flags == 1
    ax.scatter(labels[calce_mask], preds[calce_mask],
               alpha=0.3, s=10, color="steelblue",  label="CALCE")
    ax.scatter(labels[new_mask],   preds[new_mask],
               alpha=0.3, s=10, color="darkorange", label="New Battery")
    lo, hi = labels.min(), labels.max()
    ax.plot([lo, hi], [lo, hi], "--", color="red")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("Predicted vs True RUL (CNN)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("cnn_pred_vs_true.png", dpi=150)
    plt.show()
    print("Plot saved to cnn_pred_vs_true.png")


if __name__ == "__main__":
    main()
