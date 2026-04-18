import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Config  (mirrors cnn_vae_transfer.py so results are directly comparable)
# ---------------------------------------------------------------------------
DATA_PATH   = "combined_scaled_battery_data.csv"
WINDOW_SIZE = 10
BATCH_SIZE  = 256
LR          = 1e-3
EPOCHS      = 50

NUM_FINETUNE_BATTERIES = 1   # kept identical so target test split matches VAE script

# "HNEI" → train on HNEI, test on NASA.  "NASA" → train on NASA, test on HNEI.
SOURCE_DOMAIN = "HNEI"

CKPT_PATH = f"checkpoint_cnn_baseline_{SOURCE_DOMAIN}.pt"

FEATURE_COLS = [
    "Discharge Time (s)",
    "Decrement 3.6-3.4V (s)",
    "Max. Voltage Dischar. (V)",
    "Min. Voltage Charg. (V)",
    "Time at 4.15V (s)",
    "Time constant current (s)",
    "Charging time (s)",
]
NUM_FEATURES = len(FEATURE_COLS)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def split_into_battery_segments(dataframe: pd.DataFrame) -> list[pd.DataFrame]:
    diffs      = np.diff(dataframe["RUL"].values)
    boundaries = np.where(diffs > 0)[0] + 1
    starts     = np.concatenate([[0], boundaries])
    ends       = np.concatenate([boundaries, [len(dataframe)]])
    return [dataframe.iloc[s:e].reset_index(drop=True) for s, e in zip(starts, ends)]


def build_sliding_windows(
    segments: list[pd.DataFrame], window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    windows, labels = [], []
    for seg in segments:
        features = seg[FEATURE_COLS].values.astype(np.float32)
        rul      = seg["RUL"].values.astype(np.float32)
        for i in range(len(seg) - window_size):
            windows.append(features[i : i + window_size])
            labels.append(rul[i + window_size - 1])
    return np.array(windows), np.array(labels, dtype=np.float32)


class RULScaler:
    """Min-max normalisation fit on source training labels only."""
    def __init__(self):
        self.rul_min: float = 0.0
        self.rul_max: float = 1.0

    def fit(self, labels: np.ndarray) -> "RULScaler":
        self.rul_min = float(labels.min())
        self.rul_max = float(labels.max())
        return self

    def transform(self, labels: np.ndarray) -> np.ndarray:
        return (labels - self.rul_min) / (self.rul_max - self.rul_min)

    def inverse_transform(self, labels: np.ndarray) -> np.ndarray:
        return labels * (self.rul_max - self.rul_min) + self.rul_min


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(windows).permute(0, 2, 1)   # (N, C, T) for Conv1d
        self.y = torch.tensor(labels).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_loader(windows: np.ndarray, labels: np.ndarray, shuffle: bool) -> DataLoader:
    return DataLoader(
        WindowDataset(windows, labels),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CrossDomainCNN(nn.Module):
    """Plain 1D-CNN trained on source domain, applied directly to target domain.

    No transfer learning, no fine-tuning on target — this is the naive baseline
    that shows how poorly a model generalises when the target domain is unseen.
    Architecture is identical to BaselineCNN in cnn_vae_transfer.py.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(num_features, 64),
            ConvBlock(64, 128),
            nn.MaxPool1d(2),
            ConvBlock(128, 128),
            ConvBlock(128, 64),
        )
        self.pool      = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.encoder(x)).squeeze(-1)
        return self.regressor(features)


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


def eval_loss(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += criterion(model(x), y).item() * len(y)
    return total / len(loader.dataset)


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(preds).flatten(), np.concatenate(labels).flatten()


def print_metrics(name: str, preds: np.ndarray, labels: np.ndarray) -> None:
    mae  = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    r2   = r2_score(labels, preds)
    print(f"{name:40s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")

    df            = pd.read_csv(DATA_PATH)
    all_segments  = split_into_battery_segments(df)
    hnei_segments = [s for s in all_segments if s["Is_NASA"].iloc[0] == 0]
    nasa_segments = [s for s in all_segments if s["Is_NASA"].iloc[0] == 1]
    print(f"Total — HNEI: {len(hnei_segments)} batteries, NASA: {len(nasa_segments)} batteries")

    rng = np.random.default_rng(42)

    # --- assign source / target ---
    if SOURCE_DOMAIN == "HNEI":
        source_segments, target_segments = hnei_segments, nasa_segments
    else:
        source_segments, target_segments = nasa_segments, hnei_segments
    target_name = "NASA" if SOURCE_DOMAIN == "HNEI" else "HNEI"
    print(f"Training on: {SOURCE_DOMAIN}   Testing on: {target_name}\n")

    # --- source 80/20 battery-level split ---
    source_shuffled = list(source_segments)
    rng.shuffle(source_shuffled)
    n_val              = max(1, int(len(source_shuffled) * 0.2))
    source_train_segs  = source_shuffled[n_val:]
    source_val_segs    = source_shuffled[:n_val]
    print(f"{SOURCE_DOMAIN} split — train: {len(source_train_segs)}, val: {len(source_val_segs)}")

    # --- target: reserve longest batteries for fine-tune slot, rest → test ---
    # (mirrors cnn_vae_transfer.py so the test batteries are identical)
    sorted_target   = sorted(target_segments, key=lambda s: len(s), reverse=True)
    target_test     = sorted_target[NUM_FINETUNE_BATTERIES:]
    print(f"{target_name} test batteries: {len(target_test)}\n")

    # --- build windows ---
    X_train, y_train = build_sliding_windows(source_train_segs, WINDOW_SIZE)
    X_val,   y_val   = build_sliding_windows(source_val_segs,   WINDOW_SIZE)
    X_test,  y_test  = build_sliding_windows(target_test,       WINDOW_SIZE)

    # scaler fit on source training labels only
    scaler = RULScaler().fit(y_train)
    print(f"Source scaler: min={scaler.rul_min:.1f}, max={scaler.rul_max:.1f}\n")

    train_loader = make_loader(X_train, scaler.transform(y_train), shuffle=True)
    val_loader   = make_loader(X_val,   scaler.transform(y_val),   shuffle=False)
    test_loader  = make_loader(X_test,  scaler.transform(y_test),  shuffle=False)

    model     = CrossDomainCNN(NUM_FEATURES).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- train or load checkpoint ---
    if os.path.exists(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
        print(f"Loaded checkpoint from {CKPT_PATH}, skipping training.")
    else:
        print(f"Training CNN on {SOURCE_DOMAIN} for {EPOCHS} epochs...")
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            if epoch % 10 == 0 or epoch == 1:
                val_loss = eval_loss(model, val_loader, criterion, device)
                print(f"  epoch {epoch:3d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"Checkpoint saved to {CKPT_PATH}")

    # --- evaluate on source val (in-domain) and target test (cross-domain) ---
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    val_preds_norm,  val_labels_norm  = predict(model, val_loader,  device)
    test_preds_norm, test_labels_norm = predict(model, test_loader, device)

    val_preds   = scaler.inverse_transform(val_preds_norm)
    val_labels  = scaler.inverse_transform(val_labels_norm)
    test_preds  = scaler.inverse_transform(test_preds_norm)
    test_labels = scaler.inverse_transform(test_labels_norm)

    print_metrics(f"CNN in-domain  ({SOURCE_DOMAIN} val)",   val_preds,  val_labels)
    print_metrics(f"CNN cross-domain ({target_name} test)", test_preds, test_labels)

    # --- scatter plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (labels, preds, title) in zip(axes, [
        (val_labels,  val_preds,  f"In-domain: {SOURCE_DOMAIN} val"),
        (test_labels, test_preds, f"Cross-domain: {target_name} test (no transfer)"),
    ]):
        ax.scatter(labels, preds, alpha=0.3, s=10)
        lo, hi = labels.min(), labels.max()
        ax.plot([lo, hi], [lo, hi], "--", color="red")
        ax.set_xlabel("True RUL")
        ax.set_ylabel("Predicted RUL")
        ax.set_title(title)
        ax.grid(True)

    plt.tight_layout()
    out_path = f"cnn_baseline_{SOURCE_DOMAIN}_to_{target_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
