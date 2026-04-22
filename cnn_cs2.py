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
DATA_PATHS = [
    "CS2_35_Cleaned_Features.csv",
    "CS2_36_Cleaned_Features.csv",
    "CS2_37_Cleaned_Features.csv",
    "CS2_38_Cleaned_Features.csv",
]
CKPT_PATH   = "checkpoint_cnn_cs2.pt"
WINDOW_SIZE = 10
BATCH_SIZE  = 256
NUM_EPOCHS  = 50
LR          = 1e-3
RUL_CAP     = 800   # piecewise linear cap: RUL > cap is treated as cap

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

def fit_scaler(train_segs: list[pd.DataFrame]) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    vals  = pd.concat(train_segs)[FEATURE_COLS].values
    lower = np.percentile(vals, 1, axis=0)
    upper = np.percentile(vals, 99, axis=0)
    scaler = StandardScaler()
    scaler.fit(np.clip(vals, lower, upper))
    return scaler, lower, upper


def scale_segments(
    segs: list[pd.DataFrame], scaler: StandardScaler,
    lower: np.ndarray, upper: np.ndarray,
) -> list[pd.DataFrame]:
    result = []
    for seg in segs:
        s = seg.copy()
        s[FEATURE_COLS] = scaler.transform(np.clip(s[FEATURE_COLS].values, lower, upper))
        result.append(s)
    return result


def build_sliding_windows(
    segments: list[pd.DataFrame], apply_cap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    windows, labels = [], []
    for seg in segments:
        features = seg[FEATURE_COLS].values.astype(np.float32)
        rul      = seg["RUL"].values.astype(np.float32)
        if apply_cap:
            rul = np.minimum(rul, RUL_CAP)
        for i in range(len(seg) - WINDOW_SIZE):
            windows.append(features[i : i + WINDOW_SIZE])
            labels.append(rul[i + WINDOW_SIZE - 1])
    return np.array(windows), np.array(labels, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(windows).permute(0, 2, 1)
        self.y = torch.tensor(labels).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def make_loader(windows, labels, shuffle: bool) -> DataLoader:
    return DataLoader(WindowDataset(windows, labels),
                      batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


# ---------------------------------------------------------------------------
# Model (same architecture as cnn_rul.py, no domain branch needed)
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


class RULConvNet(nn.Module):
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
        return self.regressor(self.pool(self.encoder(x)).squeeze(-1))


# ---------------------------------------------------------------------------
# Train / eval
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


def evaluate(model, loader, criterion, device):
    model.eval()
    total, preds, labels = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            total += criterion(p, y).item() * len(y)
            preds.append(p.cpu().numpy())
            labels.append(y.cpu().numpy())
    return (total / len(loader.dataset),
            np.concatenate(preds).flatten(),
            np.concatenate(labels).flatten())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- load 4 batteries, last one is test ---
    segments = [pd.read_csv(p).dropna(subset=FEATURE_COLS) for p in DATA_PATHS]
    for i, (seg, path) in enumerate(zip(segments, DATA_PATHS)):
        print(f"  Battery {i+1} ({path}): {len(seg)} cycles")

    train_segs = [segments[0], segments[1], segments[3]]   # CS2_35, 36, 38
    test_segs  = [segments[2]]                             # CS2_37

    # --- StandardScaler fit on training data only ---
    scaler, lower, upper = fit_scaler(train_segs)
    train_segs = scale_segments(train_segs, scaler, lower, upper)
    test_segs  = scale_segments(test_segs,  scaler, lower, upper)

    X_train, y_train = build_sliding_windows(train_segs, apply_cap=True)
    X_test,  y_test  = build_sliding_windows(test_segs,  apply_cap=False)
    print(f"Train windows: {len(X_train)}, Test windows: {len(X_test)}")

    train_loader = make_loader(X_train, y_train, shuffle=True)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    # --- model ---
    model     = RULConvNet(num_features=len(FEATURE_COLS)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- train or load ---
    if os.path.exists(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print(f"Loaded checkpoint from {CKPT_PATH}, skipping training.")
    else:
        print(f"Training for {NUM_EPOCHS} epochs...")
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            if epoch % 10 == 0 or epoch == 1:
                val_loss, _, _ = evaluate(model, test_loader, criterion, device)
                print(f"  epoch {epoch:3d}/{NUM_EPOCHS}  train={train_loss:.2f}  val={val_loss:.2f}")
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"Checkpoint saved to {CKPT_PATH}")

    # --- evaluation ---
    _, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"\nMAE:  {mean_absolute_error(labels, preds):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(labels, preds)):.4f}")
    print(f"R²:   {r2_score(labels, preds):.4f}")

    # --- plot ---
    _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(labels, preds, alpha=0.4, s=10, color="steelblue")
    lo, hi = labels.min(), labels.max()
    ax.plot([lo, hi], [lo, hi], "--", color="red")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("CNN: Predicted vs True RUL (CS2_37 test)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("cnn_cs2_pred_vs_true.png", dpi=150)
    plt.show()
    print("Plot saved to cnn_cs2_pred_vs_true.png")


if __name__ == "__main__":
    main()
