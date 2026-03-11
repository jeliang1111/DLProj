import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH         = "combined_scaled_battery_data.csv"
WINDOW_SIZE       = 30
NUM_TEST_BATTERIES = 8
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
# Data preparation
# ---------------------------------------------------------------------------

def split_into_battery_segments(dataframe: pd.DataFrame) -> list[pd.DataFrame]:
    """Split dataframe into per-battery segments by detecting RUL resets."""
    rul_values = dataframe["RUL"].values
    rul_diffs = np.diff(rul_values)
    boundary_indices = np.where(rul_diffs > 0)[0] + 1

    segment_starts = np.concatenate([[0], boundary_indices])
    segment_ends   = np.concatenate([boundary_indices, [len(dataframe)]])

    return [
        dataframe.iloc[start:end].reset_index(drop=True)
        for start, end in zip(segment_starts, segment_ends)
    ]


def build_sliding_windows(
    segments: list[pd.DataFrame], window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create (window, rul_label) pairs from each battery segment."""
    windows = []
    labels  = []
    for seg in segments:
        feature_array = seg[FEATURE_COLS].values.astype(np.float32)
        rul_array     = seg["RUL"].values.astype(np.float32)
        for i in range(len(seg) - window_size):
            windows.append(feature_array[i : i + window_size])
            labels.append(rul_array[i + window_size - 1])
    return np.array(windows), np.array(labels, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BatteryWindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        # Conv1d expects (batch, channels, length), so permute from (N, T, C) -> (N, C, T)
        self.X = torch.tensor(windows).permute(0, 2, 1)
        self.y = torch.tensor(labels).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


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
    """1D CNN for battery RUL regression.

    Input:  (batch, num_features, window_size)
    Output: (batch, 1)
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
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pooled   = self.global_avg_pool(features)
        return self.regressor(pooled)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_y)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            total_loss += criterion(preds, batch_y).item() * len(batch_y)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.concatenate(all_preds).flatten(), np.concatenate(all_labels).flatten()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- load & split ---
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape}")

    battery_segments   = split_into_battery_segments(df)
    train_segments     = battery_segments[:-NUM_TEST_BATTERIES]
    test_segments      = battery_segments[-NUM_TEST_BATTERIES:]
    print(f"Batteries — train: {len(train_segments)}, test: {len(test_segments)}")

    X_train, y_train = build_sliding_windows(train_segments, WINDOW_SIZE)
    X_test,  y_test  = build_sliding_windows(test_segments,  WINDOW_SIZE)
    print(f"Train windows: {X_train.shape}, Test windows: {X_test.shape}")

    train_loader = DataLoader(
        BatteryWindowDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True,  num_workers=0,
    )
    test_loader = DataLoader(
        BatteryWindowDataset(X_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # --- model, loss, optimizer ---
    model     = RULConvNet(num_features=len(FEATURE_COLS)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # --- training loop ---
    train_losses, val_losses = [], []
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  train_loss={train_loss:.2f}  val_loss={val_loss:.2f}")

    # --- final evaluation ---
    _, cnn_preds, cnn_labels = evaluate(model, test_loader, criterion, device)
    print(f"\nCNN  MAE:  {mean_absolute_error(cnn_labels, cnn_preds):.4f}")
    print(f"CNN  RMSE: {np.sqrt(mean_squared_error(cnn_labels, cnn_preds)):.4f}")
    print(f"CNN  R^2:  {r2_score(cnn_labels, cnn_preds):.4f}")

    # --- plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(train_losses, label="Train")
    axes[0].plot(val_losses,   label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training Curve")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].scatter(cnn_labels, cnn_preds, alpha=0.3, s=10)
    min_val, max_val = cnn_labels.min(), cnn_labels.max()
    axes[1].plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red")
    axes[1].set_xlabel("True RUL")
    axes[1].set_ylabel("Predicted RUL")
    axes[1].set_title("Predicted vs True RUL")
    axes[1].grid(True)

    residuals = cnn_preds - cnn_labels
    axes[2].hist(residuals, bins=50, edgecolor="black")
    axes[2].axvline(0, color="red", linestyle="--")
    axes[2].set_xlabel("Residual (Predicted - True)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Residual Distribution")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("cnn_results.png", dpi=150)
    plt.show()
    print("Plot saved to cnn_results.png")


if __name__ == "__main__":
    main()
