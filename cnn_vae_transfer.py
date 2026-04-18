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
DATA_PATH          = "combined_scaled_battery_data.csv"
WINDOW_SIZE        = 30
BATCH_SIZE         = 256
LATENT_DIM         = 32
VAE_EPOCHS         = 50
FINETUNE_EPOCHS    = 100
BASELINE_EPOCHS    = 50
LR                 = 1e-3
NUM_FINETUNE_BATTERIES = 4   # NASA batteries used for fine-tuning

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
    """Split dataframe into per-battery segments by detecting RUL resets."""
    rul_values = dataframe["RUL"].values
    rul_diffs  = np.diff(rul_values)
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
    """Return (windows, rul_labels) arrays from a list of battery segments."""
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
# Datasets
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """Dataset for (window, rul) pairs. Conv1d format: (N, C, T)."""
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(windows).permute(0, 2, 1)  # (N, C, T)
        self.y = torch.tensor(labels).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn   = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ConvEncoder(nn.Module):
    """Shared 1D-CNN encoder used by both the VAE and the baseline CNN.

    Input:  (batch, num_features, window_size)
    Output: (batch, 64)
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(num_features, 64,  kernel_size=3),
            ConvBlock(64,           128, kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
            ConvBlock(128,          128, kernel_size=3),
            ConvBlock(128,          64,  kernel_size=3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.net(x)).squeeze(-1)   # (batch, 64)


class BatteryVAE(nn.Module):
    """VAE built around ConvEncoder.

    Encoder path: ConvEncoder → μ, log_var  (both shape: batch × latent_dim)
    Decoder path: latent → Linear layers → reconstructed window (batch, C, T)
    """
    def __init__(self, num_features: int, window_size: int, latent_dim: int):
        super().__init__()
        self.encoder     = ConvEncoder(num_features)
        self.fc_mu       = nn.Linear(64, latent_dim)
        self.fc_log_var  = nn.Linear(64, latent_dim)

        # Decoder reconstructs the flattened window then reshapes
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_features * window_size),
        )
        self.num_features = num_features
        self.window_size  = window_size

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h       = self.encoder(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        flat = self.decoder(z)
        return flat.view(-1, self.num_features, self.window_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        x_recon     = self.decode(z)
        return x_recon, mu, log_var


class TransferRULNet(nn.Module):
    """Frozen VAE encoder + trainable regression head.

    Only the regression head is updated during fine-tuning.
    """
    def __init__(self, frozen_encoder: ConvEncoder, latent_dim: int):
        super().__init__()
        self.encoder     = frozen_encoder
        self.fc_mu       = nn.Linear(64, latent_dim)

        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        # Freeze encoder and projection weights
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.fc_mu.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h  = self.encoder(x)
            mu = self.fc_mu(h)
        return self.regressor(mu)


class BaselineCNN(nn.Module):
    """Single-branch CNN trained on HNEI and tested directly on NASA (no transfer)."""
    def __init__(self, num_features: int):
        super().__init__()
        self.encoder = ConvEncoder(num_features)
        self.regressor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.encoder(x))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def vae_loss(
    x_recon: torch.Tensor,
    x_orig: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(x_recon, x_orig, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + 0.001 * kl_loss


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_vae_epoch(
    vae: BatteryVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    vae.train()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_recon, mu, log_var = vae(x)
        loss = vae_loss(x_recon, x, mu, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def train_regression_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def evaluate_regression(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device))
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_preds).flatten(), np.concatenate(all_labels).flatten()


def print_metrics(name: str, preds: np.ndarray, labels: np.ndarray) -> dict:
    mae  = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    r2   = r2_score(labels, preds)
    print(f"{name:30s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- load data and segment by battery ---
    df = pd.read_csv(DATA_PATH)
    battery_segments = split_into_battery_segments(df)

    hnei_segments = [s for s in battery_segments if s["Is_NASA"].iloc[0] == 0]
    nasa_segments = [s for s in battery_segments if s["Is_NASA"].iloc[0] == 1]
    print(f"HNEI batteries: {len(hnei_segments)}, NASA batteries: {len(nasa_segments)}")

    # --- split HNEI into fine-tune set and test set ---
    rng = np.random.default_rng(42)
    finetune_indices   = list(rng.choice(len(hnei_segments), size=NUM_FINETUNE_BATTERIES, replace=False))
    finetune_index_set = set(finetune_indices)
    hnei_finetune_segments = [hnei_segments[i] for i in finetune_indices]
    hnei_test_segments     = [s for i, s in enumerate(hnei_segments) if i not in finetune_index_set]
    print(f"HNEI fine-tune batteries: {len(hnei_finetune_segments)}, "
          f"HNEI test batteries: {len(hnei_test_segments)}\n")

    # --- build window arrays ---
    X_nasa,      y_nasa      = build_sliding_windows(nasa_segments,           WINDOW_SIZE)
    X_finetune,  y_finetune  = build_sliding_windows(hnei_finetune_segments,  WINDOW_SIZE)
    X_hnei_test, y_hnei_test = build_sliding_windows(hnei_test_segments,      WINDOW_SIZE)

    nasa_loader      = DataLoader(WindowDataset(X_nasa,      y_nasa),      batch_size=BATCH_SIZE, shuffle=True)
    finetune_loader  = DataLoader(WindowDataset(X_finetune,  y_finetune),  batch_size=BATCH_SIZE, shuffle=True)
    hnei_test_loader = DataLoader(WindowDataset(X_hnei_test, y_hnei_test), batch_size=BATCH_SIZE, shuffle=False)

    mse_criterion = nn.MSELoss()

    CKPT_VAE      = "checkpoint_vae.pt"
    CKPT_TRANSFER = "checkpoint_transfer.pt"
    CKPT_BASELINE = "checkpoint_baseline.pt"

    import os

    # -----------------------------------------------------------------------
    # Phase 1: Train VAE on NASA (or load checkpoint)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Phase 1: Training VAE on NASA data")
    print("=" * 60)

    vae = BatteryVAE(NUM_FEATURES, WINDOW_SIZE, LATENT_DIM).to(device)
    if os.path.exists(CKPT_VAE):
        vae.load_state_dict(torch.load(CKPT_VAE, map_location=device))
        print(f"  Loaded VAE from {CKPT_VAE}, skipping training.")
    else:
        vae_optim = torch.optim.Adam(vae.parameters(), lr=LR, weight_decay=1e-4)
        for epoch in range(1, VAE_EPOCHS + 1):
            loss = train_vae_epoch(vae, nasa_loader, vae_optim, device)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  VAE epoch {epoch:3d}/{VAE_EPOCHS}  loss={loss:.4f}")
        torch.save(vae.state_dict(), CKPT_VAE)
        print(f"  VAE checkpoint saved to {CKPT_VAE}")

    # -----------------------------------------------------------------------
    # Phase 2: Fine-tune regression head on HNEI (or load checkpoint)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 2: Fine-tuning regression head on HNEI data")
    print("=" * 60)

    transfer_model = TransferRULNet(vae.encoder, LATENT_DIM).to(device)
    transfer_model.fc_mu.load_state_dict(vae.fc_mu.state_dict())
    for param in transfer_model.fc_mu.parameters():
        param.requires_grad = False

    if os.path.exists(CKPT_TRANSFER):
        transfer_model.load_state_dict(torch.load(CKPT_TRANSFER, map_location=device))
        print(f"  Loaded transfer model from {CKPT_TRANSFER}, skipping fine-tuning.")
    else:
        finetune_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, transfer_model.parameters()),
            lr=LR, weight_decay=1e-4
        )
        for epoch in range(1, FINETUNE_EPOCHS + 1):
            loss = train_regression_epoch(transfer_model, finetune_loader, mse_criterion, finetune_optim, device)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Fine-tune epoch {epoch:3d}/{FINETUNE_EPOCHS}  loss={loss:.4f}")
        torch.save(transfer_model.state_dict(), CKPT_TRANSFER)
        print(f"  Transfer model checkpoint saved to {CKPT_TRANSFER}")

    # -----------------------------------------------------------------------
    # Baseline: CNN trained on NASA (or load checkpoint)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Baseline: CNN trained on NASA, tested directly on HNEI")
    print("=" * 60)

    baseline_model = BaselineCNN(NUM_FEATURES).to(device)
    if os.path.exists(CKPT_BASELINE):
        baseline_model.load_state_dict(torch.load(CKPT_BASELINE, map_location=device))
        print(f"  Loaded baseline from {CKPT_BASELINE}, skipping training.")
    else:
        baseline_optim     = torch.optim.Adam(baseline_model.parameters(), lr=LR, weight_decay=1e-4)
        baseline_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(baseline_optim, T_max=BASELINE_EPOCHS)
        for epoch in range(1, BASELINE_EPOCHS + 1):
            loss = train_regression_epoch(baseline_model, nasa_loader, mse_criterion, baseline_optim, device)
            baseline_scheduler.step()
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Baseline epoch {epoch:3d}/{BASELINE_EPOCHS}  loss={loss:.4f}")
        torch.save(baseline_model.state_dict(), CKPT_BASELINE)
        print(f"  Baseline checkpoint saved to {CKPT_BASELINE}")

    # -----------------------------------------------------------------------
    # Evaluation on held-out NASA test batteries
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Results on held-out HNEI test batteries")
    print("=" * 60)

    transfer_preds, transfer_labels = evaluate_regression(transfer_model, hnei_test_loader, device)
    baseline_preds,  baseline_labels  = evaluate_regression(baseline_model,  hnei_test_loader, device)

    results = [
        print_metrics("Baseline CNN (no transfer)",   baseline_preds,  baseline_labels),
        print_metrics("VAE Transfer (proposed)",      transfer_preds,  transfer_labels),
    ]

    # -----------------------------------------------------------------------
    # Plot: side-by-side predicted vs true
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_configs = [
        (baseline_labels,  baseline_preds,  "Baseline CNN (no transfer)"),
        (transfer_labels,  transfer_preds,  "VAE Transfer (proposed)"),
    ]
    for ax, (labels, preds, title) in zip(axes, plot_configs):
        ax.scatter(labels, preds, alpha=0.3, s=10)
        lo, hi = labels.min(), labels.max()
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="red")
        ax.set_xlabel("True RUL")
        ax.set_ylabel("Predicted RUL")
        ax.set_title(title)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("vae_transfer_results.png", dpi=150)
    plt.show()
    print("\nPlot saved to vae_transfer_results.png")


if __name__ == "__main__":
    main()
