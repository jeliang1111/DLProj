import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HNEI_DATA_PATH = "Battery_RUL_Cleaned.csv"
NASA_DATA_PATH = "nasa_battery_cycles.csv"

WINDOW_SIZE = 10
BATCH_SIZE  = 256
LATENT_DIM  = 32
LR          = 1e-3

VAE_EPOCHS      = 50
FINETUNE_EPOCHS = 100
BASELINE_EPOCHS = 100   # more epochs to give baseline a fair chance with limited data

CKPT_VAE      = "ckpt_fewshot_vae.pt"
CKPT_TRANSFER = "ckpt_fewshot_transfer.pt"
CKPT_BASELINE = "ckpt_fewshot_baseline.pt"

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


def fit_scaler(train_segs: list[pd.DataFrame]) -> tuple[MinMaxScaler, np.ndarray, np.ndarray]:
    vals  = pd.concat(train_segs)[FEATURE_COLS].values
    lower = np.percentile(vals, 1, axis=0)
    upper = np.percentile(vals, 99, axis=0)
    scaler = MinMaxScaler()
    scaler.fit(np.clip(vals, lower, upper))
    return scaler, lower, upper


def scale_segments(segs: list[pd.DataFrame], scaler: MinMaxScaler,
                   lower: np.ndarray, upper: np.ndarray) -> list[pd.DataFrame]:
    result = []
    for seg in segs:
        s = seg.copy()
        s[FEATURE_COLS] = scaler.transform(np.clip(s[FEATURE_COLS].values, lower, upper))
        result.append(s)
    return result


class RULScaler:
    """Min-max normalisation fit on a specific set of labels."""
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
# Dataset / loader
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(windows).permute(0, 2, 1)
        self.y = torch.tensor(labels).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_loader(windows: np.ndarray, labels: np.ndarray, shuffle: bool) -> DataLoader:
    return DataLoader(
        WindowDataset(windows, labels),
        batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0,
    )


# ---------------------------------------------------------------------------
# Model components
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


class ConvEncoder(nn.Module):
    """Shared 1D-CNN encoder used by both the VAE and the baseline."""
    def __init__(self, num_features: int):
        super().__init__()
        self.net  = nn.Sequential(
            ConvBlock(num_features, 64),
            ConvBlock(64, 128),
            nn.MaxPool1d(2),
            ConvBlock(128, 128),
            ConvBlock(128, 64),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.net(x)).squeeze(-1)


class BatteryVAE(nn.Module):
    """VAE trained on NASA to capture generalizable degradation patterns."""
    def __init__(self, num_features: int, window_size: int, latent_dim: int):
        super().__init__()
        self.encoder    = ConvEncoder(num_features)
        self.fc_mu      = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)
        self.decoder    = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_features * window_size),
        )
        self.num_features = num_features
        self.window_size  = window_size

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, self.num_features, self.window_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        return self.decode(self.reparameterize(mu, log_var)), mu, log_var


class TransferRULNet(nn.Module):
    """Frozen NASA-VAE encoder + fine-tuned regression head on 1 HNEI battery."""
    def __init__(self, frozen_encoder: ConvEncoder, fc_mu: nn.Linear, latent_dim: int):
        super().__init__()
        self.encoder   = frozen_encoder
        self.fc_mu     = fc_mu
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.fc_mu.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu = self.fc_mu(self.encoder(x))
        return self.regressor(mu)


class FewShotCNN(nn.Module):
    """CNN trained from scratch on 1 HNEI battery — the weak baseline."""
    def __init__(self, num_features: int):
        super().__init__()
        self.encoder   = ConvEncoder(num_features)
        self.regressor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.encoder(x))


# ---------------------------------------------------------------------------
# Loss / training helpers
# ---------------------------------------------------------------------------

def vae_loss(x_recon, x_orig, mu, log_var) -> torch.Tensor:
    recon = nn.functional.mse_loss(x_recon, x_orig, reduction="mean")
    kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + 0.001 * kl


def train_vae_epoch(vae, loader, optimizer, device) -> float:
    vae.train()
    total = 0.0
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_recon, mu, log_var = vae(x)
        loss = vae_loss(x_recon, x, mu, log_var)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)


def eval_vae_epoch(vae, loader, device) -> float:
    vae.eval()
    total = 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_recon, mu, log_var = vae(x)
            total += vae_loss(x_recon, x, mu, log_var).item() * len(x)
    return total / len(loader.dataset)


def train_regression_epoch(model, loader, criterion, optimizer, device) -> float:
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
    print(f"{name:45s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")


def load_or_train(model, ckpt_path: str, train_fn) -> None:
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"  Loaded from {ckpt_path}")
    else:
        train_fn()
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved to {ckpt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- load data ---
    hnei_df = pd.read_csv(HNEI_DATA_PATH)
    nasa_df = pd.read_csv(NASA_DATA_PATH)
    hnei_df["Is_NASA"] = 0
    nasa_df["Is_NASA"] = 1
    hnei_segments = split_into_battery_segments(hnei_df)
    nasa_segments = split_into_battery_segments(nasa_df)
    print(f"HNEI: {len(hnei_segments)} batteries, NASA: {len(nasa_segments)} batteries")

    # --- pick 1 HNEI battery for training (longest = most data) ---
    hnei_sorted        = sorted(hnei_segments, key=lambda s: len(s), reverse=True)
    hnei_train_battery = hnei_sorted[:1]           # 1 battery used for baseline + fine-tune
    hnei_test_segs     = hnei_sorted[1:]           # all remaining HNEI for testing
    print(f"HNEI train battery: {len(hnei_train_battery[0])} cycles")
    print(f"HNEI test batteries: {len(hnei_test_segs)}\n")

    # Each domain uses its own feature scaler so neither domain's data is distorted
    # by the other's statistics. NASA scaler fit on all NASA; HNEI scaler fit on the
    # 1 training battery (the only HNEI data we are allowed to use at training time).
    nasa_scaler, nasa_lower, nasa_upper = fit_scaler(nasa_segments)
    hnei_scaler, hnei_lower, hnei_upper = fit_scaler(hnei_train_battery)

    nasa_scaled       = scale_segments(nasa_segments,     nasa_scaler, nasa_lower, nasa_upper)
    hnei_train_scaled = scale_segments(hnei_train_battery, hnei_scaler, hnei_lower, hnei_upper)
    hnei_test_scaled  = scale_segments(hnei_test_segs,    hnei_scaler, hnei_lower, hnei_upper)

    # --- build windows ---
    X_hnei_train, y_hnei_train = build_sliding_windows(hnei_train_scaled, WINDOW_SIZE)
    X_hnei_test,  y_hnei_test  = build_sliding_windows(hnei_test_scaled,  WINDOW_SIZE)
    X_nasa,       y_nasa       = build_sliding_windows(nasa_scaled,       WINDOW_SIZE)
    print(f"Windows — HNEI train: {len(X_hnei_train)}, HNEI test: {len(X_hnei_test)}, NASA: {len(X_nasa)}")

    # RUL scalers: one per domain to avoid scale mismatch between HNEI (~1100) and NASA (~167)
    hnei_rul_scaler = RULScaler().fit(y_hnei_train)
    nasa_rul_scaler = RULScaler().fit(y_nasa)
    print(f"HNEI RUL range: {hnei_rul_scaler.rul_min:.0f} – {hnei_rul_scaler.rul_max:.0f}")
    print(f"NASA RUL range: {nasa_rul_scaler.rul_min:.0f} – {nasa_rul_scaler.rul_max:.0f}\n")

    hnei_train_loader = make_loader(X_hnei_train, hnei_rul_scaler.transform(y_hnei_train), shuffle=True)
    hnei_test_loader  = make_loader(X_hnei_test,  hnei_rul_scaler.transform(y_hnei_test),  shuffle=False)
    nasa_loader       = make_loader(X_nasa,        nasa_rul_scaler.transform(y_nasa),       shuffle=True)
    mse = nn.MSELoss()

    # -----------------------------------------------------------------------
    # Phase 1: Train VAE on all NASA batteries
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Phase 1: VAE pre-training on NASA data")
    print("=" * 60)

    vae       = BatteryVAE(NUM_FEATURES, WINDOW_SIZE, LATENT_DIM).to(device)
    vae_optim = torch.optim.Adam(vae.parameters(), lr=LR, weight_decay=1e-4)

    def train_vae():
        for epoch in range(1, VAE_EPOCHS + 1):
            train_loss = train_vae_epoch(vae, nasa_loader, vae_optim, device)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{VAE_EPOCHS}  loss={train_loss:.4f}")

    load_or_train(vae, CKPT_VAE, train_vae)

    # -----------------------------------------------------------------------
    # Phase 2: Fine-tune regression head on 1 HNEI battery
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 2: Fine-tuning on 1 HNEI battery (NASA VAE → HNEI)")
    print("=" * 60)

    transfer_model = TransferRULNet(vae.encoder, vae.fc_mu, LATENT_DIM).to(device)
    ft_optim       = torch.optim.Adam(
        filter(lambda p: p.requires_grad, transfer_model.parameters()), lr=LR, weight_decay=1e-4
    )

    def train_transfer():
        for epoch in range(1, FINETUNE_EPOCHS + 1):
            loss = train_regression_epoch(transfer_model, hnei_train_loader, mse, ft_optim, device)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{FINETUNE_EPOCHS}  loss={loss:.4f}")

    load_or_train(transfer_model, CKPT_TRANSFER, train_transfer)

    # -----------------------------------------------------------------------
    # Baseline: CNN trained from scratch on the same 1 HNEI battery only
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Baseline: CNN trained from scratch on 1 HNEI battery only")
    print("=" * 60)

    baseline_model     = FewShotCNN(NUM_FEATURES).to(device)
    baseline_optim     = torch.optim.Adam(baseline_model.parameters(), lr=LR, weight_decay=1e-4)
    baseline_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(baseline_optim, T_max=BASELINE_EPOCHS)

    def train_baseline():
        for epoch in range(1, BASELINE_EPOCHS + 1):
            loss = train_regression_epoch(baseline_model, hnei_train_loader, mse, baseline_optim, device)
            baseline_scheduler.step()
            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{BASELINE_EPOCHS}  loss={loss:.4f}")

    load_or_train(baseline_model, CKPT_BASELINE, train_baseline)

    # -----------------------------------------------------------------------
    # Evaluation on held-out HNEI test batteries
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Results on held-out HNEI test batteries")
    print("=" * 60)

    baseline_norm, labels_norm = predict(baseline_model, hnei_test_loader, device)
    transfer_norm, _           = predict(transfer_model,  hnei_test_loader, device)

    baseline_preds = hnei_rul_scaler.inverse_transform(baseline_norm)
    transfer_preds = hnei_rul_scaler.inverse_transform(transfer_norm)
    true_labels    = hnei_rul_scaler.inverse_transform(labels_norm)

    print_metrics("CNN (1 HNEI battery, no transfer)", baseline_preds, true_labels)
    print_metrics("VAE Transfer (NASA → 1 HNEI fine-tune)", transfer_preds, true_labels)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Few-shot comparison: 1 HNEI training battery", fontsize=13)

    for ax, (preds, title) in zip(axes, [
        (baseline_preds, "CNN from scratch\n(1 HNEI battery only)"),
        (transfer_preds, "VAE Transfer\n(NASA pre-trained → 1 HNEI fine-tune)"),
    ]):
        ax.scatter(true_labels, preds, alpha=0.3, s=10)
        lo, hi = true_labels.min(), true_labels.max()
        ax.plot([lo, hi], [lo, hi], "--", color="red")
        ax.set_xlabel("True RUL (cycles)")
        ax.set_ylabel("Predicted RUL (cycles)")
        ax.set_title(title)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("few_shot_comparison.png", dpi=150)
    plt.show()
    print("\nPlot saved to few_shot_comparison.png")


if __name__ == "__main__":
    main()
