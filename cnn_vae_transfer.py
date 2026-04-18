import os
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
DATA_PATH  = "combined_scaled_battery_data.csv"
WINDOW_SIZE = 10
BATCH_SIZE  = 256
LATENT_DIM  = 32
LR          = 1e-3

VAE_EPOCHS      = 50
FINETUNE_EPOCHS = 100
BASELINE_EPOCHS = 50

NUM_FINETUNE_BATTERIES = 1

# Set to "HNEI" to train on HNEI and transfer to NASA,
# or "NASA" to train on NASA and transfer to HNEI.
SOURCE_DOMAIN = "NASA"

CKPT_VAE       = f"checkpoint_vae_{SOURCE_DOMAIN}.pt"
CKPT_TRANSFER  = f"checkpoint_transfer_{SOURCE_DOMAIN}.pt"
CKPT_BASELINE  = f"checkpoint_baseline_{SOURCE_DOMAIN}.pt"
DEMO_DATA_PATH = "demo_batteries.npz"
SCALER_PATH    = f"rul_scaler_{SOURCE_DOMAIN}.npz"

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
    """Split the full dataframe into one sub-dataframe per battery.

    Batteries are identified by detecting upward jumps in the RUL column
    (RUL resets to a high value at the start of each new battery).
    """
    diffs = np.diff(dataframe["RUL"].values)
    boundaries = np.where(diffs > 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends   = np.concatenate([boundaries, [len(dataframe)]])
    return [dataframe.iloc[s:e].reset_index(drop=True) for s, e in zip(starts, ends)]


def build_sliding_windows(
    segments: list[pd.DataFrame], window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert battery segments into overlapping windows for model input.

    Each window contains `window_size` consecutive cycles of feature data.
    The label for each window is the RUL at the last cycle in that window.
    Returns arrays of shape (N, window_size, num_features) and (N,).
    """
    windows, labels = [], []
    for seg in segments:
        features = seg[FEATURE_COLS].values.astype(np.float32)
        rul      = seg["RUL"].values.astype(np.float32)
        for i in range(len(seg) - window_size):
            windows.append(features[i : i + window_size])
            labels.append(rul[i + window_size - 1])
    return np.array(windows), np.array(labels, dtype=np.float32)


class RULScaler:
    """Min-max scaler for RUL labels, fit on training data only.

    Normalises RUL to [0, 1] so the model predicts a bounded value
    instead of raw cycle counts (0–1133), which improves training stability.
    Saves rul_min and rul_max to disk so the demo app can inverse-transform
    predictions back to actual cycle counts for display.
    """
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

    def save(self, path: str) -> None:
        np.savez(path, rul_min=self.rul_min, rul_max=self.rul_max)

    @classmethod
    def load(cls, path: str) -> "RULScaler":
        data = np.load(path)
        scaler = cls()
        scaler.rul_min = float(data["rul_min"])
        scaler.rul_max = float(data["rul_max"])
        return scaler


def make_loader(windows: np.ndarray, labels: np.ndarray, shuffle: bool) -> DataLoader:
    """Wrap numpy arrays into a DataLoader ready for model training or evaluation."""
    return DataLoader(
        WindowDataset(windows, labels),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """PyTorch Dataset that holds sliding-window samples.

    Converts input from (N, T, C) to (N, C, T) as required by Conv1d.
    """
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(windows).permute(0, 2, 1)
        self.y = torch.tensor(labels).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Single Conv1d layer followed by BatchNorm and ReLU activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvEncoder(nn.Module):
    """1D-CNN that compresses a cycle window into a 64-dim feature vector.

    Shared by both the VAE (for learning latent degradation patterns)
    and the baseline CNN (for direct RUL regression).
    Input:  (batch, num_features, window_size)
    Output: (batch, 64)
    """
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
    """Variational Autoencoder for learning battery degradation patterns.

    Phase 1 of the transfer learning pipeline.
    - Encoder: ConvEncoder → μ and log_var (defines the latent distribution)
    - Reparameterize: sample z = μ + ε·σ (allows gradients to flow through sampling)
    - Decoder: z → reconstructed window (trained to reproduce the input)

    After training, the encoder captures generalizable degradation features
    that can be transferred to a new battery domain.
    """
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
    """Phase 2 model: frozen VAE encoder + trainable regression head.

    The VAE encoder (trained on the source domain) is kept fixed so that
    the learned degradation representation is not overwritten.
    Only the small regression head is fine-tuned on the target domain,
    which requires very few target-domain batteries.
    """
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


class BaselineCNN(nn.Module):
    """Baseline: CNN trained on source domain, tested directly on target domain.

    No transfer learning — used to show how poorly a model generalises
    when applied to an unseen battery type without adaptation.
    """
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
# Loss
# ---------------------------------------------------------------------------

def vae_loss(x_recon, x_orig, mu, log_var) -> torch.Tensor:
    """VAE loss = reconstruction error + KL divergence.

    Reconstruction loss forces the decoder to reproduce the input window.
    KL loss regularises the latent space to stay close to a standard normal,
    preventing the encoder from memorising individual samples.
    """
    recon = nn.functional.mse_loss(x_recon, x_orig, reduction="mean")
    kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + 0.001 * kl


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_vae_epoch(vae, loader, optimizer, device) -> float:
    """Run one epoch of VAE training and return the average loss."""
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
    """Compute VAE loss on a validation loader without updating weights."""
    vae.eval()
    total = 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_recon, mu, log_var = vae(x)
            total += vae_loss(x_recon, x, mu, log_var).item() * len(x)
    return total / len(loader.dataset)


def train_regression_epoch(model, loader, criterion, optimizer, device) -> float:
    """Run one epoch of regression training and return the average MSE loss."""
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


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a DataLoader and return (predictions, true_labels)."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(preds).flatten(), np.concatenate(labels).flatten()


def print_metrics(name: str, preds: np.ndarray, labels: np.ndarray) -> dict:
    """Print MAE, RMSE, R² for a set of predictions and return them as a dict."""
    mae  = mean_absolute_error(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    r2   = r2_score(labels, preds)
    print(f"{name:35s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def load_or_train(model, ckpt_path, train_fn):
    """Load model weights from checkpoint if it exists, otherwise run train_fn to train.

    train_fn should be a zero-argument callable that trains the model in place
    and returns nothing. After training, weights are saved to ckpt_path.
    """
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"  Loaded from {ckpt_path}, skipping training.")
    else:
        train_fn()
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint to {ckpt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- segment data by battery ---
    df = pd.read_csv(DATA_PATH)
    all_segments  = split_into_battery_segments(df)
    hnei_segments = [s for s in all_segments if s["Is_NASA"].iloc[0] == 0]
    nasa_segments = [s for s in all_segments if s["Is_NASA"].iloc[0] == 1]
    print(f"Total — HNEI: {len(hnei_segments)} batteries, NASA: {len(nasa_segments)} batteries")

    # --- reserve one battery from each domain for demo only ---
    rng = np.random.default_rng(42)
    demo_hnei = hnei_segments[rng.integers(len(hnei_segments))]
    demo_nasa = nasa_segments[rng.integers(len(nasa_segments))]
    hnei_segments = [s for s in hnei_segments if not s.equals(demo_hnei)]
    nasa_segments = [s for s in nasa_segments if not s.equals(demo_nasa)]
    print(f"Demo batteries reserved — HNEI: 1, NASA: 1")

    # save demo battery windows so the Streamlit app can load them directly
    demo_hnei_windows, demo_hnei_labels = build_sliding_windows([demo_hnei], WINDOW_SIZE)
    demo_nasa_windows, demo_nasa_labels = build_sliding_windows([demo_nasa], WINDOW_SIZE)
    np.savez(
        DEMO_DATA_PATH,
        hnei_windows=demo_hnei_windows, hnei_labels=demo_hnei_labels,
        nasa_windows=demo_nasa_windows, nasa_labels=demo_nasa_labels,
    )
    print(f"Demo data saved to {DEMO_DATA_PATH}\n")

    # --- assign source / target domains based on SOURCE_DOMAIN setting ---
    if SOURCE_DOMAIN == "HNEI":
        source_segments, target_segments = hnei_segments, nasa_segments
    else:
        source_segments, target_segments = nasa_segments, hnei_segments
    target_name = "NASA" if SOURCE_DOMAIN == "HNEI" else "HNEI"
    print(f"Transfer direction: {SOURCE_DOMAIN} → {target_name}")

    # --- split target domain: fine-tune set + test set ---
    # Select the batteries with the most cycles for fine-tuning (most data = best adaptation signal).
    sorted_target = sorted(target_segments, key=lambda s: len(s), reverse=True)
    target_finetune = sorted_target[:NUM_FINETUNE_BATTERIES]
    target_test     = sorted_target[NUM_FINETUNE_BATTERIES:]
    finetune_lengths = [len(s) for s in target_finetune]
    print(f"{target_name} split — fine-tune: {len(target_finetune)} (cycles: {finetune_lengths}), test: {len(target_test)}")

    # --- 80/20 battery-level split for source domain validation ---
    source_shuffled = list(source_segments)
    rng.shuffle(source_shuffled)
    n_source_val        = max(1, int(len(source_shuffled) * 0.2))
    source_train_segs   = source_shuffled[n_source_val:]
    source_val_segs     = source_shuffled[:n_source_val]
    print(f"{SOURCE_DOMAIN} source split — train: {len(source_train_segs)}, val: {len(source_val_segs)}")

    # --- build raw window arrays ---
    X_source,      y_source      = build_sliding_windows(source_train_segs, WINDOW_SIZE)
    X_source_val,  y_source_val  = build_sliding_windows(source_val_segs,   WINDOW_SIZE)
    X_finetune,    y_finetune    = build_sliding_windows(target_finetune,    WINDOW_SIZE)
    X_target_test, y_target_test = build_sliding_windows(target_test,        WINDOW_SIZE)

    # source_scaler: fit on source training labels → used for VAE + baseline training.
    # target_scaler: fit on fine-tune labels only → used for transfer fine-tuning + evaluation.
    # Using separate scalers prevents target domain labels (e.g. NASA max~35 cycles) from being
    # crushed into [0, 0.03] by the source domain range (HNEI max~1133), which kills the gradient.
    source_scaler = RULScaler().fit(y_source)
    target_scaler = RULScaler().fit(y_finetune)
    source_scaler.save(SCALER_PATH)
    target_scaler_path = SCALER_PATH.replace(".npz", "_target.npz")
    target_scaler.save(target_scaler_path)
    print(f"Source scaler: min={source_scaler.rul_min:.1f}, max={source_scaler.rul_max:.1f}")
    print(f"Target scaler: min={target_scaler.rul_min:.1f}, max={target_scaler.rul_max:.1f}")
    print(f"Scalers saved to {SCALER_PATH} and {target_scaler_path}\n")

    source_loader           = make_loader(X_source,      source_scaler.transform(y_source),      shuffle=True)
    source_val_loader       = make_loader(X_source_val,  source_scaler.transform(y_source_val),  shuffle=False)
    finetune_loader         = make_loader(X_finetune,    target_scaler.transform(y_finetune),    shuffle=True)
    target_test_loader_src  = make_loader(X_target_test, source_scaler.transform(y_target_test), shuffle=False)
    target_test_loader_tgt  = make_loader(X_target_test, target_scaler.transform(y_target_test), shuffle=False)
    mse = nn.MSELoss()

    # -----------------------------------------------------------------------
    # Phase 1: Train VAE on source domain
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Phase 1: VAE training on {SOURCE_DOMAIN} data")
    print("=" * 60)

    vae       = BatteryVAE(NUM_FEATURES, WINDOW_SIZE, LATENT_DIM).to(device)
    vae_optim = torch.optim.Adam(vae.parameters(), lr=LR, weight_decay=1e-4)

    def train_vae():
        for epoch in range(1, VAE_EPOCHS + 1):
            train_loss = train_vae_epoch(vae, source_loader, vae_optim, device)
            if epoch % 10 == 0 or epoch == 1:
                val_loss = eval_vae_epoch(vae, source_val_loader, device)
                print(f"  epoch {epoch:3d}/{VAE_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

    load_or_train(vae, CKPT_VAE, train_vae)

    # -----------------------------------------------------------------------
    # Phase 2: Fine-tune regression head on target domain
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Phase 2: Fine-tuning regression head on {target_name} data")
    print("=" * 60)

    transfer_model = TransferRULNet(vae.encoder, vae.fc_mu, LATENT_DIM).to(device)
    ft_optim       = torch.optim.Adam(
        filter(lambda p: p.requires_grad, transfer_model.parameters()), lr=LR, weight_decay=1e-4
    )

    def train_transfer():
        for epoch in range(1, FINETUNE_EPOCHS + 1):
            loss = train_regression_epoch(transfer_model, finetune_loader, mse, ft_optim, device)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{FINETUNE_EPOCHS}  loss={loss:.4f}")

    load_or_train(transfer_model, CKPT_TRANSFER, train_transfer)

    # -----------------------------------------------------------------------
    # Baseline: CNN trained on source domain, no transfer
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Baseline: CNN trained on {SOURCE_DOMAIN}, tested directly on {target_name}")
    print("=" * 60)

    baseline_model     = BaselineCNN(NUM_FEATURES).to(device)
    baseline_optim     = torch.optim.Adam(baseline_model.parameters(), lr=LR, weight_decay=1e-4)
    baseline_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(baseline_optim, T_max=BASELINE_EPOCHS)

    def train_baseline():
        for epoch in range(1, BASELINE_EPOCHS + 1):
            train_loss = train_regression_epoch(baseline_model, source_loader, mse, baseline_optim, device)
            baseline_scheduler.step()
            if epoch % 10 == 0 or epoch == 1:
                val_preds, val_labels = evaluate(baseline_model, source_val_loader, device)
                val_loss = float(mse(torch.tensor(val_preds), torch.tensor(val_labels)))
                print(f"  epoch {epoch:3d}/{BASELINE_EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

    load_or_train(baseline_model, CKPT_BASELINE, train_baseline)

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Results on held-out {target_name} test batteries")
    print("=" * 60)

    # Baseline uses source scaler (trained in source space); transfer uses target scaler.
    baseline_preds_norm, _ = evaluate(baseline_model, target_test_loader_src, device)
    transfer_preds_norm, labels_tgt_norm = evaluate(transfer_model, target_test_loader_tgt, device)

    baseline_preds = source_scaler.inverse_transform(baseline_preds_norm)
    transfer_preds = target_scaler.inverse_transform(transfer_preds_norm)
    true_labels    = target_scaler.inverse_transform(labels_tgt_norm)

    print_metrics("Baseline CNN (no transfer)", baseline_preds, true_labels)
    print_metrics("VAE Transfer (proposed)",    transfer_preds, true_labels)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (labels, preds, title) in zip(axes, [
        (true_labels, baseline_preds, "Baseline CNN (no transfer)"),
        (true_labels, transfer_preds, "VAE Transfer (proposed)"),
    ]):
        ax.scatter(labels, preds, alpha=0.3, s=10)
        ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], "--", color="red")
        ax.set_xlabel("True RUL")
        ax.set_ylabel("Predicted RUL")
        ax.set_title(title)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("vae_transfer_results.png", dpi=150)
    plt.show()
    print("Plot saved to vae_transfer_results.png")

    # -----------------------------------------------------------------------
    # Demo battery evaluation: RUL curve over time for one HNEI and one NASA
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Demo battery results")
    print("=" * 60)

    # Each demo battery is evaluated in its own domain's scale.
    # HNEI demo → source_scaler; NASA demo → target_scaler (or reversed when SOURCE_DOMAIN="NASA").
    demo_configs = [
        (demo_hnei, "Demo HNEI Battery", source_scaler if SOURCE_DOMAIN == "HNEI" else target_scaler),
        (demo_nasa, "Demo NASA Battery", target_scaler if SOURCE_DOMAIN == "HNEI" else source_scaler),
    ]

    _, demo_axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (demo_seg, title, demo_scaler) in zip(demo_axes, demo_configs):
        X_demo, y_demo = build_sliding_windows([demo_seg], WINDOW_SIZE)
        y_demo_norm    = demo_scaler.transform(y_demo)
        demo_loader    = make_loader(X_demo, y_demo_norm, shuffle=False)

        baseline_norm, labels_norm_demo = evaluate(baseline_model, demo_loader, device)
        transfer_norm, _                = evaluate(transfer_model,  demo_loader, device)

        true_rul      = demo_scaler.inverse_transform(labels_norm_demo)
        baseline_rul  = demo_scaler.inverse_transform(baseline_norm)
        transfer_rul  = demo_scaler.inverse_transform(transfer_norm)

        cycles = np.arange(len(true_rul))
        ax.plot(cycles, true_rul,     label="True RUL",              linewidth=2)
        ax.plot(cycles, baseline_rul, label="Baseline CNN",  linestyle="--")
        ax.plot(cycles, transfer_rul, label="VAE Transfer",  linestyle="--")
        ax.set_xlabel("Window index (cycle progression)")
        ax.set_ylabel("RUL (cycles)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        print(f"\n{title}")
        print_metrics("  Baseline CNN", baseline_rul, true_rul)
        print_metrics("  VAE Transfer", transfer_rul, true_rul)

    plt.tight_layout()
    plt.savefig("demo_battery_results.png", dpi=150)
    plt.show()
    print("\nDemo plot saved to demo_battery_results.png")


if __name__ == "__main__":
    main()
