from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import RMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


SEED = 42
MODEL_TARGET_COL = "target_scaled"
FEATURE_RENAME = {
    "Cycle": "time_idx",
    "Discharge Time (s)": "discharge_time",
    "Decrement 3.6-3.4V (s)": "decrement_36_34",
    "Max. Voltage Dischar. (V)": "max_voltage_discharge",
    "Min. Voltage Charg. (V)": "min_voltage_charge",
    "Time at 4.15V (s)": "time_at_415",
    "Time constant current (s)": "time_constant_current",
    "Charging time (s)": "charging_time",
    "RUL": "target",
}
RAW_FEATURE_COLS = [
    "discharge_time",
    "decrement_36_34",
    "max_voltage_discharge",
    "min_voltage_charge",
    "time_at_415",
    "time_constant_current",
    "charging_time",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a features.csv VAE transfer experiment: pretrain VAE+TFT on one "
            "cell, evaluate target zero-shot, then fine-tune on target cells."
        )
    )
    parser.add_argument("--data-path", type=Path, default=Path("features.csv"))
    parser.add_argument("--source-cell", type=str, default=None, help="Defaults to the longest cell trajectory.")
    parser.add_argument("--target-train-cells", type=int, default=2)
    parser.add_argument("--target-val-cells", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--vae-hidden-size", type=int, default=64)
    parser.add_argument("--vae-epochs", type=int, default=5)
    parser.add_argument("--source-epochs", type=int, default=4)
    parser.add_argument("--finetune-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vae-batch-size", type=int, default=256)
    parser.add_argument("--source-lr", type=float, default=3e-2)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-mode", choices=["all", "head_only"], default="head_only")
    parser.add_argument("--use-raw-features", action="store_true", help="Append raw scaled features to VAE latents for TFT.")
    parser.add_argument("--output-dir", type=Path, default=Path("model_checkpoints/tft_vae_features_transfer"))
    parser.add_argument("--figure-path", type=Path, default=Path("figures/tft_vae_features_before_after_finetune.png"))
    parser.add_argument("--predictions-path", type=Path, default=Path("tft_vae_features_transfer_predictions.csv"))
    parser.add_argument("--metrics-path", type=Path, default=Path("tft_vae_features_transfer_results.csv"))
    return parser.parse_args()


def set_reproducibility(seed: int = SEED) -> torch.device:
    seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def impute_feature_gaps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["battery_id", "time_idx"]).reset_index(drop=True)
    df[RAW_FEATURE_COLS] = (
        df.groupby("battery_id", group_keys=False)[RAW_FEATURE_COLS]
        .apply(lambda group: group.interpolate(method="linear", limit_direction="both"))
    )
    df[RAW_FEATURE_COLS] = df[RAW_FEATURE_COLS].fillna(df[RAW_FEATURE_COLS].median(numeric_only=True))
    return df


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns=FEATURE_RENAME).rename(columns={"cell_id": "battery_id"})
    df["battery_id"] = df["battery_id"].astype(str)
    df["time_idx"] = df["time_idx"].astype(int)
    df["target"] = df["target"].astype(float)
    df = impute_feature_gaps(df)
    required = ["battery_id", "time_idx", "target", *RAW_FEATURE_COLS]
    missing_counts = df[required].isna().sum()
    if missing_counts.any():
        raise ValueError(f"Unexpected missing values after imputation:\n{missing_counts[missing_counts > 0]}")
    return df


def choose_source_cell(df: pd.DataFrame, source_cell: str | None) -> str:
    if source_cell is not None:
        if source_cell not in set(df["battery_id"]):
            raise ValueError(f"source cell {source_cell!r} not found in features.csv")
        return source_cell
    return str(df.groupby("battery_id").size().sort_values(ascending=False).index[0])


def split_one_source_cell(source_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_df = source_df.sort_values("time_idx").copy()
    n = len(source_df)
    train_end = max(20, int(n * 0.70))
    val_end = max(train_end + 10, int(n * 0.85))
    val_end = min(val_end, n - 1)
    return source_df.iloc[:train_end].copy(), source_df.iloc[train_end:val_end].copy(), source_df.iloc[val_end:].copy()


def split_target_cells(
    df: pd.DataFrame,
    source_cell: str,
    target_train_cells: int,
    target_val_cells: int,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    target_ids = np.array(sorted(cell for cell in df["battery_id"].unique() if cell != source_cell))
    min_needed = target_train_cells + target_val_cells + 1
    if len(target_ids) < min_needed:
        raise ValueError(f"Need at least {min_needed} non-source cells, found {len(target_ids)}")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(target_ids))
    train_ids = sorted(shuffled[:target_train_cells])
    val_ids = sorted(shuffled[target_train_cells : target_train_cells + target_val_cells])
    test_ids = sorted(shuffled[target_train_cells + target_val_cells :])
    target_train = df[df["battery_id"].isin(train_ids)].copy()
    target_val = df[df["battery_id"].isin(val_ids)].copy()
    target_test = df[df["battery_id"].isin(test_ids)].copy()
    return target_train, target_val, target_test, train_ids, val_ids, test_ids


def build_feature_scaler(fit_df: pd.DataFrame) -> dict[str, object]:
    means = fit_df[RAW_FEATURE_COLS].mean().astype(float)
    stds = fit_df[RAW_FEATURE_COLS].std(ddof=0).replace(0, 1.0).astype(float)
    return {"mean": means.to_dict(), "std": stds.to_dict(), "fit_battery_ids": sorted(fit_df["battery_id"].unique())}


def apply_feature_scaler(df: pd.DataFrame, scaler: dict[str, object]) -> pd.DataFrame:
    result = df.copy()
    for col in RAW_FEATURE_COLS:
        result[col] = (result[col] - float(scaler["mean"][col])) / float(scaler["std"][col])
    return result


def build_target_scaler(fit_df: pd.DataFrame) -> dict[str, object]:
    target_min = float(fit_df["target"].min())
    target_max = float(fit_df["target"].max())
    if target_max <= target_min:
        raise ValueError("Invalid target range for target scaling")
    return {
        "target_min": target_min,
        "target_max": target_max,
        "target_range": target_max - target_min,
        "fit_battery_ids": sorted(fit_df["battery_id"].unique()),
    }


def add_normalized_target(df: pd.DataFrame, scaler: dict[str, object]) -> pd.DataFrame:
    result = df.copy()
    result[MODEL_TARGET_COL] = (result["target"] - scaler["target_min"]) / scaler["target_range"]
    return result


def inverse_target_scale(values: np.ndarray, scaler: dict[str, object]) -> np.ndarray:
    return values * scaler["target_range"] + scaler["target_min"]


class FeatureWindowDataset(Dataset):
    def __init__(self, data: pd.DataFrame, feature_cols: list[str], window_length: int):
        windows = []
        for _, group in data.groupby("battery_id", sort=False):
            values = group[feature_cols].to_numpy(dtype=np.float32)
            for end in range(len(values)):
                start = max(0, end - window_length + 1)
                window = values[start : end + 1]
                if len(window) < window_length:
                    pad = np.repeat(window[:1], window_length - len(window), axis=0)
                    window = np.vstack([pad, window])
                windows.append(window)
        self.windows = torch.tensor(np.stack(windows), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.windows[idx]


class SequenceVAE(nn.Module):
    def __init__(self, num_features: int, hidden_size: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.GRU(num_features, hidden_size, batch_first=True)
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.logvar = nn.Linear(hidden_size, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size)
        self.decoder = nn.GRU(num_features, hidden_size, batch_first=True)
        self.reconstruction = nn.Linear(hidden_size, num_features)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, h = self.encoder(x)
        h = h[-1]
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, seq_len: int, num_features: int) -> torch.Tensor:
        h0 = self.latent_to_hidden(z).unsqueeze(0)
        decoder_input = torch.zeros(z.size(0), seq_len, num_features, device=z.device)
        decoded, _ = self.decoder(decoder_input, h0)
        return self.reconstruction(decoded)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, x.size(1), x.size(2)), mu, logvar


def train_vae(
    source_train_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SequenceVAE, pd.DataFrame, Path]:
    dataset = FeatureWindowDataset(source_train_df, RAW_FEATURE_COLS, args.window_size)
    loader = DataLoader(dataset, batch_size=args.vae_batch_size, shuffle=True, num_workers=0)
    model = SequenceVAE(len(RAW_FEATURE_COLS), args.vae_hidden_size, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []

    for epoch in range(1, args.vae_epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_seen = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            recon_loss = nn.functional.mse_loss(recon, batch, reduction="mean")
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 1e-3 * kl_loss
            loss.backward()
            optimizer.step()
            n = batch.size(0)
            total_loss += float(loss.detach().cpu()) * n
            total_recon += float(recon_loss.detach().cpu()) * n
            total_kl += float(kl_loss.detach().cpu()) * n
            n_seen += n
        row = {
            "epoch": epoch,
            "loss": total_loss / n_seen,
            "reconstruction_loss": total_recon / n_seen,
            "kl_loss": total_kl / n_seen,
        }
        history.append(row)
        print(
            f"[VAE] epoch {epoch:02d}/{args.vae_epochs} "
            f"loss={row['loss']:.5f} recon={row['reconstruction_loss']:.5f} kl={row['kl_loss']:.5f}"
        )

    history_df = pd.DataFrame(history)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "source_one_cell_vae.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history_df,
            "feature_cols": RAW_FEATURE_COLS,
            "window_length": args.window_size,
            "latent_dim": args.latent_dim,
            "hidden_size": args.vae_hidden_size,
        },
        checkpoint_path,
    )
    return model, history_df, checkpoint_path


@torch.no_grad()
def add_frozen_vae_latents(
    data: pd.DataFrame,
    vae: SequenceVAE,
    args: argparse.Namespace,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[pd.DataFrame, list[str]]:
    vae.eval()
    windows = []
    row_indices = []
    for _, group in data.groupby("battery_id", sort=False):
        values = group[RAW_FEATURE_COLS].to_numpy(dtype=np.float32)
        indices = group.index.to_numpy()
        for pos, idx in enumerate(indices):
            start = max(0, pos - args.window_size + 1)
            window = values[start : pos + 1]
            if len(window) < args.window_size:
                pad = np.repeat(window[:1], args.window_size - len(window), axis=0)
                window = np.vstack([pad, window])
            windows.append(window)
            row_indices.append(idx)

    latent_chunks = []
    for start in range(0, len(windows), batch_size):
        batch = torch.tensor(np.stack(windows[start : start + batch_size]), dtype=torch.float32, device=device)
        mu, _ = vae.encode(batch)
        latent_chunks.append(mu.cpu().numpy())

    latents = np.vstack(latent_chunks)
    latent_cols = [f"vae_latent_{i:02d}" for i in range(args.latent_dim)]
    result = data.copy()
    for col in latent_cols:
        result[col] = np.nan
    result.loc[row_indices, latent_cols] = latents
    result[latent_cols] = result[latent_cols].astype(float)
    return result, latent_cols


def attach_prediction_index(dataset: TimeSeriesDataSet, data: pd.DataFrame) -> TimeSeriesDataSet:
    group_order = list(data.groupby("battery_id", sort=False).groups.keys())
    index = dataset.index.copy().reset_index(drop=True)
    if {"sequence_id", "time", "sequence_length"}.issubset(index.columns):
        prediction_index = pd.DataFrame(
            {
                "battery_id": index["sequence_id"].map(lambda value: str(group_order[int(value)])),
                "time_idx": (index["time"] + index["sequence_length"] - dataset.max_prediction_length).astype(int),
            }
        )
    else:
        decoded = dataset.decoded_index.reset_index(drop=True).copy()
        prediction_index = pd.DataFrame(
            {
                "battery_id": decoded["battery_id"].astype(str),
                "time_idx": decoded.filter(like="time_idx").iloc[:, -1].astype(int),
            }
        )
    dataset.prediction_index_ = prediction_index
    return dataset


def make_dataset(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    unknown_reals: list[str],
    args: argparse.Namespace,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=MODEL_TARGET_COL,
        group_ids=["battery_id"],
        categorical_encoders={
            "battery_id": NaNLabelEncoder(add_nan=True),
            "__group_id__battery_id": NaNLabelEncoder(add_nan=True),
        },
        min_encoder_length=1,
        max_encoder_length=args.window_size,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_reals=[],
        time_varying_unknown_reals=unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
        target_normalizer=None,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=True)
    test = TimeSeriesDataSet.from_dataset(training, test_df, predict=False, stop_randomization=True)
    attach_prediction_index(training, train_df)
    attach_prediction_index(validation, val_df)
    attach_prediction_index(test, test_df)
    return training, validation, test


def make_transfer_dataset(source_training: TimeSeriesDataSet, data: pd.DataFrame) -> TimeSeriesDataSet:
    dataset = TimeSeriesDataSet.from_dataset(source_training, data, predict=False, stop_randomization=True)
    attach_prediction_index(dataset, data)
    return dataset


def build_tft(training: TimeSeriesDataSet, learning_rate: float) -> TemporalFusionTransformer:
    return TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.2,
        hidden_continuous_size=8,
        output_size=1,
        loss=RMSE(),
        reduce_on_plateau_patience=3,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def configure_finetuning(model: TemporalFusionTransformer, mode: str) -> TemporalFusionTransformer:
    if mode == "all":
        for param in model.parameters():
            param.requires_grad = True
    elif mode == "head_only":
        for param in model.parameters():
            param.requires_grad = False
        trainable = 0
        for name, param in model.named_parameters():
            if "output_layer" in name:
                param.requires_grad = True
                trainable += param.numel()
        if trainable == 0:
            for param in model.parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"Unsupported finetune mode: {mode}")
    print(f"Fine-tuning mode={mode}; trainable parameters={count_trainable_parameters(model):,}")
    return model


def fit_tft(
    model: TemporalFusionTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
    experiment_name: str,
    args: argparse.Namespace,
) -> TemporalFusionTransformer:
    checkpoint_dir = args.output_dir / experiment_name
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename=f"{experiment_name}-{{epoch:02d}}-{{val_loss:.4f}}",
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, mode="min"),
            checkpoint,
        ],
        logger=CSVLogger("lightning_logs", name=experiment_name),
        enable_checkpointing=True,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model.best_checkpoint_path_ = checkpoint.best_model_path
    model.last_checkpoint_path_ = checkpoint.last_model_path
    print(f"Best checkpoint for {experiment_name}: {checkpoint.best_model_path}")
    return model


def tensor_from_prediction(predictions: object) -> torch.Tensor:
    if isinstance(predictions, torch.Tensor):
        return predictions
    if hasattr(predictions, "prediction"):
        return predictions.prediction
    return torch.as_tensor(predictions)


def prediction_index_from_loader(loader: DataLoader) -> pd.DataFrame:
    dataset = loader.dataset
    if not hasattr(dataset, "prediction_index_"):
        raise ValueError("Prediction dataset is missing prediction_index_.")
    return dataset.prediction_index_.reset_index(drop=True).copy()


def evaluate_tft(
    model: TemporalFusionTransformer,
    loader: DataLoader,
    target_scaler: dict[str, object],
    stage: str,
) -> dict[str, object]:
    predictions_scaled = tensor_from_prediction(
        model.predict(
            loader,
            trainer_kwargs={
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
            },
        )
    )
    actuals_scaled = torch.cat([y[0] for _, y in iter(loader)])
    preds_scaled = np.atleast_1d(predictions_scaled.squeeze().detach().cpu().numpy())
    acts_scaled = np.atleast_1d(actuals_scaled.squeeze().detach().cpu().numpy())
    preds = np.maximum(inverse_target_scale(preds_scaled, target_scaler), 0.0)
    acts = inverse_target_scale(acts_scaled, target_scaler)

    prediction_df = prediction_index_from_loader(loader).iloc[: len(preds)].copy()
    prediction_df["true_rul"] = acts
    prediction_df["predicted_rul"] = preds
    prediction_df["true_rul_scaled"] = acts_scaled
    prediction_df["predicted_rul_scaled"] = preds_scaled
    prediction_df["stage"] = stage

    metrics = {
        "stage": stage,
        "mae": float(mean_absolute_error(acts, preds)),
        "rmse": float(np.sqrt(mean_squared_error(acts, preds))),
        "r2": float(r2_score(acts, preds)),
        "mae_scaled": float(mean_absolute_error(acts_scaled, preds_scaled)),
        "rmse_scaled": float(np.sqrt(mean_squared_error(acts_scaled, preds_scaled))),
        "r2_scaled": float(r2_score(acts_scaled, preds_scaled)),
        "prediction_df": prediction_df,
        "actuals": acts,
        "preds": preds,
    }
    print(f"{stage}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")
    return metrics


def make_loader(dataset: TimeSeriesDataSet, train: bool, batch_size: int) -> DataLoader:
    return dataset.to_dataloader(train=train, batch_size=batch_size, num_workers=0)


def plot_before_after(before: dict[str, object], after: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [("Before fine-tuning", before), ("After fine-tuning", after)]
    all_actuals = np.concatenate([panel[1]["actuals"] for panel in panels])
    all_preds = np.concatenate([panel[1]["preds"] for panel in panels])
    lower = float(min(all_actuals.min(), all_preds.min()))
    upper = float(max(all_actuals.max(), all_preds.max()))
    pad = 0.05 * (upper - lower) if upper > lower else 1.0
    lower = max(0.0, lower - pad)
    upper += pad

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, (title, metrics) in zip(axes, panels):
        ax.scatter(metrics["actuals"], metrics["preds"], alpha=0.35, s=10)
        ax.plot([lower, upper], [lower, upper], "--", color="red", linewidth=1.4)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_xlabel("True RUL")
        ax.set_ylabel("Predicted RUL")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.text(
            0.04,
            0.96,
            f"MAE={metrics['mae']:.1f}\nRMSE={metrics['rmse']:.1f}\nR2={metrics['r2']:.2f}",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
    fig.suptitle("VAE latent TFT transfer on features.csv")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def json_safe_args(args: argparse.Namespace) -> dict[str, object]:
    safe = {}
    for key, value in vars(args).items():
        safe[key] = str(value) if isinstance(value, Path) else value
    return safe


def run() -> None:
    args = parse_args()
    device = set_reproducibility()
    print(f"Using device: {device}")

    df = load_features(args.data_path)
    source_cell = choose_source_cell(df, args.source_cell)
    source_all = df[df["battery_id"] == source_cell].copy()
    source_train, source_val, source_test = split_one_source_cell(source_all)
    target_train, target_val, target_test, target_train_ids, target_val_ids, target_test_ids = split_target_cells(
        df,
        source_cell,
        args.target_train_cells,
        args.target_val_cells,
    )

    print(f"Source pretraining cell: {source_cell}")
    print(f"Source rows train/val/test: {len(source_train)}/{len(source_val)}/{len(source_test)}")
    print(f"Target fine-tune cells: {target_train_ids}")
    print(f"Target val cells: {target_val_ids}")
    print(f"Target held-out test cells: {target_test_ids}")

    source_feature_scaler = build_feature_scaler(source_train)
    source_target_scaler = build_target_scaler(source_train)
    target_target_scaler = build_target_scaler(target_train)

    scaled_parts = {
        "source_train": apply_feature_scaler(source_train, source_feature_scaler),
        "source_val": apply_feature_scaler(source_val, source_feature_scaler),
        "source_test": apply_feature_scaler(source_test, source_feature_scaler),
        "target_train": apply_feature_scaler(target_train, source_feature_scaler),
        "target_val": apply_feature_scaler(target_val, source_feature_scaler),
        "target_test": apply_feature_scaler(target_test, source_feature_scaler),
    }

    vae, vae_history, vae_checkpoint = train_vae(scaled_parts["source_train"], args, device)
    for param in vae.parameters():
        param.requires_grad = False

    latent_parts = {}
    latent_cols = []
    for name, part in scaled_parts.items():
        latent_df, latent_cols = add_frozen_vae_latents(part, vae, args, device)
        latent_parts[name] = latent_df

    source_reals = (RAW_FEATURE_COLS if args.use_raw_features else []) + latent_cols
    source_train_model = add_normalized_target(latent_parts["source_train"], source_target_scaler)
    source_val_model = add_normalized_target(latent_parts["source_val"], source_target_scaler)
    source_test_model = add_normalized_target(latent_parts["source_test"], source_target_scaler)
    target_train_model = add_normalized_target(latent_parts["target_train"], target_target_scaler)
    target_val_model = add_normalized_target(latent_parts["target_val"], target_target_scaler)
    target_test_model = add_normalized_target(latent_parts["target_test"], target_target_scaler)

    source_training, source_validation, source_testing = make_dataset(
        source_train_model,
        source_val_model,
        source_test_model,
        source_reals,
        args,
    )
    print(f"TFT reals: {source_training.reals}")
    print(f"TFT samples source train/val/test: {len(source_training)}/{len(source_validation)}/{len(source_testing)}")

    source_tft = build_tft(source_training, args.source_lr)
    source_tft = fit_tft(
        source_tft,
        make_loader(source_training, train=True, batch_size=args.batch_size),
        make_loader(source_validation, train=False, batch_size=args.batch_size),
        args.source_epochs,
        "tft_vae_features_source_one_cell",
        args,
    )
    source_checkpoint = Path(source_tft.best_checkpoint_path_)
    stable_source_checkpoint = args.output_dir / "source_one_cell_tft.ckpt"
    stable_source_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if source_checkpoint.exists():
        shutil.copyfile(source_checkpoint, stable_source_checkpoint)

    target_zero_dataset = make_transfer_dataset(source_training, target_test_model)
    target_zero_loader = make_loader(target_zero_dataset, train=False, batch_size=args.batch_size)
    zero_metrics = evaluate_tft(source_tft, target_zero_loader, target_target_scaler, "before_finetuning")

    target_train_dataset = make_transfer_dataset(source_training, target_train_model)
    target_val_dataset = make_transfer_dataset(source_training, target_val_model)
    target_test_dataset = make_transfer_dataset(source_training, target_test_model)

    transfer_tft = build_tft(source_training, args.finetune_lr)
    transfer_tft.load_state_dict(copy.deepcopy(source_tft.state_dict()))
    transfer_tft = configure_finetuning(transfer_tft, args.finetune_mode)
    transfer_tft = fit_tft(
        transfer_tft,
        make_loader(target_train_dataset, train=True, batch_size=args.batch_size),
        make_loader(target_val_dataset, train=False, batch_size=args.batch_size),
        args.finetune_epochs,
        "tft_vae_features_finetuned",
        args,
    )
    transfer_checkpoint = Path(transfer_tft.best_checkpoint_path_)
    stable_transfer_checkpoint = args.output_dir / "finetuned_tft.ckpt"
    if transfer_checkpoint.exists():
        shutil.copyfile(transfer_checkpoint, stable_transfer_checkpoint)

    transfer_metrics = evaluate_tft(
        transfer_tft,
        make_loader(target_test_dataset, train=False, batch_size=args.batch_size),
        target_target_scaler,
        "after_finetuning",
    )

    prediction_df = pd.concat(
        [zero_metrics["prediction_df"], transfer_metrics["prediction_df"]],
        ignore_index=True,
    )
    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(args.predictions_path, index=False)

    metrics_rows = []
    for metrics in [zero_metrics, transfer_metrics]:
        metrics_rows.append(
            {
                "stage": metrics["stage"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "mae_scaled": metrics["mae_scaled"],
                "rmse_scaled": metrics["rmse_scaled"],
                "r2_scaled": metrics["r2_scaled"],
                "source_cell": source_cell,
                "target_train_cells": ", ".join(target_train_ids),
                "target_val_cells": ", ".join(target_val_ids),
                "target_test_cells": ", ".join(target_test_ids),
                "vae_checkpoint": str(vae_checkpoint),
                "source_tft_checkpoint": str(stable_source_checkpoint),
                "finetuned_tft_checkpoint": str(stable_transfer_checkpoint),
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.metrics_path, index=False)

    metadata = {
        "source_cell": source_cell,
        "source_rows": {"train": len(source_train), "val": len(source_val), "test": len(source_test)},
        "target_train_cells": target_train_ids,
        "target_val_cells": target_val_ids,
        "target_test_cells": target_test_ids,
        "feature_scaler_fit_cells": source_feature_scaler["fit_battery_ids"],
        "source_target_scaler": source_target_scaler,
        "target_target_scaler": target_target_scaler,
        "latent_cols": latent_cols,
        "tft_reals": source_reals,
        "vae_history": vae_history.to_dict(orient="records"),
        "args": json_safe_args(args),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    plot_before_after(zero_metrics, transfer_metrics, args.figure_path)

    print("Saved metrics:", args.metrics_path)
    print("Saved predictions:", args.predictions_path)
    print("Saved figure:", args.figure_path)
    print("Saved metadata:", args.output_dir / "metadata.json")


if __name__ == "__main__":
    run()
