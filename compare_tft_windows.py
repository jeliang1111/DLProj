import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


FEATURE_RENAME = {
    "Cycle_Index": "time_idx",
    "Discharge Time (s)": "discharge_time",
    "Decrement 3.6-3.4V (s)": "decrement_36_34",
    "Max. Voltage Dischar. (V)": "max_voltage_discharge",
    "Min. Voltage Charg. (V)": "min_voltage_charge",
    "Time at 4.15V (s)": "time_at_415",
    "Time constant current (s)": "time_constant_current",
    "Charging time (s)": "charging_time",
    "RUL": "target",
}

UNKNOWN_REALS = [
    "discharge_time",
    "decrement_36_34",
    "max_voltage_discharge",
    "min_voltage_charge",
    "time_at_415",
    "time_constant_current",
    "charging_time",
]


def load_data(path: Path, seed: int):
    df = pd.read_csv(path)
    df = df.rename(columns=FEATURE_RENAME)
    df = df.reset_index(drop=True)
    df["battery_id"] = (df["time_idx"] == 1).cumsum() - 1
    df = df.sort_values(["battery_id", "time_idx"]).reset_index(drop=True)
    df["battery_id"] = df["battery_id"].astype(str)
    df["time_idx"] = df["time_idx"].astype(int)
    df["Is_NASA"] = df["Is_NASA"].astype(float)

    battery_ids = df["battery_id"].unique()
    train_ids, test_ids = train_test_split(battery_ids, test_size=0.2, random_state=seed)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=seed)

    train_df = df[df["battery_id"].isin(train_ids)].copy()
    val_df = df[df["battery_id"].isin(val_ids)].copy()
    test_df = df[df["battery_id"].isin(test_ids)].copy()
    return train_df, val_df, test_df


def build_datasets(train_df, val_df, test_df, max_encoder_length: int):
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target",
        group_ids=["battery_id"],
        min_encoder_length=1,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=[],
        static_reals=["Is_NASA"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=UNKNOWN_REALS,
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
        target_normalizer=None,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, predict=False, stop_randomization=True
    )
    test = TimeSeriesDataSet.from_dataset(
        training, test_df, predict=False, stop_randomization=True
    )
    return training, validation, test


def tensor_from_prediction(predictions):
    if isinstance(predictions, torch.Tensor):
        return predictions
    if hasattr(predictions, "prediction"):
        return predictions.prediction
    return torch.as_tensor(predictions)


def evaluate(model, dataloader):
    predictions = tensor_from_prediction(
        model.predict(
            dataloader,
            trainer_kwargs={
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
            },
        )
    )
    actuals = torch.cat([y[0] for _, y in iter(dataloader)])

    preds = predictions.squeeze().detach().cpu().numpy()
    acts = actuals.squeeze().detach().cpu().numpy()
    return {
        "test_mae": float(mean_absolute_error(acts, preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(acts, preds))),
        "num_test_predictions": int(len(acts)),
    }


def run_window(args, train_df, val_df, test_df, window: int):
    seed_everything(args.seed, workers=True)
    training, validation, test = build_datasets(train_df, val_df, test_df, window)

    train_dataloader = training.to_dataloader(
        train=True, batch_size=args.batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=0
    )
    test_dataloader = test.to_dataloader(
        train=False, batch_size=args.batch_size, num_workers=0
    )

    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_continuous_size,
        output_size=1,
        loss=RMSE(),
        reduce_on_plateau_patience=3,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=args.patience,
        mode="min",
    )
    logger = CSVLogger(
        save_dir=str(args.log_dir),
        name=f"window_{window}",
    )
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        logger=logger,
        enable_checkpointing=False,
        deterministic=args.deterministic,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    metrics = evaluate(model, test_dataloader)
    metrics.update(
        {
            "window": window,
            "best_val_loss": float(early_stop_callback.best_score.detach().cpu()),
            "epochs_ran": int(trainer.current_epoch),
            "train_samples": len(training),
            "val_samples": len(validation),
            "test_samples": len(test),
            "log_dir": str(Path(logger.log_dir).resolve()),
        }
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare TFT performance for different sliding window lengths."
    )
    parser.add_argument("--data", type=Path, default=Path("combined_scaled_battery_data.csv"))
    parser.add_argument("--windows", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--attention-head-size", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-continuous-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("lightning_logs/sliding_window_compare"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tft_sliding_window_comparison.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_df, val_df, test_df = load_data(args.data, args.seed)

    results = []
    for window in args.windows:
        print(f"\n=== Running TFT with max_encoder_length={window} ===", flush=True)
        metrics = run_window(args, train_df, val_df, test_df, window)
        results.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)

    results_df = pd.DataFrame(results).sort_values("test_rmse")
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved comparison to {args.output.resolve()}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
