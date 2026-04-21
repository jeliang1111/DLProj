import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HNEI_DATA_PATH = "Battery_RUL.csv"
NASA_DATA_PATH = "nasa_battery_cycles.csv"
DEMO_CSV_PATH  = "demo_batteries_xgboost.csv"
NUM_TEST_BATTERIES = 8   # 4 HNEI + 4 NASA

SCALER_TYPE = "standard"   # "minmax" or "standard"
CKPT_PATH   = f"checkpoint_xgboost_cross_{SCALER_TYPE}.joblib"

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
) -> tuple[MinMaxScaler | StandardScaler, np.ndarray, np.ndarray]:
    vals  = pd.concat(train_segs)[FEATURE_COLS].values
    lower = np.percentile(vals, 1, axis=0)
    upper = np.percentile(vals, 99, axis=0)
    scaler = MinMaxScaler() if SCALER_TYPE == "minmax" else StandardScaler()
    scaler.fit(np.clip(vals, lower, upper))
    return scaler, lower, upper


def apply_scaler(
    segs: list[pd.DataFrame],
    scaler: MinMaxScaler,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[pd.DataFrame]:
    result = []
    for seg in segs:
        s = seg.copy()
        s[FEATURE_COLS] = scaler.transform(np.clip(s[FEATURE_COLS].values, lower, upper))
        result.append(s)
    return result


def segments_to_xy(segs: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    combined  = pd.concat(segs)
    features  = combined[FEATURE_COLS + ["Is_NASA"]].values.astype(np.float32)
    labels    = combined["RUL"].values.astype(np.float32)
    return features, labels


def print_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"{name:45s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- load & tag ---
    hnei_df = pd.read_csv(HNEI_DATA_PATH)
    nasa_df = pd.read_csv(NASA_DATA_PATH)
    hnei_df["Is_NASA"] = 0
    nasa_df["Is_NASA"] = 1

    hnei_segments = split_into_battery_segments(hnei_df)
    nasa_segments = split_into_battery_segments(nasa_df)
    print(f"HNEI: {len(hnei_segments)} batteries, NASA: {len(nasa_segments)} batteries")

    # --- reserve last battery of each domain for demo ---
    demo_hnei = hnei_segments[-1]
    demo_nasa = nasa_segments[-1]
    demo_df   = pd.concat([demo_hnei, demo_nasa], ignore_index=True)
    demo_df.to_csv(DEMO_CSV_PATH, index=False)
    print(f"Demo batteries saved to {DEMO_CSV_PATH} "
          f"(HNEI: {len(demo_hnei)} cycles, NASA: {len(demo_nasa)} cycles)")

    hnei_remaining = hnei_segments[:-1]
    nasa_remaining = nasa_segments[:-1]

    # --- stratified battery-level train/test split ---
    rng = np.random.default_rng(42)
    num_test_hnei = NUM_TEST_BATTERIES // 2
    num_test_nasa = NUM_TEST_BATTERIES - num_test_hnei

    hnei_test_idx = set(rng.choice(len(hnei_remaining), size=num_test_hnei, replace=False).tolist())
    nasa_test_idx = set(rng.choice(len(nasa_remaining), size=num_test_nasa, replace=False).tolist())

    hnei_train = [s for i, s in enumerate(hnei_remaining) if i not in hnei_test_idx]
    hnei_test  = [s for i, s in enumerate(hnei_remaining) if i in hnei_test_idx]
    nasa_train = [s for i, s in enumerate(nasa_remaining) if i not in nasa_test_idx]
    nasa_test  = [s for i, s in enumerate(nasa_remaining) if i in nasa_test_idx]

    print(f"HNEI — train: {len(hnei_train)}, test: {len(hnei_test)}")
    print(f"NASA  — train: {len(nasa_train)}, test: {len(nasa_test)}\n")

    # --- per-domain feature normalization (fit on training data only) ---
    hnei_scaler, hnei_lower, hnei_upper = fit_domain_scaler(hnei_train)
    nasa_scaler, nasa_lower, nasa_upper = fit_domain_scaler(nasa_train)

    hnei_train_scaled = apply_scaler(hnei_train, hnei_scaler, hnei_lower, hnei_upper)
    hnei_test_scaled  = apply_scaler(hnei_test,  hnei_scaler, hnei_lower, hnei_upper)
    nasa_train_scaled = apply_scaler(nasa_train, nasa_scaler, nasa_lower, nasa_upper)
    nasa_test_scaled  = apply_scaler(nasa_test,  nasa_scaler, nasa_lower, nasa_upper)

    # --- build arrays (Is_NASA included as a feature) ---
    X_train, y_train = segments_to_xy(hnei_train_scaled + nasa_train_scaled)
    X_test_hnei, y_test_hnei = segments_to_xy(hnei_test_scaled)
    X_test_nasa, y_test_nasa = segments_to_xy(nasa_test_scaled)
    X_test  = np.concatenate([X_test_hnei, X_test_nasa])
    y_test  = np.concatenate([y_test_hnei, y_test_nasa])

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")

    # --- train or load checkpoint ---
    if os.path.exists(CKPT_PATH):
        model = joblib.load(CKPT_PATH)
        print(f"Loaded checkpoint from {CKPT_PATH}, skipping training.")
    else:
        print("Training XGBoost...")
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )
        model.fit(X_train, y_train)
        joblib.dump(model, CKPT_PATH)
        print(f"Checkpoint saved to {CKPT_PATH}")

    # --- evaluation ---
    y_pred_hnei = model.predict(X_test_hnei)
    y_pred_nasa = model.predict(X_test_nasa)
    y_pred      = np.concatenate([y_pred_hnei, y_pred_nasa])

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print_metrics("XGBoost overall",       y_test,      y_pred)
    print_metrics("XGBoost HNEI test",     y_test_hnei, y_pred_hnei)
    print_metrics("XGBoost NASA test",     y_test_nasa, y_pred_nasa)

    # --- scatter plot color-coded by domain ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test_hnei, y_pred_hnei, alpha=0.3, s=10, color="steelblue",  label="HNEI")
    ax.scatter(y_test_nasa, y_pred_nasa, alpha=0.3, s=10, color="darkorange", label="NASA")
    lo, hi = y_test.min(), y_test.max()
    ax.plot([lo, hi], [lo, hi], "--", color="red")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("XGBoost: Predicted vs True RUL")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("xgboost_cross_pred_vs_true.png", dpi=150)
    plt.show()
    print("Plot saved to xgboost_cross_pred_vs_true.png")


if __name__ == "__main__":
    main()
