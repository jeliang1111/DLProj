import numpy as np
import pandas as pd


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    return df


def split_data(df: pd.DataFrame, test_size=0.2, val_size=0.1):
    # Time-based split to avoid data leakage
    df = df.sort_values("Cycle_Index").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - test_size - val_size))
    val_end   = int(n * (1 - test_size))

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]
    return train_df, val_df, test_df


def get_features_and_labels(df: pd.DataFrame):
    feature_cols = [
        "Cycle_Index",
        "Discharge Time (s)",
        "Decrement 3.6-3.4V (s)",
        "Max. Voltage Dischar. (V)",
        "Min. Voltage Charg. (V)",
        "Time at 4.15V (s)",
        "Time constant current (s)",
        "Charging time (s)",
    ]
    X = df[feature_cols].values
    y = df["RUL"].values
    return X, y


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    print(f"\n{'='*40}")
    print(f"  Model : {model_name}")
    print(f"  MAE   : {mae:.4f} cycles")
    print(f"  RMSE  : {rmse:.4f} cycles")
    print(f"{'='*40}")

    return {"model": model_name, "MAE": mae, "RMSE": rmse}


def compare_models(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results).set_index("model").sort_values("RMSE")
    print("\n" + "="*40)
    print("  Model Comparison (sorted by RMSE)")
    print("="*40)
    print(df.to_string(float_format="{:.4f}".format))
    print("="*40)
    return df


if __name__ == "__main__":
    df = load_data("/mnt/user-data/uploads/Battery_RUL.csv")
    train_df, val_df, test_df = split_data(df)
    X_train, y_train = get_features_and_labels(train_df)
    X_val,   y_val   = get_features_and_labels(val_df)
    X_test,  y_test  = get_features_and_labels(test_df)

    # Replace these with real model predictions
    np.random.seed(42)
    results = [
        evaluate_model(y_test, y_test + np.random.normal(0, 30, size=y_test.shape), "LSTM"),
        evaluate_model(y_test, y_test + np.random.normal(0, 45, size=y_test.shape), "CNN"),
        evaluate_model(y_test, y_test + np.random.normal(0, 20, size=y_test.shape), "TFT"),
        evaluate_model(y_test, y_test + np.random.normal(0, 60, size=y_test.shape), "XGBoost"),
    ]

    compare_models(results)