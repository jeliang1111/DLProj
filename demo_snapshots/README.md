# Demo Model Snapshots

Generated from `tft.ipynb` and `xgboost.ipynb`.

## TFT Notebook

- Model: `TemporalFusionTransformer.from_dataset(...)`
- Window: `max_encoder_length=10`, `max_prediction_length=1`
- Static real: `Is_NASA`
- Known real: `time_idx`
- Unknown reals: discharge time, voltage decrement, max discharge voltage, min charge voltage, time at 4.15V, time constant current, charging time
- Hyperparameters: learning rate `0.03`, hidden size `16`, attention heads `2`, dropout `0.1`, hidden continuous size `8`, RMSE loss
- Notebook metrics: MAE `22.6152`, RMSE `36.0381`

## XGBoost Notebook

- Model: `XGBRegressor`
- Features: discharge time, voltage decrement, max discharge voltage, min charge voltage, time at 4.15V, time constant current, charging time
- Hyperparameters: `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `objective="reg:squarederror"`
- Notebook metrics: MAE `7.4499`, RMSE `12.7233`, R2 `0.9908`

## Files

- `model_demo_snapshot_grid.png`: single slide-friendly overview of both models
- `tft_predicted_vs_true.png`: TFT predicted vs true RUL
- `tft_training_history.png`: TFT train/validation history
- `xgboost_predicted_vs_true.png`: XGBoost predicted vs true RUL with ideal line
- `xgboost_feature_importance.png`: XGBoost feature importance
- `*_notebook_output_*.png`: exact raw image outputs extracted from the notebooks
- `existing_tft_figure.png`, `existing_xgboost_figure.png`: previously saved top-level figures, copied here for convenience
