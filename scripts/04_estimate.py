"""Fit all models, robustness check, horizons -> outputs/"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.build import load_processed
from src.config import h, HORIZONS, MODEL_DIR, OUTPUT_DIR
from src.estimate import (mean_forecast, ar2_forecast, fit_ridge, fit_lasso,
                          fit_tree, robustness_no_ip, run_all_horizons)
from src.tables import comparison_table, robustness_table, horizon_tables, save_table

d = load_processed()
X, clean, data = d["X"], d["clean"], d["data"]
X_train, X_test, y_train, y_test = d["X_train"], d["X_test"], d["y_train"], d["y_test"]

# --- Models ---
mean_pred = mean_forecast(y_train, y_test)
ar2_pred = ar2_forecast(clean, y_train, y_test)
grid_ridge, ridge_pred = fit_ridge(X_train, y_train, X_test, y_test, h)
grid_lasso, lasso_pred = fit_lasso(X_train, y_train, X_test, y_test, h)
tree_model, tree_pred = fit_tree(X_train, y_train, X_test, y_test)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(grid_ridge, MODEL_DIR / "grid_ridge.joblib")
joblib.dump(grid_lasso, MODEL_DIR / "grid_lasso.joblib")
joblib.dump(tree_model, MODEL_DIR / "tree_model.joblib")

# --- Predictions and comparison table ---
preds_all = {
    "Mean": mean_pred, "AR(2) Baseline": ar2_pred,
    "Ridge": ridge_pred, "Lasso": lasso_pred, "Decision Tree": tree_pred,
}
pd.DataFrame(preds_all, index=y_test.index).assign(Actual=y_test).to_csv(OUTPUT_DIR / "predictions.csv")

df_comp = comparison_table(y_test, preds_all, h)
print(df_comp.to_string(index=False))
save_table(df_comp, "model_comparison.csv", index=False)

# --- Robustness: drop IP block ---
ridge_no_ip_pred, lasso_no_ip_pred = robustness_no_ip(X, X_train, X_test, y_train, h)
df_robustness = robustness_table(y_test, ridge_pred, lasso_pred, ridge_no_ip_pred, lasso_no_ip_pred)
print("\n=== Robustness Check Results ===")
print(df_robustness.to_string(index=False))
save_table(df_robustness, "robustness_no_ip.csv", index=False)

# --- Results across forecast horizons ---
out = run_all_horizons(data, clean, HORIZONS)
rmse_full, rel_full, rel_nocovid = horizon_tables(out, HORIZONS)

print("Test RMSE (%), full sample");     print(rmse_full.round(3))
print("Relative RMSE vs AR(2), full");   print(rel_full.round(3))
print("Relative RMSE vs AR(2), excl. COVID window"); print(rel_nocovid.round(3))

save_table(rmse_full.round(3), "horizons_rmse_full.csv")
save_table(rel_full.round(3), "horizons_rel_full.csv")
save_table(rel_nocovid.round(3), "horizons_rel_nocovid.csv")
