"""Generate every chart -> figures/"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")   # save figures without opening windows

import joblib
import pandas as pd

from src.build import load_processed
from src.config import HORIZONS, MODEL_DIR, OUTPUT_DIR
from src import plots

d = load_processed()
data, clean, X, y = d["data"], d["clean"], d["X"], d["y"]
X_train, y_test = d["X_train"], d["y_test"]

grid_ridge = joblib.load(MODEL_DIR / "grid_ridge.joblib")
grid_lasso = joblib.load(MODEL_DIR / "grid_lasso.joblib")
tree_model = joblib.load(MODEL_DIR / "tree_model.joblib")

p = pd.read_csv(OUTPUT_DIR / "predictions.csv", index_col=0, parse_dates=True)
df_comp = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
rd = lambda f: pd.read_csv(OUTPUT_DIR / f, index_col=0)
rel_full, rel_nocovid = rd("horizons_rel_full.csv"), rd("horizons_rel_nocovid.csv")
rel_full.columns = rel_nocovid.columns = HORIZONS

plots.plot_raw_vs_transformed(data, clean)
plots.plot_target_hist(y)
plots.plot_target_vs_predictors(X, y)
plots.plot_ip_heatmap(clean)
plots.plot_train_test_split(y)
plots.plot_ar2(y_test, p["AR(2) Baseline"].values)
plots.plot_ridge_cv(grid_ridge)
plots.plot_ridge_coefs(grid_ridge, X_train)
plots.plot_ridge_predictions(y_test, p["Ridge"].values)
plots.plot_lasso_cv(grid_lasso)
plots.plot_lasso_features(grid_lasso, X_train)
plots.plot_lasso_predictions(y_test, p["Lasso"].values)
plots.plot_tree_diagram(tree_model, X_train)
plots.plot_relative_rmse(df_comp)
plots.plot_forecast_errors(y_test, p["AR(2) Baseline"].values, p["Ridge"].values,
                           p["Lasso"].values, p["Decision Tree"].values)
plots.plot_horizons(rel_full, rel_nocovid, HORIZONS)
print("Figures saved to figures/")
