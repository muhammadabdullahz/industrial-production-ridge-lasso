"""Step 4: benchmarks, Ridge, Lasso, regression tree, robustness, horizons."""
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.exceptions import ConvergenceWarning

from src.config import h

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------- Benchmark 1: Mean forecast ----------
def mean_forecast(y_train, y_test):
    mean_pred = np.full(len(y_test), y_train.mean())
    rmse_mean = root_mean_squared_error(y_test, mean_pred)
    print(f"Mean Forecast Test RMSE: {rmse_mean:.4f}")
    return mean_pred


# ---------- Benchmark 2: AR(2) ----------
def ar2_forecast(clean, y_train, y_test):
    # AR(2): current and previous month's INDPRO growth (no back-filling)
    ip = clean["INDPRO"]
    ar2_features = pd.DataFrame({
        "INDPRO_lag0": ip,
        "INDPRO_lag1": ip.shift(1)
    }).dropna()

    X_ar2_train = ar2_features.loc[ar2_features.index.intersection(y_train.index)]
    y_ar2_train = y_train.loc[X_ar2_train.index]
    X_ar2_test  = ar2_features.loc[y_test.index]

    ar2_model = LinearRegression().fit(X_ar2_train, y_ar2_train)
    ar2_pred  = ar2_model.predict(X_ar2_test)
    rmse_ar2  = root_mean_squared_error(y_test, ar2_pred)
    print(f"AR(2) Benchmark Test RMSE: {rmse_ar2:.4f}")
    return ar2_pred


# ---------- Ridge ----------
def fit_ridge(X_train, y_train, X_test, y_test, h=h):
    # 1. Define Pipeline (Scaler + Model)
    pipe_ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])

    # 2. Set up Time-Series Cross-Validation with gap protection
    tscv = TimeSeriesSplit(n_splits=5, gap=h)

    # 3. Define grid search over hyperparameter alpha
    param_grid_ridge = {
        "model__alpha": np.logspace(-2, 4, 30)
    }

    grid_ridge = GridSearchCV(
        estimator=pipe_ridge,
        param_grid=param_grid_ridge,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    # 4. Fit ONLY on training data to prevent leakage
    grid_ridge.fit(X_train, y_train)

    # 5. Predict on test set and evaluate RMSE
    ridge_pred = grid_ridge.predict(X_test)
    rmse_ridge = root_mean_squared_error(y_test, ridge_pred)

    print(f"Optimal Alpha: {grid_ridge.best_params_['model__alpha']:.4f}")
    print(f"Ridge Test RMSE: {rmse_ridge:.4f}")
    return grid_ridge, ridge_pred


# ---------- Lasso ----------
def fit_lasso(X_train, y_train, X_test, y_test, h=h):
    # 1. Pipeline Definition
    pipe_lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(max_iter=10000, random_state=42))
    ])

    # 2. Time-Series CV with gap protection
    tscv = TimeSeriesSplit(n_splits=5, gap=h)

    # 3. Parameter Grid for Lasso (smaller alphas suitable for L1)
    param_grid_lasso = {
        "model__alpha": np.logspace(-4, 1, 30)
    }

    grid_lasso = GridSearchCV(
        estimator=pipe_lasso,
        param_grid=param_grid_lasso,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    # 4. Fit strictly on train set
    grid_lasso.fit(X_train, y_train)

    # 5. Predict on test set
    lasso_pred = grid_lasso.predict(X_test)
    rmse_lasso = root_mean_squared_error(y_test, lasso_pred)

    print(f"Optimal Lasso Alpha: {grid_lasso.best_params_['model__alpha']:.6f}")
    print(f"Lasso Test RMSE:     {rmse_lasso:.4f}")
    return grid_lasso, lasso_pred


# ---------- Regression tree ----------
def fit_tree(X_train, y_train, X_test, y_test):
    # 1. Initialize Decision Tree with restricted depth to prevent overfitting
    tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)

    # 2. Fit model directly on unscaled training data (trees are scale-invariant)
    tree_model.fit(X_train, y_train)

    # 3. Predict on test set and evaluate RMSE
    tree_pred = tree_model.predict(X_test)
    rmse_tree = root_mean_squared_error(y_test, tree_pred)

    print(f"Decision Tree Test RMSE: {rmse_tree:.4f}")
    return tree_model, tree_pred


# ---------- Robustness: drop the IP block ----------
# Helper function for model fitting and evaluation using GridSearchCV
def fit_predict(est, grid, Xtr, ytr, Xte, h=h):
    pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
    gs = GridSearchCV(
        pipe,
        {"model__alpha": grid},
        cv=TimeSeriesSplit(n_splits=5, gap=h),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    gs.fit(Xtr, ytr)
    return gs.predict(Xte)


def robustness_no_ip(X, X_train, X_test, y_train, h=h):
    # 1. Identify and drop all Industrial Production (IP) series from feature matrix X
    ip_block = [c for c in X.columns if c.startswith("IP") or "INDPRO" in c]
    X_train_no_ip = X_train.drop(columns=ip_block)
    X_test_no_ip = X_test.drop(columns=ip_block)

    print(f"Dropped {len(ip_block)} IP-related features: {ip_block}")

    # 2. Rerun Ridge and Lasso models without the IP block
    ridge_no_ip_pred = fit_predict(
        Ridge(),
        np.logspace(-2, 4, 30),
        X_train_no_ip,
        y_train,
        X_test_no_ip,
        h=h
    )

    lasso_no_ip_pred = fit_predict(
        Lasso(max_iter=10000, random_state=42),
        np.logspace(-4, 1, 30),
        X_train_no_ip,
        y_train,
        X_test_no_ip,
        h=h
    )
    return ridge_no_ip_pred, lasso_no_ip_pred


# ---------- All horizons ----------
def run_horizon(data, clean, h, cutoff="2009-12-01", test_start="2010-01-01"):
    """Fit all models for an h-month horizon; return test RMSEs (full and excl. COVID)."""

    # 1. Target and predictors for this horizon
    log_ip = np.log(data["INDPRO"])
    y_all = (100 * (log_ip.shift(-h) - log_ip)).loc[clean.index].dropna()
    X_all = clean.loc[y_all.index]

    # 2. Chronological split; drop the last (h-1) training rows to avoid overlap
    y_train, X_train = y_all.loc[:cutoff], X_all.loc[:cutoff]
    n_keep = len(y_train) - (h - 1)            # careful: iloc[:-(h-1)] breaks when h = 1
    y_train, X_train = y_train.iloc[:n_keep], X_train.iloc[:n_keep]
    y_test, X_test = y_all.loc[test_start:], X_all.loc[test_start:]

    preds = {}

    # 3. Mean forecast
    preds["Mean"] = np.full(len(y_test), y_train.mean())

    # 4. AR(2): current and previous month's INDPRO growth
    ip = clean["INDPRO"]
    ar_feat = pd.DataFrame({"lag0": ip, "lag1": ip.shift(1)}).dropna()
    Xa_train = ar_feat.loc[ar_feat.index.intersection(y_train.index)]
    ar = LinearRegression().fit(Xa_train, y_train.loc[Xa_train.index])
    preds["AR(2)"] = ar.predict(ar_feat.loc[y_test.index])

    # 5. Ridge and Lasso, tuned with time-series CV on the training set only
    tscv = TimeSeriesSplit(n_splits=5, gap=h)
    best_alpha = {}
    for name, est, grid in [
        ("Ridge", Ridge(), np.logspace(-2, 4, 30)),
        ("Lasso", Lasso(max_iter=10000, random_state=42), np.logspace(-4, 1, 30)),
    ]:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
        gs = GridSearchCV(pipe, {"model__alpha": grid}, cv=tscv,
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X_train, y_train)
        preds[name] = gs.predict(X_test)
        best_alpha[name] = gs.best_params_["model__alpha"]
        if best_alpha[name] in (grid.min(), grid.max()):
            print(f"  Warning (h={h}): {name} alpha is at the edge of its grid")

    # 6. Regression tree
    tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_train, y_train)
    preds["Tree"] = tree.predict(X_test)

    # 7. RMSEs. COVID window: any origin whose h-month target touches Mar-Dec 2020
    covid_start = pd.Timestamp("2020-03-01") - pd.DateOffset(months=h)
    covid_mask = (y_test.index >= covid_start) & (y_test.index <= "2020-12-31")

    full = {m: root_mean_squared_error(y_test, p) for m, p in preds.items()}
    nocovid = {m: root_mean_squared_error(y_test[~covid_mask], p[~covid_mask])
               for m, p in preds.items()}
    return {"full": full, "nocovid": nocovid, "alpha": best_alpha}


def run_all_horizons(data, clean, horizons):
    out = {}
    for hh in horizons:
        print(f"Running h = {hh} ...")
        out[hh] = run_horizon(data, clean, hh)
    return out
