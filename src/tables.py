"""Summary tables (printed and saved to outputs/)."""
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from src.config import OUTPUT_DIR, h


def split_summary(y_train, y_test):
    split_stats = pd.DataFrame({
        "Period": ["Train (1960–2009)", "Test (2010–2026)"],
        "Observations": [len(y_train), len(y_test)],
        "Mean (%)": [y_train.mean(), y_test.mean()],
        "Std Dev (%)": [y_train.std(), y_test.std()]
    }).round(2)
    return split_stats


def target_extremes(y):
    smallest = y.nsmallest(5)
    largest = y.nlargest(5)
    return smallest, largest


def multicollinearity_count(clean):
    corr = clean.corr().abs()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(mask).stack()
    n_high = int((pairs > 0.8).sum())
    return n_high, len(pairs)


def comparison_table(y_test, preds_all, h=h):
    models = list(preds_all.keys())

    # COVID window by forecast origin: targets that touch Mar-Dec 2020
    covid_start = pd.Timestamp("2020-03-01") - pd.DateOffset(months=h)
    covid_mask = (y_test.index >= covid_start) & (y_test.index <= "2020-12-31")

    rmse_all = [root_mean_squared_error(y_test, preds_all[m]) for m in models]
    rmse_no_covid = [root_mean_squared_error(y_test[~covid_mask], preds_all[m][~covid_mask]) for m in models]

    ar_i = models.index("AR(2) Baseline")
    rel_rmse_all = [r / rmse_all[ar_i] for r in rmse_all]
    rel_rmse_no_covid = [r / rmse_no_covid[ar_i] for r in rmse_no_covid]

    df_comp = pd.DataFrame({
        "Model": models,
        "RMSE (full)": rmse_all, "Relative (full)": rel_rmse_all,
        "RMSE (excl. COVID)": rmse_no_covid, "Relative (excl. COVID)": rel_rmse_no_covid,
    }).round(3)
    return df_comp


def robustness_table(y_test, ridge_pred, lasso_pred, ridge_no_ip_pred, lasso_no_ip_pred):
    rmse_ridge = root_mean_squared_error(y_test, ridge_pred)
    rmse_lasso = root_mean_squared_error(y_test, lasso_pred)
    rmse_ridge_no_ip = root_mean_squared_error(y_test, ridge_no_ip_pred)
    rmse_lasso_no_ip = root_mean_squared_error(y_test, lasso_no_ip_pred)

    # Compare full-sample and no-2020 RMSE metrics
    covid_mask = (y_test.index >= "2020-03-01") & (y_test.index <= "2020-12-31")
    y_test_no_covid = y_test[~covid_mask]

    # Sub-sample metrics for original full models
    ridge_full_nocovid = root_mean_squared_error(y_test_no_covid, ridge_pred[~covid_mask])
    lasso_full_nocovid = root_mean_squared_error(y_test_no_covid, lasso_pred[~covid_mask])

    # Sub-sample metrics for no-IP models
    ridge_no_ip_nocovid = root_mean_squared_error(y_test_no_covid, ridge_no_ip_pred[~covid_mask])
    lasso_no_ip_nocovid = root_mean_squared_error(y_test_no_covid, lasso_no_ip_pred[~covid_mask])

    df_robustness = pd.DataFrame({
        "Model": ["Ridge (Full X)", "Ridge (No IP)", "Lasso (Full X)", "Lasso (No IP)"],
        "Full Test RMSE": [rmse_ridge, rmse_ridge_no_ip, rmse_lasso, rmse_lasso_no_ip],
        "Excl. 2020 RMSE": [ridge_full_nocovid, ridge_no_ip_nocovid, lasso_full_nocovid, lasso_no_ip_nocovid]
    })
    return df_robustness


def horizon_tables(out, horizons):
    rmse_full = pd.DataFrame({hh: out[hh]["full"] for hh in horizons})
    rmse_nocovid = pd.DataFrame({hh: out[hh]["nocovid"] for hh in horizons})

    rel_full = rmse_full.div(rmse_full.loc["AR(2)"])
    rel_nocovid = rmse_nocovid.div(rmse_nocovid.loc["AR(2)"])
    return rmse_full, rel_full, rel_nocovid


def save_table(df, name, **kwargs):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / name, **kwargs)
