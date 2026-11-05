"""All charts. Each function saves to figures/ (if `name` is given) and shows the plot."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import root_mean_squared_error

from src.config import FIGURE_DIR, fred_group_map, color_map


def _finish(name):
    plt.tight_layout()
    if name:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURE_DIR / name, dpi=150, bbox_inches="tight")
    plt.show()


# Chart 1: Raw vs Transformed INDPRO
def plot_raw_vs_transformed(data, clean, name="01_raw_vs_transformed.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(data.loc[clean.index, "INDPRO"], color="#1f77b4", linewidth=1.5)
    axes[0].set_title("INDPRO: Raw Index (Level)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Index (2017=100)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(clean["INDPRO"], color="#2ca02c", linewidth=1.2)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[1].set_title("INDPRO: Log Difference (Monthly Growth)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Stationary Value")
    axes[1].grid(True, alpha=0.3)

    _finish(name)


# Chart 2: Histogram of the target
def plot_target_hist(y, name="02_target_histogram.png"):
    plt.figure(figsize=(10, 5))
    plt.hist(y, bins=50, edgecolor="white", color="#1f77b4", alpha=0.85)

    mean_val = y.mean()
    plt.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}%")

    plt.title("Distribution of Target Variable: 3-Month IP Growth (%)", fontsize=12, fontweight="bold")
    plt.xlabel("3-month IP growth (%)")
    plt.ylabel("Number of months")
    plt.legend()
    plt.grid(True, alpha=0.2)

    _finish(name)


# Chart 3: Target vs 2 predictors
def plot_target_vs_predictors(X, y, name="03_target_vs_predictors.png"):
    predictors = ["PERMIT", "T10YFFM"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, col in zip(axes, predictors):
        # Scatter plot of predictor vs. target y
        ax.scatter(X[col], y, alpha=0.3, s=10, color="#1f77b4")

        # Fit simple 1D linear polynomial (slope & intercept)
        slope, intercept = np.polyfit(X[col], y, deg=1)
        xs = np.linspace(X[col].min(), X[col].max(), 100)
        ax.plot(xs, intercept + slope * xs, color="red", linewidth=2, linestyle="--", label=f"Fit (slope={slope:.2f})")

        # Labels and formatting
        ax.set_title(f"Target (y) vs. {col}", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{col} (Transformed Indicator)")
        ax.set_ylabel("3-Month IP Growth (%)")
        ax.grid(True, alpha=0.2)
        ax.legend()

    _finish(name)


# Correlation heatmap of the IP block
def plot_ip_heatmap(clean, name="04_ip_correlation_heatmap.png"):
    ip_cols = [c for c in ["INDPRO", "IPFINAL", "IPCONGD", "IPMAT", "IPMANSICS", "IPFUELS", "CUMFNS"] if c in clean.columns]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        clean[ip_cols].corr(),
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        linewidths=0.5
    )

    plt.title("Correlation Heatmap: Industrial Production Block", fontsize=12)
    _finish(name)


# Train/test split chart
def plot_train_test_split(y, name="05_train_test_split.png"):
    plt.figure(figsize=(12, 4))

    plt.plot(y.index, y, color="#1f77b4", linewidth=1.2, label="3-Month IP Growth (%)")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

    test_start = pd.to_datetime("2010-01-01")
    test_end = y.index.max()

    plt.axvspan(test_start, test_end, color="gray", alpha=0.2, label="Test Period (2010–2026)")

    plt.title("Industrial Production 3-Month Growth: Train vs. Test Split", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("3-Month IP Growth (%)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.2)

    _finish(name)


# Chart 6: AR(2) benchmark
def plot_ar2(y_test, ar2_pred, name="06_ar2_benchmark.png"):
    rmse_ar2 = root_mean_squared_error(y_test, ar2_pred)
    plt.figure(figsize=(12, 5))

    # Plot actual target values
    plt.plot(y_test.index, y_test, color="black", label="Actual 3-Month IP Growth", linewidth=1.5)

    # Plot AR benchmark predictions
    plt.plot(y_test.index, ar2_pred, color="red", linestyle="--", label=f"AR Benchmark (RMSE: {rmse_ar2:.2f})", linewidth=1.2)

    plt.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    plt.title("Chart 6: AR Benchmark Predictions vs. Actual IP Growth (2010–2026)", fontsize=12)
    plt.xlabel("Date")
    plt.ylabel("3-Month IP Growth (%)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.2)

    _finish(name)


# Chart 7: Ridge CV error vs alpha
def plot_ridge_cv(grid_ridge, name="07_ridge_cv_alpha.png"):
    # Extract mean test scores and convert from negative RMSE to positive RMSE
    cv_rmse = -grid_ridge.cv_results_["mean_test_score"]
    alphas = grid_ridge.param_grid["model__alpha"]

    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(alphas, cv_rmse, marker="o", color="#1f77b4", linewidth=2, markersize=4)
    plt.xscale("log")
    plt.axvline(grid_ridge.best_params_["model__alpha"], color="red", linestyle="--", label=f"Optimal Alpha ({grid_ridge.best_params_['model__alpha']:.2f})")

    plt.xlabel("Alpha (L2 Penalty Parameter, Log Scale)")
    plt.ylabel("Mean CV RMSE (%)")
    plt.title("Cross-Validation Error vs. Alpha (Ridge)")
    plt.legend()
    _finish(name)


# Chart 8: Top 15 Ridge coefficients
def plot_ridge_coefs(grid_ridge, X_train, name="08_ridge_top15_coefs.png"):
    # Extract coefficients from the best estimator inside the pipeline
    ridge_coefs = grid_ridge.best_estimator_.named_steps["model"].coef_
    feature_names = X_train.columns

    # Build DataFrame and take top 15 by absolute magnitude
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coef": ridge_coefs,
        "AbsCoef": np.abs(ridge_coefs)
    }).sort_values("AbsCoef", ascending=False).head(15)

    # Sort ascending for clean horizontal bar chart plotting
    coef_df = coef_df.sort_values("AbsCoef", ascending=True)

    plt.figure(figsize=(8, 5), dpi=120)
    bars = plt.barh(coef_df["Feature"], coef_df["Coef"], color=np.where(coef_df["Coef"] > 0, "#2ca02c", "#d62728"))
    plt.axvline(0, color="black", linestyle="-", linewidth=0.8)

    plt.xlabel("Coefficient Value (Standardized Predictors)")
    plt.ylabel("Macroeconomic Feature")
    plt.title("Top 15 Ridge Coefficients by Absolute Magnitude")
    _finish(name)


# Chart 9 / 12: Predictions and residuals (Ridge, Lasso)
def plot_ridge_predictions(y_test, ridge_pred, name="09_ridge_predictions.png"):
    rmse_ridge = root_mean_squared_error(y_test, ridge_pred)

    # Calculate out-of-sample residuals
    residuals = y_test - ridge_pred

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, dpi=120)

    # Panel 1: Actual vs Predicted
    ax1.plot(y_test.index, y_test, label="Actual 3-Month IP Growth", color="black", linewidth=1.5)
    ax1.plot(y_test.index, ridge_pred, label=f"Ridge Forecast (RMSE: {rmse_ridge:.2f})", color="#1f77b4", linestyle="--", linewidth=1.5)
    ax1.set_ylabel("3-Month IP Growth (%)")
    ax1.set_title("Out-of-Sample Predictions and Residuals Over Time (Ridge)")
    ax1.legend(loc="upper left")

    # Panel 2: Residuals over time
    ax2.plot(y_test.index, residuals, color="#e377c2", linewidth=1.2, label="Residuals (y - y_hat)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Error (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")

    _finish(name)


# Chart 10: Lasso CV error vs alpha
def plot_lasso_cv(grid_lasso, name="10_lasso_cv_alpha.png"):
    cv_rmse_lasso = -grid_lasso.cv_results_["mean_test_score"]
    alphas_lasso = grid_lasso.param_grid["model__alpha"]

    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.plot(alphas_lasso, cv_rmse_lasso, marker="o", color="#1f77b4", linewidth=2, markersize=4)
    plt.xscale("log")
    plt.axvline(grid_lasso.best_params_["model__alpha"], color="red", linestyle="--",
                label=f"Optimal Alpha ({grid_lasso.best_params_['model__alpha']:.4f})")

    plt.xlabel("Alpha (L1 Penalty Parameter, Log Scale)")
    plt.ylabel("Mean CV RMSE (%)")
    plt.title("Cross-Validation Error vs. Alpha (Lasso)")
    plt.legend()
    _finish(name)


# Chart 11: Lasso selected features by group
def plot_lasso_features(grid_lasso, X_train, name="11_lasso_selected_features.png"):
    # 1. Extract non-zero coefficients from optimal model
    best_lasso_coefs = grid_lasso.best_estimator_.named_steps["model"].coef_
    selected_mask = best_lasso_coefs != 0

    selected_features = X_train.columns[selected_mask]
    selected_weights = best_lasso_coefs[selected_mask]

    # Construct DataFrame for non-zero variables
    df_selected = pd.DataFrame({
        "Feature": selected_features,
        "Weight": selected_weights,
        "Group": [fred_group_map.get(col, "Unmapped") for col in selected_features]
    })

    # Flag any unmapped series
    unmapped = df_selected[df_selected["Group"] == "Unmapped"]["Feature"].tolist()
    if unmapped:
        print(f"Warning: The following features were unmapped: {unmapped}")
    else:
        print("All selected features successfully mapped to FRED-MD groups!")

    # Plot selected features sorted by weight
    df_selected = df_selected.sort_values("Weight", ascending=True)

    plt.figure(figsize=(9, 6), dpi=120)
    bars = plt.barh(df_selected["Feature"], df_selected["Weight"],
                    color=[color_map.get(g, "black") for g in df_selected["Group"]])
    plt.axvline(0, color="black", linestyle="-", linewidth=0.8)

    # Add custom legend for groups
    handles = [plt.Rectangle((0,0),1,1, color=color_map[g]) for g in df_selected["Group"].unique()]
    plt.legend(handles, df_selected["Group"].unique(), loc="lower right")

    plt.xlabel("Lasso Coefficient Value")
    plt.ylabel("Selected Macroeconomic Feature")
    plt.title("Lasso Selected Features (Non-Zero) by FRED-MD Group")
    _finish(name)


def plot_lasso_predictions(y_test, lasso_pred, name="12_lasso_predictions.png"):
    rmse_lasso = root_mean_squared_error(y_test, lasso_pred)
    residuals_lasso = y_test - lasso_pred

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, dpi=120)

    # Panel 1: Actual vs Predicted
    ax1.plot(y_test.index, y_test, label="Actual 3-Month IP Growth", color="black", linewidth=1.5)
    ax1.plot(y_test.index, lasso_pred, label=f"Lasso Forecast (RMSE: {rmse_lasso:.2f})",
             color="#2ca02c", linestyle="--", linewidth=1.5)
    ax1.set_ylabel("3-Month IP Growth (%)")
    ax1.set_title("Out-of-Sample Predictions and Residuals Over Time (Lasso)")
    ax1.legend(loc="upper left")

    # Panel 2: Residuals over time
    ax2.plot(y_test.index, residuals_lasso, color="#d62728", linewidth=1.2, label="Residuals (y - y_hat)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Error (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")

    _finish(name)


# Decision tree diagram
def plot_tree_diagram(tree_model, X_train, name="13_decision_tree.png"):
    plt.figure(figsize=(16, 8), dpi=150)
    plot_tree(
        tree_model,
        feature_names=X_train.columns,
        filled=True,
        rounded=True,
        fontsize=10,
        precision=2
    )
    plt.title("Decision Tree Structural Diagram (max_depth=3)")
    _finish(name)


# Chart 13: Relative RMSE bar chart
def plot_relative_rmse(df_comp, name="14_relative_rmse.png"):
    models = df_comp["Model"].tolist()
    rel_rmse_all = df_comp["Relative (full)"].values
    rel_rmse_no_covid = df_comp["Relative (excl. COVID)"].values

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5), dpi=120)
    plt.bar(x - width/2, rel_rmse_all, width, label="Full Test Period", color="#1f77b4")
    plt.bar(x + width/2, rel_rmse_no_covid, width, label="Excluding 2020", color="#2ca02c")

    plt.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="AR(2) Baseline Benchmark (1.0)")
    plt.xticks(x, models, rotation=15)
    plt.ylabel("Relative RMSE (Model RMSE / AR(2) RMSE)")
    plt.title("Relative Model Performance vs. AR(2) Baseline")
    plt.ylim(0, 1.3)
    plt.legend(loc="lower right")
    _finish(name)


# Chart 14: Forecast errors across all models
def plot_forecast_errors(y_test, ar2_pred, ridge_pred, lasso_pred, tree_pred, name="15_forecast_errors.png"):
    res_ar2 = y_test - ar2_pred
    res_ridge = y_test - ridge_pred
    res_lasso = y_test - lasso_pred
    res_tree = y_test - tree_pred

    plt.figure(figsize=(11, 5.5), dpi=120)
    plt.plot(y_test.index, res_ar2, label="AR(2)", color="black", linestyle=":", linewidth=1.2)
    plt.plot(y_test.index, res_ridge, label="Ridge", color="#1f77b4", linewidth=1.2, alpha=0.8)
    plt.plot(y_test.index, res_lasso, label="Lasso", color="#2ca02c", linewidth=1.5)
    plt.plot(y_test.index, res_tree, label="Decision Tree", color="#ff7f0e", linestyle="--", linewidth=1.2)

    plt.axhline(0, color="black", linestyle="-", linewidth=0.8)
    plt.ylabel("Forecast Error (%)")
    plt.xlabel("Date")
    plt.title("Out-of-Sample Forecast Errors Over Time Across Models")
    plt.legend(loc="upper left")
    _finish(name)


# Results across horizons
def plot_horizons(rel_full, rel_nocovid, horizons, name="16_horizons.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, tbl, title in zip(axes, [rel_full, rel_nocovid],
                              ["Full test period", "Excluding COVID window"]):
        for model in ["Mean", "Ridge", "Lasso", "Tree"]:
            ax.plot(horizons, tbl.loc[model], marker="o", label=model)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="AR(2) = 1.0")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Forecast horizon (months)")
        ax.set_xticks(horizons)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("RMSE relative to AR(2)")
    axes[0].legend()
    _finish(name)
