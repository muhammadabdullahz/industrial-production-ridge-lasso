"""Step 2: transform, clean, build the target and split train/test."""
import numpy as np
import pandas as pd
from src.config import RAW_FILE, PROCESSED_DIR, h, CUTOFF


def load_raw(path=RAW_FILE):
    return pd.read_csv(path)


def parse_raw(raw):
    row = raw.iloc[0]
    series = row.drop('sasdate')
    tcodes = series.astype(int).to_dict()

    data = raw.iloc[1:].copy()
    data['sasdate'] = pd.to_datetime(data['sasdate'])
    data = data.set_index('sasdate')
    return data, tcodes


# Transformation functions
def transform(x, tcode):
    if tcode == 1:
      return x
    elif tcode == 2:
      return x.diff()
    elif tcode == 3:
      return x.diff().diff()
    elif tcode == 4:
      return np.log(x)
    elif tcode == 5:
      return np.log(x).diff()
    elif tcode == 6:
      return np.log(x).diff().diff()
    elif tcode == 7:
      return ((x / x.shift(1)) - 1).diff()
    else:
      raise ValueError(f"Unknown tcode: {tcode}")


def transform_all(data, tcodes):
    return pd.DataFrame({
        col: transform(data[col], tcodes[col])
        for col in data.columns
    })


def clean_data(transformed):
    """Drop series with >10% missing, then drop rows with any NaN."""
    share_missing = transformed.isna().mean()
    keep = share_missing[share_missing <= 0.10].index
    transformed = transformed[keep]
    clean = transformed.dropna()
    return transformed, clean


def make_target(data, clean, h=h):
    log_ip = np.log(data["INDPRO"])
    y_raw = 100 * (log_ip.shift(-h) - log_ip)

    # Restrict to clean's date index and drop trailing NaNs caused by shift(-h)
    y = y_raw.loc[clean.index].dropna()

    # Align Feature Matrix (X) to match y's exact dates
    X = clean.loc[y.index].copy()
    return X, y


def split_data(X, y, h=h, cutoff=CUTOFF, test_start="2010-01-01"):
    X_train_raw = X.loc[:cutoff]
    y_train_raw = y.loc[:cutoff]

    X_test = X.loc[test_start:]
    y_test = y.loc[test_start:]

    # Drop the last (h-1) training rows: their targets peek into the test period
    n_keep = len(y_train_raw) - (h - 1)   # iloc[:-(h-1)] breaks when h = 1
    X_train = X_train_raw.iloc[:n_keep]
    y_train = y_train_raw.iloc[:n_keep]
    return X_train, X_test, y_train, y_test


def save_processed(data, clean, X, y, folder=PROCESSED_DIR):
    folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(folder / "data.csv")
    clean.to_csv(folder / "clean.csv")
    X.to_csv(folder / "X.csv")
    y.to_csv(folder / "y.csv")


def load_processed(folder=PROCESSED_DIR, h=h):
    """Reload processed files and rebuild the train/test split."""
    read = lambda f: pd.read_csv(folder / f, index_col=0, parse_dates=True)
    data, clean, X = read("data.csv"), read("clean.csv"), read("X.csv")
    y = read("y.csv").iloc[:, 0]
    X_train, X_test, y_train, y_test = split_data(X, y, h=h)
    return dict(data=data, clean=clean, X=X, y=y, X_train=X_train,
                X_test=X_test, y_train=y_train, y_test=y_test)
