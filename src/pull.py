"""Step 1: download the raw FRED-MD file and store it unchanged."""
import pandas as pd
from src.config import DATA_URL, RAW_FILE


def pull_raw(url=DATA_URL, path=RAW_FILE):
    raw = pd.read_csv(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(path, index=False)
    return raw
