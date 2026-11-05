"""Download raw FRED-MD data -> data/raw/"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pull import pull_raw

raw = pull_raw()
print(raw.shape)
print(raw.head())
