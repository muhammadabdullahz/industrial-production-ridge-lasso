"""Transform, clean, build target -> data/processed/"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.build import (load_raw, parse_raw, transform_all, clean_data,
                       make_target, save_processed)
from src.config import h

raw = load_raw()
data, tcodes = parse_raw(raw)

print(data.shape)
print(data.index.min(), data.index.max())
print(data['INDPRO'].head())
print(data.isna().sum().sort_values(ascending=False).head())
print(data.dtypes.value_counts())

transformed = transform_all(data, tcodes)
print(transformed.shape)
print(set(tcodes) == set(data.columns))
print(transformed.isna().sum().sort_values(ascending=False).head(10))

transformed, clean = clean_data(transformed)
print(clean.shape, clean.index.min(), clean.index.max())

dropped = [c for c in data.columns if c not in clean.columns]
print(dropped)

first = transformed[clean.columns].apply(lambda s: s.first_valid_index())
last  = transformed[clean.columns].apply(lambda s: s.last_valid_index())
print(first.sort_values().tail(5))   # which series start latest
print(last.sort_values().head(5))    # which series end earliest

X, y = make_target(data, clean, h)

print("--- VERIFICATION CHECKS ---")
print(f"1. Raw INDPRO check: {data['INDPRO'].head(1).values[0]:.2f}")
print(f"2. Clean shape:                    {clean.shape}")
print(f"3. X shape:                        {X.shape}")
print(f"4. y shape:                        {y.shape}")
print(f"5. Do X and y shapes match?        {X.shape[0] == y.shape[0]}")
print(f"6. End date of y (should be -{h}m):  {y.index.max().strftime('%Y-%m')}")
print(f"7. Target units:                   Percent growth over next {h} months")
print("\nTail of y (Future IP Growth):")
print(y.tail())

save_processed(data, clean, X, y)
