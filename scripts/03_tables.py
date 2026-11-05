"""Descriptive tables: split summary, target extremes, multicollinearity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.build import load_processed
from src.tables import split_summary, target_extremes, multicollinearity_count, save_table

d = load_processed()

split_stats = split_summary(d["y_train"], d["y_test"])
print(split_stats)
save_table(split_stats, "split_summary.csv", index=False)

smallest, largest = target_extremes(d["y"])
print("--- TOP 5 SMALLEST VALUES (Extremely Negative Growth) ---")
print(smallest)
print("\n--- TOP 5 LARGEST VALUES (Extremely Positive Growth) ---")
print(largest)

n_high, n_pairs = multicollinearity_count(d["clean"])
print(n_high, "of", n_pairs, "pairs have |corr| > 0.8")
