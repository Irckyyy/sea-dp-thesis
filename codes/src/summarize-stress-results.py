"""
codes/src/summarize-stress-results.py
=====================================
Summarize SEA-DP stress-test results.

Input:
    data/processed/stress_test_results.csv

Outputs:
    data/processed/stress_test_summary_best.csv
    data/processed/stress_test_summary_failures.csv
    data/processed/stress_test_summary_paper.csv
"""

import os
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SRC_PATH = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(SRC_PATH, "..", ".."))

input_path = os.path.join(
    ROOT,
    "data",
    "processed",
    "stress_test_results.csv",
)

out_dir = os.path.join(
    ROOT,
    "data",
    "processed",
)

os.makedirs(out_dir, exist_ok=True)


# ---------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Could not find input CSV: {input_path}")

df = pd.read_csv(input_path)

print("\nLoaded stress-test results:")
print(input_path)
print(f"Rows: {len(df)}")


# ---------------------------------------------------------------------
# Keep only successful rows
# ---------------------------------------------------------------------

ok = df[df["status"] == "ok"].copy()

if ok.empty:
    raise ValueError("No successful stress-test rows found.")

numeric_cols = [
    "epsilon",
    "original_vertices",
    "std_vertices",
    "std_tec",
    "std_sec",
    "std_hd",
    "std_vrr",
    "std_time",
    "sea_vertices",
    "sea_tec",
    "sea_sec",
    "sea_hd",
    "sea_vrr",
    "sea_time",
    "sea_shared_edges",
    "sea_shared_arcs",
    "sec_improvement",
    "tec_improvement",
]

for col in numeric_cols:
    if col in ok.columns:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")


# ---------------------------------------------------------------------
# 1. Best cases: highest SEC improvement
# ---------------------------------------------------------------------

best = ok.sort_values(
    by=["sec_improvement", "sea_sec"],
    ascending=[False, False],
).copy()

best_out = os.path.join(out_dir, "stress_test_summary_best.csv")
best.to_csv(best_out, index=False)


# ---------------------------------------------------------------------
# 2. Failure / limitation cases
# ---------------------------------------------------------------------

failures = ok[
    (ok["sec_improvement"] < 0)
    | (ok["sea_tec"] > ok["std_tec"])
    | (ok["sea_sec"] < 0.90)
].copy()

failures = failures.sort_values(
    by=["sec_improvement", "sea_tec"],
    ascending=[True, False],
)

failures_out = os.path.join(out_dir, "stress_test_summary_failures.csv")
failures.to_csv(failures_out, index=False)


# ---------------------------------------------------------------------
# 3. Paper-ready compact table
#    Select one best row per dataset + pair.
# ---------------------------------------------------------------------

paper_rows = []

for (dataset, left, right), group in ok.groupby(["dataset", "left", "right"]):
    group = group.copy()

    # Prefer high SEA-DP SEC and positive improvement.
    # This picks the most defensible tolerance for each pair.
    group = group.sort_values(
        by=["sea_sec", "sec_improvement", "sea_vrr"],
        ascending=[False, False, False],
    )

    paper_rows.append(group.iloc[0])

paper = pd.DataFrame(paper_rows)

paper = paper.sort_values(
    by=["dataset", "sec_improvement"],
    ascending=[True, False],
)

paper_cols = [
    "dataset",
    "left",
    "right",
    "epsilon",
    "original_vertices",
    "std_vertices",
    "sea_vertices",
    "std_tec",
    "sea_tec",
    "std_sec",
    "sea_sec",
    "sec_improvement",
    "std_hd",
    "sea_hd",
    "std_vrr",
    "sea_vrr",
    "std_time",
    "sea_time",
    "sea_shared_edges",
    "sea_shared_arcs",
]

paper = paper[[col for col in paper_cols if col in paper.columns]]

paper_out = os.path.join(out_dir, "stress_test_summary_paper.csv")
paper.to_csv(paper_out, index=False)


# ---------------------------------------------------------------------
# Print quick summary
# ---------------------------------------------------------------------

print("\nSaved summary files:")
print(f"  Best cases       -> {best_out}")
print(f"  Failure cases    -> {failures_out}")
print(f"  Paper table      -> {paper_out}")

print("\nTop 10 best SEC improvements:")
print(
    best[
        [
            "dataset",
            "left",
            "right",
            "epsilon",
            "std_sec",
            "sea_sec",
            "sec_improvement",
            "std_tec",
            "sea_tec",
            "std_vrr",
            "sea_vrr",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nPotential stress / limitation cases:")
if failures.empty:
    print("  None detected based on current rules.")
else:
    print(
        failures[
            [
                "dataset",
                "left",
                "right",
                "epsilon",
                "std_sec",
                "sea_sec",
                "sec_improvement",
                "std_tec",
                "sea_tec",
                "std_vrr",
                "sea_vrr",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )