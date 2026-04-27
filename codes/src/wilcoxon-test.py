"""
codes/src/wilcoxon-test.py
==========================
Wilcoxon signed-rank test for Standard DP vs SEA-DP.

Input:
    data/processed/stress_test_results.csv

Outputs:
    data/processed/wilcoxon_results.csv
    data/processed/wilcoxon_results.md

Purpose:
    Tests whether SEA-DP significantly differs from Standard DP
    using paired metric values from the same dataset pair and epsilon.
"""

import os
import math
import pandas as pd
from scipy.stats import wilcoxon


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

csv_out = os.path.join(out_dir, "wilcoxon_results.csv")
md_out = os.path.join(out_dir, "wilcoxon_results.md")


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

if not os.path.exists(input_path):
    raise FileNotFoundError(
        f"Could not find input file:\n{input_path}\n\n"
        "Run stress-test.py first."
    )

df = pd.read_csv(input_path)

if "status" in df.columns:
    df = df[df["status"] == "ok"].copy()

if df.empty:
    raise ValueError("No successful rows found for Wilcoxon testing.")


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def holm_adjust(p_values):
    """
    Holm-Bonferroni correction for multiple tests.
    Returns adjusted p-values in the original order.
    """
    indexed = [
        (i, p)
        for i, p in enumerate(p_values)
        if p is not None and not math.isnan(p)
    ]

    if not indexed:
        return [None for _ in p_values]

    indexed_sorted = sorted(indexed, key=lambda x: x[1])
    m = len(indexed_sorted)

    adjusted = [None for _ in p_values]
    running_max = 0.0

    for rank, (original_index, p) in enumerate(indexed_sorted, start=1):
        adj = min((m - rank + 1) * p, 1.0)
        running_max = max(running_max, adj)
        adjusted[original_index] = running_max

    return adjusted


def run_wilcoxon_test(metric_name, std_col, sea_col, alternative, interpretation):
    """
    Runs Wilcoxon signed-rank test on SEA-DP - Standard DP.

    alternative:
        "greater"  -> tests whether SEA-DP > Standard DP
        "less"     -> tests whether SEA-DP < Standard DP
        "two-sided"-> tests whether SEA-DP != Standard DP
    """
    if std_col not in df.columns or sea_col not in df.columns:
        return {
            "metric": metric_name,
            "std_col": std_col,
            "sea_col": sea_col,
            "n": 0,
            "alternative": alternative,
            "statistic": None,
            "p_value": None,
            "p_holm": None,
            "std_median": None,
            "sea_median": None,
            "median_difference_sea_minus_std": None,
            "positive_differences": None,
            "negative_differences": None,
            "ties": None,
            "result": "missing columns",
            "interpretation": interpretation,
        }

    paired = df[[std_col, sea_col]].copy()
    paired[std_col] = pd.to_numeric(paired[std_col], errors="coerce")
    paired[sea_col] = pd.to_numeric(paired[sea_col], errors="coerce")
    paired = paired.dropna()

    if paired.empty:
        return {
            "metric": metric_name,
            "std_col": std_col,
            "sea_col": sea_col,
            "n": 0,
            "alternative": alternative,
            "statistic": None,
            "p_value": None,
            "p_holm": None,
            "std_median": None,
            "sea_median": None,
            "median_difference_sea_minus_std": None,
            "positive_differences": None,
            "negative_differences": None,
            "ties": None,
            "result": "no valid paired values",
            "interpretation": interpretation,
        }

    diff = paired[sea_col] - paired[std_col]

    n = len(diff)
    positive = int((diff > 0).sum())
    negative = int((diff < 0).sum())
    ties = int((diff == 0).sum())

    std_median = paired[std_col].median()
    sea_median = paired[sea_col].median()
    median_diff = diff.median()

    # Wilcoxon cannot run if all differences are zero.
    if positive == 0 and negative == 0:
        return {
            "metric": metric_name,
            "std_col": std_col,
            "sea_col": sea_col,
            "n": n,
            "alternative": alternative,
            "statistic": None,
            "p_value": None,
            "p_holm": None,
            "std_median": std_median,
            "sea_median": sea_median,
            "median_difference_sea_minus_std": median_diff,
            "positive_differences": positive,
            "negative_differences": negative,
            "ties": ties,
            "result": "not tested; all paired differences are zero",
            "interpretation": interpretation,
        }

    try:
        test = wilcoxon(
            paired[sea_col],
            paired[std_col],
            alternative=alternative,
            zero_method="wilcox",
            correction=False,
            mode="auto",
        )

        p_value = float(test.pvalue)
        statistic = float(test.statistic)

        if p_value < 0.05:
            result = "significant at alpha = 0.05"
        else:
            result = "not significant at alpha = 0.05"

        return {
            "metric": metric_name,
            "std_col": std_col,
            "sea_col": sea_col,
            "n": n,
            "alternative": alternative,
            "statistic": statistic,
            "p_value": p_value,
            "p_holm": None,
            "std_median": std_median,
            "sea_median": sea_median,
            "median_difference_sea_minus_std": median_diff,
            "positive_differences": positive,
            "negative_differences": negative,
            "ties": ties,
            "result": result,
            "interpretation": interpretation,
        }

    except Exception as e:
        return {
            "metric": metric_name,
            "std_col": std_col,
            "sea_col": sea_col,
            "n": n,
            "alternative": alternative,
            "statistic": None,
            "p_value": None,
            "p_holm": None,
            "std_median": std_median,
            "sea_median": sea_median,
            "median_difference_sea_minus_std": median_diff,
            "positive_differences": positive,
            "negative_differences": negative,
            "ties": ties,
            "result": f"error: {e}",
            "interpretation": interpretation,
        }


# ---------------------------------------------------------------------
# Tests to run
# ---------------------------------------------------------------------

tests = [
    {
        "metric_name": "Shared-Edge Consistency (SEC)",
        "std_col": "std_sec",
        "sea_col": "sea_sec",
        "alternative": "greater",
        "interpretation": "Tests whether SEA-DP has higher shared-edge consistency than Standard DP.",
    },
    {
        "metric_name": "Topological Error Count (TEC)",
        "std_col": "std_tec",
        "sea_col": "sea_tec",
        "alternative": "less",
        "interpretation": "Tests whether SEA-DP has lower topological error count than Standard DP.",
    },
    {
        "metric_name": "Hausdorff Distance (HD)",
        "std_col": "std_hd",
        "sea_col": "sea_hd",
        "alternative": "two-sided",
        "interpretation": "Tests whether SEA-DP and Standard DP differ in geometric deviation.",
    },
    {
        "metric_name": "Vertex Reduction Ratio (VRR)",
        "std_col": "std_vrr",
        "sea_col": "sea_vrr",
        "alternative": "two-sided",
        "interpretation": "Tests whether SEA-DP and Standard DP differ in vertex reduction.",
    },
    {
        "metric_name": "Execution Time",
        "std_col": "std_time",
        "sea_col": "sea_time",
        "alternative": "greater",
        "interpretation": "Tests whether SEA-DP requires more runtime than Standard DP.",
    },
]


# ---------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------

results = []

for t in tests:
    results.append(
        run_wilcoxon_test(
            metric_name=t["metric_name"],
            std_col=t["std_col"],
            sea_col=t["sea_col"],
            alternative=t["alternative"],
            interpretation=t["interpretation"],
        )
    )

# Holm correction for multiple comparisons.
p_values = [r["p_value"] for r in results]
p_adjusted = holm_adjust(p_values)

for r, p_holm in zip(results, p_adjusted):
    r["p_holm"] = p_holm

    if p_holm is None:
        r["holm_result"] = "not applicable"
    elif p_holm < 0.05:
        r["holm_result"] = "significant after Holm correction"
    else:
        r["holm_result"] = "not significant after Holm correction"


result_df = pd.DataFrame(results)


# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------

result_df.to_csv(csv_out, index=False)

with open(md_out, "w", encoding="utf-8") as f:
    f.write("# Wilcoxon Signed-Rank Test Results\n\n")
    f.write(
        "The Wilcoxon signed-rank test was used because Standard DP and "
        "SEA-DP were evaluated on the same polygon pairs under the same "
        "simplification tolerances, making the observations paired.\n\n"
    )

    f.write(result_df.to_markdown(index=False))

    f.write("\n\n## Notes\n\n")
    f.write("- For SEC, the alternative hypothesis is SEA-DP > Standard DP.\n")
    f.write("- For TEC, the alternative hypothesis is SEA-DP < Standard DP.\n")
    f.write("- For HD and VRR, a two-sided test is used.\n")
    f.write("- For runtime, the alternative hypothesis is SEA-DP > Standard DP.\n")
    f.write("- Holm correction is included to adjust for multiple comparisons.\n")

print("\nWilcoxon test complete.")
print(f"Saved CSV: {csv_out}")
print(f"Saved Markdown: {md_out}")

print("\nResults:")
print(result_df.to_string(index=False))