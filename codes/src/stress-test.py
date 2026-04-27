"""
codes/src/stress-test.py
========================
Batch stress test for SEA-DP.

Runs multiple polygon pairs across multiple tolerances and saves results to CSV.
"""

import os
import sys
import time
import pandas as pd
import geopandas as gpd

SRC_PATH = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(SRC_PATH, "..", ".."))

MAIN_ALGO_PATH = os.path.join(SRC_PATH, "..", "main algo")
sys.path.insert(0, os.path.abspath(MAIN_ALGO_PATH))
sys.path.insert(0, os.path.abspath(SRC_PATH))

from sea_dp import sea_dp_simplify, standard_dp_simplify
from evaluation_metrics import (
    compute_new_tec,
    compute_pair_sec,
    compute_hausdorff,
    compute_vrr,
    count_vertices_gdf,
)


# ---------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------

USA_PATH = os.path.join(
    ROOT,
    "data",
    "raw",
    "natural_earth",
    "USA",
    "ne_10m_admin_1_states_provinces",
    "ne_10m_admin_1_states_provinces.shp",
)

PH_PATH = os.path.join(
    ROOT,
    "data",
    "raw",
    "GADM",
    "philippines",
    "gadm41_PHL_1.shp",
)

COUNTRIES_PATH = os.path.join(
    ROOT,
    "data",
    "raw",
    "natural_earth",
    "Armenia-Azerbaijan",
    "ne_10m_admin_0_countries",
    "ne_10m_admin_0_countries.shp",
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def detect_name_col(gdf, candidates):
    for col in candidates:
        if col in gdf.columns:
            return col
    raise ValueError(f"No name column found. Columns: {list(gdf.columns)}")


def project_to_metric(gdf):
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    gdf_4326 = gdf.to_crs(epsg=4326)

    try:
        crs = gdf_4326.estimate_utm_crs()
        if crs is None:
            crs = "EPSG:3857"
    except Exception:
        crs = "EPSG:3857"

    return gdf_4326.to_crs(crs)


def filter_usa(gdf):
    possible_cols = ["admin", "adm0_name", "geonunit", "sovereignt", "iso_a2", "adm0_a3"]

    for col in possible_cols:
        if col in gdf.columns:
            values = gdf[col].astype(str)
            mask = (
                values.str.contains("United States", case=False, na=False)
                | values.str.fullmatch("USA", case=False, na=False)
                | values.str.fullmatch("US", case=False, na=False)
            )
            if mask.sum() > 0:
                return gdf[mask].copy()

    if "iso_3166_2" in gdf.columns:
        values = gdf["iso_3166_2"].astype(str)
        mask = values.str.startswith("US-", na=False)
        if mask.sum() > 0:
            return gdf[mask].copy()

    return gdf


def run_one_pair(dataset_name, gdf, name_col, left_name, right_name, epsilon, zone_buffer_m):
    selected = gdf[gdf[name_col].isin([left_name, right_name])].copy()

    if len(selected) != 2:
        return {
            "dataset": dataset_name,
            "left": left_name,
            "right": right_name,
            "epsilon": epsilon,
            "status": "missing_pair",
        }

    selected = selected[[name_col, "geometry"]].copy()
    selected = selected.rename(columns={name_col: "feature_name"})
    selected = selected.reset_index(drop=True)
    selected = project_to_metric(selected)

    original_vertices = count_vertices_gdf(selected)

    # Standard DP
    std_start = time.perf_counter()
    std_gdf, _ = standard_dp_simplify(selected.copy(), epsilon=epsilon)
    std_time = time.perf_counter() - std_start

    std_tec = compute_new_tec(selected, std_gdf)["new_tec"]
    std_sec = compute_pair_sec(
        selected,
        std_gdf,
        name_col="feature_name",
        left_name=left_name,
        right_name=right_name,
        zone_buffer_m=zone_buffer_m,
        match_tol_m=5.0,
    )["sec"]
    std_hd = compute_hausdorff(selected, std_gdf)["mean"]
    std_vrr_info = compute_vrr(selected, std_gdf)

    # SEA-DP
    sea_start = time.perf_counter()
    sea_gdf, sea_stats = sea_dp_simplify(
        selected.copy(),
        epsilon=epsilon,
        tol=0.01,
        verbose=False,
    )
    sea_time = time.perf_counter() - sea_start

    sea_tec = compute_new_tec(selected, sea_gdf)["new_tec"]
    sea_sec = compute_pair_sec(
        selected,
        sea_gdf,
        name_col="feature_name",
        left_name=left_name,
        right_name=right_name,
        zone_buffer_m=zone_buffer_m,
        match_tol_m=5.0,
    )["sec"]
    sea_hd = compute_hausdorff(selected, sea_gdf)["mean"]
    sea_vrr_info = compute_vrr(selected, sea_gdf)

    return {
        "dataset": dataset_name,
        "left": left_name,
        "right": right_name,
        "epsilon": epsilon,
        "original_vertices": original_vertices,

        "std_vertices": std_vrr_info["v_simplified"],
        "std_tec": std_tec,
        "std_sec": std_sec,
        "std_hd": std_hd,
        "std_vrr": std_vrr_info["vrr"],
        "std_time": std_time,

        "sea_vertices": sea_vrr_info["v_simplified"],
        "sea_tec": sea_tec,
        "sea_sec": sea_sec,
        "sea_hd": sea_hd,
        "sea_vrr": sea_vrr_info["vrr"],
        "sea_time": sea_time,

        "sea_shared_edges": sea_stats.get("n_edges_shared", None),
        "sea_shared_arcs": sea_stats.get("n_arcs_assembled", None),

        "sec_improvement": sea_sec - std_sec,
        "tec_improvement": std_tec - sea_tec,
        "status": "ok",
    }


# ---------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------

tests = []

if os.path.exists(USA_PATH):
    usa = gpd.read_file(USA_PATH)
    usa = filter_usa(usa)
    usa_name_col = detect_name_col(usa, ["name", "NAME", "name_en", "postal"])
    tests.append({
        "dataset": "USA States",
        "gdf": usa,
        "name_col": usa_name_col,
        "pairs": [
            ("California", "Nevada"),
            ("Arizona", "New Mexico"),
            ("Oregon", "Washington"),
            ("Texas", "Oklahoma"),
        ],
        "epsilons": [5000, 10000, 20000, 30000],
        "zone_ratio": 0.25,
    })

if os.path.exists(PH_PATH):
    ph = gpd.read_file(PH_PATH)
    ph_name_col = detect_name_col(ph, ["NAME_1", "NAME", "province", "PROVINCE"])
    tests.append({
        "dataset": "Philippine Provinces",
        "gdf": ph,
        "name_col": ph_name_col,
        "pairs": [
            ("Laguna", "Quezon"),
            ("Rizal", "Laguna"),
            ("Batangas", "Laguna"),
            ("Cavite", "Laguna"),
        ],
        "epsilons": [500, 1000, 2000, 3000],
        "zone_ratio": 0.25,
    })

if os.path.exists(COUNTRIES_PATH):
    countries = gpd.read_file(COUNTRIES_PATH)
    country_name_col = detect_name_col(countries, ["NAME", "ADMIN", "NAME_EN", "SOVEREIGNT"])
    tests.append({
        "dataset": "Countries",
        "gdf": countries,
        "name_col": country_name_col,
        "pairs": [
            ("Armenia", "Azerbaijan"),
            ("Sweden", "Norway"),
            ("Spain", "Portugal"),
            ("France", "Germany"),
        ],
        "epsilons": [3000, 5000, 10000, 20000],
        "zone_ratio": 0.25,
    })


# ---------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------

rows = []

for test in tests:
    for left, right in test["pairs"]:
        for eps in test["epsilons"]:
            print(f"Running {test['dataset']}: {left} vs {right}, eps={eps}")

            zone_buffer = eps * test["zone_ratio"]

            try:
                row = run_one_pair(
                    dataset_name=test["dataset"],
                    gdf=test["gdf"],
                    name_col=test["name_col"],
                    left_name=left,
                    right_name=right,
                    epsilon=eps,
                    zone_buffer_m=zone_buffer,
                )
            except Exception as e:
                row = {
                    "dataset": test["dataset"],
                    "left": left,
                    "right": right,
                    "epsilon": eps,
                    "status": f"error: {e}",
                }

            rows.append(row)


# ---------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------

out_dir = os.path.join(ROOT, "data", "processed")
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "stress_test_results.csv")

df = pd.DataFrame(rows)
df.to_csv(out_path, index=False)

print("\nStress test complete.")
print(f"Saved results to: {out_path}")

print("\nSummary:")
print(df[[
    "dataset",
    "left",
    "right",
    "epsilon",
    "std_sec",
    "sea_sec",
    "sec_improvement",
    "std_tec",
    "sea_tec",
    "tec_improvement",
    "status",
]].to_string(index=False))