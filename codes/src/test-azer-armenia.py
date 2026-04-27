"""
codes/src/test-armenia-azerbaijan-seadp.py
==========================================
Compare Standard DP vs SEA-DP on Armenia and Azerbaijan.

Metrics are computed from evaluation_metrics.py:
1. TEC  = gaps + overlaps + invalid
2. SEC  = shared-edge consistency
3. HD   = Hausdorff distance
4. VRR  = vertex reduction ratio
5. T    = execution time

This is a complex boundary / enclave test.
"""

import os
import sys
import time

import geopandas as gpd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 0. Import project modules
# ---------------------------------------------------------------------------

SRC_PATH = os.path.dirname(__file__)

MAIN_ALGO_PATH = os.path.join(SRC_PATH, "..", "main algo")
sys.path.insert(0, os.path.abspath(MAIN_ALGO_PATH))

# evaluation_metrics.py should be inside codes/src
sys.path.insert(0, os.path.abspath(SRC_PATH))

import sea_dp
from sea_dp import sea_dp_simplify, standard_dp_simplify

from evaluation_metrics import (
    compute_tec,
    compute_new_tec,
    compute_pair_sec,
    compute_hausdorff,
    compute_vrr,
    count_vertices_gdf,
    get_named_parts,
    get_named_holes,
    get_named_vertices,
)

print("Using SEA-DP from:", sea_dp.__file__)


# ---------------------------------------------------------------------------
# 1. Load Natural Earth Admin 0 countries
# ---------------------------------------------------------------------------

path_candidates = [
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "raw",
        "natural_earth",
        "Armenia-Azerbaijan",
        "ne_10m_admin_0_countries",
        "ne_10m_admin_0_countries.shp",
    ),
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "raw",
        "natural_earth",
        "Armenia-Azerbaijan",
        "ne_50m_admin_0_countries",
        "ne_50m_admin_0_countries.shp",
    ),
]

path = None
for candidate in path_candidates:
    if os.path.exists(candidate):
        path = candidate
        break

if path is None:
    raise FileNotFoundError(
        "Could not find Armenia-Azerbaijan Natural Earth shapefile."
    )

print("Using dataset:", path)

world = gpd.read_file(path)


# ---------------------------------------------------------------------------
# 2. Detect country-name column
# ---------------------------------------------------------------------------

possible_name_cols = ["NAME", "ADMIN", "SOVEREIGNT", "NAME_EN", "name"]

name_col = None
for col in possible_name_cols:
    if col in world.columns:
        name_col = col
        break

if name_col is None:
    raise ValueError(
        f"No usable country name column found. Columns: {list(world.columns)}"
    )

print("Using country name column:", name_col)


# ---------------------------------------------------------------------------
# 3. Filter Armenia and Azerbaijan
# ---------------------------------------------------------------------------

left_name = "Armenia"
right_name = "Azerbaijan"

countries = world[world[name_col].isin([left_name, right_name])].copy()
countries = countries.reset_index(drop=True)

if len(countries) != 2:
    print(world[[name_col]].drop_duplicates().sort_values(name_col).to_string())
    raise ValueError(f"Could not find exactly {left_name} and {right_name}.")

print("\nSelected countries:")
print(countries[[name_col]])


# ---------------------------------------------------------------------------
# 4. Project to metric CRS
# EPSG:32638 = UTM Zone 38N, reasonable for Armenia/Azerbaijan
# ---------------------------------------------------------------------------

countries = countries.to_crs(epsg=32638)


# ---------------------------------------------------------------------------
# 5. Parameters
# ---------------------------------------------------------------------------

tolerance = 5000        # meters
edge_match_tol = 0.01   # meters
match_tol_m = 5.0       # actual boundary matching tolerance
zone_buffer_m = 1250    # SEC window; 25% of 5000m tolerance


# ---------------------------------------------------------------------------
# 6. Display helpers only
# ---------------------------------------------------------------------------

def print_country_survival(label, gdf):
    print(f"\n=== {label} Country Survival ===")

    for country in [left_name, right_name]:
        print(f"{country}:")
        print(f"  Parts    : {get_named_parts(gdf, name_col, country)}")
        print(f"  Holes    : {get_named_holes(gdf, name_col, country)}")
        print(f"  Vertices : {get_named_vertices(gdf, name_col, country)}")


def make_metrics_text(tec, sec, hd, vrr, runtime):
    return (
        "Metrics\n"
        f"TEC : {tec}\n"
        f"SEC : {sec:.4f}\n"
        f"HD  : {hd:,.2f} m\n"
        f"VRR : {vrr:.2%}\n"
        f"T   : {runtime:.3f} s"
    )


# ---------------------------------------------------------------------------
# 7. Original metrics
# ---------------------------------------------------------------------------

orig_stats = compute_tec(countries)
orig_vertices = count_vertices_gdf(countries)

orig_sec = 1.0
orig_hd = 0.0
orig_vrr = 0.0
orig_time = 0.0

print("\n=== Original Topology ===")
print(f"  TEC       : {orig_stats['tec']}")
print(f"  Gaps      : {orig_stats['n_gaps']}")
print(f"  Overlaps  : {orig_stats['n_overlaps']}")
print(f"  Invalid   : {orig_stats['n_invalid']}")
print(f"  SEC       : {orig_sec:.4f}")
print(f"  HD        : {orig_hd:,.2f} m")
print(f"  VRR       : {orig_vrr:.2%}")
print(f"  Vertices  : {orig_vertices}")

print_country_survival("Original", countries)


# ---------------------------------------------------------------------------
# 8. Standard DP
# ---------------------------------------------------------------------------

std_start = time.perf_counter()

countries_std, std_algo_stats = standard_dp_simplify(
    countries.copy(),
    epsilon=tolerance,
)

std_time = time.perf_counter() - std_start

std_tec_info = compute_new_tec(countries, countries_std)
std_raw_stats = std_tec_info["simplified"]
std_new_tec = std_tec_info["new_tec"]

std_sec_info = compute_pair_sec(
    countries,
    countries_std,
    name_col=name_col,
    left_name=left_name,
    right_name=right_name,
    zone_buffer_m=zone_buffer_m,
    match_tol_m=match_tol_m,
)

std_sec = std_sec_info["sec"]

std_hd_info = compute_hausdorff(countries, countries_std)
std_hd = std_hd_info["mean"]

std_vrr_info = compute_vrr(countries, countries_std)
std_vrr = std_vrr_info["vrr"]
std_vertices = std_vrr_info["v_simplified"]


# ---------------------------------------------------------------------------
# 9. SEA-DP
# ---------------------------------------------------------------------------

sea_start = time.perf_counter()

countries_sea, sea_algo_stats = sea_dp_simplify(
    countries.copy(),
    epsilon=tolerance,
    tol=edge_match_tol,
    verbose=True,
)

sea_time = time.perf_counter() - sea_start

sea_tec_info = compute_new_tec(countries, countries_sea)
sea_raw_stats = sea_tec_info["simplified"]
sea_new_tec = sea_tec_info["new_tec"]

sea_sec_info = compute_pair_sec(
    countries,
    countries_sea,
    name_col=name_col,
    left_name=left_name,
    right_name=right_name,
    zone_buffer_m=zone_buffer_m,
    match_tol_m=match_tol_m,
)

sea_sec = sea_sec_info["sec"]

sea_hd_info = compute_hausdorff(countries, countries_sea)
sea_hd = sea_hd_info["mean"]

sea_vrr_info = compute_vrr(countries, countries_sea)
sea_vrr = sea_vrr_info["vrr"]
sea_vertices = sea_vrr_info["v_simplified"]


# ---------------------------------------------------------------------------
# 10. Print results
# ---------------------------------------------------------------------------

print("\n=== Standard DP Results ===")
print(f"  Raw TEC       : {std_raw_stats['tec']}")
print(f"  New TEC       : {std_new_tec}")
print(f"  SEC           : {std_sec:.4f}")
print(f"  HD            : {std_hd:,.2f} m")
print(f"  VRR           : {std_vrr:.2%}")
print(f"  Gaps          : {std_raw_stats['n_gaps']}")
print(f"  Overlaps      : {std_raw_stats['n_overlaps']}")
print(f"  Invalid       : {std_raw_stats['n_invalid']}")
print(f"  Vertices      : {std_vertices}")
print(f"  Time          : {std_time:.3f}s")
print(f"  Matched Length: {std_sec_info['l_matched']:,.2f} m")
print(f"  Total Length  : {std_sec_info['l_total']:,.2f} m")

print_country_survival("Standard DP", countries_std)

print("\n=== SEA-DP Results ===")
print(f"  Raw TEC       : {sea_raw_stats['tec']}")
print(f"  New TEC       : {sea_new_tec}")
print(f"  SEC           : {sea_sec:.4f}")
print(f"  HD            : {sea_hd:,.2f} m")
print(f"  VRR           : {sea_vrr:.2%}")
print(f"  Gaps          : {sea_raw_stats['n_gaps']}")
print(f"  Overlaps      : {sea_raw_stats['n_overlaps']}")
print(f"  Invalid       : {sea_raw_stats['n_invalid']}")
print(f"  Shared edges  : {sea_algo_stats.get('n_edges_shared', 'N/A')}")
print(f"  Shared arcs   : {sea_algo_stats.get('n_arcs_assembled', 'N/A')}")
print(f"  Vertices      : {sea_vertices}")
print(f"  Time          : {sea_time:.3f}s")
print(f"  Matched Length: {sea_sec_info['l_matched']:,.2f} m")
print(f"  Total Length  : {sea_sec_info['l_total']:,.2f} m")

print_country_survival("SEA-DP", countries_sea)


# ---------------------------------------------------------------------------
# 11. Plot with bottom metric boxes
# ---------------------------------------------------------------------------

orig_metrics = make_metrics_text(
    tec=0,
    sec=orig_sec,
    hd=orig_hd,
    vrr=orig_vrr,
    runtime=orig_time,
)

std_metrics = make_metrics_text(
    tec=std_new_tec,
    sec=std_sec,
    hd=std_hd,
    vrr=std_vrr,
    runtime=std_time,
)

sea_metrics = make_metrics_text(
    tec=sea_new_tec,
    sec=sea_sec,
    hd=sea_hd,
    vrr=sea_vrr,
    runtime=sea_time,
)

fig = plt.figure(figsize=(18, 9), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[4, 1.35])

fig.suptitle(
    "Armenia & Azerbaijan — Complex Boundary / Enclave Test",
    fontsize=14,
    fontweight="bold",
    y=1.03,
)

# Top row: maps
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

countries.plot(
    ax=ax1,
    color="whitesmoke",
    edgecolor="blue",
    linewidth=1.5,
)
ax1.set_title(
    f"Original\nVertices = {orig_vertices}",
    pad=12,
)
ax1.set_axis_off()

countries_std.plot(
    ax=ax2,
    color="whitesmoke",
    edgecolor="red",
    linewidth=1.5,
)
ax2.set_title(
    f"Standard DP ({tolerance / 1000:.0f} km)\n"
    f"Vertices = {std_vertices}",
    pad=12,
)
ax2.set_axis_off()

countries_sea.plot(
    ax=ax3,
    color="whitesmoke",
    edgecolor="green",
    linewidth=1.5,
)
ax3.set_title(
    f"SEA-DP ({tolerance / 1000:.0f} km)\n"
    f"Vertices = {sea_vertices}",
    pad=12,
)
ax3.set_axis_off()

# Bottom row: metrics
bx1 = fig.add_subplot(gs[1, 0])
bx2 = fig.add_subplot(gs[1, 1])
bx3 = fig.add_subplot(gs[1, 2])

for bx in (bx1, bx2, bx3):
    bx.set_axis_off()

box_style = dict(
    boxstyle="round,pad=0.55",
    facecolor="white",
    edgecolor="black",
    alpha=0.95,
)

bx1.text(
    0.02,
    0.98,
    orig_metrics,
    transform=bx1.transAxes,
    va="top",
    ha="left",
    fontsize=10,
    family="monospace",
    bbox=box_style,
)

bx2.text(
    0.02,
    0.98,
    std_metrics,
    transform=bx2.transAxes,
    va="top",
    ha="left",
    fontsize=10,
    family="monospace",
    bbox=box_style,
)

bx3.text(
    0.02,
    0.98,
    sea_metrics,
    transform=bx3.transAxes,
    va="top",
    ha="left",
    fontsize=10,
    family="monospace",
    bbox=box_style,
)


# ---------------------------------------------------------------------------
# 12. Save output
# ---------------------------------------------------------------------------

out_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "results",
    "figures",
)
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "armenia_azerbaijan_comparison.png")

plt.savefig(
    out_path,
    dpi=200,
    bbox_inches="tight",
    pad_inches=0.25,
)

plt.show()

print(f"\nPlot saved -> {out_path}")