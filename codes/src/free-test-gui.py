"""
codes/src/free-test-gui.py
==========================
SEA-DP Free Test GUI

Modes:
1. Countries mode:
   - Country 1
   - Country 2
   - Uses Natural Earth admin_0 countries dataset
   - Country 2 is filtered to neighbors of Country 1

2. States / Provinces mode:
   - Check "States or Provinces"
   - Choose either USA or Philippines
   - State/Province 1
   - State/Province 2
   - State/Province 2 is filtered to neighbors of State/Province 1

Runs:
- Original
- Standard DP
- SEA-DP

Metrics:
- TEC
- SEC
- HD
- VRR
- Execution Time
"""

import os
import sys
import time
import glob
import tkinter as tk
from tkinter import ttk, messagebox

import geopandas as gpd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 0. Project paths
# ---------------------------------------------------------------------------

SRC_PATH = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(SRC_PATH, "..", ".."))

MAIN_ALGO_PATH = os.path.join(SRC_PATH, "..", "main algo")
sys.path.insert(0, os.path.abspath(MAIN_ALGO_PATH))
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
)

print("Using SEA-DP from:", sea_dp.__file__)


# ---------------------------------------------------------------------------
# 1. Find dataset paths automatically
# ---------------------------------------------------------------------------

def find_first(patterns):
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        matches = [m for m in matches if os.path.exists(m)]

        if matches:
            return sorted(matches)[0]

    return None


NATURAL_EARTH_ROOT = os.path.join(
    ROOT,
    "data",
    "raw",
    "natural_earth",
)

GADM_PH_ROOT = os.path.join(
    ROOT,
    "data",
    "raw",
    "GADM",
    "philippines",
)

COUNTRIES_PATH = find_first([
    os.path.join(NATURAL_EARTH_ROOT, "**", "ne_*admin_0_countries.shp"),
])

USA_STATES_PATH = find_first([
    os.path.join(NATURAL_EARTH_ROOT, "USA", "**", "ne_*admin_1_states_provinces.shp"),
])

PH_PROVINCES_PATH = find_first([
    os.path.join(GADM_PH_ROOT, "gadm41_PHL_1.shp"),
    os.path.join(GADM_PH_ROOT, "**", "gadm41_PHL_1.shp"),
])


# ---------------------------------------------------------------------------
# 2. Utility functions
# ---------------------------------------------------------------------------

def detect_name_col(gdf, candidates):
    for col in candidates:
        if col in gdf.columns:
            return col

    raise ValueError(f"No usable name column found. Columns: {list(gdf.columns)}")


def safe_filename(text):
    out = str(text).lower()
    out = out.replace(" ", "_")
    out = out.replace(":", "")
    out = out.replace("/", "_")
    out = out.replace("\\", "_")
    out = out.replace("(", "")
    out = out.replace(")", "")
    out = out.replace(",", "")
    out = out.replace("&", "and")
    return out


def make_metrics_text(tec, sec, hd, vrr, runtime):
    return (
        "Metrics\n"
        f"TEC : {tec}\n"
        f"SEC : {sec:.4f}\n"
        f"HD  : {hd:,.2f} m\n"
        f"VRR : {vrr:.2%}\n"
        f"T   : {runtime:.3f} s"
    )


def project_to_metric(gdf):
    """
    Project selected features to local metric CRS.
    Falls back to EPSG:3857 if UTM estimation fails.
    """
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    gdf_4326 = gdf.to_crs(epsg=4326)

    try:
        metric_crs = gdf_4326.estimate_utm_crs()

        if metric_crs is None:
            metric_crs = "EPSG:3857"

    except Exception:
        metric_crs = "EPSG:3857"

    return gdf_4326.to_crs(metric_crs)


def get_selected_pair(gdf, name_col, name1, name2):
    selected = gdf[gdf[name_col].isin([name1, name2])].copy()

    if len(selected) != 2:
        found = list(selected[name_col].unique()) if name_col in selected.columns else []
        raise ValueError(
            f"Could not find exactly two selected features: {name1}, {name2}\n"
            f"Found: {found}"
        )

    selected = selected[[name_col, "geometry"]].copy()
    selected = selected.rename(columns={name_col: "feature_name"})
    selected = selected.reset_index(drop=True)

    selected = project_to_metric(selected)

    return selected


def largest_part_for_zoom(geom):
    """
    Return the largest polygon part for display zoom only.
    This prevents remote islands or tiny far-away pieces from zooming out the map.
    """
    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == "Polygon":
        return geom

    if geom.geom_type == "MultiPolygon":
        parts = [g for g in geom.geoms if g is not None and not g.is_empty]
        if parts:
            return max(parts, key=lambda g: g.area)

    if geom.geom_type == "GeometryCollection":
        polygons = []

        for g in geom.geoms:
            if g.geom_type == "Polygon":
                polygons.append(g)
            elif g.geom_type == "MultiPolygon":
                polygons.extend(list(g.geoms))

        polygons = [g for g in polygons if g is not None and not g.is_empty]

        if polygons:
            return max(polygons, key=lambda g: g.area)

    return geom


def set_map_zoom(ax, base_gdf, padding_ratio=0.08):
    """
    Zoom display to the main landmass of the selected pair.
    This only affects the plot view, not the actual computation.
    """
    zoom_geoms = []

    for geom in base_gdf.geometry:
        main_geom = largest_part_for_zoom(geom)

        if main_geom is not None and not main_geom.is_empty:
            zoom_geoms.append(main_geom)

    if not zoom_geoms:
        return

    zoom_gdf = gpd.GeoDataFrame(geometry=zoom_geoms, crs=base_gdf.crs)

    minx, miny, maxx, maxy = zoom_gdf.total_bounds

    width = maxx - minx
    height = maxy - miny

    if width <= 0 or height <= 0:
        return

    pad_x = width * padding_ratio
    pad_y = height * padding_ratio

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")


def get_neighbor_names(gdf, name_col, selected_name):
    """
    Return names of features that share a boundary with selected_name.

    Fix:
    - spatial index returns positional indices
    - use .iloc[int(pos)] instead of .loc[pos]
    """
    if selected_name is None or selected_name == "":
        return sorted(gdf[name_col].dropna().astype(str).unique())

    work = gdf.reset_index(drop=True).copy()
    rows = work[work[name_col] == selected_name]

    if rows.empty:
        return sorted(work[name_col].dropna().astype(str).unique())

    selected_geom = rows.geometry.iloc[0]

    if selected_geom is None or selected_geom.is_empty:
        return []

    neighbors = []

    try:
        sindex = work.sindex
        candidate_positions = list(sindex.intersection(selected_geom.bounds))
    except Exception:
        candidate_positions = list(range(len(work)))

    for pos in candidate_positions:
        try:
            row = work.iloc[int(pos)]
        except Exception:
            continue

        other_name = str(row[name_col])

        if other_name == selected_name:
            continue

        other_geom = row.geometry

        if other_geom is None or other_geom.is_empty:
            continue

        try:
            if not selected_geom.intersects(other_geom):
                continue

            shared_boundary = selected_geom.boundary.intersection(other_geom.boundary)

            if not shared_boundary.is_empty and shared_boundary.length > 0:
                neighbors.append(other_name)

        except Exception:
            continue

    return sorted(set(neighbors))


def filter_usa_admin1(gdf):
    """
    Natural Earth admin_1 files are global.
    This keeps only USA admin-1 rows.
    """
    work = gdf.copy()

    possible_country_cols = [
        "admin",
        "adm0_name",
        "geonunit",
        "sovereignt",
        "SOVEREIGNT",
        "ADM0_NAME",
        "ADMIN",
        "iso_a2",
        "adm0_a3",
        "sov_a3",
    ]

    for col in possible_country_cols:
        if col not in work.columns:
            continue

        values = work[col].astype(str)

        mask = (
            values.str.contains("United States", case=False, na=False)
            | values.str.fullmatch("USA", case=False, na=False)
            | values.str.fullmatch("US", case=False, na=False)
        )

        if mask.sum() > 0:
            print(f"Filtered USA states using column: {col}")
            return work[mask].copy().reset_index(drop=True)

    possible_iso_cols = [
        "iso_3166_2",
        "ISO_3166_2",
        "postal",
        "POSTAL",
    ]

    for col in possible_iso_cols:
        if col not in work.columns:
            continue

        values = work[col].astype(str)

        if col.lower() == "iso_3166_2":
            mask = values.str.startswith("US-", na=False)
        else:
            us_postals = {
                "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
                "DC",
            }
            mask = values.isin(us_postals)

        if mask.sum() > 0:
            print(f"Filtered USA states using column: {col}")
            return work[mask].copy().reset_index(drop=True)

    raise ValueError(
        "Could not filter USA states from Natural Earth admin_1 file.\n"
        f"Columns found: {list(work.columns)}"
    )


# ---------------------------------------------------------------------------
# 3. Load datasets
# ---------------------------------------------------------------------------

def load_datasets():
    datasets = {}

    # Countries
    if COUNTRIES_PATH and os.path.exists(COUNTRIES_PATH):
        countries = gpd.read_file(COUNTRIES_PATH)

        country_name_col = detect_name_col(
            countries,
            ["NAME", "ADMIN", "NAME_EN", "SOVEREIGNT", "name"],
        )

        countries = countries.dropna(subset=[country_name_col]).copy()
        countries[country_name_col] = countries[country_name_col].astype(str)
        countries = countries[countries[country_name_col].str.strip() != ""].copy()
        countries = countries.reset_index(drop=True)

        datasets["countries"] = {
            "label": "Countries",
            "path": COUNTRIES_PATH,
            "gdf": countries,
            "name_col": country_name_col,
            "names": sorted(countries[country_name_col].unique()),
        }

    # USA States
    if USA_STATES_PATH and os.path.exists(USA_STATES_PATH):
        usa_states = gpd.read_file(USA_STATES_PATH)

        usa_name_col = detect_name_col(
            usa_states,
            ["name", "NAME", "name_en", "region", "postal"],
        )

        usa_states = filter_usa_admin1(usa_states)

        usa_states = usa_states.dropna(subset=[usa_name_col]).copy()
        usa_states[usa_name_col] = usa_states[usa_name_col].astype(str)
        usa_states = usa_states[usa_states[usa_name_col].str.strip() != ""].copy()
        usa_states = usa_states.drop_duplicates(subset=[usa_name_col]).copy()
        usa_states = usa_states.reset_index(drop=True)

        datasets["usa_states"] = {
            "label": "USA",
            "path": USA_STATES_PATH,
            "gdf": usa_states,
            "name_col": usa_name_col,
            "names": sorted(usa_states[usa_name_col].unique()),
        }

    # Philippine Provinces
    if PH_PROVINCES_PATH and os.path.exists(PH_PROVINCES_PATH):
        ph_provinces = gpd.read_file(PH_PROVINCES_PATH)

        ph_name_col = detect_name_col(
            ph_provinces,
            ["NAME_1", "NAME", "name", "province", "PROVINCE"],
        )

        ph_provinces = ph_provinces.dropna(subset=[ph_name_col]).copy()
        ph_provinces[ph_name_col] = ph_provinces[ph_name_col].astype(str)
        ph_provinces = ph_provinces[ph_provinces[ph_name_col].str.strip() != ""].copy()
        ph_provinces = ph_provinces.reset_index(drop=True)

        datasets["ph_provinces"] = {
            "label": "Philippines",
            "path": PH_PROVINCES_PATH,
            "gdf": ph_provinces,
            "name_col": ph_name_col,
            "names": sorted(ph_provinces[ph_name_col].unique()),
        }

    return datasets


DATASETS = load_datasets()

if "countries" not in DATASETS:
    raise RuntimeError(
        "Natural Earth countries dataset not found. "
        f"Checked inside: {NATURAL_EARTH_ROOT}"
    )

if "usa_states" not in DATASETS and "ph_provinces" not in DATASETS:
    print("Warning: no USA states or Philippine provinces dataset found.")


# ---------------------------------------------------------------------------
# 4. Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    selected,
    feature1,
    feature2,
    title_name,
    tolerance,
    edge_match_tol,
    match_tol_m,
    zone_buffer_m,
):
    name_col = "feature_name"

    # Original
    orig_stats = compute_tec(selected)
    orig_vertices = count_vertices_gdf(selected)

    orig_sec = 1.0
    orig_hd = 0.0
    orig_vrr = 0.0
    orig_time = 0.0

    # Standard DP
    std_start = time.perf_counter()

    selected_std, std_algo_stats = standard_dp_simplify(
        selected.copy(),
        epsilon=tolerance,
    )

    std_time = time.perf_counter() - std_start

    std_tec_info = compute_new_tec(selected, selected_std)
    std_raw_stats = std_tec_info["simplified"]
    std_new_tec = std_tec_info["new_tec"]

    std_sec_info = compute_pair_sec(
        selected,
        selected_std,
        name_col=name_col,
        left_name=feature1,
        right_name=feature2,
        zone_buffer_m=zone_buffer_m,
        match_tol_m=match_tol_m,
    )

    std_sec = std_sec_info["sec"]

    std_hd_info = compute_hausdorff(selected, selected_std)
    std_hd = std_hd_info["mean"]

    std_vrr_info = compute_vrr(selected, selected_std)
    std_vrr = std_vrr_info["vrr"]
    std_vertices = std_vrr_info["v_simplified"]

    # SEA-DP
    sea_start = time.perf_counter()

    selected_sea, sea_algo_stats = sea_dp_simplify(
        selected.copy(),
        epsilon=tolerance,
        tol=edge_match_tol,
        verbose=True,
    )

    sea_time = time.perf_counter() - sea_start

    sea_tec_info = compute_new_tec(selected, selected_sea)
    sea_raw_stats = sea_tec_info["simplified"]
    sea_new_tec = sea_tec_info["new_tec"]

    sea_sec_info = compute_pair_sec(
        selected,
        selected_sea,
        name_col=name_col,
        left_name=feature1,
        right_name=feature2,
        zone_buffer_m=zone_buffer_m,
        match_tol_m=match_tol_m,
    )

    sea_sec = sea_sec_info["sec"]

    sea_hd_info = compute_hausdorff(selected, selected_sea)
    sea_hd = sea_hd_info["mean"]

    sea_vrr_info = compute_vrr(selected, selected_sea)
    sea_vrr = sea_vrr_info["vrr"]
    sea_vertices = sea_vrr_info["v_simplified"]

    # Terminal results
    print("\n" + "=" * 70)
    print(f"{title_name}: {feature1} vs {feature2}")
    print("=" * 70)

    print("\n=== Original ===")
    print(f"  TEC       : {orig_stats['tec']}")
    print(f"  Vertices  : {orig_vertices}")

    print("\n=== Standard DP ===")
    print(f"  TEC       : {std_new_tec}")
    print(f"  SEC       : {std_sec:.4f}")
    print(f"  HD        : {std_hd:,.2f} m")
    print(f"  VRR       : {std_vrr:.2%}")
    print(f"  Vertices  : {std_vertices}")
    print(f"  Time      : {std_time:.3f}s")
    print(f"  Gaps      : {std_raw_stats['n_gaps']}")
    print(f"  Overlaps  : {std_raw_stats['n_overlaps']}")
    print(f"  Invalid   : {std_raw_stats['n_invalid']}")

    print("\n=== SEA-DP ===")
    print(f"  TEC       : {sea_new_tec}")
    print(f"  SEC       : {sea_sec:.4f}")
    print(f"  HD        : {sea_hd:,.2f} m")
    print(f"  VRR       : {sea_vrr:.2%}")
    print(f"  Vertices  : {sea_vertices}")
    print(f"  Time      : {sea_time:.3f}s")
    print(f"  Gaps      : {sea_raw_stats['n_gaps']}")
    print(f"  Overlaps  : {sea_raw_stats['n_overlaps']}")
    print(f"  Invalid   : {sea_raw_stats['n_invalid']}")
    print(f"  Shared edges : {sea_algo_stats.get('n_edges_shared', 'N/A')}")
    print(f"  Shared arcs  : {sea_algo_stats.get('n_arcs_assembled', 'N/A')}")

    # Metric boxes
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

    # Plot
    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1.35])

    fig.suptitle(
        f"{feature1} & {feature2} - Shared Border Test",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    selected.plot(
        ax=ax1,
        color="whitesmoke",
        edgecolor="blue",
        linewidth=1.5,
    )
    set_map_zoom(ax1, selected)
    ax1.set_title(
        f"Original\nVertices = {orig_vertices}",
        pad=12,
    )
    ax1.set_axis_off()

    selected_std.plot(
        ax=ax2,
        color="whitesmoke",
        edgecolor="red",
        linewidth=1.5,
    )
    set_map_zoom(ax2, selected)
    ax2.set_title(
        f"Standard DP ({tolerance / 1000:.0f} km)\n"
        f"Vertices = {std_vertices}",
        pad=12,
    )
    ax2.set_axis_off()

    selected_sea.plot(
        ax=ax3,
        color="whitesmoke",
        edgecolor="green",
        linewidth=1.5,
    )
    set_map_zoom(ax3, selected)
    ax3.set_title(
        f"SEA-DP ({tolerance / 1000:.0f} km)\n"
        f"Vertices = {sea_vertices}",
        pad=12,
    )
    ax3.set_axis_off()

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

    out_dir = os.path.join(ROOT, "results", "figures")
    os.makedirs(out_dir, exist_ok=True)

    out_name = (
        f"free_test_{safe_filename(feature1)}_"
        f"{safe_filename(feature2)}.png"
    )

    out_path = os.path.join(out_dir, out_name)

    plt.savefig(
        out_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.25,
    )

    plt.show()

    return out_path


# ---------------------------------------------------------------------------
# 5. GUI logic
# ---------------------------------------------------------------------------

def update_status(extra_text=None):
    base = (
        f"Loaded: "
        f"{len(DATASETS.get('countries', {}).get('names', []))} countries, "
        f"{len(DATASETS.get('usa_states', {}).get('names', []))} USA states, "
        f"{len(DATASETS.get('ph_provinces', {}).get('names', []))} PH provinces"
    )

    if extra_text:
        status.config(text=f"{base} | {extra_text}")
    else:
        status.config(text=base)


def update_dropdowns():
    use_subnational = subnational_var.get()

    if use_subnational:
        dataset_label.grid()
        dataset_combo.grid()
        dataset_combo.config(state="readonly")

        dataset_choice = dataset_var.get()

        if dataset_choice == "USA":
            data = DATASETS.get("usa_states")
            label1_text.set("State 1")
            label2_text.set("State 2")
            tolerance_var.set("20000")
            zone_buffer_var.set("5000")

        elif dataset_choice == "Philippines":
            data = DATASETS.get("ph_provinces")
            label1_text.set("Province 1")
            label2_text.set("Province 2")
            tolerance_var.set("2000")
            zone_buffer_var.set("500")

        else:
            data = None

        values = data["names"] if data else []

    else:
        dataset_combo.config(state="disabled")
        dataset_label.grid_remove()
        dataset_combo.grid_remove()

        data = DATASETS["countries"]
        values = data["names"]

        label1_text.set("Country 1")
        label2_text.set("Country 2")
        tolerance_var.set("5000")
        zone_buffer_var.set("1250")

    combo1["values"] = values
    combo2["values"] = values

    selected1_var.set("")
    selected2_var.set("")

    update_status()


def get_active_dataset():
    if subnational_var.get():
        dataset_choice = dataset_var.get()

        if dataset_choice == "USA":
            if "usa_states" not in DATASETS:
                raise ValueError("USA states dataset not found.")
            return DATASETS["usa_states"], "USA States"

        if dataset_choice == "Philippines":
            if "ph_provinces" not in DATASETS:
                raise ValueError("Philippine provinces dataset not found.")
            return DATASETS["ph_provinces"], "Philippine Provinces"

        raise ValueError("Please choose USA or Philippines.")

    return DATASETS["countries"], "Countries"


def on_first_selection_changed(event=None):
    """
    When Country/State/Province 1 changes, filter dropdown 2
    to only neighboring features.
    """
    name1 = selected1_var.get()
    selected2_var.set("")

    if not name1:
        return

    try:
        data, _ = get_active_dataset()

        neighbor_names = get_neighbor_names(
            data["gdf"],
            data["name_col"],
            name1,
        )

        combo2["values"] = neighbor_names

        if neighbor_names:
            update_status(f"Neighbors detected for {name1}: {len(neighbor_names)}")
        else:
            update_status(f"No neighbor detected for {name1}")
            messagebox.showinfo(
                "No neighbor detected",
                (
                    f"No neighboring polygon was detected for '{name1}'.\n\n"
                    "This can happen for islands, isolated features, or datasets "
                    "where boundaries do not physically touch."
                ),
            )

    except Exception as e:
        messagebox.showerror("Neighbor filter error", str(e))


def on_run():
    name1 = selected1_var.get()
    name2 = selected2_var.get()

    if not name1 or not name2:
        messagebox.showerror("Missing selection", "Please choose both items.")
        return

    if name1 == name2:
        messagebox.showerror("Same selection", "Please choose two different items.")
        return

    try:
        tolerance = float(tolerance_var.get())
        edge_match_tol = float(edge_match_tol_var.get())
        match_tol_m = float(match_tol_var.get())
        zone_buffer_m = float(zone_buffer_var.get())

    except ValueError:
        messagebox.showerror(
            "Invalid parameter",
            "Tolerance values must be valid numbers.",
        )
        return

    try:
        data, title_name = get_active_dataset()

        selected = get_selected_pair(
            data["gdf"],
            data["name_col"],
            name1,
            name2,
        )

        run_button.config(state="disabled")
        root.update_idletasks()

        out_path = run_experiment(
            selected=selected,
            feature1=name1,
            feature2=name2,
            title_name=title_name,
            tolerance=tolerance,
            edge_match_tol=edge_match_tol,
            match_tol_m=match_tol_m,
            zone_buffer_m=zone_buffer_m,
        )

        messagebox.showinfo(
            "Done",
            f"Experiment finished.\n\nSaved figure:\n{out_path}",
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))

    finally:
        run_button.config(state="normal")


# ---------------------------------------------------------------------------
# 6. GUI layout
# ---------------------------------------------------------------------------

root = tk.Tk()
root.title("SEA-DP Free Test GUI")
root.geometry("840x470")

main = ttk.Frame(root, padding=18)
main.pack(fill="both", expand=True)

title = ttk.Label(
    main,
    text="SEA-DP Free Test",
    font=("Segoe UI", 16, "bold"),
)
title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

subtitle = ttk.Label(
    main,
    text="Choose a first polygon, then the second dropdown will show only detected neighbors.",
)
subtitle.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 14))

subnational_var = tk.BooleanVar(value=False)

states_check = ttk.Checkbutton(
    main,
    text="States or Provinces",
    variable=subnational_var,
    command=update_dropdowns,
)
states_check.grid(row=2, column=0, sticky="w", pady=(0, 8))

dataset_var = tk.StringVar()

available_subnational = []

if "usa_states" in DATASETS:
    available_subnational.append("USA")

if "ph_provinces" in DATASETS:
    available_subnational.append("Philippines")

if available_subnational:
    dataset_var.set(available_subnational[0])
else:
    dataset_var.set("")

dataset_label = ttk.Label(main, text="Dataset")
dataset_combo = ttk.Combobox(
    main,
    textvariable=dataset_var,
    values=available_subnational,
    width=30,
    state="disabled",
)
dataset_combo.bind("<<ComboboxSelected>>", lambda event: update_dropdowns())

dataset_label.grid(row=2, column=1, sticky="e", padx=(10, 6), pady=(0, 8))
dataset_combo.grid(row=2, column=2, sticky="w", pady=(0, 8))

label1_text = tk.StringVar(value="Country 1")
label2_text = tk.StringVar(value="Country 2")

selected1_var = tk.StringVar()
selected2_var = tk.StringVar()

label1 = ttk.Label(main, textvariable=label1_text)
label1.grid(row=3, column=0, sticky="w")

combo1 = ttk.Combobox(
    main,
    textvariable=selected1_var,
    width=58,
    state="readonly",
)
combo1.grid(row=3, column=1, columnspan=2, sticky="ew", pady=4)
combo1.bind("<<ComboboxSelected>>", on_first_selection_changed)

label2 = ttk.Label(main, textvariable=label2_text)
label2.grid(row=4, column=0, sticky="w")

combo2 = ttk.Combobox(
    main,
    textvariable=selected2_var,
    width=58,
    state="readonly",
)
combo2.grid(row=4, column=1, columnspan=2, sticky="ew", pady=4)

ttk.Separator(main).grid(row=5, column=0, columnspan=3, sticky="ew", pady=14)

tolerance_var = tk.StringVar(value="5000")
edge_match_tol_var = tk.StringVar(value="0.01")
match_tol_var = tk.StringVar(value="5.0")
zone_buffer_var = tk.StringVar(value="1250")

ttk.Label(main, text="DP tolerance / epsilon").grid(row=6, column=0, sticky="w")
ttk.Entry(main, textvariable=tolerance_var, width=18).grid(row=6, column=1, sticky="w")
ttk.Label(main, text="meters; countries example 5000, USA 20000, PH 2000").grid(row=6, column=2, sticky="w")

ttk.Label(main, text="SEA-DP edge match tol").grid(row=7, column=0, sticky="w")
ttk.Entry(main, textvariable=edge_match_tol_var, width=18).grid(row=7, column=1, sticky="w")
ttk.Label(main, text="meters; usually 0.01").grid(row=7, column=2, sticky="w")

ttk.Label(main, text="SEC match tolerance").grid(row=8, column=0, sticky="w")
ttk.Entry(main, textvariable=match_tol_var, width=18).grid(row=8, column=1, sticky="w")
ttk.Label(main, text="meters; usually 5").grid(row=8, column=2, sticky="w")

ttk.Label(main, text="SEC zone buffer").grid(row=9, column=0, sticky="w")
ttk.Entry(main, textvariable=zone_buffer_var, width=18).grid(row=9, column=1, sticky="w")
ttk.Label(main, text="meters; countries 1250, USA 5000, PH 500").grid(row=9, column=2, sticky="w")

run_button = ttk.Button(
    main,
    text="Run Standard DP vs SEA-DP",
    command=on_run,
)
run_button.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(18, 8))

status = ttk.Label(main, text="")
status.grid(row=11, column=0, columnspan=3, sticky="w")

path_status = ttk.Label(
    main,
    text=(
        f"Countries: {COUNTRIES_PATH}\n"
        f"USA: {USA_STATES_PATH}\n"
        f"Philippines: {PH_PROVINCES_PATH}"
    ),
    font=("Segoe UI", 8),
)
path_status.grid(row=12, column=0, columnspan=3, sticky="w", pady=(8, 0))

main.columnconfigure(1, weight=1)
main.columnconfigure(2, weight=1)

update_dropdowns()

root.mainloop()