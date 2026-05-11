"""
intro_to_ai_baseline.py
=======================
Baseline Douglas-Peucker (DP) — Intro to AI Course
Polytechnic University of the Philippines

Runs ONLY the standard (baseline) Douglas-Peucker algorithm.
Same datasets, same paths, same visual style as free-test-gui.py —
but shows only: Original | Standard DP (no SEA-DP).

Architecture modules (from flowchart):
  Module 0  — INPUT LAYER:      Douglas_Peucker(P, epsilon)
  Module 1  — Initialize:       dmax=0, index=0, end=length(P)
  Module 2  — Loop:             for i=2 to end-1
  Module 3  — Distance:         compute perpendicular distance d
  Module 4  — Compare:          d > dmax? → update max point
  Module 5  — Prepare output:   ResultList = empty
  Module 6  — Check threshold:  dmax > epsilon?
  Module 7a — Recursive:        split and recurse on both halves
  Module 7b — Base case:        keep endpoints only {P[0], P[n+1]}
  Module 8  — OUTPUT LAYER:     return ResultList

Usage:
    python intro_to_ai_baseline.py
"""

from __future__ import annotations

import os
import sys
import glob
import time
import math
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

# ---------------------------------------------------------------------------
# 0. Paths — same structure as free-test-gui.py
# ---------------------------------------------------------------------------

SRC_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(SRC_PATH, "..", ".."))

MAIN_ALGO_PATH = os.path.join(SRC_PATH, "..", "main algo")
sys.path.insert(0, os.path.abspath(MAIN_ALGO_PATH))
sys.path.insert(0, os.path.abspath(SRC_PATH))

from evaluation_metrics import (
    compute_tec,
    compute_new_tec,
    compute_pair_sec,
    compute_hausdorff,
    compute_vrr,
    count_vertices_gdf,
)

NATURAL_EARTH_ROOT = os.path.join(ROOT, "data", "raw", "natural_earth")
GADM_PH_ROOT       = os.path.join(ROOT, "data", "raw", "GADM", "philippines")


def find_first(patterns):
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        matches = [m for m in matches if os.path.exists(m)]
        if matches:
            return sorted(matches)[0]
    return None


COUNTRIES_PATH    = find_first([os.path.join(NATURAL_EARTH_ROOT, "**", "ne_*admin_0_countries.shp")])
USA_STATES_PATH   = find_first([os.path.join(NATURAL_EARTH_ROOT, "USA", "**", "ne_*admin_1_states_provinces.shp")])
PH_PROVINCES_PATH = find_first([
    os.path.join(GADM_PH_ROOT, "gadm41_PHL_1.shp"),
    os.path.join(GADM_PH_ROOT, "**", "gadm41_PHL_1.shp"),
])


# ===========================================================================
# DOUGLAS-PEUCKER CORE — structured to match flowchart exactly
# ===========================================================================

def _perpendicular_distance(point: Tuple, start: Tuple, end: Tuple) -> float:
    """
    Compute perpendicular distance from 'point' to the line defined by
    'start' and 'end'. Used in Module 3 (Distance Computation).
    """
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - sx, py - sy)
    t = ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)
    return math.hypot(px - (sx + t * dx), py - (sy + t * dy))


def douglas_peucker(coords: List[Tuple], epsilon: float) -> List[Tuple]:
    """
    Douglas-Peucker recursive simplification.

    Module 1  — INPUT LAYER:   Douglas_Peucker(P, epsilon)
    Module 2  — Initialize:    dmax=0, index=0, end=length(P)
    Module 3  — Loop:          for i=2 to end-1
    Module 4  — Distance:      compute perpendicular distance d
    Module 5  — Compare:       d > dmax? update max point
    Module 6  — Prepare:       ResultList = empty
    Module 7  — Threshold:     dmax > epsilon?
    Module 8a — YES: recurse on both halves
    Module 8b — NO:  keep endpoints only
    Module 9  — return ResultList
    """
    # MODULE 2: INITIALIZE
    if len(coords) < 3:
        return list(coords)
    start = coords[0]
    end   = coords[-1]
    dmax  = 0.0
    index = 0

    # MODULES 3-5: LOOP + DISTANCE + COMPARE
    for i in range(1, len(coords) - 1):
        d = _perpendicular_distance(coords[i], start, end)   # Module 3
        if d > dmax:                                          # Module 4
            dmax  = d
            index = i

    # MODULE 6: ResultList = empty (implicit)

    # MODULE 7: dmax > epsilon?
    if dmax > epsilon:
        # MODULE 8a: YES - recursive simplification
        left  = douglas_peucker(coords[:index + 1], epsilon)
        right = douglas_peucker(coords[index:],     epsilon)
        return left[:-1] + right   # ResultList = recResults1[0..-2] + recResults2
    else:
        # MODULE 8b: NO - keep endpoints only {P[0], P[n+1]}
        return [start, end]

    # MODULE 9: return ResultList (handled by returns above)


def _simplify_polygon(geom, epsilon: float):
    """Apply DP to every ring of a polygon geometry."""
    if geom is None or geom.is_empty:
        return geom

    def simplify_ring(ring_coords):
        coords = list(ring_coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) < 3:
            return list(ring_coords)
        simplified = douglas_peucker(coords, epsilon)
        if len(simplified) > 1 and simplified[0] != simplified[-1]:
            simplified.append(simplified[0])
        return simplified if len(simplified) >= 4 else list(ring_coords)

    def simplify_one(poly):
        ext  = simplify_ring(list(poly.exterior.coords))
        ints = [simplify_ring(list(r.coords)) for r in poly.interiors]
        ints = [r for r in ints if len(r) >= 4]
        try:
            result = Polygon(ext, ints)
            return make_valid(result) if not result.is_valid else result
        except Exception:
            return poly

    if geom.geom_type == 'Polygon':
        return simplify_one(geom)
    if geom.geom_type == 'MultiPolygon':
        parts = []

    for p in geom.geoms:
        simplified = simplify_one(p)

        if simplified is None or simplified.is_empty:
            continue

        # Flatten nested MultiPolygons
        if simplified.geom_type == 'Polygon':
            parts.append(simplified)

        elif simplified.geom_type == 'MultiPolygon':
            parts.extend([
                g for g in simplified.geoms
                if g is not None and not g.is_empty
            ])

    if not parts:
        return geom

    return parts[0] if len(parts) == 1 else MultiPolygon(parts)
    


def _preprocess(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    def clean(g):
        if g is None or g.is_empty:
            return None
        if not g.is_valid:
            g = make_valid(g)
        if g.geom_type in ('Polygon', 'MultiPolygon'):
            return g
        if g.geom_type == 'GeometryCollection':
            parts = [p for p in g.geoms if p.geom_type in ('Polygon', 'MultiPolygon')]
            if not parts:
                return None
            return parts[0] if len(parts) == 1 else MultiPolygon(parts)
        return None

    gdf['geometry'] = gdf['geometry'].apply(clean)
    gdf = gdf[~(gdf['geometry'].isna() | gdf['geometry'].is_empty)].copy()
    return gdf.reset_index(drop=True)


def baseline_dp_simplify(gdf: gpd.GeoDataFrame, epsilon: float):
    """Apply standard DP to all polygons independently."""
    t0 = time.perf_counter()
    working_gdf = _preprocess(gdf)
    result_gdf  = working_gdf.copy()
    result_gdf['geometry'] = result_gdf['geometry'].apply(
        lambda g: _simplify_polygon(g, epsilon)
    )
    elapsed = time.perf_counter() - t0
    return result_gdf, {'epsilon': epsilon, 'execution_time_s': elapsed}


# ===========================================================================
# Dataset utilities — same as free-test-gui.py
# ===========================================================================

def detect_name_col(gdf, candidates):
    for col in candidates:
        if col in gdf.columns:
            return col
    raise ValueError(f"No usable name column found. Columns: {list(gdf.columns)}")


def safe_filename(text):
    out = str(text).lower()
    for old, new in [(' ','_'),(':',''),('/','_'),('\\','_'),
                     ('(',''),(')', ''),(',',''),('&','and')]:
        out = out.replace(old, new)
    return out


def project_to_metric(gdf):
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    gdf_4326 = gdf.to_crs(epsg=4326)
    try:
        metric_crs = gdf_4326.estimate_utm_crs() or "EPSG:3857"
    except Exception:
        metric_crs = "EPSG:3857"
    return gdf_4326.to_crs(metric_crs)


def get_selected_pair(gdf, name_col, name1, name2):
    selected = gdf[gdf[name_col].isin([name1, name2])].copy()
    if len(selected) != 2:
        found = list(selected[name_col].unique()) if name_col in selected.columns else []
        raise ValueError(f"Could not find exactly two features: {name1}, {name2}\nFound: {found}")
    selected = selected[[name_col, 'geometry']].copy()
    selected = selected.rename(columns={name_col: 'feature_name'})
    selected = selected.reset_index(drop=True)
    return project_to_metric(selected)


def largest_part_for_zoom(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == 'Polygon':
        return geom
    if geom.geom_type == 'MultiPolygon':
        parts = [g for g in geom.geoms if g and not g.is_empty]
        return max(parts, key=lambda g: g.area) if parts else None
    if geom.geom_type == 'GeometryCollection':
        polys = []
        for g in geom.geoms:
            if g.geom_type == 'Polygon':
                polys.append(g)
            elif g.geom_type == 'MultiPolygon':
                polys.extend(list(g.geoms))
        polys = [g for g in polys if g and not g.is_empty]
        return max(polys, key=lambda g: g.area) if polys else None
    return geom


def set_map_zoom(ax, base_gdf, padding_ratio=0.08):
    zoom_geoms = [largest_part_for_zoom(g) for g in base_gdf.geometry]
    zoom_geoms = [g for g in zoom_geoms if g is not None and not g.is_empty]
    if not zoom_geoms:
        return
    zoom_gdf = gpd.GeoDataFrame(geometry=zoom_geoms, crs=base_gdf.crs)
    minx, miny, maxx, maxy = zoom_gdf.total_bounds
    w, h = maxx - minx, maxy - miny
    if w <= 0 or h <= 0:
        return
    ax.set_xlim(minx - w * padding_ratio, maxx + w * padding_ratio)
    ax.set_ylim(miny - h * padding_ratio, maxy + h * padding_ratio)
    ax.set_aspect('equal')


def get_neighbor_names(gdf, name_col, selected_name):
    if not selected_name:
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
            shared = selected_geom.boundary.intersection(other_geom.boundary)
            if not shared.is_empty and shared.length > 0:
                neighbors.append(other_name)
        except Exception:
            continue
    return sorted(set(neighbors))


def filter_usa_admin1(gdf):
    work = gdf.copy()
    for col in ['admin','adm0_name','geonunit','sovereignt','SOVEREIGNT',
                'ADM0_NAME','ADMIN','iso_a2','adm0_a3','sov_a3']:
        if col not in work.columns:
            continue
        values = work[col].astype(str)
        mask = (values.str.contains('United States', case=False, na=False)
                | values.str.fullmatch('USA', case=False, na=False)
                | values.str.fullmatch('US',  case=False, na=False))
        if mask.sum() > 0:
            print(f"Filtered USA states using column: {col}")
            return work[mask].copy().reset_index(drop=True)
    for col in ['iso_3166_2','ISO_3166_2','postal','POSTAL']:
        if col not in work.columns:
            continue
        values = work[col].astype(str)
        if col.lower() == 'iso_3166_2':
            mask = values.str.startswith('US-', na=False)
        else:
            us_postals = {
                'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID',
                'IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS',
                'MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK',
                'OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV',
                'WI','WY','DC',
            }
            mask = values.isin(us_postals)
        if mask.sum() > 0:
            print(f"Filtered USA states using column: {col}")
            return work[mask].copy().reset_index(drop=True)
    raise ValueError(f"Could not filter USA states.\nColumns: {list(work.columns)}")


# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------

def load_datasets():
    datasets = {}

    if COUNTRIES_PATH and os.path.exists(COUNTRIES_PATH):
        countries = gpd.read_file(COUNTRIES_PATH)
        nc = detect_name_col(countries, ['NAME','ADMIN','NAME_EN','SOVEREIGNT','name'])
        countries = countries.dropna(subset=[nc]).copy()
        countries[nc] = countries[nc].astype(str)
        countries = countries[countries[nc].str.strip() != ''].reset_index(drop=True)
        datasets['countries'] = {
            'label': 'Countries', 'path': COUNTRIES_PATH,
            'gdf': countries, 'name_col': nc,
            'names': sorted(countries[nc].unique()),
        }

    if USA_STATES_PATH and os.path.exists(USA_STATES_PATH):
        usa = gpd.read_file(USA_STATES_PATH)
        nc  = detect_name_col(usa, ['name','NAME','name_en','region','postal'])
        usa = filter_usa_admin1(usa)
        usa = usa.dropna(subset=[nc]).copy()
        usa[nc] = usa[nc].astype(str)
        usa = usa[usa[nc].str.strip() != ''].drop_duplicates(subset=[nc]).reset_index(drop=True)
        datasets['usa_states'] = {
            'label': 'USA', 'path': USA_STATES_PATH,
            'gdf': usa, 'name_col': nc,
            'names': sorted(usa[nc].unique()),
        }

    if PH_PROVINCES_PATH and os.path.exists(PH_PROVINCES_PATH):
        ph = gpd.read_file(PH_PROVINCES_PATH)
        nc = detect_name_col(ph, ['NAME_1','NAME','name','province','PROVINCE'])
        ph = ph.dropna(subset=[nc]).copy()
        ph[nc] = ph[nc].astype(str)
        ph = ph[ph[nc].str.strip() != ''].reset_index(drop=True)
        datasets['ph_provinces'] = {
            'label': 'Philippines', 'path': PH_PROVINCES_PATH,
            'gdf': ph, 'name_col': nc,
            'names': sorted(ph[nc].unique()),
        }

    return datasets


DATASETS = load_datasets()

if 'countries' not in DATASETS:
    raise RuntimeError(
        f"Natural Earth countries dataset not found.\nChecked inside: {NATURAL_EARTH_ROOT}"
    )


# ===========================================================================
# Experiment runner + plotting
# ===========================================================================

def make_metrics_text(tec, sec, hd, vrr, runtime):
    return (
        "Metrics\n"
        f"TEC : {tec}\n"
        f"SEC : {sec:.4f}\n"
        f"HD  : {hd:,.2f} m\n"
        f"VRR : {vrr:.2%}\n"
        f"T   : {runtime:.3f} s"
    )


def run_experiment(selected, feature1, feature2, title_name,
                   tolerance, match_tol_m, zone_buffer_m):
    name_col = 'feature_name'

    # Original
    orig_stats    = compute_tec(selected)
    orig_vertices = count_vertices_gdf(selected)

    # Baseline Standard DP
    std_start = time.perf_counter()
    selected_std, _ = baseline_dp_simplify(selected.copy(), epsilon=tolerance)
    std_time = time.perf_counter() - std_start

    std_tec_info = compute_new_tec(selected, selected_std)
    std_new_tec  = std_tec_info['new_tec']

    std_sec_info = compute_pair_sec(
        selected, selected_std,
        name_col=name_col,
        left_name=feature1, right_name=feature2,
        zone_buffer_m=zone_buffer_m, match_tol_m=match_tol_m,
    )
    std_sec = std_sec_info['sec']

    std_hd_info  = compute_hausdorff(selected, selected_std)
    std_hd       = std_hd_info['mean']

    std_vrr_info = compute_vrr(selected, selected_std)
    std_vrr      = std_vrr_info['vrr']
    std_vertices = std_vrr_info['v_simplified']

    # Terminal output
    print("\n" + "=" * 60)
    print(f"  BASELINE DP — {title_name}: {feature1} vs {feature2}")
    print("=" * 60)
    print(f"\n  Original vertices : {orig_vertices}")
    print(f"  Original TEC      : {orig_stats['tec']}")
    print(f"\n  [Standard DP]  epsilon = {tolerance} m")
    print(f"  TEC      : {std_new_tec}")
    print(f"  SEC      : {std_sec:.4f}")
    print(f"  HD       : {std_hd:,.2f} m")
    print(f"  VRR      : {std_vrr:.2%}")
    print(f"  Vertices : {std_vertices}")
    print(f"  Time     : {std_time:.3f}s")
    print("=" * 60 + "\n")

    # Metric text
    orig_metrics = make_metrics_text(tec=0, sec=1.0, hd=0.0, vrr=0.0, runtime=0.0)
    std_metrics  = make_metrics_text(tec=std_new_tec, sec=std_sec,
                                     hd=std_hd, vrr=std_vrr, runtime=std_time)

    # Plot — same style as free-test-gui.py, 2 columns instead of 3
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs  = fig.add_gridspec(2, 2, height_ratios=[4, 1.35])

    fig.suptitle(
        f"{feature1} & {feature2}  —  Baseline DP Test\n"
        f"epsilon = {tolerance:,.0f} m  ({tolerance/1000:.1f} km)",
        fontsize=13, fontweight='bold', y=1.03,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Original — with gold shared boundary highlight
    selected.plot(ax=ax1, color='whitesmoke', edgecolor='blue', linewidth=1.5)
    set_map_zoom(ax1, selected)
    ax1.set_title(f"Original\nVertices = {orig_vertices}", pad=12)
    ax1.set_axis_off()
    try:
        g1 = selected[selected['feature_name'] == feature1].iloc[0].geometry
        g2 = selected[selected['feature_name'] == feature2].iloc[0].geometry
        shared = g1.boundary.intersection(g2.boundary)
        if not shared.is_empty:
            gpd.GeoSeries([shared], crs=selected.crs).plot(
                ax=ax1, color='gold', linewidth=2.5, zorder=5)
    except Exception:
        pass

    # Standard DP
    selected_std.plot(ax=ax2, color='whitesmoke', edgecolor='red', linewidth=1.5)
    set_map_zoom(ax2, selected)
    ax2.set_title(
        f"Standard DP  ({tolerance/1000:.1f} km)\nVertices = {std_vertices}  (VRR {std_vrr:.1%})",
        pad=12,
    )
    ax2.set_axis_off()

    # Metric boxes
    bx1 = fig.add_subplot(gs[1, 0])
    bx2 = fig.add_subplot(gs[1, 1])
    for bx in (bx1, bx2):
        bx.set_axis_off()

    box_style = dict(boxstyle='round,pad=0.55', facecolor='white',
                     edgecolor='black', alpha=0.95)

    bx1.text(0.02, 0.98, orig_metrics,
             transform=bx1.transAxes, va='top', ha='left',
             fontsize=10, family='monospace', bbox=box_style)
    bx2.text(0.02, 0.98, std_metrics,
             transform=bx2.transAxes, va='top', ha='left',
             fontsize=10, family='monospace', bbox=box_style)

    bx1.text(0.5, 0.02, 'Original',
             transform=bx1.transAxes, ha='center', va='bottom',
             fontsize=9, color='blue', fontweight='bold')
    bx2.text(0.5, 0.02, 'Standard DP (Baseline)',
             transform=bx2.transAxes, ha='center', va='bottom',
             fontsize=9, color='red', fontweight='bold')

    # Save
    out_dir = os.path.join(ROOT, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"baseline_{safe_filename(feature1)}_{safe_filename(feature2)}.png"
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.25)
    plt.show()
    return out_path


# ===========================================================================
# GUI — same Tkinter layout as free-test-gui.py
# ===========================================================================

def get_active_dataset():
    if subnational_var.get():
        choice = dataset_var.get()
        if choice == 'USA':
            if 'usa_states' not in DATASETS:
                raise ValueError('USA states dataset not found.')
            return DATASETS['usa_states'], 'USA States'
        if choice == 'Philippines':
            if 'ph_provinces' not in DATASETS:
                raise ValueError('Philippine provinces dataset not found.')
            return DATASETS['ph_provinces'], 'Philippine Provinces'
        raise ValueError('Please choose USA or Philippines.')
    return DATASETS['countries'], 'Countries'


def update_status(extra=None):
    base = (
        f"Loaded: "
        f"{len(DATASETS.get('countries',{}).get('names',[]))} countries, "
        f"{len(DATASETS.get('usa_states',{}).get('names',[]))} USA states, "
        f"{len(DATASETS.get('ph_provinces',{}).get('names',[]))} PH provinces"
    )
    status.config(text=f"{base} | {extra}" if extra else base)


def update_dropdowns():
    if subnational_var.get():
        dataset_label.grid()
        dataset_combo.grid()
        dataset_combo.config(state='readonly')
        choice = dataset_var.get()
        if choice == 'USA':
            data = DATASETS.get('usa_states')
            label1_text.set('State 1')
            label2_text.set('State 2')
            tolerance_var.set('20000')
            zone_buffer_var.set('5000')
        elif choice == 'Philippines':
            data = DATASETS.get('ph_provinces')
            label1_text.set('Province 1')
            label2_text.set('Province 2')
            tolerance_var.set('2000')
            zone_buffer_var.set('500')
        else:
            data = None
        values = data['names'] if data else []
    else:
        dataset_combo.config(state='disabled')
        dataset_label.grid_remove()
        dataset_combo.grid_remove()
        data = DATASETS['countries']
        values = data['names']
        label1_text.set('Country 1')
        label2_text.set('Country 2')
        tolerance_var.set('5000')
        zone_buffer_var.set('1250')

    combo1['values'] = values
    combo2['values'] = values
    selected1_var.set('')
    selected2_var.set('')
    update_status()


def on_first_selection_changed(event=None):
    name1 = selected1_var.get()
    selected2_var.set('')
    if not name1:
        return
    try:
        data, _ = get_active_dataset()
        neighbors = get_neighbor_names(data['gdf'], data['name_col'], name1)
        combo2['values'] = neighbors
        if neighbors:
            update_status(f"Neighbors of {name1}: {len(neighbors)}")
        else:
            update_status(f"No neighbors found for {name1}")
            messagebox.showinfo('No neighbors',
                f"No neighboring polygon found for '{name1}'.\n"
                "This can happen for islands or isolated features.")
    except Exception as e:
        messagebox.showerror('Neighbor filter error', str(e))


def on_run():
    name1 = selected1_var.get()
    name2 = selected2_var.get()
    if not name1 or not name2:
        messagebox.showerror('Missing selection', 'Please choose both items.')
        return
    if name1 == name2:
        messagebox.showerror('Same selection', 'Please choose two different items.')
        return
    try:
        tolerance     = float(tolerance_var.get())
        match_tol_m   = float(match_tol_var.get())
        zone_buffer_m = float(zone_buffer_var.get())
    except ValueError:
        messagebox.showerror('Invalid parameter', 'Tolerance values must be valid numbers.')
        return
    try:
        data, title_name = get_active_dataset()
        selected = get_selected_pair(data['gdf'], data['name_col'], name1, name2)
        run_button.config(state='disabled')
        root.update_idletasks()
        out_path = run_experiment(
            selected=selected, feature1=name1, feature2=name2,
            title_name=title_name, tolerance=tolerance,
            match_tol_m=match_tol_m, zone_buffer_m=zone_buffer_m,
        )
        messagebox.showinfo('Done', f"Baseline DP finished.\n\nSaved:\n{out_path}")
    except Exception as e:
        messagebox.showerror('Error', str(e))
    finally:
        run_button.config(state='normal')


# GUI layout

root = tk.Tk()
root.title("Baseline DP — Intro to AI")
root.geometry("840x430")

main = ttk.Frame(root, padding=18)
main.pack(fill='both', expand=True)

ttk.Label(main, text="Baseline Douglas-Peucker",
          font=('Segoe UI', 16, 'bold')).grid(
    row=0, column=0, columnspan=3, sticky='w', pady=(0, 4))

ttk.Label(main,
    text="Standard DP baseline — no shared-edge detection. "
         "Select a polygon pair to compare Original vs Simplified.",
    font=('Segoe UI', 9)).grid(
    row=1, column=0, columnspan=3, sticky='w', pady=(0, 12))

subnational_var = tk.BooleanVar(value=False)
ttk.Checkbutton(main, text='States or Provinces',
                variable=subnational_var,
                command=update_dropdowns).grid(
    row=2, column=0, sticky='w', pady=(0, 8))

dataset_var = tk.StringVar()
available_subnational = []
if 'usa_states'   in DATASETS: available_subnational.append('USA')
if 'ph_provinces' in DATASETS: available_subnational.append('Philippines')
dataset_var.set(available_subnational[0] if available_subnational else '')

dataset_label = ttk.Label(main, text='Dataset')
dataset_combo = ttk.Combobox(main, textvariable=dataset_var,
                              values=available_subnational,
                              width=30, state='disabled')
dataset_combo.bind('<<ComboboxSelected>>', lambda e: update_dropdowns())
dataset_label.grid(row=2, column=1, sticky='e', padx=(10,6), pady=(0,8))
dataset_combo.grid(row=2, column=2, sticky='w', pady=(0,8))

label1_text   = tk.StringVar(value='Country 1')
label2_text   = tk.StringVar(value='Country 2')
selected1_var = tk.StringVar()
selected2_var = tk.StringVar()

ttk.Label(main, textvariable=label1_text).grid(row=3, column=0, sticky='w')
combo1 = ttk.Combobox(main, textvariable=selected1_var, width=58, state='readonly')
combo1.grid(row=3, column=1, columnspan=2, sticky='ew', pady=4)
combo1.bind('<<ComboboxSelected>>', on_first_selection_changed)

ttk.Label(main, textvariable=label2_text).grid(row=4, column=0, sticky='w')
combo2 = ttk.Combobox(main, textvariable=selected2_var, width=58, state='readonly')
combo2.grid(row=4, column=1, columnspan=2, sticky='ew', pady=4)

ttk.Separator(main).grid(row=5, column=0, columnspan=3, sticky='ew', pady=12)

tolerance_var   = tk.StringVar(value='5000')
match_tol_var   = tk.StringVar(value='5.0')
zone_buffer_var = tk.StringVar(value='1250')

ttk.Label(main, text='DP tolerance / epsilon').grid(row=6, column=0, sticky='w')
ttk.Entry(main, textvariable=tolerance_var, width=18).grid(row=6, column=1, sticky='w')
ttk.Label(main, text='meters  |  countries: 5000, USA: 20000, PH: 2000').grid(row=6, column=2, sticky='w')

ttk.Label(main, text='SEC match tolerance').grid(row=7, column=0, sticky='w')
ttk.Entry(main, textvariable=match_tol_var, width=18).grid(row=7, column=1, sticky='w')
ttk.Label(main, text='meters  |  usually 5').grid(row=7, column=2, sticky='w')

ttk.Label(main, text='SEC zone buffer').grid(row=8, column=0, sticky='w')
ttk.Entry(main, textvariable=zone_buffer_var, width=18).grid(row=8, column=1, sticky='w')
ttk.Label(main, text='meters  |  countries: 1250, USA: 5000, PH: 500').grid(row=8, column=2, sticky='w')

run_button = ttk.Button(main, text='Run Baseline DP', command=on_run)
run_button.grid(row=9, column=0, columnspan=3, sticky='ew', pady=(16, 8))

status = ttk.Label(main, text='')
status.grid(row=10, column=0, columnspan=3, sticky='w')

ttk.Label(main,
    text=(f"Countries : {COUNTRIES_PATH}\n"
          f"USA       : {USA_STATES_PATH}\n"
          f"PH        : {PH_PROVINCES_PATH}"),
    font=('Segoe UI', 8)).grid(
    row=11, column=0, columnspan=3, sticky='w', pady=(8, 0))

main.columnconfigure(1, weight=1)
main.columnconfigure(2, weight=1)

update_dropdowns()
root.mainloop()
