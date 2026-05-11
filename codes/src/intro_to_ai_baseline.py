"""
intro_to_ai_baseline.py
=======================
Baseline Douglas-Peucker (DP) Algorithm
Intro to AI Course — Polytechnic University of the Philippines

Topology-Preserving Polygon Simplification Using a
Shared-Edge-Aware Douglas-Peucker Algorithm

This file implements ONLY the standard (baseline) Douglas-Peucker algorithm,
structured to directly match the architecture flowchart provided in the course.

Architecture modules (from the flowchart):
  [INPUT LAYER]
    Module 1 — Input: Douglas_Peucker(P, epsilon)

  [PROCESSING LAYER]
    Module 2 — Initialize:       dmax = 0, index = 0, end = length(P)
    Module 3 — Loop:             for i = 2 to end - 1
    Module 4 — Distance:         Compute perpendicular distance d
    Module 5 — Compare:          d > dmax?  →  Update max point
    Module 6 — Prepare output:   ResultList = empty
    Module 7 — Check threshold:  dmax > epsilon?
    Module 8a — Recursive:       Split and recurse on both halves
    Module 8b — Base case:       Keep endpoints only

  [OUTPUT LAYER]
    Module 9 — Output:           return ResultList

Usage:
    python intro_to_ai_baseline.py
    python intro_to_ai_baseline.py path/to/your/file.shp 500
"""

from __future__ import annotations

import math
import time
from typing import List, Tuple

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

# Type alias for clarity
Coords = List[Tuple[float, float]]


# =============================================================================
# MODULE 1 — INPUT LAYER
# Douglas_Peucker(P, epsilon)
# Accepts a polyline P (list of coordinate points) and a tolerance epsilon.
# =============================================================================

# NOTE: The actual function entry point is douglas_peucker() below.
# This comment block marks the INPUT LAYER from the architecture diagram.


# =============================================================================
# HELPER — Perpendicular Distance
# Used inside Module 3 (Distance Computation).
#
# Computes the perpendicular distance from a point to the baseline
# connecting 'start' and 'end'.
#
# This is the core geometric operation of DP:
#   - If start == end (degenerate segment), returns straight Euclidean distance
#   - Otherwise, projects the point onto the line and measures the deviation
# =============================================================================

def _perpendicular_distance(point: Tuple, start: Tuple, end: Tuple) -> float:
    """
    Compute perpendicular distance from 'point' to the line defined by
    'start' and 'end'.

    This directly implements the formula:
        t = ((P - A) · (B - A)) / |B - A|^2
        closest = A + t * (B - A)
        distance = |P - closest|

    Where A = start, B = end, P = point.
    """
    px, py = point
    sx, sy = start
    ex, ey = end

    dx = ex - sx
    dy = ey - sy

    # Degenerate case: start and end are the same point
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - sx, py - sy)

    # Project point onto the line, find the scalar parameter t
    t = ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)

    # Compute distance from point to its projection on the line
    return math.hypot(px - (sx + t * dx), py - (sy + t * dy))


# =============================================================================
# MODULES 1-9 — PROCESSING + OUTPUT LAYER
# The full Douglas-Peucker recursive algorithm.
#
# Directly maps to the flowchart boxes:
#   Module 2: Initialize dmax, index, end
#   Module 3: Loop through all intermediate points
#   Module 4: Compute perpendicular distance d
#   Module 5: d > dmax? → update max point (index, dmax)
#   Module 6: ResultList = empty (prepare output container)
#   Module 7: dmax > epsilon?
#   Module 8a: YES → Recursive simplification on both halves
#   Module 8b: NO  → Keep endpoints only {P[0], P[n+1]}
#   Module 9: return ResultList
# =============================================================================

def douglas_peucker(coords: Coords, epsilon: float) -> Coords:
    """
    Douglas-Peucker line simplification algorithm.

    Parameters
    ----------
    coords  : list of (x, y) tuples — the polyline to simplify
    epsilon : float — maximum allowable perpendicular deviation (tolerance)

    Returns
    -------
    list of (x, y) tuples — the simplified polyline
    """

    # ── MODULE 2: INITIALIZE ─────────────────────────────────────────────────
    # dmax = 0, index = 0, end = length(P)
    # Base case: if fewer than 3 points, nothing to simplify
    if len(coords) < 3:
        return list(coords)

    start = coords[0]        # P[1]   — fixed start point
    end   = coords[-1]       # P[end] — fixed end point

    dmax  = 0.0              # Maximum perpendicular distance found so far
    index = 0                # Index of the point with maximum distance

    # ── MODULES 3, 4, 5: LOOP + DISTANCE + COMPARE ───────────────────────────
    # Loop through all intermediate points: for i = 2 to end - 1
    # Compute perpendicular distance d for each point
    # If d > dmax, update the max point (index, dmax)
    for i in range(1, len(coords) - 1):

        # Module 3: Compute perpendicular distance
        d = _perpendicular_distance(coords[i], start, end)

        # Module 4: d > dmax? → update
        if d > dmax:
            dmax  = d
            index = i

    # ── MODULE 6: PREPARE OUTPUT CONTAINER ───────────────────────────────────
    # ResultList = empty
    # (Python implicitly prepares the return value in the recursive calls below)

    # ── MODULE 7: CHECK THRESHOLD — dmax > epsilon? ───────────────────────────
    if dmax > epsilon:

        # ── MODULE 8a: YES → RECURSIVE SIMPLIFICATION ────────────────────────
        # Split at the farthest point and recurse on both halves:
        #   recResults1 = DouglasPeucker(P[1..index], epsilon)
        #   recResults2 = DouglasPeucker(P[index..end], epsilon)
        left  = douglas_peucker(coords[:index + 1], epsilon)   # P[1..index]
        right = douglas_peucker(coords[index:],     epsilon)   # P[index..end]

        # Build result list: combine, dropping duplicate middle point
        #   ResultList = recResults1[0..-2] + recResults2
        return left[:-1] + right

    else:

        # ── MODULE 8b: NO → BASE CASE: KEEP ENDPOINTS ONLY ───────────────────
        # All intermediate points are within epsilon of the baseline.
        # Drop everything in between.
        #   ResultList = {P[0], P[n+1]}
        return [start, end]

    # ── MODULE 9: return ResultList ───────────────────────────────────────────
    # (handled by the return statements above)


# =============================================================================
# POLYGON SIMPLIFICATION WRAPPER
# Applies DP to each ring of a Polygon or MultiPolygon.
# This is the geospatial layer on top of the core DP algorithm.
# =============================================================================

def _simplify_polygon(geom, epsilon: float):
    """
    Apply Douglas-Peucker to every ring of a polygon geometry.
    Handles both Polygon and MultiPolygon types.
    Holes (interior rings) are also simplified independently.
    """
    if geom is None or geom.is_empty:
        return geom

    def simplify_ring(ring_coords: list) -> list:
        """Simplify one ring, ensuring closure and minimum vertex count."""
        coords = list(ring_coords)

        # Remove closing duplicate if present
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]

        # Need at least 3 points to form a polygon ring
        if len(coords) < 3:
            return list(ring_coords)

        # Run DP on the open ring
        simplified = douglas_peucker(coords, epsilon)

        # Re-close the ring
        if len(simplified) > 1 and simplified[0] != simplified[-1]:
            simplified.append(simplified[0])

        # If DP collapsed it below 4 points (invalid polygon), keep original
        if len(simplified) < 4:
            return list(ring_coords)

        return simplified

    def simplify_one_polygon(poly: Polygon) -> Polygon:
        """Simplify exterior ring + all interior rings of one polygon."""
        # Simplify exterior boundary
        ext = simplify_ring(list(poly.exterior.coords))

        # Simplify each interior ring (hole) independently
        ints = []
        for interior in poly.interiors:
            simp_int = simplify_ring(list(interior.coords))
            if len(simp_int) >= 4:
                ints.append(simp_int)

        try:
            result = Polygon(ext, ints)
            if not result.is_valid:
                result = make_valid(result)
            return result
        except Exception:
            return poly  # fallback to original on failure

    if geom.geom_type == 'Polygon':
        return simplify_one_polygon(geom)

    if geom.geom_type == 'MultiPolygon':
        parts = [simplify_one_polygon(p) for p in geom.geoms]
        parts = [p for p in parts if p and not p.is_empty]
        if not parts:
            return geom
        if len(parts) == 1:
            return parts[0]
        return MultiPolygon(parts)

    return geom


# =============================================================================
# PREPROCESSING
# Validates and repairs raw geometries before simplification.
# Same approach used in sea_dp.py — make_valid() handles self-intersections,
# degenerate rings, and other common shapefile quality issues.
# =============================================================================

def _preprocess(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Validate and repair input polygon geometries."""
    gdf = gdf.copy()

    def clean(g):
        if g is None or g.is_empty:
            return None
        if not g.is_valid:
            g = make_valid(g)
        # Keep only polygonal parts (make_valid may return GeometryCollection)
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
    gdf = gdf.reset_index(drop=True)
    return gdf


# =============================================================================
# TOPOLOGY VALIDATION
# Counts topological errors in the simplified output:
#   - Invalid geometries (self-intersections, etc.)
#   - Overlapping regions between adjacent polygons
#   - Enclosed gaps (interior holes in the dissolved union)
# =============================================================================

def _topology_validation(gdf: gpd.GeoDataFrame, area_tol: float = 1e-10) -> dict:
    """Compute Topological Error Count (TEC) for a GeoDataFrame."""
    work  = gdf[~(gdf['geometry'].isna() | gdf['geometry'].is_empty)].copy()
    geoms = list(work['geometry'])

    n_invalid  = sum(1 for g in geoms if not g.is_valid)
    n_overlaps = 0
    n_gaps     = 0

    if len(work) > 1:
        sindex = work.sindex
        for i, g in enumerate(geoms):
            for j in list(sindex.intersection(g.bounds)):
                if j <= i:
                    continue
                try:
                    inter = g.intersection(geoms[j])
                    if not inter.is_empty and inter.area > area_tol:
                        n_overlaps += 1
                except Exception:
                    n_overlaps += 1

    try:
        union_geom = unary_union(geoms)
        polys = list(union_geom.geoms) if union_geom.geom_type == 'MultiPolygon' else [union_geom]
        for poly in polys:
            if poly.geom_type == 'Polygon':
                n_gaps += len(list(poly.interiors))
    except Exception:
        n_gaps = -1

    tec = max(n_gaps, 0) + n_overlaps + n_invalid
    return {'tec': tec, 'n_gaps': n_gaps, 'n_overlaps': n_overlaps, 'n_invalid': n_invalid}


# =============================================================================
# VERTEX COUNTER
# =============================================================================

def _count_vertices(geom) -> int:
    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type == 'Polygon':
        return len(list(geom.exterior.coords)) + sum(len(list(r.coords)) for r in geom.interiors)
    if geom.geom_type == 'MultiPolygon':
        return sum(_count_vertices(p) for p in geom.geoms)
    return 0


# =============================================================================
# MAIN BASELINE FUNCTION
# Full pipeline: preprocess → simplify → validate → report
# =============================================================================

def baseline_dp_simplify(
    gdf: gpd.GeoDataFrame,
    epsilon: float,
    verbose: bool = True,
) -> Tuple[gpd.GeoDataFrame, dict]:
    """
    Apply the standard Douglas-Peucker algorithm to a polygon GeoDataFrame.

    This is the baseline model — no shared-edge awareness, no topology
    constraints. Each polygon boundary is simplified independently.

    Parameters
    ----------
    gdf     : input GeoDataFrame (polygon dataset)
    epsilon : simplification tolerance in the CRS units
              (use meters after reprojecting to UTM)
    verbose : print progress to console

    Returns
    -------
    result_gdf : simplified GeoDataFrame
    stats      : dict of performance metrics
    """
    t0 = time.perf_counter()

    if verbose:
        print("=" * 55)
        print("  Baseline Douglas-Peucker — Standard DP")
        print("=" * 55)

    # ── STEP 1: PREPROCESS ────────────────────────────────────────────────────
    if verbose:
        print("\n[Step 1/3] Preprocessing — validating geometries...")
    working_gdf = _preprocess(gdf)
    v_original  = sum(_count_vertices(g) for g in working_gdf['geometry'])

    if verbose:
        print(f"           Polygons : {len(working_gdf)}")
        print(f"           Vertices : {v_original:,}  (before simplification)")

    # ── STEP 2: SIMPLIFY — APPLY DP TO EACH POLYGON ───────────────────────────
    if verbose:
        print(f"\n[Step 2/3] Simplifying with epsilon = {epsilon}...")

    result_gdf = working_gdf.copy()
    result_gdf['geometry'] = result_gdf['geometry'].apply(
        lambda g: _simplify_polygon(g, epsilon)
    )

    v_simplified = sum(_count_vertices(g) for g in result_gdf['geometry'])
    vrr = (v_original - v_simplified) / v_original if v_original > 0 else 0.0

    if verbose:
        print(f"           Vertices after : {v_simplified:,}")
        print(f"           VRR            : {vrr:.1%}  ({v_original - v_simplified:,} vertices removed)")

    # ── STEP 3: TOPOLOGY VALIDATION ───────────────────────────────────────────
    if verbose:
        print("\n[Step 3/3] Topology validation...")
    tec_results = _topology_validation(result_gdf)

    elapsed = time.perf_counter() - t0

    stats = {
        'epsilon'         : epsilon,
        'n_polygons'      : len(working_gdf),
        'v_original'      : v_original,
        'v_simplified'    : v_simplified,
        'vrr'             : vrr,
        'execution_time_s': elapsed,
        **tec_results,
    }

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"  RESULTS")
        print(f"{'=' * 55}")
        print(f"  Epsilon          : {epsilon}")
        print(f"  Vertices before  : {v_original:,}")
        print(f"  Vertices after   : {v_simplified:,}")
        print(f"  VRR              : {vrr:.2%}")
        print(f"  TEC              : {tec_results['tec']}")
        print(f"    Gaps           : {tec_results['n_gaps']}")
        print(f"    Overlaps       : {tec_results['n_overlaps']}")
        print(f"    Invalid geoms  : {tec_results['n_invalid']}")
        print(f"  Execution time   : {elapsed:.4f}s")
        print(f"{'=' * 55}\n")

    return result_gdf, stats


# =============================================================================
# DATASET CONFIGURATION
# These are the exact same datasets used in the thesis evaluation.
# Update the paths below if your folder structure is different.
# =============================================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

PATH_US = BASE_DIR / "data/raw/natural_earth/USA/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp"
PATH_PH = BASE_DIR / "data/raw/GADM/philippines/gadm41_PHL_1.shp"
PATH_INTL = BASE_DIR / "data/raw/natural_earth/Armenia-Azerbaijan/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

# Epsilon values to test (in meters, after UTM reprojection)
EPSILONS = [500, 1000, 2000, 5000]

# ── TIER 1: LOW COMPLEXITY — US State pairs ───────────────────────────────────
# Natural Earth Admin 1 — name column: 'name'
# UTM: varies per state pair, auto-detected from centroid
US_PAIRS = [
    ('California',  'Nevada'),
    ('Arizona',     'New Mexico'),
    ('Oregon',      'Washington'),
    ('Texas',       'Oklahoma'),
]

# ── TIER 2: MODERATE COMPLEXITY — Philippine Province pairs ───────────────────
# GADM Level 1 (PHL_1) — name column: 'NAME_1'
# UTM: Zone 51N — EPSG:32651
PH_PAIRS = [
    ('Laguna',  'Quezon'),
    ('Rizal',   'Laguna'),
    ('Cavite',  'Laguna'),
    ('Rizal',   'Metro Manila'),
]

# ── TIER 3: HIGH COMPLEXITY — International boundary pairs ────────────────────
# Natural Earth Admin 0 — name column: 'NAME_EN' (fallback: 'ADMIN' or 'NAME')
# UTM: auto-detected from centroid
INTL_PAIRS = [
    ('Armenia',              'Azerbaijan'),
    ('United Arab Emirates', 'Oman'),
    ('Kyrgyzstan',           'Tajikistan'),
    ('Uzbekistan',           'Kyrgyzstan'),
]


# =============================================================================
# HELPERS — Dataset loading and pair extraction
# =============================================================================

def _auto_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to the appropriate UTM zone for its centroid."""
    if gdf.crs and not gdf.crs.is_geographic:
        return gdf  # already projected, leave as-is
    centroid     = gdf.geometry.union_all().centroid
    utm_zone     = int((centroid.x + 180) / 6) + 1
    hemisphere   = 32600 if centroid.y >= 0 else 32700
    epsg         = hemisphere + utm_zone
    print(f"           Auto UTM: EPSG:{epsg}")
    return gdf.to_crs(epsg=epsg)


def _load_and_reproject(path: str, epsg: int = None) -> gpd.GeoDataFrame:
    """Load a shapefile and reproject to metric CRS."""
    print(f"  Loading: {path}")
    gdf = gpd.read_file(path)
    print(f"  CRS    : {gdf.crs}  |  Features: {len(gdf)}")
    if epsg:
        gdf = gdf.to_crs(epsg=epsg)
        print(f"  Reprojected to EPSG:{epsg}")
    else:
        gdf = _auto_utm(gdf)
    return gdf


def _detect_name_col(gdf: gpd.GeoDataFrame, candidates: list) -> str:
    """Find the first matching name column in a GeoDataFrame."""
    for col in candidates:
        if col in gdf.columns:
            return col
    raise ValueError(
        f"None of the candidate name columns {candidates} found.\n"
        f"Available columns: {list(gdf.columns)}"
    )


def _extract_pair(gdf: gpd.GeoDataFrame, name_col: str,
                  left: str, right: str) -> gpd.GeoDataFrame:
    """Filter GeoDataFrame to just the two named features."""
    subset = gdf[gdf[name_col].isin([left, right])].copy()
    if len(subset) < 2:
        raise ValueError(
            f"Could not find both '{left}' and '{right}' in column '{name_col}'.\n"
            f"Found: {list(subset[name_col])}"
        )
    return subset.reset_index(drop=True)


# =============================================================================
# TIER RUNNERS
# Each function runs the baseline DP on all pairs in one complexity tier
# across all configured epsilon values.
# =============================================================================

def run_tier_ph(verbose: bool = True) -> list:
    """TIER 2 — Philippine Provinces (GADM Level 1, UTM 51N)."""
    print("\n" + "=" * 60)
    print("  TIER 2 — MODERATE COMPLEXITY: Philippine Provinces")
    print("=" * 60)

    gdf = _load_and_reproject(PATH_PH, epsg=32651)

    # GADM Level 1 Philippines: province name is in 'NAME_1'
    name_col = _detect_name_col(gdf, ['NAME_1', 'NAME_2', 'name'])
    print(f"  Name column: '{name_col}'")

    all_results = []
    for left, right in PH_PAIRS:
        for eps in EPSILONS:
            print(f"\n  Pair: {left} / {right}  |  epsilon = {eps}m")
            try:
                pair_gdf       = _extract_pair(gdf, name_col, left, right)
                result, stats  = baseline_dp_simplify(pair_gdf, epsilon=eps,
                                                      verbose=verbose)
                stats.update({'tier': 'moderate', 'left': left,
                              'right': right, 'name_col': name_col})
                all_results.append((left, right, eps, result, stats))
            except Exception as e:
                print(f"  [!] Skipped ({e})")

    return all_results


def run_tier_us(verbose: bool = True) -> list:
    """TIER 1 — US States (Natural Earth Admin 1, auto UTM)."""
    print("\n" + "=" * 60)
    print("  TIER 1 — LOW COMPLEXITY: US States")
    print("=" * 60)

    gdf = _load_and_reproject(PATH_US)

    # Natural Earth Admin 1: state name is in 'name' or 'NAME'
    name_col = _detect_name_col(gdf, ['name', 'NAME', 'NAME_EN', 'admin'])
    print(f"  Name column: '{name_col}'")

    all_results = []
    for left, right in US_PAIRS:
        for eps in EPSILONS:
            print(f"\n  Pair: {left} / {right}  |  epsilon = {eps}m")
            try:
                pair_gdf = _extract_pair(gdf, name_col, left, right)
                # Re-detect best UTM per pair (states span different zones)
                pair_gdf       = _auto_utm(pair_gdf)
                result, stats  = baseline_dp_simplify(pair_gdf, epsilon=eps,
                                                      verbose=verbose)
                stats.update({'tier': 'low', 'left': left,
                              'right': right, 'name_col': name_col})
                all_results.append((left, right, eps, result, stats))
            except Exception as e:
                print(f"  [!] Skipped ({e})")

    return all_results


def run_tier_intl(verbose: bool = True) -> list:
    """TIER 3 — International Boundaries (Natural Earth Admin 0, auto UTM)."""
    print("\n" + "=" * 60)
    print("  TIER 3 — HIGH COMPLEXITY: International Boundaries")
    print("=" * 60)

    gdf = _load_and_reproject(PATH_INTL)

    # Natural Earth Admin 0: country name may be in several columns
    name_col = _detect_name_col(gdf, ['NAME_EN', 'ADMIN', 'NAME', 'name'])
    print(f"  Name column: '{name_col}'")

    all_results = []
    for left, right in INTL_PAIRS:
        for eps in EPSILONS:
            print(f"\n  Pair: {left} / {right}  |  epsilon = {eps}m")
            try:
                pair_gdf       = _extract_pair(gdf, name_col, left, right)
                pair_gdf       = _auto_utm(pair_gdf)
                result, stats  = baseline_dp_simplify(pair_gdf, epsilon=eps,
                                                      verbose=verbose)
                stats.update({'tier': 'high', 'left': left,
                              'right': right, 'name_col': name_col})
                all_results.append((left, right, eps, result, stats))
            except Exception as e:
                print(f"  [!] Skipped ({e})")

    return all_results


# =============================================================================
# SYNTHETIC DEMO — no shapefile needed
# =============================================================================

def _demo_synthetic():
    """Quick demo on a tiny synthetic curve. No shapefile needed."""
    print("\n" + "=" * 55)
    print("  SYNTHETIC DEMO — No shapefile needed")
    print("=" * 55)

    curve = [
        (0.0, 0.0),   # A — start
        (1.0, 0.5),   # B — small deviation
        (2.0, 1.8),   # C — larger deviation
        (3.0, 0.9),   # D — medium deviation
        (4.0, 0.1),   # E — small deviation
        (5.0, 0.0),   # F — end
    ]

    print("\nOriginal polyline:")
    for i, pt in enumerate(curve):
        print(f"  P[{i}] = {pt}")

    for eps in [0.3, 1.0, 2.0]:
        result  = douglas_peucker(curve, epsilon=eps)
        removed = len(curve) - len(result)
        print(f"\nepsilon = {eps}  →  {len(result)} points kept, {removed} removed")
        print(f"  Kept: {result}")

    print("\nNote: larger epsilon = more aggressive = fewer points kept.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import sys

    # ── Command-line usage ────────────────────────────────────────────────────
    # python intro_to_ai_baseline.py              → synthetic demo
    # python intro_to_ai_baseline.py ph           → run PH tier only
    # python intro_to_ai_baseline.py us           → run US tier only
    # python intro_to_ai_baseline.py intl         → run INTL tier only
    # python intro_to_ai_baseline.py all          → run all three tiers
    # python intro_to_ai_baseline.py <file> <eps> → run on any shapefile

    arg = sys.argv[1].lower() if len(sys.argv) > 1 else ''

    if arg == 'ph':
        run_tier_ph(verbose=True)

    elif arg == 'us':
        run_tier_us(verbose=True)

    elif arg == 'intl':
        run_tier_intl(verbose=True)

    elif arg == 'all':
        run_tier_us(verbose=True)
        run_tier_ph(verbose=True)
        run_tier_intl(verbose=True)

    elif arg.endswith('.shp') and len(sys.argv) >= 3:
        # Generic single-file run: python intro_to_ai_baseline.py file.shp 500
        path = sys.argv[1]
        eps  = float(sys.argv[2])
        print(f"\nLoading: {path}")
        gdf = gpd.read_file(path)
        gdf = _auto_utm(gdf)
        baseline_dp_simplify(gdf, epsilon=eps, verbose=True)

    else:
        # No args or unrecognised → synthetic demo
        _demo_synthetic()
        print("\nTip: run with 'ph', 'us', 'intl', or 'all' to use your real datasets.")
        print("     e.g.  python intro_to_ai_baseline.py ph")
