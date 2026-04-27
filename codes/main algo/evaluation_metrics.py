"""
evaluation_metrics.py
=====================
Evaluation Metrics for SEA-DP vs Standard Douglas-Peucker Comparison

Metrics:
1. Topological Error Count (TEC)   = N_gaps + N_overlaps + N_invalid
2. Shared-Edge Consistency  (SEC)  = L_matched / L_total
3. Hausdorff Distance       (HD)
4. Vertex Reduction Ratio   (VRR)  = (V_orig - V_simp) / V_orig
5. Execution Time           (T)    = T_end - T_start

Important:
- TEC detects invalid geometries, overlaps, and enclosed gaps.
- TEC may miss open gaps along a shared border in isolated 2-polygon tests.
- For 2-polygon shared-border experiments, use compute_pair_sec().
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from shapely.ops import unary_union


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def count_vertices_geom(geom) -> int:
    """Count all coordinate vertices in a Polygon or MultiPolygon."""
    if geom is None or geom.is_empty:
        return 0

    if geom.geom_type == "Polygon":
        return (
            len(list(geom.exterior.coords))
            + sum(len(list(r.coords)) for r in geom.interiors)
        )

    if geom.geom_type == "MultiPolygon":
        return sum(count_vertices_geom(part) for part in geom.geoms)

    return 0


def count_vertices_gdf(gdf: gpd.GeoDataFrame) -> int:
    """Count all vertices in a GeoDataFrame."""
    return sum(count_vertices_geom(g) for g in gdf.geometry)


def count_parts_geom(geom) -> int:
    """Count polygon parts in a Polygon or MultiPolygon."""
    if geom is None or geom.is_empty:
        return 0

    if geom.geom_type == "Polygon":
        return 1

    if geom.geom_type == "MultiPolygon":
        return len(geom.geoms)

    return 0


def count_holes_geom(geom) -> int:
    """Count interior rings / holes in a Polygon or MultiPolygon."""
    if geom is None or geom.is_empty:
        return 0

    if geom.geom_type == "Polygon":
        return len(geom.interiors)

    if geom.geom_type == "MultiPolygon":
        return sum(len(part.interiors) for part in geom.geoms)

    return 0


def line_length(geom) -> float:
    """Return total line length from LineString, MultiLineString, or GeometryCollection."""
    if geom is None or geom.is_empty:
        return 0.0

    if geom.geom_type in ("LineString", "LinearRing"):
        return geom.length

    if geom.geom_type == "MultiLineString":
        return sum(line_length(g) for g in geom.geoms)

    if geom.geom_type == "GeometryCollection":
        return sum(line_length(g) for g in geom.geoms)

    return 0.0


def get_named_geom(gdf: gpd.GeoDataFrame, name_col: str, name: str):
    """Get one geometry from a GeoDataFrame using a name column."""
    rows = gdf[gdf[name_col] == name]

    if rows.empty:
        raise ValueError(f"{name} not found in column {name_col}.")

    return rows.geometry.iloc[0]


def get_named_vertices(gdf: gpd.GeoDataFrame, name_col: str, name: str) -> int:
    return count_vertices_geom(get_named_geom(gdf, name_col, name))


def get_named_parts(gdf: gpd.GeoDataFrame, name_col: str, name: str) -> int:
    return count_parts_geom(get_named_geom(gdf, name_col, name))


def get_named_holes(gdf: gpd.GeoDataFrame, name_col: str, name: str) -> int:
    return count_holes_geom(get_named_geom(gdf, name_col, name))


# ===========================================================================
# METRIC 1: TOPOLOGICAL ERROR COUNT
# ===========================================================================

def compute_tec(gdf: gpd.GeoDataFrame, area_tol: float = 1e-10) -> Dict:
    """
    TEC = N_gaps + N_overlaps + N_invalid

    Notes:
    - Gaps are detected as holes in the dissolved union.
    - This detects enclosed gaps.
    - It may not detect open gaps that leak into the outside void.
    """
    work = gdf.copy()
    work = work[~(work["geometry"].isna() | work["geometry"].is_empty)].copy()
    work = work.reset_index(drop=True)

    geoms = list(work.geometry)

    n_invalid = sum(1 for g in geoms if not g.is_valid)
    n_overlaps = 0
    n_gaps = 0

    if len(work) > 1:
        sindex = work.sindex

        for i, g in enumerate(geoms):
            candidates = list(sindex.intersection(g.bounds))

            for j in candidates:
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

        polys = (
            list(union_geom.geoms)
            if union_geom.geom_type == "MultiPolygon"
            else [union_geom]
        )

        for poly in polys:
            if poly.geom_type == "Polygon":
                n_gaps += len(list(poly.interiors))

    except Exception:
        n_gaps = -1

    tec = max(n_gaps, 0) + n_overlaps + n_invalid

    return {
        "tec": tec,
        "n_gaps": n_gaps,
        "n_overlaps": n_overlaps,
        "n_invalid": n_invalid,
    }


def compute_new_tec(
    original_gdf: gpd.GeoDataFrame,
    simplified_gdf: gpd.GeoDataFrame,
    area_tol: float = 1e-10,
) -> Dict:
    """
    Compute raw TEC and new TEC relative to the original input.

    new_tec = max(0, simplified_tec - original_tec)
    """
    orig = compute_tec(original_gdf, area_tol=area_tol)
    simp = compute_tec(simplified_gdf, area_tol=area_tol)

    return {
        "original_tec": orig["tec"],
        "raw_tec": simp["tec"],
        "new_tec": max(0, simp["tec"] - orig["tec"]),
        "original": orig,
        "simplified": simp,
    }


# ===========================================================================
# METRIC 2A: GENERAL SHARED-EDGE CONSISTENCY
# ===========================================================================

def _round_pt(p: Tuple[float, float], prec: int = 6) -> Tuple[float, float]:
    return round(p[0], prec), round(p[1], prec)


def _edge_key(p1, p2, prec: int = 6):
    a = _round_pt(p1, prec)
    b = _round_pt(p2, prec)
    return (a, b) if a <= b else (b, a)


def compute_sec(
    original_gdf: gpd.GeoDataFrame,
    simplified_gdf: gpd.GeoDataFrame,
    prec: int = 6,
) -> Dict:
    """
    General edge-key SEC.

    This is useful for exact segment-level checking, but it can be too strict
    after simplification because original small segments may disappear.
    For isolated 2-polygon visual tests, prefer compute_pair_sec().
    """

    def build_edge_map(gdf):
        edge_map: Dict = {}

        for idx, row in gdf.iterrows():
            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]

            for part in parts:
                rings = [part.exterior] + list(part.interiors)

                for ring in rings:
                    pts = list(ring.coords)

                    if len(pts) < 2:
                        continue

                    if pts[0] == pts[-1]:
                        pts = pts[:-1]

                    n = len(pts)

                    for i in range(n):
                        a = pts[i]
                        b = pts[(i + 1) % n]
                        key = _edge_key(a, b, prec)

                        if key not in edge_map:
                            edge_map[key] = {}

                        edge_map[key][idx] = (a, b)

        return edge_map

    orig_map = build_edge_map(original_gdf)
    simp_map = build_edge_map(simplified_gdf)

    shared_keys = {
        key for key, owners in orig_map.items()
        if len(owners) >= 2
    }

    l_total = 0.0
    l_matched = 0.0
    n_consistent = 0
    n_inconsistent = 0

    for key in shared_keys:
        a_orig, b_orig = key
        seg_len = math.hypot(
            b_orig[0] - a_orig[0],
            b_orig[1] - a_orig[1],
        )

        l_total += seg_len

        if key not in simp_map or len(simp_map[key]) < 2:
            n_inconsistent += 1
            continue

        endpoint_set = set()

        for _, (a, b) in simp_map[key].items():
            endpoint_set.add(_edge_key(a, b, prec))

        if len(endpoint_set) == 1:
            l_matched += seg_len
            n_consistent += 1
        else:
            n_inconsistent += 1

    sec = l_matched / l_total if l_total > 0 else 1.0

    return {
        "sec": sec,
        "l_matched": l_matched,
        "l_total": l_total,
        "n_shared_edges": len(shared_keys),
        "n_consistent_edges": n_consistent,
        "n_inconsistent_edges": n_inconsistent,
    }


# ===========================================================================
# METRIC 2B: PAIRWISE SEC FOR TWO-POLYGON SHARED-BORDER TESTS
# ===========================================================================

def compute_pair_sec(
    original_gdf: gpd.GeoDataFrame,
    simplified_gdf: gpd.GeoDataFrame,
    name_col: str,
    left_name: str,
    right_name: str,
    zone_buffer_m: float,
    match_tol_m: float = 5.0,
) -> Dict:
    """
    Pairwise Shared-Edge Consistency.

    This directly checks whether two simplified polygon boundaries remain
    coincident near their original shared boundary.

    Use this for:
    - California-Nevada
    - Rizal-Laguna
    - Armenia-Azerbaijan pair tests
    - isolated two-polygon visual demos

    SEC = L_matched / L_total
    """
    orig_left = get_named_geom(original_gdf, name_col, left_name)
    orig_right = get_named_geom(original_gdf, name_col, right_name)

    simp_left = get_named_geom(simplified_gdf, name_col, left_name)
    simp_right = get_named_geom(simplified_gdf, name_col, right_name)

    original_shared = orig_left.boundary.intersection(orig_right.boundary)

    if original_shared.is_empty:
        return {
            "sec": 0.0,
            "l_matched": 0.0,
            "l_total": 0.0,
            "original_shared_length": 0.0,
        }

    shared_zone = original_shared.buffer(zone_buffer_m)

    left_near = simp_left.boundary.intersection(shared_zone)
    right_near = simp_right.boundary.intersection(shared_zone)

    left_len = line_length(left_near)
    right_len = line_length(right_near)

    matched_geom = left_near.intersection(right_near.buffer(match_tol_m))
    l_matched = line_length(matched_geom)

    l_total = max(left_len, right_len)

    sec = l_matched / l_total if l_total > 0 else 0.0
    sec = max(0.0, min(1.0, sec))

    return {
        "sec": sec,
        "l_matched": l_matched,
        "l_total": l_total,
        "original_shared_length": line_length(original_shared),
    }


# ===========================================================================
# METRIC 3: HAUSDORFF DISTANCE
# ===========================================================================

def compute_hausdorff(
    original_gdf: gpd.GeoDataFrame,
    simplified_gdf: gpd.GeoDataFrame,
) -> Dict:
    """
    Compute per-polygon Hausdorff distance using polygon boundaries.

    Returns:
    - hd_values
    - mean
    - std
    - max
    - min
    """
    hd_values = []

    for orig, simp in zip(original_gdf.geometry, simplified_gdf.geometry):
        if orig is None or simp is None:
            continue

        if orig.is_empty or simp.is_empty:
            continue

        try:
            hd = orig.boundary.hausdorff_distance(simp.boundary)
            hd_values.append(float(hd))
        except Exception:
            continue

    hd_arr = np.array(hd_values, dtype=float)

    return {
        "hd_values": hd_values,
        "mean": float(np.mean(hd_arr)) if len(hd_arr) else float("nan"),
        "std": float(np.std(hd_arr)) if len(hd_arr) else float("nan"),
        "max": float(np.max(hd_arr)) if len(hd_arr) else float("nan"),
        "min": float(np.min(hd_arr)) if len(hd_arr) else float("nan"),
    }


# ===========================================================================
# METRIC 4: VERTEX REDUCTION RATIO
# ===========================================================================

def compute_vrr(
    original_gdf: gpd.GeoDataFrame,
    simplified_gdf: gpd.GeoDataFrame,
) -> Dict:
    """
    VRR = (V_original - V_simplified) / V_original
    """
    v_original = count_vertices_gdf(original_gdf)
    v_simplified = count_vertices_gdf(simplified_gdf)

    vrr = (
        (v_original - v_simplified) / v_original
        if v_original > 0
        else 0.0
    )

    return {
        "vrr": vrr,
        "v_original": v_original,
        "v_simplified": v_simplified,
    }


# ===========================================================================
# METRIC 5: TIMER
# ===========================================================================

class Timer:
    """Simple wall-clock timer."""

    def __init__(self):
        self._start = None
        self._end = None

    def start(self):
        self._start = time.perf_counter()

    def stop(self) -> float:
        self._end = time.perf_counter()
        return self._end - self._start

    def result(self) -> Dict:
        if self._start is None or self._end is None:
            return {"execution_time_s": float("nan")}

        return {"execution_time_s": self._end - self._start}


# ===========================================================================
# FULL EVALUATION
# ===========================================================================

def evaluate_pair(
    original_gdf: gpd.GeoDataFrame,
    sea_dp_gdf: gpd.GeoDataFrame,
    standard_dp_gdf: gpd.GeoDataFrame,
    sea_dp_time: float,
    std_dp_time: float,
    epsilon: float,
    verbose: bool = True,
    pairwise_sec: bool = False,
    name_col: Optional[str] = None,
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
    zone_buffer_m: Optional[float] = None,
    match_tol_m: float = 5.0,
) -> pd.DataFrame:
    """
    Evaluate SEA-DP and Standard DP using common metrics.

    If pairwise_sec=True, this uses compute_pair_sec() instead of compute_sec().
    This is recommended for isolated two-polygon shared-border tests.
    """
    if verbose:
        print(f"\n[Eval] epsilon = {epsilon}")
        print("[Eval] Computing metrics")

    orig_tec = compute_tec(original_gdf)

    tec_sea_raw = compute_tec(sea_dp_gdf)
    tec_std_raw = compute_tec(standard_dp_gdf)

    new_tec_sea = max(0, tec_sea_raw["tec"] - orig_tec["tec"])
    new_tec_std = max(0, tec_std_raw["tec"] - orig_tec["tec"])

    if pairwise_sec:
        if name_col is None or left_name is None or right_name is None or zone_buffer_m is None:
            raise ValueError(
                "pairwise_sec=True requires name_col, left_name, "
                "right_name, and zone_buffer_m."
            )

        sec_sea = compute_pair_sec(
            original_gdf,
            sea_dp_gdf,
            name_col=name_col,
            left_name=left_name,
            right_name=right_name,
            zone_buffer_m=zone_buffer_m,
            match_tol_m=match_tol_m,
        )

        sec_std = compute_pair_sec(
            original_gdf,
            standard_dp_gdf,
            name_col=name_col,
            left_name=left_name,
            right_name=right_name,
            zone_buffer_m=zone_buffer_m,
            match_tol_m=match_tol_m,
        )

    else:
        sec_sea = compute_sec(original_gdf, sea_dp_gdf)
        sec_std = compute_sec(original_gdf, standard_dp_gdf)

    hd_sea = compute_hausdorff(original_gdf, sea_dp_gdf)
    hd_std = compute_hausdorff(original_gdf, standard_dp_gdf)

    vrr_sea = compute_vrr(original_gdf, sea_dp_gdf)
    vrr_std = compute_vrr(original_gdf, standard_dp_gdf)

    rows = [
        {
            "algorithm": "SEA-DP",
            "epsilon": epsilon,
            "tec": tec_sea_raw["tec"],
            "new_tec": new_tec_sea,
            "n_gaps": tec_sea_raw["n_gaps"],
            "n_overlaps": tec_sea_raw["n_overlaps"],
            "n_invalid": tec_sea_raw["n_invalid"],
            "sec": sec_sea["sec"],
            "l_matched": sec_sea["l_matched"],
            "l_total": sec_sea["l_total"],
            "hausdorff_mean": hd_sea["mean"],
            "hausdorff_max": hd_sea["max"],
            "vrr": vrr_sea["vrr"],
            "v_original": vrr_sea["v_original"],
            "v_simplified": vrr_sea["v_simplified"],
            "execution_time_s": sea_dp_time,
        },
        {
            "algorithm": "Standard-DP",
            "epsilon": epsilon,
            "tec": tec_std_raw["tec"],
            "new_tec": new_tec_std,
            "n_gaps": tec_std_raw["n_gaps"],
            "n_overlaps": tec_std_raw["n_overlaps"],
            "n_invalid": tec_std_raw["n_invalid"],
            "sec": sec_std["sec"],
            "l_matched": sec_std["l_matched"],
            "l_total": sec_std["l_total"],
            "hausdorff_mean": hd_std["mean"],
            "hausdorff_max": hd_std["max"],
            "vrr": vrr_std["vrr"],
            "v_original": vrr_std["v_original"],
            "v_simplified": vrr_std["v_simplified"],
            "execution_time_s": std_dp_time,
        },
    ]

    df = pd.DataFrame(rows)

    if verbose:
        print(df.to_string(index=False))

    return df


# ===========================================================================
# STATISTICAL TESTS
# ===========================================================================

def statistical_tests(
    results_df: pd.DataFrame,
    metric: str = "sec",
    alpha: float = 0.05,
    verbose: bool = True,
) -> Dict:
    """
    Paired statistical comparison of SEA-DP vs Standard-DP on one metric.
    """
    sea_vals = results_df[results_df["algorithm"] == "SEA-DP"][metric].values
    std_vals = results_df[results_df["algorithm"] == "Standard-DP"][metric].values

    if len(sea_vals) < 3:
        return {"error": "Need at least 3 pairs for meaningful testing."}

    sw_sea = scipy_stats.shapiro(sea_vals)
    sw_std = scipy_stats.shapiro(std_vals)

    both_normal = (sw_sea.pvalue > alpha) and (sw_std.pvalue > alpha)

    ttest = scipy_stats.ttest_rel(sea_vals, std_vals)

    try:
        wilcox = scipy_stats.wilcoxon(sea_vals, std_vals)
        wstat = wilcox.statistic
        wpval = wilcox.pvalue
    except Exception:
        wstat = float("nan")
        wpval = float("nan")

    result = {
        "metric": metric,
        "alpha": alpha,
        "n_pairs": len(sea_vals),
        "sea_dp": {
            "mean": float(np.mean(sea_vals)),
            "std": float(np.std(sea_vals)),
            "min": float(np.min(sea_vals)),
            "max": float(np.max(sea_vals)),
        },
        "standard_dp": {
            "mean": float(np.mean(std_vals)),
            "std": float(np.std(std_vals)),
            "min": float(np.min(std_vals)),
            "max": float(np.max(std_vals)),
        },
        "shapiro_sea_dp": {
            "stat": float(sw_sea.statistic),
            "p": float(sw_sea.pvalue),
        },
        "shapiro_standard_dp": {
            "stat": float(sw_std.statistic),
            "p": float(sw_std.pvalue),
        },
        "both_normal": bool(both_normal),
        "paired_ttest": {
            "stat": float(ttest.statistic),
            "p": float(ttest.pvalue),
            "significant": bool(ttest.pvalue < alpha),
        },
        "wilcoxon": {
            "stat": float(wstat),
            "p": float(wpval),
            "significant": bool(wpval < alpha) if not math.isnan(wpval) else None,
        },
    }

    if verbose:
        print(f"\n{'=' * 55}")
        print(f" Statistical Test - {metric.upper()}")
        print(f"{'=' * 55}")
        print(f" SEA-DP      mean={result['sea_dp']['mean']:.4f}")
        print(f" Standard-DP mean={result['standard_dp']['mean']:.4f}")
        print(f" Paired t-test p={result['paired_ttest']['p']:.4f}")
        print(f" Wilcoxon    p={result['wilcoxon']['p']:.4f}")

    return result


def pearson_sec_vs_epsilon(
    results_df: pd.DataFrame,
    verbose: bool = True,
) -> Dict:
    """
    Pearson correlation between epsilon and SEA-DP SEC.
    """
    sub = results_df[results_df["algorithm"] == "SEA-DP"]

    eps = sub["epsilon"].values
    sec = sub["sec"].values

    if len(eps) < 2:
        return {"error": "Need at least 2 epsilon values."}

    r, p = scipy_stats.pearsonr(eps, sec)

    if verbose:
        print(f"\nPearson r epsilon vs SEC [SEA-DP]: r={r:.4f}, p={p:.4f}")

    return {
        "pearson_r": float(r),
        "p_value": float(p),
    }


# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot TEC, SEC, HD, VRR, and execution time across tolerances.
    """
    epsilons = sorted(results_df["epsilon"].unique())

    sea_rows = (
        results_df[results_df["algorithm"] == "SEA-DP"]
        .sort_values("epsilon")
    )

    std_rows = (
        results_df[results_df["algorithm"] == "Standard-DP"]
        .sort_values("epsilon")
    )

    x = np.arange(len(epsilons))
    width = 0.35

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "SEA-DP vs Standard Douglas-Peucker",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.bar(x - width / 2, sea_rows["new_tec"], width, label="SEA-DP")
    ax.bar(x + width / 2, std_rows["new_tec"], width, label="Standard DP")
    ax.set_title("New Topological Error Count")
    ax.set_xlabel("Tolerance")
    ax.set_ylabel("New TEC")
    ax.set_xticks(x)
    ax.set_xticklabels(epsilons, rotation=45)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    ax = axes[0, 1]
    ax.plot(epsilons, sea_rows["sec"], "o-", label="SEA-DP")
    ax.plot(epsilons, std_rows["sec"], "s--", label="Standard DP")
    ax.set_title("Shared-Edge Consistency")
    ax.set_xlabel("Tolerance")
    ax.set_ylabel("SEC")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.5)

    ax = axes[0, 2]
    ax.plot(epsilons, sea_rows["hausdorff_mean"], "o-", label="SEA-DP")
    ax.plot(epsilons, std_rows["hausdorff_mean"], "s--", label="Standard DP")
    ax.set_title("Hausdorff Distance")
    ax.set_xlabel("Tolerance")
    ax.set_ylabel("HD")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.5)

    ax = axes[1, 0]
    ax.bar(x - width / 2, sea_rows["vrr"], width, label="SEA-DP")
    ax.bar(x + width / 2, std_rows["vrr"], width, label="Standard DP")
    ax.set_title("Vertex Reduction Ratio")
    ax.set_xlabel("Tolerance")
    ax.set_ylabel("VRR")
    ax.set_xticks(x)
    ax.set_xticklabels(epsilons, rotation=45)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    ax = axes[1, 1]
    ax.bar(x - width / 2, sea_rows["execution_time_s"], width, label="SEA-DP")
    ax.bar(x + width / 2, std_rows["execution_time_s"], width, label="Standard DP")
    ax.set_title("Execution Time")
    ax.set_xlabel("Tolerance")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x)
    ax.set_xticklabels(epsilons, rotation=45)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    ax = axes[1, 2]
    ax.axis("off")

    summary_cols = [
        "algorithm",
        "epsilon",
        "new_tec",
        "sec",
        "hausdorff_mean",
        "vrr",
        "execution_time_s",
    ]

    summary = results_df[summary_cols].copy()
    summary = summary.round(4)

    table = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    ax.set_title("Summary Table")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved -> {save_path}")
    else:
        plt.show()


def plot_tec_breakdown(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot gaps, overlaps, and invalid counts across tolerances.
    """
    epsilons = sorted(results_df["epsilon"].unique())

    sea_rows = (
        results_df[results_df["algorithm"] == "SEA-DP"]
        .sort_values("epsilon")
    )

    std_rows = (
        results_df[results_df["algorithm"] == "Standard-DP"]
        .sort_values("epsilon")
    )

    x = np.arange(len(epsilons))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("TEC Breakdown", fontsize=12, fontweight="bold")

    for offset, rows, label in [
        (-width / 2, sea_rows, "SEA-DP"),
        (width / 2, std_rows, "Standard DP"),
    ]:
        bottom = np.zeros(len(epsilons))

        for component, col in [
            ("gaps", "n_gaps"),
            ("overlaps", "n_overlaps"),
            ("invalid", "n_invalid"),
        ]:
            vals = rows[col].values.astype(float)

            ax.bar(
                x + offset,
                vals,
                width,
                bottom=bottom,
                label=f"{label} - {component}",
            )

            bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(epsilons, rotation=45)
    ax.set_xlabel("Tolerance")
    ax.set_ylabel("Count")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved -> {save_path}")
    else:
        plt.show()


# ===========================================================================
# BATCH EVALUATION
# ===========================================================================

def batch_evaluate(
    input_path: str,
    epsilons: Sequence[float],
    tol: float = 1e-6,
    verbose: bool = True,
    pairwise_sec: bool = False,
    name_col: Optional[str] = None,
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
    zone_buffer_factor: float = 0.25,
    match_tol_m: float = 5.0,
) -> pd.DataFrame:
    """
    Run SEA-DP and Standard-DP across multiple epsilon values.
    """
    from sea_dp import sea_dp_simplify, standard_dp_simplify

    original_gdf = gpd.read_file(input_path)

    all_rows = []

    for eps in epsilons:
        if verbose:
            print(f"\n{'-' * 60}")
            print(f"epsilon = {eps}")

        t0 = time.perf_counter()
        sea_result, _ = sea_dp_simplify(
            original_gdf.copy(),
            epsilon=eps,
            tol=tol,
            verbose=False,
        )
        sea_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        std_result, _ = standard_dp_simplify(
            original_gdf.copy(),
            epsilon=eps,
        )
        std_time = time.perf_counter() - t0

        zone_buffer_m = eps * zone_buffer_factor if pairwise_sec else None

        df_eps = evaluate_pair(
            original_gdf=original_gdf,
            sea_dp_gdf=sea_result,
            standard_dp_gdf=std_result,
            sea_dp_time=sea_time,
            std_dp_time=std_time,
            epsilon=eps,
            verbose=verbose,
            pairwise_sec=pairwise_sec,
            name_col=name_col,
            left_name=left_name,
            right_name=right_name,
            zone_buffer_m=zone_buffer_m,
            match_tol_m=match_tol_m,
        )

        all_rows.append(df_eps)

    return pd.concat(all_rows, ignore_index=True)


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SEA-DP Evaluation Metrics"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input Shapefile or GeoJSON",
    )

    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=[0.002, 0.005, 0.01, 0.02, 0.05],
        help="Simplification tolerances",
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="SEA-DP edge matching tolerance",
    )

    parser.add_argument(
        "--output",
        default="evaluation_results.csv",
        help="Output CSV path",
    )

    parser.add_argument(
        "--pairwise-sec",
        action="store_true",
        help="Use pairwise SEC for two-polygon tests",
    )

    parser.add_argument(
        "--name-col",
        default=None,
        help="Name column for pairwise SEC",
    )

    parser.add_argument(
        "--left-name",
        default=None,
        help="Left polygon name for pairwise SEC",
    )

    parser.add_argument(
        "--right-name",
        default=None,
        help="Right polygon name for pairwise SEC",
    )

    parser.add_argument(
        "--zone-buffer-factor",
        type=float,
        default=0.25,
        help="SEC window as a fraction of epsilon",
    )

    parser.add_argument(
        "--match-tol-m",
        type=float,
        default=5.0,
        help="Boundary match tolerance in CRS units",
    )

    args = parser.parse_args()

    results = batch_evaluate(
        input_path=args.input,
        epsilons=args.epsilons,
        tol=args.tol,
        verbose=True,
        pairwise_sec=args.pairwise_sec,
        name_col=args.name_col,
        left_name=args.left_name,
        right_name=args.right_name,
        zone_buffer_factor=args.zone_buffer_factor,
        match_tol_m=args.match_tol_m,
    )

    results.to_csv(args.output, index=False)

    print(f"\nSaved results -> {args.output}")

    plot_comparison(
        results,
        save_path=args.output.replace(".csv", "_comparison.png"),
    )

    plot_tec_breakdown(
        results,
        save_path=args.output.replace(".csv", "_tec_breakdown.png"),
    )