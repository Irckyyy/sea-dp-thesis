"""
sea_dp.py
=========
Shared-Edge-Aware Douglas-Peucker (SEA-DP)

Topology-preserving polygon simplification for shared boundaries.

Main design:
- Shared exterior boundaries are detected, assembled into arcs, simplified once,
  and reused by adjacent polygons.
- Interior rings / holes are NOT processed as shared-edge arcs.
- Holes are preserved separately and safely reinserted during reconstruction.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Dict, FrozenSet, List, Tuple

import geopandas as gpd
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid


Coords = List[Tuple[float, float]]
SegKey = Tuple[Tuple[float, float], Tuple[float, float]]
ArcKey = Tuple[SegKey, ...]
OwnerSet = FrozenSet[int]


# ===========================================================================
# 1. DOUGLAS-PEUCKER CORE
# ===========================================================================

def _perpendicular_distance(point, start, end) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end

    dx = ex - sx
    dy = ey - sy

    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - sx, py - sy)

    t = ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)

    return math.hypot(px - (sx + t * dx), py - (sy + t * dy))


def douglas_peucker(coords: Coords, epsilon: float) -> Coords:
    if len(coords) < 3:
        return list(coords)

    start = coords[0]
    end = coords[-1]

    max_dist = 0.0
    max_idx = 0

    for i in range(1, len(coords) - 1):
        d = _perpendicular_distance(coords[i], start, end)

        if d > max_dist:
            max_dist = d
            max_idx = i

    if max_dist > epsilon:
        left = douglas_peucker(coords[:max_idx + 1], epsilon)
        right = douglas_peucker(coords[max_idx:], epsilon)

        return left[:-1] + right

    return [start, end]


def simplify_chain(coords: Coords, epsilon: float) -> Coords:
    if len(coords) < 3:
        return list(coords)

    if coords[0] == coords[-1]:
        try:
            line = LineString(coords)
            simp = line.simplify(epsilon, preserve_topology=True)
            out = list(simp.coords)

            if len(out) < 4:
                return list(coords)

            if out[0] != out[-1]:
                out.append(out[0])

            return out

        except Exception:
            return list(coords)

    return douglas_peucker(coords, epsilon)


# ===========================================================================
# 2. COORDINATE UTILITIES
# ===========================================================================

def _rnd(v: float, prec: int) -> float:
    return round(v, prec)


def _rnd_pt(p, prec: int):
    return (_rnd(p[0], prec), _rnd(p[1], prec))


def _same_pt(a, b, prec: int) -> bool:
    return _rnd_pt(a, prec) == _rnd_pt(b, prec)


def _seg_key(a, b, prec: int) -> SegKey:
    ra = _rnd_pt(a, prec)
    rb = _rnd_pt(b, prec)

    return (ra, rb) if ra <= rb else (rb, ra)


def _run_key(seg_keys: List[SegKey]) -> ArcKey:
    return tuple(sorted(seg_keys))


def _extract_rings(geom: Polygon) -> List[Coords]:
    """
    Return exterior ring only for SEA-DP arc assembly.

    Interior rings / holes are preserved later during reconstruction.
    They should not be processed as shared-edge arcs because they can collapse,
    invert, or become fake polygon pieces after repair.
    """
    return [list(geom.exterior.coords)]


def _polygon_parts(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "Polygon":
        return [geom]

    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)

    if geom.geom_type == "GeometryCollection":
        parts = []

        for g in geom.geoms:
            parts.extend(_polygon_parts(g))

        return parts

    return []


def _to_polygonal(geom):
    parts = _polygon_parts(geom)

    if not parts:
        return None

    if len(parts) == 1:
        return parts[0]

    return MultiPolygon(parts)


def _count_vertices(geom) -> int:
    if geom is None or geom.is_empty:
        return 0

    if geom.geom_type == "Polygon":
        return (
            len(list(geom.exterior.coords))
            + sum(len(list(r.coords)) for r in geom.interiors)
        )

    if geom.geom_type == "MultiPolygon":
        return sum(_count_vertices(p) for p in geom.geoms)

    return 0


# ===========================================================================
# 3. PREPROCESSING
# ===========================================================================

def _preprocess(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    def clean(g):
        if g is None or g.is_empty:
            return None

        if not g.is_valid:
            g = make_valid(g)

        return _to_polygonal(g)

    gdf["geometry"] = gdf["geometry"].apply(clean)
    gdf = gdf[~(gdf["geometry"].isna() | gdf["geometry"].is_empty)].copy()
    gdf = gdf.reset_index(drop=True)

    return gdf


# ===========================================================================
# 4. SEGMENT MAP
# ===========================================================================

def _build_segment_map(
    gdf: gpd.GeoDataFrame,
    prec: int,
) -> Tuple[
    Dict[SegKey, List[int]],
    Dict[int, List[List[Tuple[SegKey, Coords]]]],
]:
    seg_owners: Dict[SegKey, List[int]] = defaultdict(list)
    poly_seg_rings: Dict[int, List[List[Tuple[SegKey, Coords]]]] = {}

    for idx, row in gdf.iterrows():
        geom = row.geometry
        poly_seg_rings[idx] = []

        for part in _polygon_parts(geom):
            for ring_coords in _extract_rings(part):
                pts = list(ring_coords)

                if len(pts) < 4:
                    continue

                if pts[0] == pts[-1]:
                    pts = pts[:-1]

                n = len(pts)
                ring_segs = []

                for i in range(n):
                    a = pts[i]
                    b = pts[(i + 1) % n]

                    key = _seg_key(a, b, prec)

                    ring_segs.append((key, [a, b]))
                    seg_owners[key].append(idx)

                poly_seg_rings[idx].append(ring_segs)

    return seg_owners, poly_seg_rings


def _owners_frozenset(key: SegKey, seg_owners: Dict[SegKey, List[int]]) -> OwnerSet:
    return frozenset(seg_owners.get(key, []))


def _is_shared_seg(key: SegKey, seg_owners: Dict[SegKey, List[int]]) -> bool:
    return len(set(seg_owners.get(key, []))) > 1


def _seg_state(key: SegKey, seg_owners: Dict[SegKey, List[int]]):
    is_shared = _is_shared_seg(key, seg_owners)
    owners = _owners_frozenset(key, seg_owners) if is_shared else None

    return is_shared, owners


# ===========================================================================
# 5. ARC ASSEMBLY
# ===========================================================================

def assemble_arcs(
    poly_seg_rings: Dict[int, List[List[Tuple[SegKey, Coords]]]],
    seg_owners: Dict[SegKey, List[int]],
    prec: int,
) -> Tuple[Dict[ArcKey, Coords], Dict[int, List[List]]]:

    arc_coords: Dict[ArcKey, Coords] = {}
    poly_ring_arcs: Dict[int, List[List]] = {}

    for poly_idx, rings in poly_seg_rings.items():
        poly_ring_arcs[poly_idx] = []

        for ring_segs in rings:
            n = len(ring_segs)

            if n == 0:
                poly_ring_arcs[poly_idx].append([])
                continue

            states = [_seg_state(k, seg_owners) for k, _ in ring_segs]

            start_idx = 0

            for i in range(n):
                if states[i] != states[i - 1]:
                    start_idx = i
                    break

            ring_segs = ring_segs[start_idx:] + ring_segs[:start_idx]
            states = states[start_idx:] + states[:start_idx]

            items = []

            first_key, first_coords = ring_segs[0]
            run_coords = [first_coords[0], first_coords[1]]
            run_keys = [first_key]
            run_shared, run_owners = states[0]

            def flush_run():
                nonlocal run_coords, run_keys, run_shared

                akey = _run_key(run_keys)

                if run_shared and akey not in arc_coords:
                    arc_coords[akey] = list(run_coords)

                items.append((akey, list(run_coords), run_shared))

            for i in range(1, n):
                seg_key_i, seg_coords_i = ring_segs[i]
                seg_shared_i, seg_owners_i = states[i]

                same_label = seg_shared_i == run_shared
                same_owners = seg_owners_i == run_owners if run_shared else True
                can_extend = same_label and same_owners

                if can_extend:
                    run_coords.append(seg_coords_i[1])
                    run_keys.append(seg_key_i)
                else:
                    flush_run()

                    run_coords = [seg_coords_i[0], seg_coords_i[1]]
                    run_keys = [seg_key_i]
                    run_shared = seg_shared_i
                    run_owners = seg_owners_i

            flush_run()

            poly_ring_arcs[poly_idx].append(items)

    return arc_coords, poly_ring_arcs


# ===========================================================================
# 6. SIMPLIFICATION
# ===========================================================================

def simplify_arcs(
    arc_coords: Dict[ArcKey, Coords],
    poly_ring_arcs: Dict[int, List[List]],
    epsilon: float,
    prec: int,
) -> Dict[ArcKey, Coords]:

    simplified_arcs: Dict[ArcKey, Coords] = {}

    for akey, coords in arc_coords.items():
        simp = simplify_chain(coords, epsilon)

        if len(simp) < 2:
            simp = [coords[0], coords[-1]]

        simplified_arcs[akey] = simp

    return simplified_arcs


# ===========================================================================
# 7. RECONSTRUCTION
# ===========================================================================

def _build_directed_arc(
    akey: ArcKey,
    simp_arcs: Dict[ArcKey, Coords],
    orig_directed: Coords,
    prec: int,
) -> Coords:

    simp = simp_arcs.get(akey)

    if simp is None or len(simp) < 2:
        return list(orig_directed)

    orig_start = _rnd_pt(orig_directed[0], prec)
    simp_start = _rnd_pt(simp[0], prec)
    simp_end = _rnd_pt(simp[-1], prec)

    if orig_start == simp_start:
        return list(simp)

    if orig_start == simp_end:
        return list(reversed(simp))

    return list(orig_directed)


def _append_chain(
    target: Coords,
    chain: Coords,
    fallback: Coords,
    prec: int,
) -> None:

    if not chain:
        return

    if not target:
        target.extend(chain)
        return

    last = _rnd_pt(target[-1], prec)
    chain_start = _rnd_pt(chain[0], prec)
    chain_end = _rnd_pt(chain[-1], prec)

    if last == chain_start:
        target.extend(chain[1:])
        return

    if last == chain_end:
        rev = list(reversed(chain))
        target.extend(rev[1:])
        return

    fb = list(fallback)

    if not fb:
        return

    fb_start = _rnd_pt(fb[0], prec)
    fb_end = _rnd_pt(fb[-1], prec)

    if last == fb_start:
        target.extend(fb[1:])
    elif last == fb_end:
        fb.reverse()
        target.extend(fb[1:])
    else:
        target.extend(fb)


def _clean_ring(coords: Coords, prec: int) -> Coords | None:
    if not coords:
        return None

    pts = []

    for p in coords:
        if not pts or not _same_pt(pts[-1], p, prec):
            pts.append(p)

    if len(pts) < 3:
        return None

    if not _same_pt(pts[0], pts[-1], prec):
        pts.append(pts[0])

    if len(pts) < 4:
        return None

    try:
        test = Polygon(pts)

        if test.is_empty or test.area <= 0:
            return None

    except Exception:
        return None

    return pts


def _safe_simplify_holes(
    original_part: Polygon,
    new_exterior: Coords,
    epsilon: float,
    prec: int,
) -> List[Coords]:
    """
    Preserve original holes as holes.

    For complex lake / island / interior-ring cases such as Laguna de Bay,
    holes are kept from the original geometry instead of being simplified.
    This prevents hole boundaries from collapsing, spiking, or becoming
    fake polygon pieces.
    """
    holes = []

    try:
        shell = Polygon(new_exterior)

        if not shell.is_valid:
            shell = shell.buffer(0)

        if shell is None or shell.is_empty:
            return []

        for interior in original_part.interiors:
            original_hole = list(interior.coords)

            if len(original_hole) < 4:
                continue

            original_hole = _clean_ring(original_hole, prec)

            if original_hole is None:
                continue

            try:
                hole_poly = Polygon(original_hole)

                if (
                    hole_poly.is_valid
                    and not hole_poly.is_empty
                    and hole_poly.area > 0
                    and shell.covers(hole_poly.representative_point())
                ):
                    holes.append(original_hole)

            except Exception:
                continue

    except Exception:
        return []

    return holes


def _largest_polygon_only(geom, fallback=None):
    """
    Extract the largest polygon from repaired geometry.

    This prevents fake fragments from holes/lakes from becoming extra polygons.
    """
    if geom is None or geom.is_empty:
        return fallback

    if geom.geom_type == "Polygon":
        return geom

    if geom.geom_type == "MultiPolygon":
        parts = [p for p in geom.geoms if p is not None and not p.is_empty]

        if not parts:
            return fallback

        return max(parts, key=lambda p: p.area)

    if geom.geom_type == "GeometryCollection":
        parts = []

        for g in geom.geoms:
            if g.geom_type == "Polygon":
                parts.append(g)
            elif g.geom_type == "MultiPolygon":
                parts.extend(list(g.geoms))

        parts = [p for p in parts if p is not None and not p.is_empty]

        if not parts:
            return fallback

        return max(parts, key=lambda p: p.area)

    return fallback


def reconstruct_polygons(
    gdf: gpd.GeoDataFrame,
    poly_seg_rings: Dict[int, List[List[Tuple]]],
    poly_ring_arcs: Dict[int, List[List]],
    simplified_arcs: Dict[ArcKey, Coords],
    seg_owners: Dict[SegKey, List[int]],
    epsilon: float,
    prec: int,
) -> List:
    """
    Rebuild each polygon.

    SEA-DP reconstruction is applied to exterior/shared boundaries only.
    Holes are preserved separately as holes.
    Holes are never allowed to become independent polygon pieces.
    """
    new_geoms = []

    for idx, row in gdf.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            new_geoms.append(None)
            continue

        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        ring_arc_groups = poly_ring_arcs.get(idx, [])
        arc_iter = iter(ring_arc_groups)

        new_parts = []

        for part in parts:
            try:
                exterior_items = next(arc_iter)
            except StopIteration:
                new_parts.append(part)
                continue

            new_exterior: Coords = []

            for akey, orig_directed, is_shared in exterior_items:
                if is_shared:
                    simp = _build_directed_arc(
                        akey,
                        simplified_arcs,
                        orig_directed,
                        prec,
                    )
                else:
                    outer_epsilon = min(epsilon * 0.25, 500.0)
                    simp = simplify_chain(orig_directed, epsilon)

                if len(simp) < 2:
                    simp = [orig_directed[0], orig_directed[-1]]

                _append_chain(new_exterior, simp, orig_directed, prec)

            new_exterior = _clean_ring(new_exterior, prec)

            if new_exterior is None:
                new_parts.append(part)
                continue

            holes = _safe_simplify_holes(
                original_part=part,
                new_exterior=new_exterior,
                epsilon=epsilon,
                prec=prec,
            )

            try:
                new_poly = Polygon(new_exterior, holes)

                if not new_poly.is_valid:
                    repaired = new_poly.buffer(0)
                else:
                    repaired = new_poly

                repaired = _largest_polygon_only(repaired, fallback=part)

                if repaired is None or repaired.is_empty:
                    new_parts.append(part)
                else:
                    new_parts.append(repaired)

            except Exception:
                new_parts.append(part)

        cleaned_parts = []

        for p in new_parts:
            if p is None or p.is_empty:
                continue

            if p.geom_type == "Polygon":
                cleaned_parts.append(p)
            elif p.geom_type == "MultiPolygon":
                cleaned_parts.extend(list(p.geoms))

        if not cleaned_parts:
            new_geoms.append(None)
        elif len(cleaned_parts) == 1:
            new_geoms.append(cleaned_parts[0])
        else:
            new_geoms.append(MultiPolygon(cleaned_parts))

    return new_geoms


# ===========================================================================
# 8. TOPOLOGY VALIDATION
# ===========================================================================

def topology_validation(
    gdf: gpd.GeoDataFrame,
    area_tol: float = 1e-10,
) -> Dict:

    work = gdf.copy()
    work = work[~(work["geometry"].isna() | work["geometry"].is_empty)].copy()
    work = work.reset_index(drop=True)

    geoms = list(work["geometry"])

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


# ===========================================================================
# 9. MAIN SEA-DP
# ===========================================================================

def sea_dp_simplify(
    gdf: gpd.GeoDataFrame,
    epsilon: float,
    tol: float = 1e-6,
    verbose: bool = True,
) -> Tuple[gpd.GeoDataFrame, Dict]:

    t0 = time.perf_counter()

    if tol <= 0:
        raise ValueError("tol must be greater than 0")

    prec = max(0, -int(math.floor(math.log10(tol))))

    if verbose:
        print("[SEA-DP] Step 1/6 - Preprocessing")

    working_gdf = _preprocess(gdf)

    n_polygons = len(working_gdf)
    v_original = sum(_count_vertices(g) for g in working_gdf["geometry"])

    if verbose:
        print(f"           {n_polygons} polygons, {v_original} vertices")

    if verbose:
        print("[SEA-DP] Step 2/6 - Building segment map")

    seg_owners, poly_seg_rings = _build_segment_map(working_gdf, prec)

    n_segs_total = len(seg_owners)
    n_segs_shared = sum(
        1 for owners in seg_owners.values()
        if len(set(owners)) > 1
    )
    n_segs_outer = n_segs_total - n_segs_shared

    if verbose:
        print(f"           Total segments : {n_segs_total}")
        print(f"           Shared segments: {n_segs_shared}")
        print(f"           Outer segments : {n_segs_outer}")

    if verbose:
        print("[SEA-DP] Step 3/6 - Assembling arcs")

    arc_coords, poly_ring_arcs = assemble_arcs(
        poly_seg_rings,
        seg_owners,
        prec,
    )

    n_arcs = len(arc_coords)
    arc_lengths = [len(c) for c in arc_coords.values()]
    avg_arc_len = sum(arc_lengths) / n_arcs if n_arcs else 0.0

    if verbose:
        print(f"           Shared arcs assembled: {n_arcs}")
        print(f"           Avg vertices per arc : {avg_arc_len:.1f}")

    if verbose:
        print("[SEA-DP] Step 4/6 - Simplifying shared arcs")

    simplified_arcs = simplify_arcs(
        arc_coords,
        poly_ring_arcs,
        epsilon,
        prec,
    )

    v_arcs_before = sum(len(c) for c in arc_coords.values())
    v_arcs_after = sum(len(c) for c in simplified_arcs.values())

    if verbose and v_arcs_before > 0:
        print(
            f"           Arc vertex reduction: "
            f"{(v_arcs_before - v_arcs_after) / v_arcs_before:.1%}"
        )

    if verbose:
        print("[SEA-DP] Step 5/6 - Reconstructing polygons")

    new_geoms = reconstruct_polygons(
        working_gdf,
        poly_seg_rings,
        poly_ring_arcs,
        simplified_arcs,
        seg_owners,
        epsilon,
        prec,
    )

    result_gdf = working_gdf.copy()
    result_gdf["geometry"] = new_geoms
    result_gdf.crs = gdf.crs

    if verbose:
        print("[SEA-DP] Step 6/6 - Topology validation")

    tec_results = topology_validation(result_gdf)

    v_simplified = sum(_count_vertices(g) for g in result_gdf["geometry"])
    elapsed = time.perf_counter() - t0

    stats = {
        "n_polygons": n_polygons,
        "n_segments_total": n_segs_total,
        "n_segments_shared": n_segs_shared,
        "n_segments_outer": n_segs_outer,
        "n_arcs_assembled": n_arcs,
        "v_original": v_original,
        "v_simplified": v_simplified,
        "vrr": (
            (v_original - v_simplified) / v_original
            if v_original > 0
            else 0.0
        ),
        "epsilon": epsilon,
        "execution_time_s": elapsed,
        "n_edges_shared": n_segs_shared,
        "n_edges_outer": n_segs_outer,
        **tec_results,
    }

    if verbose:
        print(f"\n[SEA-DP] Done in {elapsed:.3f}s")
        print(f"          TEC      : {stats['tec']}")
        print(f"          Gaps     : {stats['n_gaps']}")
        print(f"          Overlaps : {stats['n_overlaps']}")
        print(f"          Invalid  : {stats['n_invalid']}")
        print(
            f"          Vertices : {v_original} -> {v_simplified} "
            f"(VRR {stats['vrr']:.1%})"
        )

    return result_gdf, stats


# ===========================================================================
# 10. STANDARD DP BASELINE
# ===========================================================================

def standard_dp_simplify(
    gdf: gpd.GeoDataFrame,
    epsilon: float,
) -> Tuple[gpd.GeoDataFrame, Dict]:

    t0 = time.perf_counter()

    result = gdf.copy()

    result["geometry"] = result["geometry"].apply(
        lambda g: (
            g.simplify(epsilon, preserve_topology=True)
            if g is not None and not g.is_empty
            else g
        )
    )

    elapsed = time.perf_counter() - t0
    tec_results = topology_validation(result)

    return result, {
        "epsilon": epsilon,
        "execution_time_s": elapsed,
        **tec_results,
    }


# ===========================================================================
# 11. CLI
# ===========================================================================

def run_sea_dp(
    input_path: str,
    output_path: str,
    epsilon: float,
    tol: float = 1e-6,
    verbose: bool = True,
) -> Dict:

    gdf = gpd.read_file(input_path)

    result, stats = sea_dp_simplify(
        gdf,
        epsilon=epsilon,
        tol=tol,
        verbose=verbose,
    )

    result.to_file(output_path)

    if verbose:
        print(f"[SEA-DP] Output written -> {output_path}")

    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python sea_dp.py <input_shapefile> <epsilon> [tol]")
        print("Example degrees: python sea_dp.py ph.shp 0.005")
        print("Example meters : python sea_dp.py ph.shp 2000 0.01")
        sys.exit(0)

    input_file = sys.argv[1]
    eps = float(sys.argv[2])
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6

    output_file = input_file.replace(".shp", f"_seadp_e{eps}.shp")

    stats = run_sea_dp(
        input_file,
        output_file,
        epsilon=eps,
        tol=tol,
        verbose=True,
    )

    print("\n=== SEA-DP Summary ===")

    for k, v in stats.items():
        print(f"  {k:<25}: {v}")