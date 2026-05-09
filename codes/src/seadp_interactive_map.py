"""
seadp_interactive_map.py
========================
Generates a self-contained interactive HTML map of the SEA-DP datasets.
Open the output HTML file in any browser — no installation needed on viewer's end.

Usage:
    python seadp_interactive_map.py

Output:
    seadp_dataset_explorer.html
"""

import geopandas as gpd
import folium
from folium.plugins import GroupedLayerControl
import json

# ── UPDATE THESE PATHS ──────────────────────────────────────────────────────
PATH_PH = 'data/raw/GADM/philippines/gadm41_PHL_1.shp'
PATH_US = 'data/raw/natural_earth/USA/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp'
PATH_INTL = 'data/raw/natural_earth/Armenia-Azerbaijan/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp'
# ────────────────────────────────────────────────────────────────────────────

OUTPUT_HTML = 'seadp_dataset_explorer.html'

# All pairs per tier
PAIRS = {
    'low': {
        'label': 'Low Complexity — US States',
        'gdf_path': PATH_US,
        'name_col': 'name',
        'color_a': '#2980B9',
        'color_b': '#1ABC9C',
        'pairs': [
            ('California', 'Nevada'),
            ('Arizona', 'New Mexico'),
            ('Oregon', 'Washington'),
            ('Texas', 'Oklahoma'),
        ]
    },
    'moderate': {
        'label': 'Moderate Complexity — Philippine Provinces',
        'gdf_path': PATH_PH,
        'name_col': 'NAME_1',
        'color_a': '#27AE60',
        'color_b': '#F39C12',
        'pairs': [
            ('Laguna', 'Quezon'),
            ('Rizal', 'Laguna'),
            ('Cavite', 'Laguna'),
            ('Rizal', 'Metropolitan Manila'),
        ]
    },
    'high': {
        'label': 'High Complexity — International Boundaries',
        'gdf_path': PATH_INTL,
        'name_col': 'NAME_EN',
        'color_a': '#E74C3C',
        'color_b': '#9B59B6',
        'pairs': [
            ('Armenia', 'Azerbaijan'),
            ('United Arab Emirates', 'Oman'),
            ('Kyrgyzstan', 'Tajikistan'),
            ('Uzbekistan', 'Kyrgyzstan'),
        ]
    }
}


def count_vertices(geom):
    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type == 'Polygon':
        return len(list(geom.exterior.coords)) + sum(len(list(r.coords)) for r in geom.interiors)
    if geom.geom_type == 'MultiPolygon':
        return sum(count_vertices(p) for p in geom.geoms)
    return 0


def make_popup(name, geom, tier):
    verts = count_vertices(geom)
    holes = (len(list(geom.interiors)) if geom.geom_type == 'Polygon'
             else sum(len(list(p.interiors)) for p in geom.geoms))
    parts = len(geom.geoms) if geom.geom_type == 'MultiPolygon' else 1
    valid = '✅ Valid' if geom.is_valid else '⚠️ Invalid (repaired in preprocessing)'
    bounds = geom.bounds

    html = f"""
    <div style="font-family:Arial,sans-serif;min-width:240px;font-size:13px">
      <div style="background:#1F4E79;color:white;padding:8px 12px;border-radius:6px 6px 0 0;
                  font-weight:bold;font-size:14px">{name}</div>
      <div style="padding:10px 12px;background:#f9f9f9;border-radius:0 0 6px 6px">
        <table style="width:100%;border-collapse:collapse">
          <tr><td style="color:#666;padding:3px 0">Tier</td>
              <td style="font-weight:500;padding:3px 0">{tier}</td></tr>
          <tr><td style="color:#666;padding:3px 0">Geometry</td>
              <td style="font-weight:500;padding:3px 0">{geom.geom_type}</td></tr>
          <tr><td style="color:#666;padding:3px 0">Parts</td>
              <td style="font-weight:500;padding:3px 0">{parts}</td></tr>
          <tr><td style="color:#666;padding:3px 0">Vertices</td>
              <td style="font-weight:500;padding:3px 0">{verts:,}</td></tr>
          <tr><td style="color:#666;padding:3px 0">Interior rings</td>
              <td style="font-weight:500;padding:3px 0">{holes}</td></tr>
          <tr><td style="color:#666;padding:3px 0">Validity</td>
              <td style="font-weight:500;padding:3px 0">{valid}</td></tr>
        </table>
        <div style="margin-top:8px;font-size:11px;color:#888">
          Bounds: ({bounds[0]:.3f}, {bounds[1]:.3f}) → ({bounds[2]:.3f}, {bounds[3]:.3f})
        </div>
      </div>
    </div>
    """
    return folium.Popup(html, max_width=300)


def add_shared_boundary(m, geom_a, geom_b, name_a, name_b, layer_group):
    try:
        geom_a_wgs = geom_a
        geom_b_wgs = geom_b
        shared = geom_a_wgs.boundary.intersection(geom_b_wgs.boundary)
        if shared.is_empty:
            return
        shared_len = shared.length

        # Convert to GeoJSON for Folium
        shared_gdf = gpd.GeoSeries([shared], crs='EPSG:4326')
        shared_json = json.loads(shared_gdf.to_json())

        popup_html = f"""
        <div style="font-family:Arial,sans-serif;font-size:13px;min-width:200px">
          <div style="background:#FFD700;color:#333;padding:6px 10px;border-radius:4px;
                      font-weight:bold">⚡ Shared Boundary</div>
          <div style="padding:8px 10px;background:#fffbe6;border-radius:0 0 4px 4px">
            <b>{name_a}</b> ↔ <b>{name_b}</b><br>
            <span style="color:#666;font-size:12px">
              Length: {shared_len:.4f}° (~{shared_len * 111:.1f} km)<br>
              This is the boundary SEA-DP must simplify once<br>
              and reuse for both polygons.
            </span>
          </div>
        </div>
        """

        folium.GeoJson(
            shared_json,
            style_function=lambda x: {
                'color': '#FFD700',
                'weight': 4,
                'opacity': 1.0,
            },
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f'Shared: {name_a} ↔ {name_b}',
        ).add_to(layer_group)

    except Exception as e:
        print(f'  [!] Could not compute shared boundary for {name_a}/{name_b}: {e}')


def build_map():
    print('Building interactive map...')

    m = folium.Map(
        location=[20, 20],
        zoom_start=3,
        tiles='CartoDB positron',
        control_scale=True,
    )

    # Custom title overlay
    title_html = """
    <div style="position:fixed;top:12px;left:60px;z-index:9999;background:white;
                padding:12px 18px;border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,0.15);
                font-family:Arial,sans-serif;max-width:360px">
      <div style="font-size:14px;font-weight:bold;color:#1F4E79">
        SEA-DP Dataset Explorer
      </div>
      <div style="font-size:11px;color:#666;margin-top:4px">
        Raw geospatial data — before any simplification.<br>
        <b style="color:#FFD700">Yellow lines</b> = shared boundaries SEA-DP targets.<br>
        Click any polygon or boundary for details.
      </div>
      <div style="margin-top:8px;font-size:11px">
        <span style="background:#2980B9;color:white;padding:2px 6px;border-radius:3px;margin-right:4px">Low</span>
        US States
        &nbsp;
        <span style="background:#27AE60;color:white;padding:2px 6px;border-radius:3px;margin-right:4px">Moderate</span>
        PH Provinces
        &nbsp;
        <span style="background:#E74C3C;color:white;padding:2px 6px;border-radius:3px;margin-right:4px">High</span>
        International
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    layer_groups = {}

    for tier_key, tier_info in PAIRS.items():
        print(f'  Processing {tier_info["label"]}...')

        try:
            gdf = gpd.read_file(tier_info['gdf_path'])
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
        except Exception as e:
            print(f'  [!] Could not load {tier_info["gdf_path"]}: {e}')
            continue

        name_col  = tier_info['name_col']
        label     = tier_info['label']
        color_a   = tier_info['color_a']
        color_b   = tier_info['color_b']

        # One FeatureGroup per tier
        tier_group = folium.FeatureGroup(name=f'📍 {label}', show=True)
        shared_group = folium.FeatureGroup(name=f'⚡ Shared Boundaries — {label.split("—")[0].strip()}', show=True)

        colors_cycle = [color_a, color_b]
        seen = {}  # name → color already assigned

        for left, right in tier_info['pairs']:
            left_row  = gdf[gdf[name_col] == left]
            right_row = gdf[gdf[name_col] == right]

            for name, row, default_color in [(left, left_row, color_a), (right, right_row, color_b)]:
                if name in seen:
                    continue
                if row.empty:
                    print(f'    [!] {name} not found in dataset')
                    continue

                geom = row.iloc[0].geometry
                if geom is None or geom.is_empty:
                    continue

                seen[name] = default_color

                geom_json = json.loads(gpd.GeoSeries([geom], crs='EPSG:4326').to_json())

                folium.GeoJson(
                    geom_json,
                    style_function=lambda x, c=default_color: {
                        'fillColor': c,
                        'color': '#333333',
                        'weight': 1.2,
                        'fillOpacity': 0.45,
                    },
                    highlight_function=lambda x, c=default_color: {
                        'fillColor': c,
                        'color': '#111111',
                        'weight': 2.5,
                        'fillOpacity': 0.7,
                    },
                    tooltip=folium.Tooltip(
                        f'<b>{name}</b><br><span style="font-size:11px;color:#666">'
                        f'{tier_info["label"].split("—")[1].strip()} | '
                        f'{count_vertices(geom):,} vertices</span>',
                        sticky=False
                    ),
                    popup=make_popup(name, geom, label),
                ).add_to(tier_group)

            # Shared boundary
            if not left_row.empty and not right_row.empty:
                add_shared_boundary(
                    m,
                    left_row.iloc[0].geometry,
                    right_row.iloc[0].geometry,
                    left, right,
                    shared_group
                )

        tier_group.add_to(m)
        shared_group.add_to(m)
        layer_groups[tier_key] = (tier_group, shared_group)

    folium.LayerControl(collapsed=False, position='topright').add_to(m)
    m.save(OUTPUT_HTML)
    print(f'\nSaved -> {OUTPUT_HTML}')
    print('Open this file in any browser — no internet required!')


if __name__ == '__main__':
    build_map()
