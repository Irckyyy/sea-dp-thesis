import geopandas as gpd
import shapely
from shapely.geometry import Point, LineString, Polygon
from sklearn.preprocessing import MinMaxScaler

gdf = gpd.read_file("data/raw/GADM/philippines/gadm41_PHL_0.shp")

if gdf.crs is None:
    print("Warning: No CRS found. Assuming WGS84.")
    gdf = gdf.set_crs("EPSG:4326")
else:
    print(f"Original CRS: {gdf.crs}")