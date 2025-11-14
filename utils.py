from shapely.geometry import Polygon, mapping
import geopandas as gpd
import numpy as np

def mask_to_geojson(mask, bbox=(150.3, 69.1, 150.4, 69.2)):
    # bbox = (min_lon, min_lat, max_lon, max_lat)
    h, w = mask.shape
    geojson_features = []
    # простой вариант: bounding boxes для connected components
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_lon, min_lat, max_lon, max_lat = bbox
    for cnt in contours:
        coords = []
        for point in cnt[:,0,:]:
            px, py = point
            lon = min_lon + (px/w)*(max_lon-min_lon)
            lat = min_lat + (py/h)*(max_lat-min_lat)
            coords.append([lon, lat])
        if coords:
            coords.append(coords[0])
            poly = Polygon(coords)
            geojson_features.append({
                "type":"Feature",
                "geometry":mapping(poly),
                "properties":{"class":"pollution","confidence":0.9}
            })
    return {"type":"FeatureCollection", "features":geojson_features}

