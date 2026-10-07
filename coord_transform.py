"""
GCJ-02 与 WGS-84 坐标系互转。

高德地图 API 返回 GCJ-02（国测局）坐标；OSM Overpass 与 Natural Earth 使用
WGS-84。两者在中国境内相差约 300–600m，直接混算会导致几何错位（区划边界与
海岸线不重合、底图上整体偏移）。

约定：**本项目内部统一使用 WGS-84**。所有高德 API 的返回结果在出口处立即
转换，其余模块只处理 WGS-84。

算法：
  - 正向 wgs84_to_gcj02 使用公开的解析加偏公式
  - 反向 gcj02_to_wgs84 无解析解，使用不动点迭代逼近

参考文献：业界通用的 GCJ-02 加偏公式（eviltransform / coordTransform 等实现）
"""

import math

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

# Krasovsky 1940 椭球参数，GCJ-02 加偏公式依赖这两个常数
_SEMI_MAJOR_AXIS = 6378245.0
_ECCENTRICITY_SQUARED = 0.00669342162296594323


def _out_of_china(lon: float, lat: float) -> bool:
    """粗判坐标是否在中国境外。

    GCJ-02 加偏只在中国境内生效，境外坐标应原样返回。这里用矩形范围粗判，
    会包含部分邻国区域，属业界通行近似。
    """
    return not (73.66 < lon < 135.05 and 3.86 < lat < 53.55)


def _raw_lat_offset(x: float, y: float) -> float:
    """加偏公式的纬度分量（未换算为单位度）。"""
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320.0 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _raw_lon_offset(x: float, y: float) -> float:
    """加偏公式的经度分量（未换算为单位度）。"""
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def _offset_degrees(lon: float, lat: float) -> tuple[float, float]:
    """计算 WGS-84 坐标 (lon, lat) 处的 GCJ-02 偏移量（度）。"""
    dlat = _raw_lat_offset(lon - 105.0, lat - 35.0)
    dlon = _raw_lon_offset(lon - 105.0, lat - 35.0)

    rad_lat = lat / 180.0 * math.pi
    magic = math.sin(rad_lat)
    magic = 1 - _ECCENTRICITY_SQUARED * magic * magic
    sqrt_magic = math.sqrt(magic)

    dlat = (dlat * 180.0) / ((_SEMI_MAJOR_AXIS * (1 - _ECCENTRICITY_SQUARED)) / (magic * sqrt_magic) * math.pi)
    dlon = (dlon * 180.0) / (_SEMI_MAJOR_AXIS / sqrt_magic * math.cos(rad_lat) * math.pi)
    return dlon, dlat


def wgs84_to_gcj02(lon: float, lat: float) -> tuple[float, float]:
    """WGS-84 → GCJ-02（正向加偏）。

    Args:
        lon: WGS-84 经度。
        lat: WGS-84 纬度。

    Returns:
        GCJ-02 经纬度；坐标在中国境外时原样返回。
    """
    if _out_of_china(lon, lat):
        return lon, lat
    dlon, dlat = _offset_degrees(lon, lat)
    return lon + dlon, lat + dlat


def gcj02_to_wgs84(lon: float, lat: float, *, iterations: int = 5) -> tuple[float, float]:
    """GCJ-02 → WGS-84（迭代反解）。

    加偏公式没有解析反函数，改用不动点迭代：以 GCJ 值为初值，反复用正向公式
    计算残差并修正。5 次迭代后误差小于 1e-9 度（约 0.1mm），远超实际需求。

    Args:
        lon: GCJ-02 经度。
        lat: GCJ-02 纬度。
        iterations: 迭代次数。

    Returns:
        WGS-84 经纬度；坐标在中国境外时原样返回。
    """
    if _out_of_china(lon, lat):
        return lon, lat

    wgs_lon, wgs_lat = lon, lat
    for _ in range(iterations):
        gcj_lon, gcj_lat = wgs84_to_gcj02(wgs_lon, wgs_lat)
        wgs_lon += lon - gcj_lon
        wgs_lat += lat - gcj_lat
    return wgs_lon, wgs_lat


def _convert_position(position: list, convert) -> list:
    """转换单个 GeoJSON 位置 [lon, lat, ...]，保留其余维度。"""
    lon, lat = convert(float(position[0]), float(position[1]))
    return [lon, lat, *position[2:]]


def _convert_coordinates(coords, convert):
    """递归转换 GeoJSON 的嵌套坐标结构。"""
    if coords and isinstance(coords[0], (int, float)):
        return _convert_position(coords, convert)
    return [_convert_coordinates(child, convert) for child in coords]


def transform_geojson(geojson_geom: dict, *, to_wgs84: bool = True) -> dict:
    """转换 GeoJSON 几何字典中的坐标，保留其余字段。

    Args:
        geojson_geom: 含 "type" 与 "coordinates" 的几何字典。
        to_wgs84: True 表示 GCJ-02 → WGS-84；False 表示反向。

    Returns:
        转换后的新字典，非坐标字段（如 level/adcode）原样保留。
    """
    convert = gcj02_to_wgs84 if to_wgs84 else wgs84_to_gcj02

    if geojson_geom.get("type") == "GeometryCollection":
        return {
            **geojson_geom,
            "geometries": [
                transform_geojson(g, to_wgs84=to_wgs84) for g in geojson_geom["geometries"]
            ],
        }

    return {
        **geojson_geom,
        "coordinates": _convert_coordinates(geojson_geom["coordinates"], convert),
    }


def transform_geometry(geom: BaseGeometry, *, to_wgs84: bool = True) -> BaseGeometry:
    """对 Shapely 几何的所有坐标执行坐标系转换。

    支持 Point / MultiPoint / LineString / MultiLineString / Polygon /
    MultiPolygon / GeometryCollection。

    Args:
        geom: 任意 Shapely 几何对象。
        to_wgs84: True 表示 GCJ-02 → WGS-84；False 表示反向。

    Returns:
        转换后的几何，类型与原几何一致；空几何原样返回。
    """
    if geom is None or geom.is_empty:
        return geom
    return shape(transform_geojson(geom.__geo_interface__, to_wgs84=to_wgs84))


if __name__ == "__main__":
    # 目视校验：北京天安门、深圳人才公园的加偏量应落在 300–600m 量级
    for place, lon, lat in [
        ("北京天安门", 116.3975, 39.9087),
        ("深圳人才公园", 113.9450, 22.5120),
    ]:
        gcj_lon, gcj_lat = wgs84_to_gcj02(lon, lat)
        back_lon, back_lat = gcj02_to_wgs84(gcj_lon, gcj_lat)
        dlon_m = (gcj_lon - lon) * 111320 * math.cos(math.radians(lat))
        dlat_m = (gcj_lat - lat) * 110540
        offset_m = math.hypot(dlon_m, dlat_m)
        err_m = math.hypot((back_lon - lon) * 111320, (back_lat - lat) * 110540)
        print(
            f"{place}: GCJ=({gcj_lon:.6f}, {gcj_lat:.6f}) "
            f"偏移={offset_m:.1f}m 往返误差={err_m:.6f}m"
        )

    # 境外坐标应原样返回
    print("纽约原样返回:", gcj02_to_wgs84(-74.0060, 40.7128))
