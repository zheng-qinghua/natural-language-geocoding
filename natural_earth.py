"""
Natural Earth 海岸线数据处理模块。

提供海岸线几何查询功能，用于 CoastOf 和 OffTheCoastOf 空间操作。

数据来源：Natural Earth 10m Physical Coastline
  https://raw.githubusercontent.com/martynafford/natural-earth-geojson/master/10m/physical/ne_10m_coastline.json

与源项目的对应关系：
  源项目 natural_earth.py → 本模块
"""

import json
import os
import sys
from functools import lru_cache
from shapely.geometry import GeometryCollection, shape

# 海岸线数据文件路径
# 开发环境：与 natural_earth.py 同目录
# PyInstaller打包：sys._MEIPASS 临时解压目录
if getattr(sys, 'frozen', False):
    _DATA_DIR = sys._MEIPASS
else:
    _DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_COASTLINE_FILE = os.path.join(_DATA_DIR, "ne_10m_coastline.json")


def download_coastline_file():
    """下载 Natural Earth 海岸线 GeoJSON 数据。"""
    import urllib.request

    url = ("https://raw.githubusercontent.com/martynafford/"
           "natural-earth-geojson/master/10m/physical/ne_10m_coastline.json")
    print(f"[下载] 正在下载全球海岸线数据...")
    print(f"[下载] {url}")
    urllib.request.urlretrieve(url, _COASTLINE_FILE)
    print(f"[下载] 已保存到 {_COASTLINE_FILE}")


def _load_coastline_geojson() -> GeometryCollection:
    """加载海岸线 GeoJSON 文件并解析为 Shapely GeometryCollection。"""
    if not os.path.exists(_COASTLINE_FILE):
        download_coastline_file()

    with open(_COASTLINE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # GeoJSON FeatureCollection → 提取每个feature的geometry → GeometryCollection
    geometries = []
    for feature in data.get("features", []):
        geom = shape(feature["geometry"])
        geometries.append(geom)

    return GeometryCollection(geometries)


@lru_cache(maxsize=1)
def _get_coastlines() -> GeometryCollection:
    """获取全球海岸线几何体（带缓存，首次调用时加载文件）。"""
    return _load_coastline_geojson()


def coastline_of(geometry) -> object:
    """
    计算给定几何范围内的海岸线。

    Args:
        geometry: Shapely 几何对象（通常是地点几何加2km缓冲区）

    Returns:
        海岸线几何体（GeometryCollection 或 MultiLineString），
        无海岸线时返回 None。
    """
    coastlines = _get_coastlines()
    intersection = geometry.intersection(coastlines)
    if intersection.is_empty:
        return None
    return intersection
