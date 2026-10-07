"""
按方位切分几何。

移植自源项目 natural_language_geocoding.splitter：take_compass_subset 把几何按
质心切成东西南北四半之一，是 DirectionalSubset 节点（"北半球的东半部分"这类
语义）的唯一依赖。

切分线取"质心"而非"外接矩形中心"：行政区往往极不规则，用质心切出来的两半
面积上更接近，也更贴近"北部/南部"的直觉。
"""

from shapely.geometry import GeometryCollection, MultiPolygon, Point, box
from shapely.geometry.base import BaseGeometry

# 主导子几何的判定阈值：面积占比超过这个值，就用它决定切分线的位置
_DOMINANT_AREA_RATIO = 0.50

_DIRECTIONS = ("west", "east", "north", "south")


def take_compass_subset(direction: str, geom: BaseGeometry) -> BaseGeometry:
    """取几何在指定方位上的那一半。

    Args:
        direction: "west" / "east" / "north" / "south"。
        geom: 任意 Shapely 几何。

    Returns:
        位于该方位的那部分几何；Point 原样返回。

    Raises:
        ValueError: direction 不是四个方位之一。

    Examples:
        >>> poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        >>> take_compass_subset("west", poly).area   # 西半边
        50.0
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"不支持的方位 '{direction}'，可选：{list(_DIRECTIONS)}")

    if isinstance(geom, Point):
        return geom

    centroid = _centroid_source(geom).centroid
    west, south, east, north = geom.bounds

    # 用整体外接矩形与质心构造半平面遮罩，再与原几何求交。
    # 外接矩形一定覆盖整个几何，所以求交结果不会超出原范围。
    match direction:
        case "west":
            mask = box(west, south, centroid.x, north)
        case "east":
            mask = box(centroid.x, south, east, north)
        case "south":
            mask = box(west, south, east, centroid.y)
        case "north":
            mask = box(west, centroid.y, east, north)

    return geom.intersection(mask)


def _centroid_source(geom: BaseGeometry) -> BaseGeometry:
    """挑选用于计算切分线位置的几何。

    多部件几何（MultiPolygon / GeometryCollection）直接取整体质心会被"小岛群"
    拉偏——比如省份主体在某处、一串离岛在另一处。因此优先用面积占比过半的
    主导子几何来定位；没有主导子几何时（多个体量相当的部件）才退回整体质心。
    """
    if not isinstance(geom, (MultiPolygon, GeometryCollection)):
        return geom

    total_area = geom.area
    for child in geom.geoms:
        if total_area > 0 and child.area > _DOMINANT_AREA_RATIO * total_area:
            return child
    return geom
