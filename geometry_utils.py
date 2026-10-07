"""
几何后处理：顶点数简化。

移植自源项目 e84_geoai_common.geometry：
  - simplify_geometry(geom, max_points)：顶点数超限时逐步放宽 simplify 容差
  - remove_extraneous_geoms(geom, max_points)：容差用尽仍超限时，按面积丢弃子几何

为什么需要：
  地理编码服务返回的行政区多边形可达数万顶点（高德区划边界、OSM 省级边界），
  后续 buffer / intersection / BorderOf 等操作都会被顶点数拖慢，数值误差也会
  随顶点数放大。源项目在 NamedPlace 出口处统一简化到 18500 点以内。

与源项目的差异：
  源项目用 `simplified = simplified.simplify(tolerance)` 逐级在上一轮结果上再简化，
  这里每一级都作用在原始几何上。反复简化会让形状在每级容差上叠加失真，而在原
  几何上取容差同样能达到"刚好压到限内"的效果。
"""

import shapely
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseMultipartGeometry

# 源项目的经验值：18500 点是"小岛极多的国家/地区也能压进限内"的临界点，
# 再往下调，那些由上千个子多边形组成的区域就压不动了。
DEFAULT_MAX_POINTS = 18_500

# 容差梯度，单位：度。1e-5 约 1m，1e-1 约 11km。逐级放宽，取第一个够用的。
_TOLERANCES = [10.0**power for power in range(-5, 0)]

# 子多边形数量超过此值时，先丢掉面积小于阈值的碎片（源项目的性能优化阈值）
_MANY_SUBPOLYGONS = 100
# "一英亩"的近似值，单位平方度（源项目取值）
_MIN_SUBPOLYGON_AREA = 0.000001


def count_points(geom) -> int:
    """统计几何的顶点数（含所有环与子几何）。"""
    if geom is None or geom.is_empty:
        return 0
    return int(shapely.count_coordinates(geom))


def simplify_geometry(geom, max_points: int = DEFAULT_MAX_POINTS):
    """把几何的顶点数压到 max_points 以内，尽量保持形状。

    Args:
        geom: 任意 Shapely 几何。
        max_points: 顶点数上限。

    Returns:
        简化后的几何；未超限时原样返回。压不进上限时抛 ValueError。

    Raises:
        ValueError: 顶点数压不到上限以内。
    """
    if geom is None or geom.is_empty:
        return geom
    if count_points(geom) < max_points:
        return geom

    # 子多边形成百上千时（海洋国家、群岛省），逐个简化收效甚微，先丢碎片
    if isinstance(geom, MultiPolygon) and len(geom.geoms) > _MANY_SUBPOLYGONS:
        kept = [g for g in geom.geoms if g.area > _MIN_SUBPOLYGON_AREA]
        if kept:
            geom = MultiPolygon(kept)

    for tolerance in _TOLERANCES:
        simplified = geom.simplify(tolerance)
        if count_points(simplified) < max_points:
            return simplified

    # 容差放到 0.1 度还压不下来（如上万个独立小岛），改为丢弃子几何
    return remove_extraneous_geoms(simplified, max_points=max_points)


def remove_extraneous_geoms(geom, *, max_points: int):
    """按面积从大到小保留子几何，直到顶点数降到上限内。

    用于 simplify 压不下来的情形：顶点的减少来自"少画几个小岛"，而不是
    "把小岛画得更粗"。

    Args:
        geom: 任意 Shapely 几何。
        max_points: 顶点数上限。

    Returns:
        由保留下来的子几何组成的几何。

    Raises:
        ValueError: 连最大的子几何都放不进上限。
    """
    if count_points(geom) <= max_points:
        return geom

    parts = sorted(_geometry_parts(geom), key=_bounding_area, reverse=True)

    # 外环 → 它的内环列表。内环只有在外环被保留时才跟着保留，
    # 否则会得到一个"没有外框的洞"。
    poly_rings: dict[tuple, tuple] = {}
    other_geoms = []
    num_points = 0

    for path, part, is_polygon_ring in parts:
        points = count_points(part)
        if num_points + points > max_points:
            break  # 已按面积降序，后面只会更小，但这一步是"装不下就停"
        if is_polygon_ring:
            parent_path, ring_index = path[:-1], path[-1]
            if ring_index == 0:
                poly_rings[parent_path] = (part, [])
            elif parent_path in poly_rings:
                poly_rings[parent_path][1].append(part)
        else:
            other_geoms.append(part)
        num_points += points

    if not poly_rings and not other_geoms:
        raise ValueError(f"无法把几何压缩到 {max_points} 个顶点以内")

    geoms = list(other_geoms)
    if poly_rings:
        geoms.append(
            MultiPolygon([Polygon(exterior, holes) for exterior, holes in poly_rings.values()])
        )
    return _combine_geometries(geoms)


def _geometry_parts(geom, path: tuple = ()) -> list[tuple[tuple, object, bool]]:
    """把几何递归拆成最小部件，返回 (下标路径, 部件, 是否属于某个多边形) 列表。

    路径的含义：Polygon 的第 0 个下标是外环，其余是内环。靠这个约定，
    重建时才能把内环配回它的外环，而不是把内环当成独立多边形。
    """
    if isinstance(geom, (Point, LinearRing, LineString)):
        return [(path, geom, False)]
    if isinstance(geom, Polygon):
        parts = [(path + (0,), geom.exterior, True)]
        parts += [(path + (i + 1,), ring, True) for i, ring in enumerate(geom.interiors)]
        return parts
    if isinstance(geom, BaseMultipartGeometry):
        parts = []
        for idx, child in enumerate(geom.geoms):
            parts += _geometry_parts(child, path + (idx,))
        return parts
    raise TypeError(f"不支持的几何类型: {type(geom).__name__}")


def _bounding_area(part: tuple) -> float:
    """按外接矩形面积排序。

    源项目用外接矩形而非真实面积：自相交的几何算真实面积可能失败或极慢，
    而这里只需要一个稳定的排序依据。
    """
    minx, miny, maxx, maxy = part[1].bounds
    return (maxx - minx) * (maxy - miny)


def _combine_geometries(geoms: list):
    """把多个几何合成一个：同类合成 Multi*，异类装进 GeometryCollection。"""
    if not geoms:
        raise ValueError("至少要传入一个几何")
    if len(geoms) == 1:
        return geoms[0]

    types = {type(g) for g in geoms}
    if types == {Point}:
        return MultiPoint(geoms)
    if types == {LineString}:
        return MultiLineString(geoms)
    if types == {Polygon}:
        return MultiPolygon(geoms)
    return GeometryCollection(geoms)
