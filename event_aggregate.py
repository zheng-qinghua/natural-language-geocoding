"""
事件点聚合：把一堆事件点变成"受影响的区域"。

上游是 `event_data.py`（NASA EONET，只给点和轨迹），下游是 `geocoding.py` 的
`EventAffected` 节点。本模块只做纯几何，**不发任何网络请求、不依赖任何外部
数据**，所以可以用合成点集完整离线自检。

要解决的核心矛盾：EONET 给的是"某年某月某地报了一次火情"这种离散点，而用户问
的是"被火灾影响的亚马逊雨林"——一个**区域**。两者之间需要一次聚合，而聚合方式
决定了答案的性质：

  - 不能是"所有点的凸包"：亚马逊的火点分东西两片，整体凸包会把中间没烧的
    雨林一并算进去
  - 不能是"行政边界"：那不是用户问的东西，也丢掉了"哪一片真的烧过"这个信息
  - 采用：**网格聚类 → 每簇凸包 → 与基础地点的真实边界求交 → 简化 → 合并**

为什么用网格法而不是 DBSCAN：本项目暂缓引入 sklearn/scipy（见 开发规划.md 第四节），
而网格 + 连通分量无新依赖、行为完全可预测、可离线复现。代价是"单链效应"——一串
刚好首尾相接的格子会把很长的区域连成一簇，DBSCAN 的 eps/minPts 能更好地区分。
这个代价由 `min_cluster_size`（密度门槛）部分抵消，剩余部分接受。

坐标处理的固有限制（全项目一致，不是本模块特有）：所有运算在经纬度平面上做，
不投影。由此带来两处近似：
  - `buffer` 的半径按度算，在不同纬度对应的实地距离不同（45° 处经度方向偏短约
    30%）。只用于把"单点/共线"这类零面积凸包撑开，量级小，影响可忽略
  - 一个几何体**自身**横跨 180° 经线时，平面表示本身就是错的。本模块对**事件点**
    横跨 180° 的情形做了处理（见 `_split_straddling`），但不负责修正 `base_geom`
    自身的这种情形（那需要先投影，属于全项目的架构级改动）

还有一处与几何无关、但会误导排查的坑：**Shapely 的拓扑判定在这种"裁剪出来的"
几何上不可靠**。求交（或简化）产生的顶点常常正落在 base_geom 的边界上，此时
`within` / `covered_by` 会返回 False，而 `difference(base).area` 精确等于 0——
几何其实一个点都没出去。本模块的自检因此**一律用面积判定包含关系**，不用
谓词；这个坑在全项目里通用（geocoding 的降级链同样要小心）。
"""

import math
from dataclasses import dataclass

from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from errors import GeocodeError
from geometry_utils import DEFAULT_MAX_POINTS, simplify_geometry

# 自适应网格：把 base_geom 的 bbox 按最长边切成 target_cells 份。
# 20 意味着"约 20×20 个网格"，配合上游的量级（亚马逊 1071 个点、全球降采样后
# 2000 个点）大约是每格几个点，既不会把整片雨林算成一格，也不会碎成上千块。
DEFAULT_TARGET_CELLS = 20

# 网格边长的**物理上限**（度，0.25°≈28km）。纯相对网格（span/target_cells）在
# 大范围查询上是错的：亚马逊 bbox 长边 30°，除以 20 得 1.5°（166km）的格子，
# 实测 1071 个火点里有 1046 个（98%）落进同一簇——"被火灾影响的亚马逊雨林"
# 会因此聚成一整块覆盖半个南美。原因是一次火灾/风暴的实际尺度是 10~50km，
# 与查询容器多大无关，所以格子不该随容器无限变粗。
# 上限值的实测依据（4 邻接，最大簇占点数比例）：
#   帕拉州 230 点：0.20°→19(8%)  0.25°→30(13%)  0.30°→57(25%)
#   亚马逊 1071 点：0.20°→54(5%) 0.25°→76(7%)   0.30°→142(13%)
# 0.30° 起单链粘连抬头，0.20° 以下碎块过多，取 0.25°。
# 注意这个上限只在 span > target_cells*0.25° = 5°（约 550km）时才起作用；
# 更小的查询仍由相对网格主导，城市级查询照样能拿到细网格。
MAX_GRID_DEG = 0.25

# 簇的密度门槛：一条事件周围一个格子内没有第二条就丢掉。
# 这是主要的降噪旋钮——孤立单点更可能是单次上报误差，而不是"成片受影响"。
DEFAULT_MIN_CLUSTER_SIZE = 2

# 网格连通方式。用 4 邻接而不是 8：8 邻接允许对角相接，两个簇只要在角上碰到
# 就并成一片，单链效应明显更强。实测（grid=0.25°，min_cluster_size=2）：
#   帕拉州 230 点：4 邻接最大簇 30(13%)，8 邻接 63(27%)
#   亚马逊 1071 点：4 邻接最大簇 76(7%)，8 邻接 175(16%)
# 两个尺度上 8 邻接都把最大簇翻了一倍多，所以取 4。
DEFAULT_CONNECTIVITY = 4

# 面积门槛，单位是"网格数的平方"：不足 1/4 个网格的碎块丢弃。
# 用网格面积而非绝对平方度，是为了让门槛随查询的空间尺度自动伸缩——
# 城市查询和亚马逊查询用的是同一个语义。
MIN_AREA_IN_CELLS = 0.25

# 求 longitude 方向格子宽度时用的参考纬度上限。纬度 80° 处 cos≈0.17，
# 经度格宽会是纬度格宽的 5.8 倍；再靠近极点 cos→0 会让格宽发散。
_MAX_ABS_REF_LAT = 80.0

# 点集的经度跨度超过这个值就认定"跨了 180° 经线"。
_LON_SPAN_FOR_WRAP = 180.0

# 退化网格的兜底值（度）：base_geom 的 bbox 退化成一条线/一个点时用。
_FALLBACK_GRID_DEG = 0.1
# 网格边长下限（度，约 11m）：防止病态的小 bbox 配大 target_cells 产生海量格子。
_MIN_GRID_DEG = 1e-4

# 把零面积凸包撑成面时用的圆近似段数（每 1/4 圆）。
_HULL_BUFFER_QUAD_SEGS = 8


@dataclass(frozen=True)
class EventAggregate:
    """一次事件聚合的结果。

    除了几何本身，还带上一组计数：S24 要把它们回显给用户（"命中 N 条 → M 块"），
    让"换算成绝对时间区间 + 回显"这条决策里"这个答案建立在多少证据上"可见。
    没有这些计数，S24 只能自己重算一遍簇数，那就会出现两处口径不一的数字。
    """

    geometry: Polygon | MultiPolygon
    grid_deg: float
    min_area_deg2: float
    clusters: int          # 过完 min_cluster_size 的簇数
    pieces: int            # 与 base 求交并过完面积门槛后保留的块数
    dropped_pieces: int    # 因面积过小被丢弃的块数
    points_total: int      # 参与聚合的位置点总数
    points_on_base: int    # 其中落在 base_geom 内的点数


# =============================================================================
# 网格参数
# =============================================================================
def adaptive_grid_deg(base_geom, target_cells: int = DEFAULT_TARGET_CELLS,
                      max_grid_deg: float = MAX_GRID_DEG) -> float:
    """按 base_geom 的尺度推导网格边长（度）。

    固定网格边长是行不通的：同一条 `grid_deg` 用在城市级查询上会得到几千个碎块，
    用在洲级查询上会把整片区域算成一格。所以边长必须由"容器"的尺寸决定。

    用 bbox 的**最长边**除以 target_cells，保证两个方向的格子在度数上等宽
    （经度方向的实地宽度由 cluster_events 单独补偿）。用最长边而不是各自除，
    是为了让格子保持近似正方形——否则细长 bbox 会得到细长格子。

    但相对值要再夹一个**物理上限** `max_grid_deg`：格子代表的是"一次事件的
    空间尺度"，它不该随着容器变大而无限变粗（见 MAX_GRID_DEG 的实测说明）。
    传 `max_grid_deg=None` 可以关掉上限，退回纯相对网格。
    """
    minx, miny, maxx, maxy = base_geom.bounds
    span = max(maxx - minx, maxy - miny)
    if span <= 0.0 or target_cells <= 0:
        return _FALLBACK_GRID_DEG
    grid = span / target_cells
    if max_grid_deg is not None:
        grid = min(grid, max_grid_deg)
    return max(grid, _MIN_GRID_DEG)


# =============================================================================
# 聚类
# =============================================================================
def cluster_events(points, grid_deg: float,
                   min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
                   connectivity: int = DEFAULT_CONNECTIVITY
                   ) -> list[list[tuple[float, float]]]:
    """把事件点按空间密度聚成簇。

    做法：按网格分桶 → 相邻的桶合并成连通分量 → 按点数过滤小簇。

    Args:
        points: [(lon, lat), ...]。
        grid_deg: 网格边长（度，纬度方向）。经度方向的格子宽会被换算成
            `grid_deg / cos(参考纬度)`，让格子在实地近似正方形；否则高纬度的簇
            会在经度方向被不成比例地拉长。
        min_cluster_size: 簇内点数下限。
        connectivity: 4 或 8。4 只认上下左右相邻（默认，理由见 DEFAULT_CONNECTIVITY），
            8 额外认对角相邻。

    Returns:
        每簇一个点列表。**每个返回的簇的经度跨度都不超过 180°**，可以直接求
        凸包——跨 180° 经线的簇在这里就切成两半了（见 `_split_straddling`）。

    Notes:
        过滤与切分的顺序是"先按合并后的整簇计数、再切 180°"，所以一个返回的
        子簇可能小于 min_cluster_size——横跨 180° 的两点相距可能只有几公里，
        密度上不该被当成两个孤立点。
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity 只能是 4 或 8，收到 {connectivity}")

    points = [(float(lon), float(lat)) for lon, lat in points]
    if not points or grid_deg <= 0:
        return []

    reference_lat = _reference_lat(points)
    lon_cell_deg = grid_deg / math.cos(math.radians(reference_lat))
    grid_lons = _prepare_lons(points)

    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for (lon, lat), grid_lon in zip(points, grid_lons):
        key = (math.floor(lat / grid_deg), math.floor(grid_lon / lon_cell_deg))
        buckets.setdefault(key, []).append((lon, lat))

    # 4 邻接只走上下左右；8 邻接额外走四个对角
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    clusters: list[list[tuple[float, float]]] = []
    visited: set[tuple[int, int]] = set()
    for start in buckets:
        if start in visited:
            continue
        visited.add(start)
        stack = [start]
        members: list[tuple[float, float]] = []
        while stack:
            row, col = stack.pop()
            members.extend(buckets[(row, col)])
            for d_row, d_col in offsets:
                neighbour = (row + d_row, col + d_col)
                if neighbour in buckets and neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        if len(members) >= min_cluster_size:
            clusters.extend(_split_straddling(members))
    return clusters


def _reference_lat(points) -> float:
    """格宽换算用的参考纬度：点集的平均纬度，钳到 ±80°。"""
    mean_lat = sum(lat for _, lat in points) / len(points)
    return max(min(mean_lat, _MAX_ABS_REF_LAT), -_MAX_ABS_REF_LAT)


def _prepare_lons(points) -> list[float]:
    """把点集的经度铺开到连续空间，供分格使用。

    点集经度跨度超过 180°（即认定它跨了 180° 经线）时，把负经度 +360，
    得到一段连续区间——这样 179.5E 和 179.5W 才会落到相邻的格子里，
    而不是分居网格两端。
    """
    lons = [lon for lon, _ in points]
    if max(lons) - min(lons) <= _LON_SPAN_FOR_WRAP:
        return lons
    return [lon + 360.0 if lon < 0.0 else lon for lon in lons]


def _split_straddling(points) -> list[list[tuple[float, float]]]:
    """把横跨 180° 经线的簇切成东、西两半。

    平面的凸包算法无法表达横跨 180° 的区域：19 个点在 179.5E 和 179.5W，
    它们真实的凸包是一条窄带，但按平面坐标算会得到一圈绕地球的宽条。
    切分是唯一正确的做法——这样的区域在平面表示下本来就该是两个部分。
    """
    lons = [lon for lon, _ in points]
    if max(lons) - min(lons) <= _LON_SPAN_FOR_WRAP:
        return [points]
    east = [p for p in points if p[0] >= 0.0]
    west = [p for p in points if p[0] < 0.0]
    return [part for part in (east, west) if part]


# =============================================================================
# 聚合
# =============================================================================
def aggregate_events(points, base_geom, *, base_label: str = "该范围",
                     grid_deg: float | None = None,
                     target_cells: int = DEFAULT_TARGET_CELLS,
                     min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
                     min_area_deg2: float | None = None,
                     max_points: int = DEFAULT_MAX_POINTS) -> EventAggregate:
    """把事件点聚合成"落在 base_geom 内的受影响区域"。

    Args:
        points: [(lon, lat), ...]，**按位置而非按事件**撒点。一个事件可以带多个
            位置（S22 的 `EventRecord.points()`），按事件取代表点会让一条横跨
            几百公里的飓风轨迹塌缩成一个点，也会让长轨迹上的不同受灾段丢失。
        base_geom: 基础地点的真实边界（Polygon/MultiPolygon）。
        base_label: 出现在错误提示里的地点名，例如 "亚马逊雨林"。
        grid_deg: 网格边长；None 表示按 base_geom 自适应（`adaptive_grid_deg`）。
        min_area_deg2: 面积门槛（平方度）；None 表示取 1/4 个网格的面积。

    Returns:
        EventAggregate。

    Raises:
        GeocodeError: 四种"算不出区域"的情形，各有独立的 user_message——
            没查到事件 / 基础地点没有边界 / 事件全在范围外 / 事件过于零散。
            这四种都必须抛错而不是返回空几何：空几何会一路走到渲染层，
            最后表现为"查到了但地图是空的"，用户无从判断是没数据还是算错了。
    """
    total = len(points)
    if total == 0:
        raise GeocodeError(
            "该时间窗内没有查到事件，无法给出受影响区域。",
            detail="aggregate_events 收到 0 个位置点（上游 EONET 查询返回空）",
        )

    # base_geom 可能是 Point：geocoding 的降级链在拿不到真实边界时只给中心点。
    # 中心点定不出"范围"，所以这里明确拒绝，而不是把整个地球当范围。
    base = _as_areal(base_geom)
    if base is None:
        raise GeocodeError(
            f"{base_label}没有可用的边界（只拿到了中心点），"
            f"无法计算受影响区域。",
            detail=f"base_geom 类型 = {getattr(base_geom, 'geom_type', type(base_geom).__name__)}，"
                   f"没有有面积的部分（bounds={getattr(base_geom, 'bounds', None)}）",
        )

    if grid_deg is None:
        grid_deg = adaptive_grid_deg(base, target_cells)
    if min_area_deg2 is None:
        min_area_deg2 = grid_deg * grid_deg * MIN_AREA_IN_CELLS

    points_on_base = sum(1 for lon, lat in points if base.covers(Point(lon, lat)))

    clusters = cluster_events(points, grid_deg, min_cluster_size)
    if not clusters:
        raise GeocodeError(
            f"查到的 {total} 条事件过于零散，在{base_label}范围内无法形成成片的区域。",
            detail=f"grid_deg={grid_deg:.6f}, min_cluster_size={min_cluster_size}："
                   f"没有任何格子簇达到这个密度",
        )

    pieces: list[Polygon | MultiPolygon] = []
    dropped_area = 0
    for cluster in clusters:
        piece = _as_areal(_cluster_hull(cluster, grid_deg).intersection(base))
        if piece is None:
            # 簇的凸包与该地点只擦边或完全在界外，求交得到线/点，没有面积可贡献
            continue
        piece = simplify_geometry(piece, max_points=max_points)
        if piece.area < min_area_deg2:
            dropped_area += 1
            continue
        pieces.append(piece)

    if not pieces:
        if points_on_base == 0:
            mean_lon = sum(p[0] for p in points) / total
            mean_lat = sum(p[1] for p in points) / total
            raise GeocodeError(
                f"查到的 {total} 条事件都不在{base_label}范围内，"
                f"无法给出受影响区域。",
                detail=f"事件平均位置 ({mean_lon:.4f}, {mean_lat:.4f})，"
                       f"{base_label} bounds={tuple(round(v, 4) for v in base.bounds)}",
            )
        raise GeocodeError(
            f"查到的 {total} 条事件过于零散，在{base_label}范围内无法形成成片的区域。",
            detail=f"{len(clusters)} 个簇与{base_label}求交后，{dropped_area} 块"
                   f"因面积不足 {min_area_deg2:.8f} 平方度被丢弃",
        )

    geometry = simplify_geometry(unary_union(pieces), max_points=max_points)
    return EventAggregate(
        geometry=geometry,
        grid_deg=grid_deg,
        min_area_deg2=min_area_deg2,
        clusters=len(clusters),
        pieces=len(pieces),
        dropped_pieces=dropped_area,
        points_total=total,
        points_on_base=points_on_base,
    )


def _cluster_hull(points, grid_deg: float):
    """一簇点的凸包；零面积时撑成一个面。

    单点（凸包是 Point）与共线点（凸包是 LineString）的凸包没有面积，既不能与
    base 求交得到区域、也会被面积门槛直接滤掉。一条孤立的火情上报**是**一片
    受影响区域，只是分辨率只有一个点，所以按半个网格的半径撑开——"这一点周围
    半个格子内算受影响"是与网格分辨率一致的解读。
    """
    hull = MultiPoint(points).convex_hull
    if hull.area > 0.0:
        return hull
    return hull.buffer(grid_deg / 2.0, quad_segs=_HULL_BUFFER_QUAD_SEGS)


def _as_areal(geom):
    """取出几何里"有面积"的部分；没有则返回 None。

    `intersection` 对混合维度的输入会返回 GeometryCollection（例如同时包含
    一块面与一条擦边的线），`unary_union` 会把这些线段一并带进结果，而它们
    无法序列化成用户要的区域。所以只放行 Polygon / MultiPolygon。
    """
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    polygons = []
    for part in getattr(geom, "geoms", ()):
        if isinstance(part, Polygon):
            polygons.append(part)
        elif isinstance(part, MultiPolygon):
            polygons.extend(part.geoms)
    if not polygons:
        return None
    return MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]


# =============================================================================
# 离线自检（python event_aggregate.py）
# =============================================================================
def _self_check() -> int:
    """不联网的自检：网格参数、聚类、四种失败语义、跨 180° 经线。

    刻意不引入 pytest（同 event_data.py / geocoding.py 的简易测试入口）：
    一条命令复现，且不需要任何网络与外部数据。
    """
    import sys

    from shapely.geometry import box

    failures: list[str] = []

    def check(name: str, condition: bool, extra: str = ""):
        if condition:
            print(f"  [通过] {name}")
        else:
            print(f"  [失败] {name} {extra}")
            failures.append(name)

    def error_of(func, *args, **kwargs):
        """执行并返回 GeocodeError 的 user_message；没报错返回 None。"""
        try:
            func(*args, **kwargs)
            return None
        except GeocodeError as e:
            return e.user_message

    # ---------------- 网格参数 ----------------
    print("== 网格参数 ==")
    check("网格边长随 bbox 缩放（跨度大 → 格子大）",
          adaptive_grid_deg(box(0, 0, 4, 0.5)) > adaptive_grid_deg(box(0, 0, 2, 0.5)),
          f"{adaptive_grid_deg(box(0, 0, 4, 0.5))} vs "
          f"{adaptive_grid_deg(box(0, 0, 2, 0.5))}")
    check("用最长边切分（细长 bbox 不会得到细长格子）",
          abs(adaptive_grid_deg(box(0, 0, 4, 0.1)) - 0.2) < 1e-9,
          str(adaptive_grid_deg(box(0, 0, 4, 0.1))))
    check("大范围查询被物理上限截住（不随容器无限变粗）",
          abs(adaptive_grid_deg(box(0, 0, 30, 30)) - MAX_GRID_DEG) < 1e-12,
          f"实得 {adaptive_grid_deg(box(0, 0, 30, 30))}，纯相对应为 1.5")
    check("上限可关掉（退回纯相对网格）",
          abs(adaptive_grid_deg(box(0, 0, 30, 30), max_grid_deg=None) - 1.5) < 1e-12,
          str(adaptive_grid_deg(box(0, 0, 30, 30), max_grid_deg=None)))
    check("上限只在超过它时才起作用（小查询仍是相对网格）",
          abs(adaptive_grid_deg(box(0, 0, 1, 1)) - 0.05) < 1e-12,
          str(adaptive_grid_deg(box(0, 0, 1, 1))))
    check("退化成线的 bbox 有兜底值",
          adaptive_grid_deg(box(5, 5, 5, 5)) == _FALLBACK_GRID_DEG)
    check("病态小 bbox 受下限保护",
          adaptive_grid_deg(box(0, 0, 1e-9, 1e-9)) >= _MIN_GRID_DEG)

    # ---------------- 聚类 ----------------
    print("== 聚类 ==")
    base_amazon = box(-75.0, -20.0, -45.0, 5.0)

    def blob(center_lon, center_lat, n, spacing=0.2):
        """一个 n×n 的方阵点块。"""
        side = int(math.ceil(math.sqrt(n)))
        return [(center_lon + i * spacing, center_lat + j * spacing)
                for i in range(side) for j in range(side)][:n]

    two_blobs = blob(-70.0, -10.0, 25) + blob(-50.0, 0.0, 25)
    clusters = cluster_events(two_blobs, grid_deg=1.0)
    check("相距很远的两片点聚成 2 簇", len(clusters) == 2, f"实得 {len(clusters)}")
    check("每簇点数正确", sorted(len(c) for c in clusters) == [25, 25],
          str(sorted(len(c) for c in clusters)))

    # 两个点分别落在 (0,0) 和 (1,1) 格：对角相邻。默认 4 邻接不该把它们并起来，
    # 8 邻接才并——这正是实测里 8 邻接会把最大簇翻倍的机制。
    diagonal = [(0.5, 0.5), (1.5, 1.5)]
    check("对角相邻的格子在 4 邻接下分属两簇",
          len(cluster_events(diagonal, grid_deg=1.0, min_cluster_size=1)) == 2,
          str(cluster_events(diagonal, grid_deg=1.0, min_cluster_size=1)))
    check("对角相邻的格子在 8 邻接下并成一簇",
          len(cluster_events(diagonal, grid_deg=1.0, min_cluster_size=1,
                             connectivity=8)) == 1,
          str(cluster_events(diagonal, grid_deg=1.0, min_cluster_size=1,
                             connectivity=8)))
    try:
        cluster_events(diagonal, 1.0, connectivity=5)
        bad_connectivity = False
    except ValueError:
        bad_connectivity = True
    check("非法 connectivity 直接报错", bad_connectivity, "")

    far = cluster_events([(0.1, 0.1), (5.0, 5.0)], grid_deg=1.0, min_cluster_size=1)
    check("相距很远的两个点算两簇", len(far) == 2, f"实得 {len(far)}")

    check("min_cluster_size 过滤孤立点",
          cluster_events([(0.1, 0.1), (5.0, 5.0)], grid_deg=1.0,
                         min_cluster_size=2) == [])
    check("空输入返回空", cluster_events([], grid_deg=1.0) == [])

    print("== 物理上限：大容器不并成一团 ==")
    # 复现"纯相对网格"的失真：一条点间距 0.5° 的斜向点链，横跨 20°。
    # 纯相对网格（30°/20 = 1.5°）下相邻点必然同格或邻格，单链效应把 41 个点
    # 串成一簇；夹上 0.25° 上限后间距变成两个格子，链断开。
    wide_base = box(0.0, 0.0, 30.0, 30.0)
    chain = [(i * 0.5, i * 0.5) for i in range(41)]
    relative = cluster_events(chain, adaptive_grid_deg(wide_base, max_grid_deg=None),
                              min_cluster_size=1)
    capped = cluster_events(chain, adaptive_grid_deg(wide_base), min_cluster_size=1)
    check("纯相对网格把跨 20° 的点链串成 1 簇（这就是要修的失真）",
          len(relative) == 1, f"实得 {len(relative)} 簇")
    check("加物理上限后点链断开",
          len(capped) > 30, f"实得 {len(capped)} 簇")

    print("== 纬度补偿 ==")
    # 同样的 2° 经度差：在赤道处跨了两个格子（格子宽 1°），
    # 在 70°N 处落在同一列（格宽被放大到 1/cos70° ≈ 2.92°）。
    # 必须用"隔一格"的距离来测，紧挨着的格子无论宽窄都会被连通分量合并掉。
    high = cluster_events([(0.0, 70.0), (2.0, 70.0)], grid_deg=1.0, min_cluster_size=1)
    low = cluster_events([(0.0, 0.0), (2.0, 0.0)], grid_deg=1.0, min_cluster_size=1)
    check("2° 经度差在 70°N 处同格（格宽被放大）", len(high) == 1, f"实得 {len(high)}")
    check("同样差在赤道处跨格（格宽未放大）", len(low) == 2, f"实得 {len(low)}")

    # ---------------- 正常聚合 ----------------
    print("== 聚合：两片区域 ==")
    result = aggregate_events(two_blobs, base_amazon, base_label="亚马逊雨林",
                              grid_deg=1.0)
    check("返回 2 块（两片点各成一块）", result.pieces == 2,
          f"实得 pieces={result.pieces}, clusters={result.clusters}")
    check("结果是 MultiPolygon 且含 2 个子多边形",
          result.geometry.geom_type == "MultiPolygon"
          and len(result.geometry.geoms) == 2,
          f"{result.geometry.geom_type} / {getattr(result.geometry, 'geoms', None) is not None}")
    check("结果完全落在地点边界内（按面积判定）",
          result.geometry.difference(base_amazon).area == 0.0,
          f"越界面积={result.geometry.difference(base_amazon).area:.3e}")
    check("计数：50 个点全部在边界内", result.points_on_base == 50,
          str(result.points_on_base))
    check("计数：点数与簇数一致", result.points_total == 50 and result.clusters == 2,
          f"{result.points_total}/{result.clusters}")

    print("== 聚合：单点与重复点 ==")
    single = aggregate_events([(-60.0, -10.0)], base_amazon, base_label="亚马逊雨林",
                              grid_deg=1.0, min_cluster_size=1)
    check("单点在 min_cluster_size=1 时得到一个小面",
          result.geometry.geom_type is not None and single.geometry.area > 0,
          f"area={single.geometry.area}")
    check("单点的面被撑到约半个网格的半径",
          0.6 < single.geometry.area / (1.0 * 1.0) < 1.0,
          f"面积={single.geometry.area:.4f} 网格面积=1.0")
    check("单点在默认 min_cluster_size=2 时被判为过于零散",
          "过于零散" in (error_of(aggregate_events, [(-60.0, -10.0)], base_amazon,
                                  base_label="亚马逊雨林", grid_deg=1.0) or ""),
          str(error_of(aggregate_events, [(-60.0, -10.0)], base_amazon,
                       base_label="亚马逊雨林", grid_deg=1.0)))

    dupes = [(-60.0, -10.0)] * 5
    dup_result = aggregate_events(dupes, base_amazon, base_label="亚马逊雨林",
                                  grid_deg=1.0)
    check("5 个重合点过得了默认密度门槛", dup_result.pieces == 1,
          f"实得 {dup_result.pieces}")
    check("重合点的凸包是零面积，被撑成面", dup_result.geometry.area > 0,
          f"area={dup_result.geometry.area}")

    # ---------------- 四种失败语义 ----------------
    print("== 四种失败语义 ==")
    empty_msg = error_of(aggregate_events, [], base_amazon, base_label="亚马逊雨林")
    check("空点集 → 提示没查到事件", empty_msg is not None and "没有查到事件" in empty_msg,
          str(empty_msg))

    point_base_msg = error_of(aggregate_events, two_blobs, Point(-60.0, -10.0),
                              base_label="亚马逊雨林")
    check("base 只有中心点 → 明确拒绝而非当成全球",
          point_base_msg is not None and "中心点" in point_base_msg, str(point_base_msg))

    outside = blob(-30.0, 30.0, 9)   # 完全在亚马逊 bbox 之外
    outside_msg = error_of(aggregate_events, outside, base_amazon,
                           base_label="亚马逊雨林", grid_deg=1.0)
    check("事件全在界外 → 提示都不在范围内",
          outside_msg is not None and "都不在" in outside_msg and "9" in outside_msg,
          str(outside_msg))

    scattered_msg = error_of(aggregate_events, [(-70.0, -10.0), (-50.0, 0.0)],
                             base_amazon, base_label="亚马逊雨林", grid_deg=1.0)
    check("事件过于零散 → 提示无法形成区域",
          scattered_msg is not None and "过于零散" in scattered_msg,
          str(scattered_msg))
    check("四种提示互不相同",
          len({empty_msg, point_base_msg, outside_msg, scattered_msg}) == 4,
          str({empty_msg, point_base_msg, outside_msg, scattered_msg}))

    # ---------------- 面积门槛 ----------------
    print("== 面积门槛 ==")
    # 一簇点横跨边界东缘（-45）：2 个点在界内、2 个点在界外。凸包与地点求交后
    # 只剩 0.02° × 0.5° 的一条，低于"1/4 个网格"的门槛，应被丢弃。
    # 注意这与"事件全在界外"不同——界内那 2 个点说明地点没问错，只是证据太细。
    edge = [(-45.02, -10.0), (-45.02, -10.5), (-44.5, -10.0), (-44.5, -10.5)]
    edge_msg = error_of(aggregate_events, edge, base_amazon, base_label="亚马逊雨林",
                        grid_deg=1.0)
    check("界内只剩细条时被面积门槛丢弃（而不是报'都在范围外'）",
          edge_msg is not None and "过于零散" in edge_msg, str(edge_msg))

    # ---------------- 跨 180° 经线 ----------------
    print("== 跨 180° 经线 ==")
    far_east_base = box(170.0, -20.0, 180.0, 20.0)
    far_west_base = box(-180.0, -20.0, -170.0, 20.0)
    # 180° 两侧各一个 2×2 的点块。相邻的经度必须落在相邻的格子里：
    # 179.95E 与 179.85W 在铺开后的空间里只差 0.2°，一个格子宽。
    dateline = [(179.85, 0.0), (179.85, 0.3), (179.95, 0.0), (179.95, 0.3),
                (-179.95, 0.0), (-179.95, 0.3), (-179.85, 0.0), (-179.85, 0.3)]
    dateline_clusters = cluster_events(dateline, grid_deg=0.2)
    check("跨 180° 的点聚成 1 簇（铺开经度后相邻）",
          len(dateline_clusters) == 2, f"实得 {len(dateline_clusters)} 个平面子簇")
    check("其中东西两半各 4 个点",
          sorted(len(c) for c in dateline_clusters) == [4, 4],
          str(sorted(len(c) for c in dateline_clusters)))
    check("切开后每半的经度跨度都不超过 180°",
          all(max(p[0] for p in half) - min(p[0] for p in half) <= _LON_SPAN_FOR_WRAP
              for half in dateline_clusters),
          str(dateline_clusters))

    # 东西两侧各有一块地点，两侧都应得到结果，而不是绕地球一圈的宽条
    both_sides = far_east_base.union(far_west_base)
    dateline_result = aggregate_events(dateline, both_sides, base_label="斐济",
                                       grid_deg=0.2, min_cluster_size=1)
    check("跨 180° 的聚合结果是 2 块（两侧各一），不是绕地球的条",
          dateline_result.pieces == 2, f"实得 {dateline_result.pieces}")
    check("2 块分别靠近两侧的 180°",
          dateline_result.geometry.geom_type == "MultiPolygon"
          and len(dateline_result.geometry.geoms) == 2,
          dateline_result.geometry.geom_type)
    check("结果完全落在地点边界内（按面积判定）",
          dateline_result.geometry.difference(both_sides).area == 0.0,
          f"越界面积={dateline_result.geometry.difference(both_sides).area:.3e}")

    print()
    if failures:
        print(f"自检失败 {len(failures)} 项：{failures}")
        return 1
    print("自检全部通过（离线，未发网络请求）")
    return 0


if __name__ == "__main__":
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.exit(_self_check())
