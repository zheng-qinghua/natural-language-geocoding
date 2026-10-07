"""
地理编码核心模块：将自然语言地点描述转换为空间几何数据。

核心流程：
  用户输入自然语言文本 → DeepSeek大模型解析为结构化JSON树（经 pydantic 模型校验）→
  查找真实多边形/精确坐标 → 使用Shapely生成几何对象 → 输出GeoJSON/地图

技术栈：
  - DeepSeek API：国内可直接访问的大模型（https://api.deepseek.com）
  - pydantic：节点模型与输出校验（models.py）
  - Shapely：Python地理几何计算库（通过 pip install shapely 安装）
  - 无需 Nominatim/OpenStreetMap（国内被墙），所有坐标由大模型提供

与源项目 (natural-language-geocoding) 的关键差异：
  - 源项目用 Claude(AWS Bedrock) + Nominatim/OpenSearch 地理编码数据库
  - 本模块用 DeepSeek + 高德/OSM 查找真实边界
  - NamedPlace 只接受真实多边形；无真实边界时降级为 Point 并明确报出，不伪造边界
  - 支持14种空间操作类型（对齐源项目12种，另加事件影响区域 EventAffected
    与用地类型区域 LandUseArea）
  - 失败一律抛 errors.GeocodeError（含面向用户的中文 user_message）
"""

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from openai import OpenAI
from pydantic import ValidationError
from shapely.geometry import Point, box, GeometryCollection, shape
from shapely.ops import unary_union
import math

# 导入系统提示词
from prompts import SYSTEM_PROMPT
from geometry_utils import simplify_geometry
from splitter import take_compass_subset
from models import (
    LANDUSE_LABELS,
    NODE_TYPES,
    Between,
    BorderBetween,
    BorderOf,
    Buffer,
    CoastOf,
    Difference,
    DirectionalConstraint,
    DirectionalSubset,
    EventAffected,
    Intersection,
    LandUseArea,
    NamedPlace,
    OffTheCoastOf,
    SpatialNode,
    SpatialNodeTree,
    Union,
)
from place_lookup import PlaceLookup, PlaceSearchRequest
from errors import GeocodeError
from candidate_score import (
    SOURCE_BOUNDS as _SOURCE_BOUNDS,
    SOURCE_EVENT as _SOURCE_EVENT,
    SOURCE_LANDUSE as _SOURCE_LANDUSE,
    SOURCE_POINT as _SOURCE_POINT,
    SOURCE_POLYGON as _SOURCE_POLYGON,
    GeometryEvidence,
    score_candidate,
    score_candidates,
)

# 高德地图地理编码（可选，有 API Key 时启用）
try:
    from amap_geocoder import AmapGeocoder
    _AMAP_AVAILABLE = True
except ImportError:
    _AMAP_AVAILABLE = False

# OSM Polygon Place Lookup（真实多边形边界，对齐源项目架构）
try:
    from osm_place_lookup import OsmPolygonLookup
    _OSM_AVAILABLE = True
except ImportError:
    _OSM_AVAILABLE = False

# 事件数据（EONET）+ 事件点聚合（可选，EventAffected 节点需要）
try:
    from event_data import EonetClient, EventQuery
    from event_aggregate import aggregate_events
    _EVENTS_AVAILABLE = True
except ImportError:
    _EVENTS_AVAILABLE = False

# 自然地球海岸线数据（可选）
try:
    from natural_earth import coastline_of
    _COASTLINE_AVAILABLE = True
except ImportError:
    _COASTLINE_AVAILABLE = False

# LLM 输出未通过模型校验时的重试次数（重试会把错误回喂给模型）
MAX_PARSE_RETRIES = 2

# 事件查询 bbox 相对于基础地点边界的外扩量（度）。EONET 的 bbox 是"位置点是否
# 落在框内"的粗筛，不外扩会把正好压在边界上的上报漏掉；外扩过多只是多筛掉一些
# 点（求交时会被裁掉），代价很小。
_EVENT_BBOX_PAD_DEG = 0.05

# 数据源覆盖缺口的提示门槛（天）：最早事件晚于查询起点这么多天就提示。
# 取 30 天是为了放过"月初查询、当月还没事件"这类正常抖动，只提示真正的存档边界
# （如亚马逊野火最早只到 2024-05-27，而查询起点是 2023-01-01）。
_EVENT_COVERAGE_GAP_DAYS = 30


def _base_label(child: SpatialNode) -> str:
    """基础地点在错误提示与回显里的名字。"""
    return getattr(child, "name", None) or "该范围"


def _days_between(earlier: str, later: str) -> int:
    """两个 ISO 日期之间的天数（later - earlier）。"""
    from datetime import date
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


def _walk_nodes(root: SpatialNode):
    """深度优先遍历整棵节点树（含根）。"""
    yield root
    for key in ("child_node", "child_node_1", "child_node_2"):
        child = getattr(root, key, None)
        if child is not None:
            yield from _walk_nodes(child)
    for child in getattr(root, "child_nodes", None) or []:
        yield from _walk_nodes(child)


# 几何来源的强弱次序：数值越大越弱。整棵树取**最弱的一环**作为口径——树里只要
# 有一个 NamedPlace 降级成了中心点，最终几何的可信度就被它拖住；报最强的那一环
# 会让降级消失在平均里。事件排在 polygon 之后、point 之前：事件几何本身是真实
# 的（点凸包 ∩ 真实边界），但它依赖基础地点，而基础地点若只是中心点，整条链的
# 可信度就是中心点的（所以"中心点上的事件"正确地报 point，而不是 event）。
_SOURCE_WEAKNESS: dict[str, int] = {
    _SOURCE_POLYGON: 0,
    _SOURCE_EVENT: 1,
    _SOURCE_POINT: 2,
    _SOURCE_BOUNDS: 3,
}


def collect_evidence(node: SpatialNode,
                     geometry: object | None = None) -> GeometryEvidence:
    """从已查找、已构建完的节点树上采集一个候选的可比较证据。

    这是 S25 打分的输入。**只读程序里已经存在的事实**：`NamedPlace._geometry_source`
    与 `_hierarchy_confirmed`、`EventAffected._event_count` 等私有属性都是在实际做
    决定的地方写下的，这里不重新推断，也不解析任何 stdout 文本（打印的措辞是给人
    看的，会变）。

    Args:
        node: 已跑过 `_enrich_with_place_lookup` + `_enrich_with_events`
            + `_enrich_with_landuse` 的根节点。
        geometry: 构建出的 Shapely 几何；None 表示还没构建（面积按 0 记，会被硬淘汰）。

    Returns:
        GeometryEvidence，可直接交给 `candidate_score.score_candidate`。
    """
    sources: list[str] = []
    hierarchy_flags: list[bool] = []
    degradations = 0
    requested_country: str | None = None
    requested_region: str | None = None

    event_categories: tuple[str, ...] = ()
    event_count: int | None = None
    event_pieces: int | None = None
    event_window: str | None = None
    event_window_covered: bool | None = None

    landuse_ok = False

    for current in _walk_nodes(node):
        if isinstance(current, NamedPlace):
            source = current._geometry_source
            if source:
                sources.append(source)
                if source != _SOURCE_POLYGON:
                    degradations += 1
            if current._hierarchy_confirmed is not None:
                hierarchy_flags.append(current._hierarchy_confirmed)
            # 只填第一个非空的：同一棵树的多个 NamedPlace 通常共享同一套层级
            # 上下文（LLM 逐节点写的），取最外层那个即可，不拼字符串。
            if requested_country is None:
                requested_country = current.in_country
            if requested_region is None:
                requested_region = current.in_region

        elif isinstance(current, EventAffected):
            event_categories = tuple(current.categories)
            event_window = f"{current.time_start} ~ {current.time_end}"
            if current._event_aggregate is not None:
                sources.append(_SOURCE_EVENT)
                event_count = current._event_count
                event_pieces = current._event_aggregate.pieces
                event_window_covered = current._event_window_covered
            elif current._event_error is not None:
                # **硬淘汰判据在这里才可达**：只有"查询真的发出去了、数据源明确
                # 答了没有"才记 0 条（淘汰）；"数据源不可达"与"查询根本没发出去"
                # （基础地点没查到，链在 event_base_geometry 就断了）都必须记 None，
                # 否则候选会被按一个与它无关的理由淘汰。
                if current._event_queried and not current._event_error.service_unavailable:
                    event_count = 0
                else:
                    event_count = None

        elif isinstance(current, LandUseArea):
            # 结果几何来自用地多边形，child_node 只负责圈范围。所以这里**不**让
            # child 的降级拖低结果的可信度——"中心点撑成的圆盘"只是选区，答案本身
            # 是 OSM 里真实存在的地块，报 point 会把它冤枉成一个只知道位置的结果。
            # 降级次数照记（下面按 NamedPlace 计），因为"选区建立在中心点上"确实
            # 是一处该提示给用户的回退。
            if current._landuse_geometry is not None:
                landuse_ok = True

    if landuse_ok:
        source = _SOURCE_LANDUSE
    elif sources:
        source = max(sources, key=lambda s: _SOURCE_WEAKNESS.get(s, 0))
    else:
        # 构建成功必然至少有一个 NamedPlace（唯一的叶子节点），因此取不到来源只
        # 出现在"没跑构建就采证据"的调用上。按最强处理，避免凭空扣分。
        source = _SOURCE_POLYGON

    if False in hierarchy_flags:
        hierarchy_confirmed = False       # 有一个节点冲突就是冲突
    elif True in hierarchy_flags:
        hierarchy_confirmed = True
    else:
        hierarchy_confirmed = None        # 一个都没核验过 ≠ 冲突

    return GeometryEvidence(
        source=source,
        area_deg2=geometry.area if geometry is not None else 0.0,
        requested_country=requested_country,
        requested_region=requested_region,
        hierarchy_confirmed=hierarchy_confirmed,
        event_categories=event_categories,
        event_count=event_count,
        event_pieces=event_pieces,
        event_window=event_window,
        event_window_covered=event_window_covered,
        degradations=degradations,
    )


@dataclass
class BatchItem:
    """批量查询里的一条结果：成功带几何，失败带 GeocodeError。"""

    index: int              # 在输入里的序号（从 1 开始，与打印的行号一致）
    text: str
    geometry: object | None = None
    error: GeocodeError | None = None

    @property
    def ok(self) -> bool:
        return self.geometry is not None


# =============================================================================
# DeepSeek 大模型客户端
# =============================================================================
class DeepSeekClient:
    """
    DeepSeek API 封装，使用 OpenAI 兼容接口。
    DeepSeek 服务部署在国内（api.deepseek.com），无需 VPN 即可访问。
    """

    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: DeepSeek API密钥。若为None则依次从环境变量、config.json读取。
        """
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            self.api_key = self._read_key_from_config()
        if not self.api_key:
            raise GeocodeError(
                "未配置 DeepSeek API Key，无法调用大模型。"
                "请在 config.json 中设置 deepseek_api_key，"
                "或设置环境变量 DASHSCOPE_API_KEY。"
            )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat"

    @staticmethod
    def _read_key_from_config() -> str:
        """从 config.json 中读取 DeepSeek API Key（多路径搜索）。"""
        import sys
        search_dirs = [os.path.dirname(os.path.abspath(__file__))]
        if getattr(sys, 'frozen', False):
            search_dirs.insert(0, os.path.dirname(os.path.abspath(sys.executable)))
            search_dirs.insert(1, os.path.dirname(search_dirs[0]))
        for d in search_dirs:
            cfg = os.path.join(d, "config.json")
            if os.path.exists(cfg):
                try:
                    with open(cfg, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        key = data.get("deepseek_api_key", "")
                        if key:
                            return key
                except Exception:
                    pass
        return ""

    def chat(self, user_text: str, system_prompt: str = None) -> str:
        """
        发送对话请求到 DeepSeek。

        Args:
            user_text: 用户输入的自然语言地点描述。
            system_prompt: 系统提示词，为None则使用默认的 SYSTEM_PROMPT。

        Returns:
            大模型返回的文本（期望是纯JSON字符串）。
        """
        prompt = system_prompt or SYSTEM_PROMPT
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0.1,   # 低温度，提高输出一致性和稳定性
            max_tokens=4096,
        )
        return response.choices[0].message.content


# =============================================================================
# 空间几何生成器
# =============================================================================
class GeometryBuilder:
    """
    将大模型输出的空间节点树（models.py 的 pydantic 模型）递归转换为 Shapely 几何对象。

    支持的节点类型（12种，对齐源项目）：
      - NamedPlace：单个命名地点
      - Buffer：围绕子节点的指定距离缓冲区
      - DirectionalConstraint：子节点某个方向（north/south/east/west）的 X 公里范围
      - DirectionalSubset：子节点自身在某个方位上的那一半
      - Intersection：多个子区域的交集
      - Union：多个子区域的并集
      - Difference：区域差集（child_node_1 减去 child_node_2）
      - Between：两个地点之间的区域（凸包差集法）
      - BorderBetween：两个相邻区域的边界带
      - BorderOf：区域的边界线
      - CoastOf：区域的海岸线
      - OffTheCoastOf：离岸海域
    """

    # 节点类型 → 构建方法名。键集合必须与 models.NODE_TYPES 一致，
    # 模块末尾有断言把关：新增节点类型时漏改这里会直接报错。
    _BUILDER_METHODS = {
        "NamedPlace": "_build_named_place",
        "Buffer": "_build_buffer",
        "DirectionalConstraint": "_build_directional",
        "DirectionalSubset": "_build_directional_subset",
        "Intersection": "_build_intersection",
        "Union": "_build_union",
        "Difference": "_build_difference",
        "Between": "_build_between",
        "BorderBetween": "_build_border_between",
        "BorderOf": "_build_border_of",
        "CoastOf": "_build_coast_of",
        "OffTheCoastOf": "_build_off_the_coast_of",
        "LandUseArea": "_build_landuse_area",
        "EventAffected": "_build_event_affected",
    }

    @staticmethod
    def _km_to_degrees(km: float, latitude: float) -> float:
        """
        将公里近似转换为经纬度度数。
        1度纬度 ≈ 111.32 km
        1度经度 ≈ 111.32 * cos(lat) km

        Args:
            km: 距离（公里）
            latitude: 参考纬度（用于经度换算）

        Returns:
            近似的度数（取纬度方向和经度方向的平均值）
        """
        lat_deg = km / 111.32
        lon_deg = km / (111.32 * math.cos(math.radians(abs(latitude))) + 1e-10)
        # 返回两者中较大者，确保缓冲区覆盖足够范围
        return max(lat_deg, lon_deg)

    @classmethod
    def build_geometry(cls, node: SpatialNode) -> object:
        """
        递归构建几何对象（主入口）。

        Args:
            node: models.py 里的空间节点对象（NamedPlace / Buffer / ... 之一）。
                  经 NaturalLanguageGeocoder.geocode 调用时，节点已通过
                  SpatialNodeTree 校验并完成了地点查找。

        Returns:
            Shapely 几何对象。

        Raises:
            GeocodeError: 节点类型不支持或缺少必要字段。
        """
        node_type = getattr(node, "node_type", None)
        method_name = cls._BUILDER_METHODS.get(node_type)
        if method_name is None:
            raise GeocodeError(
                "无法识别的空间操作类型，无法生成几何。",
                detail=f"不支持的节点类型: {node_type}。支持的类型: {list(NODE_TYPES)}",
            )
        return getattr(cls, method_name)(node)

    # -------------------------------------------------------------------------
    @classmethod
    def _build_named_place(cls, node: NamedPlace) -> object:
        """
        构建命名地点几何。三级降级链，按可信度从高到低：

          1. PlaceLookup 的真实多边形（高德行政区划 / OSM）——唯一可信的边界
          2. 高德地理编码的精确坐标——只降级为中心点，不伪造边界
          3. LLM 提供的 bounds 矩形——最后兜底，仅在没有高德 Key 且 OSM 不可达时
             才会走到这里（这类场景下 LLM 的省/国家级 bounds 是唯一可用的范围信息）

        每一级降级都会打印出来，不会静默返回一个"看起来像边界"的东西。

        Returns:
            Polygon/MultiPolygon（真实边界）、Point（仅知中心点）或
            Polygon（LLM bounds 矩形，兜底）。

        Raises:
            GeocodeError: 三级都拿不到几何。
        """
        place_name = node.name

        # 层 1：真实多边形边界（来自 PlaceLookup：高德 District API / OSM Overpass）
        polygon_data = node._polygon
        if polygon_data:
            try:
                geom = shape(polygon_data)
                if not geom.is_empty:
                    # 原始边界可达数万顶点，先简化再交给后续的方向/缓冲/求交操作
                    return simplify_geometry(geom)
            except Exception as e:
                # 边界已经查到却用不了（几何非法、顶点数压不下来），要说清原因：
                # 否则最终只看到"降级为中心点"，无从判断是哪一环出了问题
                print(f"[警告] NamedPlace '{place_name}' 的真实边界无法使用：{e}")

        # 层 2：只知道中心点 → 明确降级为 Point，不伪造边界
        center_lon = node.center_lon
        center_lat = node.center_lat
        if center_lon is not None and center_lat is not None:
            print(
                f"[降级] NamedPlace '{place_name}' 未取得真实边界，"
                f"返回中心点 ({center_lon:.4f}, {center_lat:.4f})"
            )
            node._geometry_source = _SOURCE_POINT
            return Point(center_lon, center_lat)

        # 层 3：LLM bounds 兜底
        bounds = node.bounds
        if bounds:
            try:
                (min_lat, min_lon), (max_lat, max_lon) = bounds
                if min_lat < max_lat and min_lon < max_lon:
                    print(
                        f"[兜底] NamedPlace '{place_name}' 无真实边界与中心坐标，"
                        f"改用 LLM bounds 矩形 "
                        f"({min_lon:.4f}, {min_lat:.4f})-({max_lon:.4f}, {max_lat:.4f})"
                    )
                    # 兜底来源由**构建阶段**决定（查找阶段只知道 bounds 还在），
                    # 所以这一处必须在构建器里写
                    node._geometry_source = _SOURCE_BOUNDS
                    return box(min_lon, min_lat, max_lon, max_lat)
            except (TypeError, ValueError):
                pass

        raise GeocodeError(
            f"找不到地点「{place_name}」的边界，也没有可用的中心坐标或范围。",
            detail=(
                f"NamedPlace '{place_name}' 三级几何来源全部不可用"
                f"（无真实多边形、无高德坐标、无有效 bounds）。"
                f"请检查 PlaceLookup、高德地理编码与 LLM 输出。"
            ),
        )

    # -------------------------------------------------------------------------
    @classmethod
    def _build_buffer(cls, node: Buffer) -> object:
        """
        构建缓冲区几何。

        在子几何周围扩展指定公里数。由于经纬度是球面坐标，
        使用参考点的纬度来做近似的度-公里转换。
        """
        child_geom = cls.build_geometry(node.child_node)
        centroid = child_geom.centroid
        degree_distance = cls._km_to_degrees(node.distance_km, centroid.y)
        return child_geom.buffer(degree_distance)

    # -------------------------------------------------------------------------
    @classmethod
    def _build_directional(cls, node: DirectionalConstraint) -> object:
        """
        构建方向约束几何。

        例如"深圳大学以南5公里" → 从深圳大学南边界向南延伸5公里的矩形。

        direction 只能是 north/south/east/west（模型层已用 Literal 约束）。
        复合方向（西南等）由嵌套的 DirectionalSubset 表达。

        max_distance_km 控制方向延伸距离：模型里可选，缺省按 3km 处理，
        避免产生覆盖半个地球的矩形。
        """
        direction = node.direction
        max_distance_km = node.max_distance_km if node.max_distance_km is not None else 3.0

        child_geom = cls.build_geometry(node.child_node)
        minx, miny, maxx, maxy = child_geom.bounds
        centroid = child_geom.centroid

        degree_dist = cls._km_to_degrees(max_distance_km, centroid.y)

        # 构建受限方向矩形：
        # - 延伸方向：从子几何边缘向外延伸 max_distance_km
        # - 垂直方向：仅覆盖子几何在该维度上的范围（不加额外边距）
        #   这样 Intersection(north, west) 的结果才是紧凑的西北角矩形
        if direction == "north":
            return box(minx, maxy, maxx, maxy + degree_dist)
        elif direction == "south":
            return box(minx, miny - degree_dist, maxx, miny)
        elif direction == "east":
            return box(maxx, miny, maxx + degree_dist, maxy)
        elif direction == "west":
            return box(minx - degree_dist, miny, minx, maxy)

    # -------------------------------------------------------------------------
    @classmethod
    def _build_directional_subset(cls, node: DirectionalSubset) -> object:
        """
        构建方位子集几何（DirectionalSubset）。

        与 DirectionalConstraint 的区别：DirectionalConstraint 是"在该地点某个
        方向的 X 公里范围内"（位置在原地点之外，靠 max_distance_km 圈定），
        DirectionalSubset 是"在该地点本身之内取某一半"（结果是原几何的一部分）。

        例如"广东省的南半部分"→ DirectionalSubset(south, 广东省)，
        "北半球的东半部分"→ DirectionalSubset(east, DirectionalSubset(north, 北半球))。
        """
        child_geom = cls.build_geometry(node.child_node)
        return take_compass_subset(node.direction, child_geom)

    # -------------------------------------------------------------------------
    @classmethod
    def _build_intersection(cls, node: Intersection) -> object:
        """
        构建交集几何：多个子区域的重叠部分。

        与源项目 border_between 思路一致：对每个子区域加一个小缓冲区(0.005度≈500m)
        再求交集。这样即使两个区域在几何上不完全重合（如两个相邻省份的边界），
        也能得到有意义的交集区域。

        用于：
          - 明确交界处（"四川和云南的交界"）
          - 多条件约束（"新墨西哥州，阿尔伯克基以西"）
        """
        # 微小缓冲区确保相邻区域可产生交集
        # 0.005° ≈ 500m，足够处理城市级和省级的交集场景
        BORDER_BUFFER_DEG = 0.005

        result = None
        for child in node.child_nodes:
            child_geom = cls.build_geometry(child)
            # 对每个子几何加微小缓冲区，确保相邻区域可以产生交集
            buffered = child_geom.buffer(BORDER_BUFFER_DEG)
            result = buffered if result is None else result.intersection(buffered)

        if result is None or result.is_empty:
            raise GeocodeError(
                "这些区域之间没有重叠部分，请确认它们确实相邻。",
                detail="交集为空：子区域之间没有重叠部分。请检查地点是否确实相邻。",
            )

        return result

    # -------------------------------------------------------------------------
    @classmethod
    def _build_union(cls, node: Union) -> object:
        """
        构建并集几何：多个子区域的合并。

        用于 "A和B" 这类并列查询。
        """
        geoms = [cls.build_geometry(child) for child in node.child_nodes]
        return unary_union(geoms)

    # -------------------------------------------------------------------------
    @classmethod
    def _build_between(cls, node: Between) -> object:
        """
        构建两地点之间的区域。

        参照源项目使用凸包差集法：
          convex_hull(g1 ∪ g2) - convex_hull(g1) - convex_hull(g2)
        这样可以精确获得两个区域之间的空隙。
        """
        geom1 = cls.build_geometry(node.child_node_1)
        geom2 = cls.build_geometry(node.child_node_2)
        coll = GeometryCollection([geom1, geom2])
        convex = coll.convex_hull
        result = convex.difference(geom1.convex_hull).difference(geom2.convex_hull)
        if result.is_empty:
            raise GeocodeError(
                "这两个地点之间没有可识别的区域（两者紧挨或重合）。",
                detail="Between 的凸包差集为空：两地点之间没有区域。",
            )
        return result

    # -------------------------------------------------------------------------
    @classmethod
    def _build_difference(cls, node: Difference) -> object:
        """
        构建差集几何：child_node_1 减去 child_node_2。

        用于 "法国除巴黎外" 这类排除查询。
        """
        geom1 = cls.build_geometry(node.child_node_1)
        geom2 = cls.build_geometry(node.child_node_2)
        result = geom1.difference(geom2)
        if result.is_empty:
            raise GeocodeError(
                "相减之后没有剩余区域：被减掉的区域完全覆盖了原区域。",
                detail="差集为空：被减区域完全覆盖了源区域。",
            )
        return result

    # -------------------------------------------------------------------------
    @classmethod
    def _build_border_between(cls, node: BorderBetween) -> object:
        """
        构建两区域交界带。

        参照源项目：对两个子几何各缓冲3.5km后求交集，
        得到覆盖共享边界的长条形区域。
        """
        # BORDER_BUFFER_SIZE = 3.5 km，与源项目一致
        BORDER_BUFFER_KM = 3.5

        geom1 = cls.build_geometry(node.child_node_1)
        geom2 = cls.build_geometry(node.child_node_2)

        centroid1 = geom1.centroid
        degree_dist1 = cls._km_to_degrees(BORDER_BUFFER_KM, centroid1.y)
        buffered1 = geom1.buffer(degree_dist1)

        centroid2 = geom2.centroid
        degree_dist2 = cls._km_to_degrees(BORDER_BUFFER_KM, centroid2.y)
        buffered2 = geom2.buffer(degree_dist2)

        if not buffered1.intersects(buffered2):
            raise GeocodeError(
                "这两个区域不相邻，没有共同边界。",
                detail="BorderBetween：两区域缓冲 3.5km 后仍不相交。",
            )

        result = buffered1.intersection(buffered2)
        return result

    # -------------------------------------------------------------------------
    @classmethod
    def _build_border_of(cls, node: BorderOf) -> object:
        """
        构建区域边界线。

        返回子几何的边界（LineString/MultiLineString）。
        注意：此操作返回线几何，非面几何。
        """
        geom = cls.build_geometry(node.child_node)
        boundary = geom.boundary
        if boundary.is_empty:
            raise GeocodeError(
                "无法获取该区域的边界（该几何不存在边界线）。",
                detail=f"BorderOf：{getattr(geom, 'geom_type', type(geom))} 的 boundary 为空。",
            )
        return boundary

    # -------------------------------------------------------------------------
    @classmethod
    def _build_coast_of(cls, node: CoastOf) -> object:
        """
        构建区域海岸线。

        使用 Natural Earth 全球海岸线数据，与子几何求交集。
        需要 natural_earth.py 模块和 coastline GeoJSON 数据文件。
        """
        if not _COASTLINE_AVAILABLE:
            raise GeocodeError(
                "当前环境缺少海岸线数据，无法计算海岸线。",
                detail="CoastOf 需要 natural_earth.py 模块和 Natural Earth 海岸线数据。",
            )

        geom = cls.build_geometry(node.child_node)
        # 对子几何加2km缓冲区后与全球海岸线求交集
        centroid = geom.centroid
        degree_2km = cls._km_to_degrees(2.0, centroid.y)
        buffered = geom.buffer(degree_2km)

        coast = coastline_of(buffered)
        if coast is None or coast.is_empty:
            raise GeocodeError(
                "该区域附近没有找到海岸线。",
                detail="CoastOf：与全球海岸线求交后为空。",
            )
        return coast

    # -------------------------------------------------------------------------
    @classmethod
    def _build_off_the_coast_of(cls, node: OffTheCoastOf) -> object:
        """
        构建离岸海域。

        参照源项目：海岸线向外缓冲指定距离，再减去陆地部分。
        """
        if not _COASTLINE_AVAILABLE:
            raise GeocodeError(
                "当前环境缺少海岸线数据，无法计算离岸区域。",
                detail="OffTheCoastOf 需要 natural_earth.py 模块和 Natural Earth 海岸线数据。",
            )

        geom = cls.build_geometry(node.child_node)
        centroid = geom.centroid

        # 获取海岸线
        degree_2km = cls._km_to_degrees(2.0, centroid.y)
        buffered = geom.buffer(degree_2km)

        coast = coastline_of(buffered)
        if coast is None or coast.is_empty:
            raise GeocodeError(
                "该区域附近没有找到海岸线，无法确定离岸位置。",
                detail="OffTheCoastOf：与全球海岸线求交后为空。",
            )

        # 海岸线向外缓冲 → 减去陆地
        degree_dist = cls._km_to_degrees(node.distance_km, centroid.y)
        buffered_coast = coast.buffer(degree_dist)
        result = buffered_coast.difference(geom)
        if result.is_empty:
            raise GeocodeError(
                "离岸区域为空，没有可输出的范围。",
                detail="OffTheCoastOf：海岸线外扩后减去陆地结果为空。",
            )
        return result

    # -------------------------------------------------------------------------
    @classmethod
    def area_geometry(cls, child: SpatialNode, purpose: str) -> object:
        """取 child_node 的几何，并保证它"有面积"——面状查询的求交底图。

        降级链第 2 层只给一个中心点，中心点定不出范围，也就没法定事件查询的 bbox、
        没法在用地查询里圈范围。有 `radius_km` 时按半径撑成一个面（与 Buffer 同一个
        公里→度换算），没有就明确报错，而不是把整个地球当范围。

        `purpose` 是"拿这个面积干什么"的中文短语（"计算受影响区域"/"圈定用地范围"），
        只进打印与报错文案：同一次降级在不同查询里的后果不同，报错必须说清断在哪一步。

        放在 GeometryBuilder 而不是查找阶段：它只做几何换算，不发任何请求。

        Raises:
            GeocodeError: 子几何没有面积且没有 radius_km 可撑开。
        """
        geom = cls.build_geometry(child)
        if getattr(geom, "area", 0.0) > 0.0:
            return geom

        radius_km = getattr(child, "radius_km", None)
        if isinstance(geom, Point) and radius_km:
            deg = cls._km_to_degrees(float(radius_km), geom.y)
            print(f"[{purpose}] 基础地点只有中心点，按 radius_km={radius_km} "
                  f"（≈{deg:.4f}°）撑成面后求交")
            return geom.buffer(deg)

        label = _base_label(child)
        raise GeocodeError(
            f"{label}没有可用的边界（只拿到了中心点），无法{purpose}。",
            detail=f"child_node={getattr(child, 'node_type', '?')} 的几何是 "
                   f"{getattr(geom, 'geom_type', type(geom).__name__)}，"
                   f"没有面积也没有 radius_km 可供撑开",
        )

    @classmethod
    def event_base_geometry(cls, child: SpatialNode) -> object:
        """事件聚合的求交底图（`area_geometry` 的事件封装，保留原名以免既有引用改名）。"""
        return cls.area_geometry(child, "计算受影响区域")

    @classmethod
    def _build_event_affected(cls, node: EventAffected) -> object:
        """构建"被某类事件影响过的区域"几何。

        几何不是在这里算的——事件查询要发网络请求，而构建阶段必须保持纯几何
        （`_polygon` 已确立的架构约束）。`_enrich_with_events` 在查找阶段把聚合结果
        写进 `node._event_aggregate`，这里只负责取出来与报错。

        两种"拿不到几何"的情形必须分开报（S20 "数据源没问到 vs 答无此地"
        这条判据的第三次应用）：
          - 数据源不可达 → service_unavailable=True，提示稍后重试
          - 问到了但这个时间窗/范围内没有事件 → 提示换时间或换区域

        Raises:
            GeocodeError: `_event_aggregate` 为空。
        """
        aggregate = node._event_aggregate
        if aggregate is not None:
            return aggregate.geometry

        # 查找阶段失败时会把原始异常留在节点上，这里原样抛出：
        # 它已经带着正确的 service_unavailable 与 user_message，重新构造一份
        # 只会让两处口径有机会不一致。
        stored = node._event_error
        if stored is not None:
            raise stored

        raise GeocodeError(
            "事件区域计算失败，无法生成几何。",
            detail="EventAffected._event_aggregate 为空，且查找阶段没有记录失败原因。"
                   "通常意味着 geocode() 跳过了 _enrich_with_events。",
        )

    # -------------------------------------------------------------------------
    @classmethod
    def _build_landuse_area(cls, node: LandUseArea) -> object:
        """构建"某范围内某一类用地"的几何。

        与 EventAffected 同一个模式：几何不是在这里算的（要发 Overpass 请求，
        构建阶段必须保持纯几何）。`_enrich_with_landuse` 在查找阶段把裁剪、并集
        后的结果写进 `node._landuse_geometry`，这里只负责取出来与报错。

        两种"拿不到几何"的情形分开报（与事件判据同源）：
          - 数据源不可达 → service_unavailable=True，提示稍后重试
          - 数据源答了、但该范围内没有标注这类用地 → 提示换范围或换类别

        Raises:
            GeocodeError: `_landuse_geometry` 为空。
        """
        geometry = node._landuse_geometry
        if geometry is not None:
            return geometry

        stored = node._landuse_error
        if stored is not None:
            raise stored

        raise GeocodeError(
            "用地类型查询没有算出几何。",
            detail="LandUseArea._landuse_geometry 为空，且查找阶段没有记录失败原因。"
                   "通常意味着 geocode() 跳过了 _enrich_with_landuse。",
        )


# 构建器与节点模型必须一一对应：新增节点类型时漏改任一边，这里在导入阶段就报出来
assert set(GeometryBuilder._BUILDER_METHODS) == set(NODE_TYPES), (
    f"构建器与 models.NODE_TYPES 不一致："
    f"构建器多出 {set(GeometryBuilder._BUILDER_METHODS) - set(NODE_TYPES)}，"
    f"模型多出 {set(NODE_TYPES) - set(GeometryBuilder._BUILDER_METHODS)}"
)


# =============================================================================
# 地理编码主类
# =============================================================================
class NaturalLanguageGeocoder:
    """
    自然语言地理编码器。

    将自然语言地点描述转换为可用的地理几何数据（Shapely对象、GeoJSON）。

    精度机制（参照源项目的多层保障思路）：
      层1 - LLM语义解析：DeepSeek将自然语言转换为空间节点树（JSON结构）
      层2 - 地点查找：PlaceLookup 取真实多边形；失败时用高德精确坐标（WGS-84）
            降级为中心点，二者皆无则明确报错，不伪造边界
      层3 - 空间几何算法：Shapely执行Buffer/Intersection/Union等专业运算

    使用示例:
        gc = NaturalLanguageGeocoder(amap_api_key="your_key")
        geometry = gc.geocode("深圳人才公园")
        geojson = gc.to_geojson(geometry)
    """

    def __init__(self, api_key: str = None, amap_api_key: str = None,
                 place_lookup: PlaceLookup = None):
        """
        Args:
            api_key: DeepSeek API密钥。为None则使用默认配置。
            amap_api_key: 高德地图Web服务API Key。为None则跳过精确地理编码，
                          仅依赖LLM提供的坐标（精度较低）。
                          免费注册: https://lbs.amap.com
            place_lookup: 自定义的地名查找后端（place_lookup.PlaceLookup 实例）。
                          为None则用默认的 OsmPolygonLookup（高德 District + OSM Overpass）。
                          传入自定义实现即可整体替换数据源，无需改动本类。
        """
        self.llm = DeepSeekClient(api_key=api_key)
        self.builder = GeometryBuilder()
        # EONET 客户端懒建（见 _eonet_client）：不用事件节点的查询不必付构造代价
        self._eonet = None
        self.amap = None
        if amap_api_key and _AMAP_AVAILABLE:
            self.amap = AmapGeocoder(amap_api_key)
        elif amap_api_key and not _AMAP_AVAILABLE:
            print("[警告] amap_geocoder.py 未找到，无法使用高德地理编码")

        # PlaceLookup：优先用注入的实现，否则用默认的 Overpass 后端
        if place_lookup is not None:
            self.place_lookup = place_lookup
        elif _OSM_AVAILABLE:
            self.place_lookup = OsmPolygonLookup(amap_geocoder=self.amap)
        else:
            self.place_lookup = None

    def _enrich_with_place_lookup(self, node: SpatialNode):
        """
        递归遍历空间节点树，用 PlaceLookup 获取每个 NamedPlace 的真实多边形几何。

        对齐源项目架构：每个 NamedPlace 通过 PlaceLookup.search() 获取真实的
        Polygon/MultiPolygon 边界，不使用矩形近似或圆形缓冲区。

        Geometry 以 GeoJSON dict 形式存储在 node._polygon 中（NamedPlace 的私有属性），
        _build_named_place 会将其转为 Shapely geometry。

        三级降级链（与 _build_named_place 一致）：
          1. PlaceLookup 真实多边形 → node._polygon，此时 LLM 的 bounds 已无用，丢弃
          2. 高德精确坐标 → node.center_lon/lat（附 radius_km 元数据）
          3. 都没有 → 只有在"这次压根没问到任何数据源"（无 PlaceLookup 且无高德，
             或 PlaceLookup 服务不可用且无高德）时才保留 LLM 的 center/bounds，
             交给 _build_named_place 兜底；否则清空它们直接报错——数据源已答复
             "查无此地"时，不允许再拿大模型自己给的坐标冒充结果。

        Args:
            node: 空间节点树的一个节点（原地修改）
        """
        if isinstance(node, NamedPlace):
            name = node.name

            # 数据源是否真的答过话：决定下面能不能退回"大模型自己给的坐标"
            lookup_attempted = False
            service_unavailable = False

            if self.place_lookup:
                lookup_attempted = True
                try:
                    request = PlaceSearchRequest(
                        name=name,
                        in_continent=node.in_continent,
                        in_country=node.in_country,
                        in_region=node.in_region,
                    )
                    candidate = self.place_lookup.search_candidate(request)
                    geom = candidate.geometry
                    # 存储 GeoJSON dict 以便序列化/调试
                    from shapely import to_geojson
                    node._polygon = json.loads(to_geojson(geom))
                    # 同时存储中心坐标（用于方向约束等需要坐标的操作）
                    node.center_lon = geom.centroid.x
                    node.center_lat = geom.centroid.y
                    # 真实多边形已到手，LLM 给的粗略 bounds 不再需要
                    node.bounds = None
                    # 给 S25 的证据采集留痕：来源等级与层级核验结果
                    node._geometry_source = _SOURCE_POLYGON
                    node._hierarchy_confirmed = candidate.hierarchy_confirmed
                    return
                except GeocodeError as e:
                    service_unavailable = e.service_unavailable
                    print(f"[PlaceLookup 失败] {name}: {e}")
                except Exception as e:
                    print(f"[PlaceLookup 失败] {name}: {e}")

            # 层 2：无真实多边形时回退到高德，只取中心坐标（不伪造边界）
            if self.amap:
                precise = self.amap.get_precise_bounds(name, node.in_region, node.in_country)
                if precise:
                    node.center_lon = precise["center_lon"]
                    node.center_lat = precise["center_lat"]
                    node.radius_km = precise["radius_km"]
                    node._amap_level = precise.get("level", "")
                    # 只拿到中心点：几何来源降一级。层级由高德在请求时按
                    # in_region/in_country 过滤过，但本类拿不到"是否真的核验过"
                    # 的信号，所以留 None（未核验）而不是替它宣称一致。
                    node._geometry_source = _SOURCE_POINT
                    return

            # 层 3 的兜底资格，只看一件事：这次有没有"问到"数据源。
            #   - 服务不可用（超时/被拒/限流）→ 没问到 → 按原设计退回 LLM 的粗略范围
            #   - 压根没有 PlaceLookup → 没问到 → 同上
            #   - 数据源答复了"查无此地"→ 问到了 → 不许再让 LLM 的坐标顶上来
            # 高德配没配**不**参与这个判断：高德查不到"France"是它不覆盖国外，
            # 属于"没问到"，不是"法国不存在"。早先把 `and not self.amap` 一起写进来，
            # 结果 Overpass 三个镜像被限流时，连"法国和西班牙的边界"都直接报错了。
            can_trust_llm = not lookup_attempted or service_unavailable
            if not can_trust_llm:
                node.center_lon = None
                node.center_lat = None
                node.radius_km = None
                node.bounds = None
                node._amap_level = ""
                print(
                    f"[查无此地] NamedPlace '{name}'：数据源已答复但无此地点，"
                    f"丢弃大模型给出的中心坐标/范围，交由上层报错。"
                )
            return

        # 递归处理子节点
        for key in ("child_node", "child_node_1", "child_node_2"):
            child = getattr(node, key, None)
            if child is not None:
                self._enrich_with_place_lookup(child)
        for child in getattr(node, "child_nodes", None) or []:
            self._enrich_with_place_lookup(child)

    # -------------------------------------------------------------------------
    # 事件影响区域（EventAffected）
    # -------------------------------------------------------------------------
    def _enrich_with_events(self, node: SpatialNode):
        """递归遍历节点树，为每个 EventAffected 计算"受影响区域"。

        **调用顺序是硬约束**：必须在 `_enrich_with_place_lookup` 跑完之后调用。
        事件区域 = 事件点凸包 ∩ 基础地点的真实边界，边界没查到就无从求交，也没法
        定事件查询的 bbox。地点查找本身是幂等代价很高（一次 Overpass 几十秒）的
        操作，所以这里**不**自己再调一遍，只依赖调用方的顺序——`geocode()` 保证它。

        结果写进 `node._event_aggregate`；失败时把原始 GeocodeError 留在
        `node._event_error` 上而不是当场抛出，让构建阶段统一决定报哪个错
        （见 `GeometryBuilder._build_event_affected`）。

        Args:
            node: 空间节点树的一个节点（原地修改）
        """
        if isinstance(node, EventAffected):
            self._resolve_events(node)
            return

        for key in ("child_node", "child_node_1", "child_node_2"):
            child = getattr(node, key, None)
            if child is not None:
                self._enrich_with_events(child)
        for child in getattr(node, "child_nodes", None) or []:
            self._enrich_with_events(child)

    # -------------------------------------------------------------------------
    # 用地类型区域（LandUseArea）
    # -------------------------------------------------------------------------
    def _enrich_with_landuse(self, node: SpatialNode):
        """递归遍历节点树，为每个 LandUseArea 查出"该范围内这类用地的真实地块"。

        **顺序是硬约束**：必须排在 `_enrich_with_place_lookup` 与 `_enrich_with_events`
        之后。这个节点的几何 = child_node 的几何 ∩ OSM `landuse=*` 的多边形，
        child 的几何要先查完/算完才谈得上求交；child 若又是一个 EventAffected，
        也已经在上一轮拿到了 `_event_aggregate`。

        **后序（先子后父）**：求交要用 child 的几何，而 `build_geometry(child)` 会
        递归到孙子节点；若父子都是 LandUseArea，父必须先等子解析完，否则会在子节点
        上抛"查找阶段没记录失败原因"。子树结构固定，递归天然是后序。

        结果写进 `node._landuse_geometry`；失败时把原始 GeocodeError 留在
        `node._landuse_error` 上而不是当场抛出，让构建阶段统一决定报哪个错
        （与 EventAffected 同一模式）。

        Args:
            node: 空间节点树的一个节点（原地修改）
        """
        for key in ("child_node", "child_node_1", "child_node_2"):
            child = getattr(node, key, None)
            if child is not None:
                self._enrich_with_landuse(child)
        for child in getattr(node, "child_nodes", None) or []:
            self._enrich_with_landuse(child)

        if isinstance(node, LandUseArea):
            self._resolve_landuse(node)

    def _resolve_landuse(self, node: LandUseArea):
        """查用地多边形 → 与范围求交并集 → 写回 node。失败只记录，不抛出。"""
        label = LANDUSE_LABELS.get(node.landuse, node.landuse)
        if self.place_lookup is None or not hasattr(self.place_lookup, "search_landuse"):
            node._landuse_error = GeocodeError(
                f"当前数据源不支持按用地类型查询「{label}」。",
                detail="LandUseArea 需要实现了 search_landuse 的 PlaceLookup"
                       "（OsmPolygonLookup）；当前注入的 lookup 没有这个方法。",
            )
            return

        try:
            scope = GeometryBuilder.area_geometry(node.child_node, "圈定用地范围")
            geometry = self.place_lookup.search_landuse(node.landuse, scope, label=label)
        except GeocodeError as e:
            # 数据源不可达与"答了、但该范围内没有这类用地"在这一层不做区分——
            # 两者都原样带到构建阶段再报，避免同一个判据写两遍。
            print(f"[用地] {e.user_message}")
            node._landuse_error = e
            return

        node._landuse_geometry = geometry
        node._landuse_pieces = (
            len(geometry.geoms) if geometry.geom_type == "MultiPolygon" else 1
        )
        print(f"[用地回显] {label}（landuse={node.landuse}）："
              f"命中 {node._landuse_pieces} 块，面积 {geometry.area:.6g} 平方度")

    def _resolve_events(self, node: EventAffected):
        """查事件 → 聚合成面 → 写回 node。失败只记录，不抛出。"""
        if not _EVENTS_AVAILABLE:
            node._event_error = GeocodeError(
                "当前环境缺少事件数据模块，无法计算受影响区域。",
                detail="EventAffected 需要 event_data.py 与 event_aggregate.py。",
            )
            return

        try:
            base = GeometryBuilder.event_base_geometry(node.child_node)
            minx, miny, maxx, maxy = base.bounds
            pad = _EVENT_BBOX_PAD_DEG
            query = EventQuery(
                categories=tuple(node.categories),
                time_start=node.time_start,
                time_end=node.time_end,
                bbox=(minx - pad, miny - pad, maxx + pad, maxy + pad),
            )
            records = self._eonet_client().fetch(query)
            # 查询真的发出去了（数据源给了答复，哪怕是空列表）。之后的任何失败
            # 才谈得上"数据源答了没有事件"；在这行之前失败的都是"没问到"。
            node._event_queried = True
            points = [pt for record in records for pt in record.points()]
            aggregate = aggregate_events(
                points, base, base_label=_base_label(node.child_node),
            )
        except GeocodeError as e:
            # 数据源不可达（service_unavailable）与"问到但没有事件"在这一层
            # 不做区分——两者都原样带到构建阶段再报，避免同一个判据写两遍。
            print(f"[事件] {e.user_message}")
            node._event_error = e
            return

        node._event_aggregate = aggregate
        # 给 S25 的证据采集留痕：命中条数与时间窗覆盖情况（回显里算过一次，
        # 存下来让打分不必再把回显文本读一遍）
        node._event_count = len(records)
        node._event_window_covered = (
            _days_between(node.time_start, records[0].date[:10])
            <= _EVENT_COVERAGE_GAP_DAYS
        )
        self._print_event_echo(node, records, aggregate)

    def _eonet_client(self):
        """懒建的 EONET 客户端，跨节点复用（内部有查询级磁盘缓存）。"""
        client = getattr(self, "_eonet", None)
        if client is None:
            client = EonetClient()
            self._eonet = client
        return client

    @staticmethod
    def _print_event_echo(node: EventAffected, records, aggregate):
        """把解析出的时间窗与证据量回显出来。

        这是决策里"相对时间换算成绝对区间 + 回显"的落地点：用户说"过去两三年"，
        最终用的是哪个绝对区间、命中多少条、聚成几块，必须打印出来。模糊表述的
        歧义只有暴露在明处才能被用户发现并纠正。

        另外要暴露**数据源的覆盖缺口**：EONET 的亚马逊野火存档最早只到
        2024-05-27（见 event_data.py 的模块说明），"过去两三年"实际只有约 1.3 年
        证据。不提示的话，用户会以为 2023 年真的没烧过。
        """
        categories = "、".join(node.categories)
        window = f"{node.time_start} ~ {node.time_end}"
        if not records:
            print(f"[时间回显] {window}；{categories} 命中 0 条 → 无区域")
            return

        latest = max(record.date for record in records)[:10]
        print(f"[时间回显] {window}；{categories} 命中 {len(records)} 条"
              f"（{records[0].date[:10]} ~ {latest}）"
              f"→ {aggregate.pieces} 块，{aggregate.points_total} 个位置点，"
              f"网格 {aggregate.grid_deg:.4f}°（≈{aggregate.grid_deg * 111:.0f}km）")

        # 判据读 `_event_window_covered`（在 `_resolve_events` 里写下、供 S25 采证据
        # 用同一个值），不在这里重算：同一个判断写两遍就会在改动时分叉。天数只是
        # 为了把话说清楚，重新算一次不影响结论。
        if not node._event_window_covered:
            earliest = records[0].date[:10]
            print(f"[时间回显] 注意：数据源在 {node.time_start} ~ "
                  f"{earliest} 这 {_days_between(node.time_start, earliest)} 天内"
                  f"没有该类事件记录，该区间不受这张图覆盖。")

    def parse_text(self, text: str) -> SpatialNode:
        """
        调用大模型把自然语言解析为空间节点树，并用 pydantic 模型校验。

        校验不过时把"上次输出 + 具体错误"回喂给大模型重试，最多 MAX_PARSE_RETRIES 次。
        纯 JSON 语法错误（json.loads 失败）同样走重试：模型常因多余文字或多写逗号翻车。

        Args:
            text: 自然语言地点描述。

        Returns:
            校验通过的根节点（models.py 中的某个 SpatialNode 子类）。

        Raises:
            GeocodeError: 重试次数用尽仍未得到合法节点树，或无法连接大模型。
        """
        prompt = text
        raw_response = ""
        for attempt in range(MAX_PARSE_RETRIES + 1):
            try:
                raw_response = self.llm.chat(prompt)
            except GeocodeError:
                raise
            except Exception as e:
                # 网络/鉴权问题不会因为重试同样的请求而好转，直接报出来
                raise GeocodeError(
                    "无法连接大模型服务，请检查网络与 API Key 配置。",
                    detail=f"调用 DeepSeek 失败：{type(e).__name__}: {e}",
                ) from e
            try:
                return self._parse_response(raw_response)
            except (GeocodeError, ValidationError) as e:
                if attempt == MAX_PARSE_RETRIES:
                    raise GeocodeError(
                        "大模型连续多次未能返回符合格式的解析结果，无法继续。",
                        detail=(
                            f"大模型输出连续 {MAX_PARSE_RETRIES + 1} 次未通过节点模型校验。\n"
                            f"最后一次的错误：{e}\n"
                            f"最后一次的原始返回内容：\n{raw_response}"
                        ),
                    ) from e
                print(f"[重试 {attempt + 1}/{MAX_PARSE_RETRIES}] 输出不合法：{e}")
                prompt = (
                    f"{text}\n\n"
                    f"你上一次的输出不合法，请修正后重新输出完整 JSON。\n"
                    f"上一次输出：\n{raw_response}\n\n"
                    f"错误信息：\n{e}"
                )
        raise AssertionError("unreachable")

    @staticmethod
    def _parse_response(raw_response: str) -> SpatialNode:
        """去掉 markdown 围栏 → 校验为节点树 → 返回根节点。"""
        text = raw_response.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise GeocodeError(
                "大模型返回的内容不是有效 JSON。",
                detail=f"大模型返回的不是有效JSON：{e}。原始内容：{text[:200]}",
            ) from e

        return SpatialNodeTree.model_validate(payload).root

    def geocode_with_evidence(self, text: str) -> tuple[object, GeometryEvidence]:
        """
        与 geocode 同一条管线，但把"这个答案是怎么来的"一并返回（诊断入口）。

        两个入口共用一个实现：`geocode()` 只取几何，返回类型与打印行为保持不变；
        需要给候选打分、给用户解释来源的调用方（扩展查询 agent 排名、CLI 诊断
        输出）从这条入口取证据。

        证据由程序直接产出（见 collect_evidence），**不允许回头解析 stdout**：
        打印是给人看的，措辞一改，靠解析它的代码就静默失效。

        Args:
            text: 自然语言地点描述。

        Returns:
            (Shapely 几何对象, GeometryEvidence)。

        Raises:
            GeocodeError: 解析或几何构建失败（含面向用户的中文 user_message）。
        """
        spatial_node = self.parse_text(text)
        # 用 PlaceLookup 获取每个 NamedPlace 的真实多边形边界（对齐源项目架构）
        self._enrich_with_place_lookup(spatial_node)
        # 再算事件影响区域：必须在地点查找**之后**（事件要跟真实边界求交）
        self._enrich_with_events(spatial_node)
        # 最后查用地类型区域：要与 child_node 算好的几何求交，故排在前两步之后
        self._enrich_with_landuse(spatial_node)
        print(f"[解析结果] {spatial_node.model_dump_json(indent=2, exclude_none=True)}")
        geometry = self.builder.build_geometry(spatial_node)
        return geometry, collect_evidence(spatial_node, geometry)

    def geocode(self, text: str) -> object:
        """
        将自然语言描述转换为 Shapely 几何对象（核心方法）。

        流程：文本 → LLM解析（含模型校验与重试）→ 查找真实边界/精确坐标 →
              递归构建几何 → Shapely对象

        Args:
            text: 自然语言地点描述。

        Returns:
            Shapely 几何对象（Point/Polygon/MultiPolygon等）。

        Raises:
            GeocodeError: 解析或几何构建失败（含面向用户的中文 user_message）。
        """
        return self.geocode_with_evidence(text)[0]

    @staticmethod
    def to_geojson(geometry: object) -> str:
        """
        将 Shapely 几何对象转换为 GeoJSON Feature 字符串。

        Args:
            geometry: Shapely 几何对象。

        Returns:
            格式化后的 GeoJSON 字符串。
        """
        from shapely import to_geojson
        feature = {
            "type": "Feature",
            "geometry": json.loads(to_geojson(geometry)),
            "properties": {"name": "查询区域"}
        }
        return json.dumps(feature, ensure_ascii=False, indent=2)

    def geocode_to_geojson(self, text: str) -> str:
        """
        一站式方法：自然语言直接输出 GeoJSON 字符串。

        Args:
            text: 自然语言地点描述。

        Returns:
            GeoJSON Feature 字符串。
        """
        geometry = self.geocode(text)
        return self.to_geojson(geometry)

    def geocode_batch(self, texts: Iterable[str]) -> list[BatchItem]:
        """
        逐条编码一批文本，单条失败不影响后续条目。

        返回 BatchItem 列表而非 list[geometry]：批量场景里"哪几条失败、为什么失败"
        和几何同等重要——只回一个几何列表，调用方无从分辨哪一条失败，
        也就没法单独报告与重跑。

        Args:
            texts: 待编码的文本序列（每项一条查询）。

        Returns:
            BatchItem 列表，顺序与输入一致；失败项的 geometry 为 None，error 带原因。
        """
        items = list(texts)
        results: list[BatchItem] = []
        for index, text in enumerate(items, 1):
            print(f"[批量 {index}/{len(items)}] {text}")
            try:
                geometry = self.geocode(text)
            except GeocodeError as e:
                print(f"   [失败] {e.user_message}")
                results.append(BatchItem(index=index, text=text, error=e))
            else:
                results.append(BatchItem(index=index, text=text, geometry=geometry))
        succeeded = sum(1 for item in results if item.ok)
        print(f"[批量完成] 成功 {succeeded} / {len(results)} 条")
        return results

    @staticmethod
    def to_feature_collection(items: Sequence[BatchItem]) -> str:
        """
        把批量结果导出为一个 GeoJSON FeatureCollection（只含成功的条目）。

        失败条目不进 FeatureCollection：空几何会污染下游工具（QGIS/geopandas
        会对空几何报错或警告）；失败信息在 BatchItem.error 里，由调用方单独报告。

        Args:
            items: geocode_batch 的返回值。

        Returns:
            FeatureCollection 的 JSON 字符串，每个 Feature 的 properties 带
            原始查询文本与序号，便于在 QGIS 里逐条核对。
        """
        from shapely import to_geojson
        features = []
        for item in items:
            if not item.ok:
                continue
            features.append({
                "type": "Feature",
                "geometry": json.loads(to_geojson(item.geometry)),
                "properties": {"index": item.index, "query": item.text},
            })
        return json.dumps(
            {"type": "FeatureCollection", "features": features},
            ensure_ascii=False, indent=2,
        )


# =============================================================================
# 离线自检：EventAffected 的接线（python geocoding.py --self-check）
# =============================================================================
def _self_check_events() -> int:
    """不联网、不调大模型的自检：只验证 EventAffected 的三处接线。

    刻意只覆盖**接线**，不覆盖 LLM 解析（那需要网络与 Key）：
      1. 节点模型能否校验（含时间格式与顺序）
      2. `_enrich_with_place_lookup` → `_enrich_with_events` → `build_geometry`
         这条链在四种不同结局下分别给出什么
      3. 时间回显与覆盖缺口提示是否真的打印出来

    数据源用假实现：PlaceLookup 返回固定多边形，EONET 客户端返回固定事件记录。
    这样"事件区域 = 点凸包 ∩ 真实边界"这条逻辑可以完全离线复现。
    """
    import io
    from contextlib import redirect_stdout

    failures: list[str] = []

    def check(name: str, condition: bool, extra: str = ""):
        if condition:
            print(f"  [通过] {name}")
        else:
            print(f"  [失败] {name} {extra}")
            failures.append(name)

    # ---- 假数据源 ----
    AMZ = box(-70.0, -12.0, -60.0, -2.0)

    class _FakeLookup(PlaceLookup):
        """总是返回同一个矩形，不发网络请求。"""

        def __init__(self, geometry=AMZ):
            self.geometry = geometry

        def search_for_places(self, request, limit=5):
            from place_lookup import PlaceCandidate
            return [PlaceCandidate(geometry=self.geometry, name=request.name,
                                   source="fake", score=1.0)]

    class _FakeEonet:
        """按构造时给的记录返回；records=None 表示模拟"数据源不可达"。"""

        def __init__(self, records=(), error=None):
            self.records = records
            self.error = error
            self.queries = []

        def fetch(self, query, use_cache=True):
            self.queries.append(query)
            if self.error is not None:
                raise self.error
            return list(self.records)

    def event(lon, lat, date_str="2024-06-01"):
        """造一条带单个位置的假事件记录。"""
        from event_data import EventRecord
        return EventRecord(id=f"{lon},{lat}", title="fake", category="wildfires",
                           date=date_str, closed=None, lon=lon, lat=lat,
                           magnitude=None)

    def events_at(cx, cy, n=9, spacing=0.3, date_str="2024-06-01"):
        """在 (cx, cy) 周围造一个 n 点方阵的假事件集。

        必须成"方阵"而不是一两个点：默认 min_cluster_size=2 会把孤立点判成
        零散（那是 S23 刻意的降噪行为），测事件接线时不能踩到它。
        """
        side = int(math.ceil(math.sqrt(n)))
        return [event(cx + i * spacing, cy + j * spacing, date_str)
                for i in range(side) for j in range(side)][:n]

    def make_geocoder(lookup, eonet):
        """绕过 __init__（那需要 API Key 与 OpenAI 客户端），只装自检需要的部件。"""
        geocoder = NaturalLanguageGeocoder.__new__(NaturalLanguageGeocoder)
        geocoder.place_lookup = lookup
        geocoder.amap = None
        geocoder._eonet = eonet
        return geocoder

    def tree(**overrides):
        payload = {
            "node_type": "EventAffected",
            "child_node": {"node_type": "NamedPlace", "name": "亚马逊雨林"},
            "categories": ["wildfires"],
            "time_start": "2023-09-20",
            "time_end": "2026-09-20",
        }
        payload.update(overrides)
        return SpatialNodeTree.model_validate(payload).root

    def run(node, lookup=None, eonet=None):
        """跑完整条链，返回 (geometry, 打印出来的文本)。"""
        geocoder = make_geocoder(lookup or _FakeLookup(), eonet or _FakeEonet())
        geocoder._enrich_with_place_lookup(node)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            geocoder._enrich_with_events(node)
        try:
            geometry = GeometryBuilder.build_geometry(node)
        except GeocodeError as e:
            geometry = e
        return geometry, buffer.getvalue()

    # ---- 节点校验 ----
    print("== EventAffected 节点校验 ==")
    node = tree()
    check("能校验出 EventAffected 节点", node.node_type == "EventAffected", node.node_type)
    check("child_node 是 NamedPlace", node.child_node.node_type == "NamedPlace",
          node.child_node.node_type)
    check("categories 是列表", node.categories == ["wildfires"], str(node.categories))

    for bad, why in (
        ({"categories": []}, "空 categories"),
        ({"time_start": "2023-02-30"}, "不存在的日期"),
        ({"time_start": "2026-01-01", "time_end": "2025-01-01"}, "end 早于 start"),
        ({"time_start": "去年"}, "不是绝对日期"),
    ):
        try:
            tree(**bad)
            check(f"{why} 被拦截", False, "没报错")
        except Exception:
            check(f"{why} 被拦截", True)

    # ---- 正常聚合 ----
    print("== 事件区域聚合 ==")
    geometry, echoed = run(tree(), lookup=_FakeLookup(),
                           eonet=_FakeEonet(events_at(-65.0, -8.0)))
    check("得到有面积的几何",
          not isinstance(geometry, GeocodeError) and geometry.area > 0,
          str(geometry)[:80])
    check("结果落在基础地点边界内（按面积判定）",
          not isinstance(geometry, GeocodeError)
          and geometry.difference(AMZ).area == 0.0,
          f"越界面积={getattr(geometry, 'difference', lambda x: type('x',(),{'area':-1})())(AMZ).area}")
    check("时间窗原样传给数据源",
          "2023-09-20 ~ 2026-09-20" in echoed, echoed[:120])
    check("类别原样传给数据源", "wildfires" in echoed, echoed[:120])
    check("回显里有命中条数与块数",
          "命中 9 条" in echoed and "块" in echoed, echoed[:200])

    # ---- 覆盖缺口提示 ----
    print("== 数据源覆盖缺口 ==")
    # 事件都在 2025 年，查询起点却是 2023-09-20 → 中途那 1.5 年没有任何记录
    _, echoed = run(tree(), eonet=_FakeEonet(events_at(-65.0, -8.0, date_str="2025-06-01")))
    check("最早事件远晚于查询起点时给出缺口提示",
          "没有该类事件记录" in echoed and "2023-09-20" in echoed, echoed[:300])
    # 起点附近就有事件 → 不该提示
    _, echoed_ok = run(tree(), eonet=_FakeEonet(events_at(-65.0, -8.0, date_str="2023-10-01")))
    check("起点附近就有事件时不提示缺口",
          "没有该类事件记录" not in echoed_ok, echoed_ok[:300])

    # ---- 失败语义 ----
    print("== 失败语义 ==")
    query_error = GeocodeError("事件数据源暂时不可用，请稍后重试。",
                               detail="fake", service_unavailable=True)
    geometry, _ = run(tree(), eonet=_FakeEonet(error=query_error))
    check("数据源不可达 → service_unavailable 保留",
          isinstance(geometry, GeocodeError) and geometry.service_unavailable is True,
          str(geometry)[:100])

    # 窗口内有记录但都在基础地点之外：aggregate_events 报"都不在范围内"
    geometry, _ = run(tree(), eonet=_FakeEonet(events_at(-30.0, 30.0)))
    check("事件全在界外 → 提示都不在范围内",
          isinstance(geometry, GeocodeError) and "都不在" in geometry.user_message,
          str(geometry)[:120])
    check("界外不算数据源故障", isinstance(geometry, GeocodeError)
          and geometry.service_unavailable is False, str(geometry)[:100])

    geometry, _ = run(tree(), eonet=_FakeEonet([]))
    check("窗口内 0 条事件 → 提示没查到事件",
          isinstance(geometry, GeocodeError) and "没有查到事件" in geometry.user_message,
          str(geometry)[:120])

    # 基础地点只有中心点、没有 radius_km → 明确拒绝
    no_radius = tree(child_node={"node_type": "NamedPlace", "name": "亚马逊雨林",
                                 "center_lon": -65.0, "center_lat": -8.0})
    geometry, echoed = run(no_radius, lookup=_FakeLookup(Point(-65.0, -8.0)))
    check("基础地点只有中心点且无 radius_km → 明确拒绝",
          isinstance(geometry, GeocodeError) and "中心点" in geometry.user_message,
          str(geometry)[:140])

    # 有 radius_km → 撑成面后照常聚合。
    # 点距要用 0.05°：撑出来的面半径 0.544°，自适应网格是 0.054°，
    # 0.3° 的点距（= 5 个格子）在 4 邻接下会全部散成孤立点。
    with_radius = tree(child_node={"node_type": "NamedPlace", "name": "亚马逊雨林",
                                   "center_lon": -65.0, "center_lat": -8.0,
                                   "radius_km": 60.0})
    geometry, echoed = run(with_radius, lookup=_FakeLookup(Point(-65.0, -8.0)),
                           eonet=_FakeEonet(events_at(-65.0, -8.0, n=16, spacing=0.05)))
    check("有 radius_km 时按半径撑成面后求交",
          not isinstance(geometry, GeocodeError) and geometry.area > 0,
          f"{str(geometry)[:80]} / 提示={echoed[:120]}")
    check("撑面这件事有打印出来", "撑成面" in echoed, echoed[:160])

    # ---- 非事件查询不触发事件查找 ----
    print("== 非事件查询不该触发事件查找 ==")
    plain = SpatialNodeTree.model_validate({
        "node_type": "BorderBetween",
        "child_node_1": {"node_type": "NamedPlace", "name": "四川"},
        "child_node_2": {"node_type": "NamedPlace", "name": "云南"},
    }).root
    eonet = _FakeEonet()
    geocoder = make_geocoder(_FakeLookup(), eonet)
    geocoder._enrich_with_place_lookup(plain)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        geocoder._enrich_with_events(plain)
    check("遍历后没有发出任何事件查询", eonet.queries == [], str(eonet.queries))
    check("也没有任何事件相关打印", "事件" not in buffer.getvalue(),
          buffer.getvalue()[:200])

    print()
    if failures:
        print(f"事件接线自检失败 {len(failures)} 项：{failures}")
        return 1
    print("事件接线自检全部通过（离线，未发网络请求、未调大模型）")
    return 0


def _self_check_evidence() -> int:
    """不联网、不调大模型的自检：证据采集（S25）是否忠实反映真实来源。

    覆盖四件事：
      1. 三级降级链的三档来源各被记成什么（polygon / point / bounds）
      2. 一棵树里出现降级时，整棵树的口径取**最弱的一环**
      3. 事件的两个"没命中"必须区分：数据源不可达（条数未知，不淘汰）
         vs 数据源明确答复没有（0 条，淘汰）
      4. `geocode()` 与 `geocode_with_evidence()` 对外行为一致

    做法：`parse_text` 换成"直接返回预置节点树"，于是"查找 → 构建 → 采证据"整条
    链可以离线跑完；数据源全是假实现。
    """
    import io
    from contextlib import redirect_stdout
    from dataclasses import replace

    from shapely.geometry.base import BaseGeometry

    failures: list[str] = []

    def check(name: str, condition: bool, extra: str = ""):
        if condition:
            print(f"  [通过] {name}")
        else:
            print(f"  [失败] {name} {extra}")
            failures.append(name)

    BOX = box(-70.0, -12.0, -60.0, -2.0)        # 面积 100 平方度

    class _FakeLookup(PlaceLookup):
        """按名字返回预置几何；名字不在表里就当作"数据源没问到"。"""

        def __init__(self, geometries=(), hierarchy_confirmed=None,
                     flags=None, service_unavailable=True):
            self.geometries = dict(geometries)
            self.hierarchy_confirmed = hierarchy_confirmed
            self.flags = dict(flags or {})   # 按名字覆盖 hierarchy_confirmed
            self.service_unavailable = service_unavailable

        def search_for_places(self, request, limit=5):
            from place_lookup import PlaceCandidate
            geom = self.geometries.get(request.name)
            if geom is None:
                raise GeocodeError(
                    f"找不到地点「{request.name}」的边界或范围信息。",
                    detail="fake", service_unavailable=self.service_unavailable,
                )
            confirmed = self.flags.get(request.name, self.hierarchy_confirmed)
            return [PlaceCandidate(geometry=geom, name=request.name, source="fake",
                                   score=1.0, hierarchy_confirmed=confirmed)]

    class _FakeAmap:
        """只给精确中心点，不给边界（对齐 amap_geocoder 的降级路径）。"""

        def __init__(self, lon=-65.0, lat=-8.0, radius_km=60.0):
            self.lon = lon
            self.lat = lat
            self.radius_km = radius_km

        def get_precise_bounds(self, name, in_region=None, in_country=None):
            return {"center_lon": self.lon, "center_lat": self.lat,
                    "radius_km": self.radius_km, "level": "兴趣点"}

    class _FakeEonet:
        """records=() 表示"答复了但没有事件"；error 表示"数据源不可达"。"""

        def __init__(self, records=(), error=None):
            self.records = records
            self.error = error

        def fetch(self, query, use_cache=True):
            if self.error is not None:
                raise self.error
            return list(self.records)

    def event(lon, lat, date_str):
        from event_data import EventRecord
        return EventRecord(id=f"{lon},{lat}", title="fake", category="wildfires",
                           date=date_str, closed=None, lon=lon, lat=lat,
                           magnitude=None)

    def events_at(cx, cy, n=9, spacing=0.3, date_str="2023-10-01"):
        side = int(math.ceil(math.sqrt(n)))
        return [event(cx + i * spacing, cy + j * spacing, date_str)
                for i in range(side) for j in range(side)][:n]

    def wire(node, lookup=None, amap=None, eonet=None):
        """绕过 __init__（那需要 API Key），只装自检需要的部件。"""
        geocoder = NaturalLanguageGeocoder.__new__(NaturalLanguageGeocoder)
        geocoder.place_lookup = lookup
        geocoder.amap = amap
        geocoder.builder = GeometryBuilder()
        geocoder._eonet = eonet
        geocoder.parse_text = lambda text: node
        return geocoder

    def full(factory, lookup=None, amap=None, eonet=None):
        """跑完整条链，返回 (geometry, evidence, 打印文本)。"""
        geocoder = wire(factory(), lookup, amap, eonet)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            geometry, evidence = geocoder.geocode_with_evidence("测试")
        return geometry, evidence, buffer.getvalue()

    def enriched(node, lookup=None, amap=None, eonet=None):
        """只跑查找阶段（不构建），返回 (node, 打印文本)。

        事件节点构建失败时不会走到采证据那一步（`_build_event_affected` 直接抛），
        所以"为什么出局"这类证据只能这样取——先查找，再直接采证据。
        """
        geocoder = wire(node, lookup, amap, eonet)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            geocoder._enrich_with_place_lookup(node)
            geocoder._enrich_with_events(node)
        return node, buffer.getvalue()

    def named(name="测试地", **extra):
        payload = {"node_type": "NamedPlace", "name": name}
        payload.update(extra)
        return lambda: SpatialNodeTree.model_validate(payload).root

    def event_tree(name="有边界的"):
        return lambda: SpatialNodeTree.model_validate({
            "node_type": "EventAffected",
            "child_node": {"node_type": "NamedPlace", "name": name},
            "categories": ["wildfires"],
            "time_start": "2023-09-20",
            "time_end": "2026-09-20",
        }).root

    # ---- 三档几何来源 ----
    print("== 三档几何来源 ==")
    _, ev_poly, _ = full(named("深圳"), lookup=_FakeLookup({"深圳": BOX}))
    check("真实边界 → polygon", ev_poly.source == "polygon", ev_poly.source)
    check("真实边界不算降级", ev_poly.degradations == 0, str(ev_poly.degradations))
    check("面积直接取几何面积", abs(ev_poly.area_deg2 - BOX.area) < 1e-9,
          str(ev_poly.area_deg2))

    _, ev_point, _ = full(named("深圳"), amap=_FakeAmap())
    check("只有高德中心点 → point", ev_point.source == "point", ev_point.source)
    check("中心点算一次降级", ev_point.degradations == 1, str(ev_point.degradations))

    _, ev_bounds, _ = full(named("亚马逊雨林",
                                 bounds=[[-12.0, -70.0], [-2.0, -60.0]]))
    check("LLM bounds 兜底 → bounds", ev_bounds.source == "bounds", ev_bounds.source)
    check("bounds 算一次降级", ev_bounds.degradations == 1, str(ev_bounds.degradations))

    # ---- 降级口径：取最弱的一环 ----
    print("== 降级口径：取最弱的一环 ==")
    # 一个孩子查到真实边界、另一个只拿到中心点 → 整棵树报 point
    mixed = lambda: SpatialNodeTree.model_validate({
        "node_type": "Union",
        "child_nodes": [
            {"node_type": "NamedPlace", "name": "有边界的"},
            {"node_type": "NamedPlace", "name": "只有中心点的"},
        ],
    }).root
    _, ev_mixed, _ = full(mixed, lookup=_FakeLookup({"有边界的": BOX}),
                          amap=_FakeAmap())
    check("一强一弱时整棵树报最弱的一档", ev_mixed.source == "point", ev_mixed.source)
    check("降级计数只数降级的那一个", ev_mixed.degradations == 1,
          str(ev_mixed.degradations))
    check("取最弱不影响面积口径", abs(ev_mixed.area_deg2 - BOX.area) < 1e-9,
          str(ev_mixed.area_deg2))

    # ---- 层级一致性三态 ----
    print("== 层级一致性 ==")
    for flag, expect in ((True, True), (False, False), (None, None)):
        _, evidence, _ = full(
            named("深圳"),
            lookup=_FakeLookup({"深圳": BOX}, hierarchy_confirmed=flag),
        )
        check(f"数据源报 {flag} → 证据保留 {expect}",
              evidence.hierarchy_confirmed is expect, str(evidence.hierarchy_confirmed))

    both = lambda: SpatialNodeTree.model_validate({
        "node_type": "Union",
        "child_nodes": [
            {"node_type": "NamedPlace", "name": "冲突的"},
            {"node_type": "NamedPlace", "name": "一致的"},
        ],
    }).root
    # 一个孩子报一致、另一个没核验过（查不到，走了中心点）→ 取一致
    _, evidence_both, _ = full(
        both, lookup=_FakeLookup({"一致的": BOX}, hierarchy_confirmed=True,
                                 flags={"冲突的": None}),
        amap=_FakeAmap(),
    )
    check("有节点报了核验结果、其它节点没核验 → 采用那个结果",
          evidence_both.hierarchy_confirmed is True,
          str(evidence_both.hierarchy_confirmed))

    # 一个报冲突、另一个报一致 → 整棵树按冲突处理（冲突比一致重要）
    _, evidence_clash, _ = full(
        both, lookup=_FakeLookup({"一致的": BOX, "冲突的": BOX},
                                 flags={"冲突的": False, "一致的": True}),
    )
    check("有一个节点冲突就是冲突",
          evidence_clash.hierarchy_confirmed is False,
          str(evidence_clash.hierarchy_confirmed))

    # ---- 事件维度 ----
    print("== 事件证据 ==")
    _, ev_event, _ = full(event_tree(), lookup=_FakeLookup({"有边界的": BOX}),
                          eonet=_FakeEonet(events_at(-65.0, -8.0)))
    check("事件命中 → 来源是 event", ev_event.source == "event", ev_event.source)
    check("事件候选不算降级", ev_event.degradations == 0, str(ev_event.degradations))
    check("命中的事件条数被记下来", ev_event.event_count == 9,
          str(ev_event.event_count))
    check("块数被记下来", ev_event.event_pieces == 1, str(ev_event.event_pieces))
    check("时间窗原样进证据",
          ev_event.event_window == "2023-09-20 ~ 2026-09-20",
          str(ev_event.event_window))
    check("起点附近就有记录 → 覆盖完整",
          ev_event.event_window_covered is True,
          str(ev_event.event_window_covered))

    # 基础地点只有中心点时，整棵树报 point 而不是 event（事件的真实度受基础地点拖累）
    point_base = lambda: SpatialNodeTree.model_validate({
        "node_type": "EventAffected",
        "child_node": {"node_type": "NamedPlace", "name": "只有中心点的"},
        "categories": ["wildfires"],
        "time_start": "2023-09-20",
        "time_end": "2026-09-20",
    }).root
    _, ev_point_event, _ = full(point_base, amap=_FakeAmap(),
                                eonet=_FakeEonet(events_at(-65.0, -8.0, n=16,
                                                           spacing=0.05)))
    check("中心点上的事件 → 报 point 不报 event",
          ev_point_event.source == "point", ev_point_event.source)
    check("中心点上的事件仍记了命中条数", ev_point_event.event_count == 16,
          str(ev_point_event.event_count))

    # ---- 0 条事件：数据源没问到 vs 明确答复没有 ----
    print("== 0 条事件：数据源没问到 vs 明确答复没有 ==")
    node, _ = enriched(event_tree()(), lookup=_FakeLookup({"有边界的": BOX}),
                       eonet=_FakeEonet([]))
    ev_zero = collect_evidence(node, None)
    check("明确答复没有 → 记 0 条", ev_zero.event_count == 0,
          str(ev_zero.event_count))
    check("没构建几何 → 面积记 0", ev_zero.area_deg2 == 0.0, str(ev_zero.area_deg2))
    score_zero = score_candidate(ev_zero)
    check("0 条事件 → 硬淘汰", score_zero.eliminated,
          str(score_zero.elimination_reason))
    check("淘汰理由点名了类别而不是笼统说空几何",
          bool(score_zero.elimination_reason)
          and "wildfires" in score_zero.elimination_reason,
          str(score_zero.elimination_reason))

    node, _ = enriched(event_tree()(), lookup=_FakeLookup({"有边界的": BOX}),
                       eonet=_FakeEonet(error=GeocodeError(
                           "事件数据源暂时不可用，请稍后重试。", detail="fake",
                           service_unavailable=True)))
    ev_down = collect_evidence(node, None)
    check("数据源不可达 → 条数未知（None）", ev_down.event_count is None,
          str(ev_down.event_count))
    # 没几何 → 出局，但**理由必须是空几何而不是"命中 0 条"**：数据源压根没答复，
    # 不能反过来赖候选"没命中事件"。判成 0 条就会给出一个错误的淘汰理由。
    reason_down = score_candidate(ev_down).elimination_reason
    check("数据源不可达 → 出局理由是空几何而不是命中 0 条",
          score_candidate(ev_down).eliminated and "空几何" in (reason_down or ""),
          str(reason_down))
    check("查不到事件不影响几何来源口径", ev_down.source == "polygon",
          ev_down.source)

    # 基础地点查不到 → 链在 event_base_geometry 就断了，事件查询根本没发出去。
    # 这时**不能**记成"数据源答了 0 条"（那会把候选按一个与它无关的理由淘汰）。
    node, log = enriched(event_tree("查不到的")(), lookup=_FakeLookup({}),
                         eonet=_FakeEonet([]))
    ev_no_base = collect_evidence(node, None)
    check("基础地点没查到 → 条数未知而不是 0", ev_no_base.event_count is None,
          str(ev_no_base.event_count))
    reason_no_base = score_candidate(ev_no_base).elimination_reason
    check("基础地点没查到 → 出局理由是空几何而不是命中 0 条",
          score_candidate(ev_no_base).eliminated
          and "空几何" in (reason_no_base or ""),
          str(reason_no_base))
    check("这条链确实没把事件查询发出去", node._event_queried is False,
          str(node._event_queried))
    check("基础地点没查到会明说是哪个地点", "查不到的" in log, log[-200:])

    # ---- 排序：同面积下 polygon > point > bounds ----
    print("== 打分排序 ==")
    # point / bounds 的原生几何一个没有面积、一个面积不同，要比较来源得分
    # 就得先把面积对齐（否则 point 会按"空几何"直接出局，比的是别的东西）
    same_area_bounds = replace(ev_bounds, area_deg2=ev_poly.area_deg2)
    same_area_point = replace(ev_point, area_deg2=ev_poly.area_deg2)
    scores = score_candidates([same_area_bounds, ev_poly, same_area_point])
    order = [round(s.total) for s in scores]
    check("同面积下 polygon > point > bounds",
          scores[1].total > scores[2].total > scores[0].total, str(order))
    check("score_candidates 保持输入顺序（调用方靠下标回映射）",
          scores[0].total == score_candidate(same_area_bounds).total, str(order))
    check("出局的候选排序键一定更小",
          score_zero.sort_key < score_candidate(ev_poly).sort_key,
          f"{score_zero.sort_key} / {score_candidate(ev_poly).sort_key}")

    # ---- 两个入口对外行为一致 ----
    print("== 两个入口对外行为一致 ==")
    factory = named("深圳")
    geocoder = wire(factory(), _FakeLookup({"深圳": BOX}))
    geocoder.parse_text = lambda text: factory()
    with redirect_stdout(io.StringIO()):
        geometry_full, evidence_full = geocoder.geocode_with_evidence("深圳")
        geometry_plain = geocoder.geocode("深圳")
    check("geocode() 返回的就是 geocode_with_evidence() 的几何",
          geometry_plain.equals(geometry_full),
          f"{geometry_plain} / {geometry_full}")
    check("geocode() 的返回类型没变（单个 Shapely 几何）",
          isinstance(geometry_plain, BaseGeometry) and not isinstance(geometry_plain, tuple),
          type(geometry_plain).__name__)
    check("诊断入口同时给出几何与证据",
          isinstance(geometry_full, BaseGeometry)
          and isinstance(evidence_full, GeometryEvidence),
          f"{type(geometry_full).__name__} / {type(evidence_full).__name__}")

    print()
    if failures:
        print(f"证据采集自检失败 {len(failures)} 项：{failures}")
        return 1
    print("证据采集自检全部通过（离线，未发网络请求、未调大模型）")
    return 0


# =============================================================================
# 离线自检：LandUseArea 的接线（python geocoding.py --self-check）
# =============================================================================
def _self_check_landuse() -> int:
    """不联网、不调大模型的自检：只验证 LandUseArea 的接线。

    刻意只覆盖**接线**，不覆盖 LLM 解析与真实 Overpass 请求（那需要网络与 Key）：
      1. 节点模型能否校验（landuse 只能取 LANDUSE_CATEGORIES 里的值）
      2. `_enrich_with_place_lookup` → `_enrich_with_landuse` → `build_geometry`
         这条链在"命中 / 数据源不可达 / 答了但无此地块 / 数据源不支持"四种结局下
         分别给出什么
      3. 选区（child_node 撑成的面）真的按 Buffer 之后的几何传给了 search_landuse
      4. 证据口径：landuse 命中即来源记 landuse，且不受 child 降级拖累

    数据源用假实现：search_for_places 返回固定矩形，search_landuse 返回固定地块。
    """
    import io
    from contextlib import redirect_stdout

    failures: list[str] = []

    def check(name: str, condition: bool, extra: str = ""):
        if condition:
            print(f"  [通过] {name}")
        else:
            print(f"  [失败] {name} {extra}")
            failures.append(name)

    # ---- 假数据源 ----
    PLACE = box(-70.0, -12.0, -60.0, -2.0)          # NamedPlace 拿到的"真实边界"
    # 两块用地地块，都落在 PLACE 之内；Buffer 撑出来的选区能整个包住它们
    BLOCKS = unary_union([box(-68.0, -10.0, -66.0, -8.0),
                          box(-65.0, -9.0, -63.0, -7.0)])
    EXCLUDE = box(-67.0, -9.0, -65.0, -8.0)         # 与第一块部分重叠的排除区

    class _FakeLookup(PlaceLookup):
        """NamedPlace 返回固定矩形；search_landuse 按构造时的配置返回或抛错。"""

        def __init__(self, landuse_result=BLOCKS, landuse_error=None, by_name=None):
            self._landuse_result = landuse_result
            self._landuse_error = landuse_error
            self._by_name = by_name or {}
            self.landuse_calls: list[tuple] = []

        def search_for_places(self, request, limit=5):
            from place_lookup import PlaceCandidate
            geom = self._by_name.get(request.name, PLACE)
            return [PlaceCandidate(geometry=geom, name=request.name,
                                   source="fake", score=1.0)]

        def search_landuse(self, landuse, scope, label=""):
            self.landuse_calls.append((landuse, scope, label))
            if self._landuse_error is not None:
                raise self._landuse_error
            # 真实实现返回的是"与 scope 求交"后的地块，这里也求交，保持口径一致
            return self._landuse_result.intersection(scope)

    class _LookupWithoutLanduse(PlaceLookup):
        """没有 search_landuse 的 lookup：模拟旧数据源。"""

        def search_for_places(self, request, limit=5):
            from place_lookup import PlaceCandidate
            return [PlaceCandidate(geometry=PLACE, name=request.name,
                                   source="fake", score=1.0)]

    def make_geocoder(lookup):
        """绕过 __init__（那需要 API Key 与 OpenAI 客户端），只装自检需要的部件。"""
        geocoder = NaturalLanguageGeocoder.__new__(NaturalLanguageGeocoder)
        geocoder.place_lookup = lookup
        geocoder.amap = None
        return geocoder

    def tree(landuse="residential", child=None):
        payload = {
            "node_type": "LandUseArea",
            "landuse": landuse,
            "child_node": child or {
                "node_type": "Buffer",
                "distance_km": 3.0,
                "child_node": {"node_type": "NamedPlace", "name": "某地"},
            },
        }
        return SpatialNodeTree.model_validate(payload).root

    def run(node, lookup):
        """跑完整条链，返回 (geometry 或 GeocodeError, 打印出来的文本)。"""
        geocoder = make_geocoder(lookup)
        geocoder._enrich_with_place_lookup(node)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            geocoder._enrich_with_landuse(node)
        try:
            geometry = GeometryBuilder.build_geometry(node)
        except GeocodeError as e:
            geometry = e
        return geometry, buffer.getvalue()

    # ---- 节点校验 ----
    print("== LandUseArea 节点校验 ==")
    node = tree()
    check("能校验出 LandUseArea 节点", node.node_type == "LandUseArea", node.node_type)
    check("landuse 原样保留", node.landuse == "residential", node.landuse)
    check("child_node 是 Buffer", node.child_node.node_type == "Buffer",
          node.child_node.node_type)

    for bad, why in (
        ({"landuse": "商圈"}, "未支持的类别词"),
        ({"landuse": "Residential"}, "大小写不符"),
        ({"landuse": ""}, "空类别"),
    ):
        try:
            tree(**bad)
            check(f"{why} 被拦截", False, "没报错")
        except Exception:
            check(f"{why} 被拦截", True)

    # ---- 正常命中 ----
    print("== 正常命中 ==")
    lookup = _FakeLookup()
    geometry, echoed = run(tree(), lookup)
    check("得到有面积的几何",
          not isinstance(geometry, GeocodeError) and geometry.area > 0,
          str(geometry)[:80])
    check("几何就是命中的用地地块",
          not isinstance(geometry, GeocodeError) and geometry.equals(BLOCKS),
          f"{str(geometry)[:80]} / blocks={BLOCKS.area:.4g}")
    check("landuse 类别原样传给数据源",
          len(lookup.landuse_calls) == 1 and lookup.landuse_calls[0][0] == "residential",
          str([c[0] for c in lookup.landuse_calls]))
    check("选区是 Buffer 撑出来的面（比原地点大）",
          len(lookup.landuse_calls) == 1 and lookup.landuse_calls[0][1].area > PLACE.area,
          f"scope.area={lookup.landuse_calls[0][1].area if lookup.landuse_calls else 'n/a'}")
    check("选区是整个包住地块的那块面",
          bool(lookup.landuse_calls)
          and lookup.landuse_calls[0][1].covers(BLOCKS),
          "scope 未覆盖地块")
    check("标签（中文类别名）传给了数据源",
          bool(lookup.landuse_calls) and "居民" in lookup.landuse_calls[0][2],
          str(lookup.landuse_calls[0][2] if lookup.landuse_calls else ""))
    check("回显里有命中块数与面积",
          "[用地回显]" in echoed and "块" in echoed, echoed[:200])

    # ---- 证据口径 ----
    print("== 证据口径 ==")
    geocoder = make_geocoder(_FakeLookup())
    evidence_node = tree()
    geocoder._enrich_with_place_lookup(evidence_node)
    with redirect_stdout(io.StringIO()):
        geocoder._enrich_with_landuse(evidence_node)
        evidence_geometry = GeometryBuilder.build_geometry(evidence_node)
    evidence = collect_evidence(evidence_node, evidence_geometry)
    check("用地命中时来源记为 landuse", evidence.source == "landuse", evidence.source)
    check("child 拿到真实边界时没有降级", evidence.degradations == 0,
          str(evidence.degradations))

    # ---- Difference 包一层（报障句的结构：3km 内的用地，去掉参照物本体）----
    print("== 与 Difference 组合 ==")
    diff_tree = SpatialNodeTree.model_validate({
        "node_type": "Difference",
        "child_node_1": {
            "node_type": "LandUseArea",
            "landuse": "residential",
            "child_node": {
                "node_type": "Buffer",
                "distance_km": 3.0,
                "child_node": {"node_type": "NamedPlace", "name": "某地"},
            },
        },
        "child_node_2": {"node_type": "NamedPlace", "name": "排除地"},
    }).root
    lookup = _FakeLookup(by_name={"排除地": EXCLUDE})
    geocoder = make_geocoder(lookup)
    geocoder._enrich_with_place_lookup(diff_tree)
    with redirect_stdout(io.StringIO()):
        geocoder._enrich_with_landuse(diff_tree)
        diff_geometry = GeometryBuilder.build_geometry(diff_tree)
    check("组合后仍能建出几何",
          not isinstance(diff_geometry, GeocodeError) and diff_geometry.area > 0,
          str(diff_geometry)[:100])
    check("减掉的确实是排除区覆盖的那部分",
          not isinstance(diff_geometry, GeocodeError)
          and diff_geometry.area < BLOCKS.area,
          f"diff.area={getattr(diff_geometry, 'area', -1)} / blocks={BLOCKS.area:.4g}")
    diff_evidence = collect_evidence(diff_tree, diff_geometry)
    check("组合后来源仍是 landuse", diff_evidence.source == "landuse",
          diff_evidence.source)

    # ---- 失败语义 ----
    print("== 失败语义 ==")
    unavailable = GeocodeError("用地数据源暂时不可用，请稍后重试。",
                               detail="fake", service_unavailable=True)
    geometry, echoed = run(tree(), _FakeLookup(landuse_error=unavailable))
    check("数据源不可达 → service_unavailable 保留",
          isinstance(geometry, GeocodeError) and geometry.service_unavailable is True,
          str(geometry)[:100])
    check("不可达时会打印原因", "暂时不可用" in echoed, echoed[:120])

    no_block = GeocodeError("没有查到标注为「居民区、住宅区、居住区、生活区」的用地。",
                            detail="fake", service_unavailable=False)
    geometry, _ = run(tree(), _FakeLookup(landuse_error=no_block))
    check("答了但范围内无该类用地 → 提示换范围/类别",
          isinstance(geometry, GeocodeError) and "没有查到" in geometry.user_message,
          str(geometry)[:120])
    check("无地块不算数据源故障",
          isinstance(geometry, GeocodeError) and geometry.service_unavailable is False,
          str(geometry)[:100])

    geometry, _ = run(tree(), _LookupWithoutLanduse())
    check("数据源没有 search_landuse → 明确说不支持",
          isinstance(geometry, GeocodeError) and "不支持" in geometry.user_message,
          str(geometry)[:120])

    # ---- 非用地查询不触发用地查找 ----
    print("== 非用地查询不该触发用地查找 ==")
    plain = SpatialNodeTree.model_validate({
        "node_type": "NamedPlace", "name": "深圳",
    }).root
    lookup = _FakeLookup()
    geocoder = make_geocoder(lookup)
    geocoder._enrich_with_place_lookup(plain)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        geocoder._enrich_with_landuse(plain)
    check("遍历后没有发出任何用地查询", lookup.landuse_calls == [],
          str(lookup.landuse_calls))
    check("也没有任何用地相关打印", "用地" not in buffer.getvalue(),
          buffer.getvalue()[:200])

    print()
    if failures:
        print(f"用地接线自检失败 {len(failures)} 项：{failures}")
        return 1
    print("用地接线自检全部通过（离线，未发网络请求、未调大模型）")
    return 0


# =============================================================================
# 简易测试入口
# =============================================================================
if __name__ == "__main__":
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--self-check" in _sys.argv:
        _sys.exit(_self_check_events() or _self_check_evidence()
                  or _self_check_landuse())

    geocoder = NaturalLanguageGeocoder()
    test_cases = [
        "深圳大学",
        "深圳大学西南方向的公园",
        "北京天安门广场附近5公里范围内",
    ]
    for text in test_cases:
        print(f"\n{'='*60}")
        print(f"输入: {text}")
        print("-" * 40)
        try:
            geometry = geocoder.geocode(text)
            print(f"几何类型: {geometry.geom_type}")
            print(f"几何中心: ({geometry.centroid.x:.4f}, {geometry.centroid.y:.4f})")
            if hasattr(geometry, 'area'):
                print(f"面积(平方度): {geometry.area:.6f}")
        except GeocodeError as e:
            print(f"错误: {e.user_message}")
            print(f"详情: {e.detail}")
        except Exception as e:
            print(f"未预期的内部错误: {type(e).__name__}: {e}")
