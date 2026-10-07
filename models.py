"""
空间节点模型（pydantic）。

对齐源项目 natural_language_geocoding.geometry.spatial_node：把 LLM 输出的 JSON
树从裸 dict 变成有类型约束的模型，字段写错在"解析"阶段就报出明确错误，而不是
等到构建几何时才抛出含糊的 KeyError。

命名说明：
  SpatialNode          13 种节点的公共基类（承载 model_config 与 node_type）
  AnySpatialNodeType   带判别字段的联合类型，递归引用靠前向引用 "AnySpatialNodeType"
  SpatialNodeTree      整棵树的根模型，对应源项目的 RootModel[SpatialNode]
  （源项目把整棵树直接叫 SpatialNode；这里 SpatialNode 已经是基类名，故根模型取
   SpatialNodeTree，避免同名冲突。）

与源项目的已知差异（有意保留，记录在此避免后续被当成 bug）：
  - 距离字段统一用 `distance_km`，没有拆成源项目的 `distance` + `distance_unit`
    两字段。当前提示词与全部构建逻辑都以公里为单位，拆单位需要同时改提示词、
    构建器与全部示例，收益仅是形式对齐，故暂不拆。
  - `center_lon` / `center_lat` 为可选：它们是"第二级几何来源"（高德中心点），
    真实多边形命中时用不上，LLM bounds 兜底时也用不上。提示词仍要求必填，
    模型层保持可选以忠实表达三级降级链。

运行期状态用 pydantic 私有属性承载：
  `_polygon`（真实多边形 GeoJSON）、`_amap_level`（高德返回的等级）由
  NaturalLanguageGeocoder._enrich_with_place_lookup 在查找阶段写入。
  私有属性不是模型字段：不进 JSON、不参与校验、也不受 extra="forbid" 影响，
  因此"先查找后构建"不会因为多出这两个键而校验失败。
"""

from datetime import date
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, RootModel, field_validator

# 方向：只允许四个基本方位。复合方向（西南等）由嵌套的 DirectionalSubset 表达，
# 不作为 direction 的取值。
DirectionLiteral = Literal["north", "south", "east", "west"]

# bounds 的形状：[[min_lat, min_lon], [max_lat, max_lon]]
_BoundsType = list[list[float]]

# 用地类别：只能是 OSM `landuse=*` 里真实存在的值，且要有常见中文说法。
# 取值用英文 OSM 值而不是中文——"居民区""住宅区""居住区"是同一个值的三种说法，
# 别名属于**提示词侧**的知识（与事件类别同一个取舍，见 EventAffected.categories），
# 模型层不维护第二份别名表，只做"这个值合不合法"的校验。
LandUseCategory = Literal[
    "residential",
    "commercial",
    "retail",
    "industrial",
    "farmland",
    "forest",
    "meadow",
    "orchard",
    "cemetery",
    "quarry",
    "military",
    "construction",
    "reservoir",
]

# 类别 → 中文说法（提示词渲染用，见 prompts.render_landuse_categories）。
# 这是"哪些用地类型可查"的唯一来源：与上面的 Literal 必须同键，模块末尾有断言把关。
LANDUSE_LABELS: dict[str, str] = {
    "residential": "居民区、住宅区、居住区、生活区",
    "commercial": "商业区、商务区",
    "retail": "商铺集中区、零售区",
    "industrial": "工业区、工厂区",
    "farmland": "农田、耕地、农用地",
    "forest": "林地、森林",
    "meadow": "草地、牧场",
    "orchard": "果园",
    "cemetery": "墓地、公墓",
    "quarry": "采石场、矿区",
    "military": "军事区、军事用地",
    "construction": "在建工地、施工区",
    "reservoir": "水库",
}

LANDUSE_CATEGORIES: tuple[str, ...] = get_args(LandUseCategory)


class SpatialNode(BaseModel):
    """所有空间节点的基类：只声明判别字段与全局校验策略。"""

    model_config = ConfigDict(strict=True, extra="forbid")

    node_type: str


# =============================================================================
# 14 种节点
# =============================================================================
class NamedPlace(SpatialNode):
    """具名地点。几何来源见 geocoding.GeometryBuilder._build_named_place 的三级降级链。"""

    node_type: Literal["NamedPlace"] = "NamedPlace"

    name: str
    # 第二级来源：高德地理编码的精确中心点
    center_lon: float | None = None
    center_lat: float | None = None
    # 第三级来源：LLM 给的粗略矩形（仅省/国家等大型区域建议填）
    bounds: _BoundsType | None = None
    # 维度元数据：当前不参与几何构建，仅随节点向下传递
    radius_km: float | None = None
    # 层级信息：查找数据源时用于消歧（同名地点的省份/国家约束）
    in_continent: str | None = None
    in_country: str | None = None
    in_region: str | None = None

    # ---- 运行期状态（不进 JSON、不参与校验，由查找阶段写入）----
    # 真实多边形（GeoJSON dict，来自高德 District API / OSM Overpass）
    _polygon: dict | None = PrivateAttr(default=None)
    # 高德返回的等级（省/市/区县/兴趣点），仅用于诊断输出
    _amap_level: str = PrivateAttr(default="")
    # 实际用上的是三级降级链里的哪一级（"polygon"/"point"/"bounds"），空串表示
    # 还没构建过。**刻意显式记录而不是从 `_polygon`/`_amap_level` 反推**：反推要
    # 猜"高德没给 level 时那个点算哪一级"，而这里只需要在实际做决定的地方写一次。
    # 供 S25 的 `collect_evidence` 读，是"几何来源等级"这一维度的唯一来源。
    _geometry_source: str = PrivateAttr(default="")
    # 层级核验结果（来自 PlaceCandidate.hierarchy_confirmed）：True/False/None
    _hierarchy_confirmed: bool | None = PrivateAttr(default=None)

    @field_validator("bounds")
    @classmethod
    def _validate_bounds_shape(cls, value: _BoundsType | None) -> _BoundsType | None:
        """只校验形状：必须是 [[a, b], [c, d]]。

        不校验 min < max：那属于几何语义，交给 _build_named_place 判断，
        避免同一个约束在两个地方各写一遍、报两种错误。
        """
        if value is None:
            return None
        if len(value) != 2 or any(len(pair) != 2 for pair in value):
            raise ValueError("bounds 必须是 [[min_lat, min_lon], [max_lat, max_lon]] 两对坐标")
        return value


class Buffer(SpatialNode):
    """在子几何周围扩展 distance_km 公里。"""

    node_type: Literal["Buffer"] = "Buffer"

    distance_km: float = 1.0
    child_node: "AnySpatialNodeType"


class DirectionalConstraint(SpatialNode):
    """方向条带：位于子几何某方向 **之外** 的带状区域，宽度由 max_distance_km 决定。"""

    node_type: Literal["DirectionalConstraint"] = "DirectionalConstraint"

    direction: DirectionLiteral
    # 为 None 时构建器按 3km 处理（提示词要求显式给出）
    max_distance_km: float | None = None
    child_node: "AnySpatialNodeType"


class DirectionalSubset(SpatialNode):
    """方位子集：子几何 **之内** 的某一半；嵌套可表达象限（西北 = 北半的西半）。"""

    node_type: Literal["DirectionalSubset"] = "DirectionalSubset"

    direction: DirectionLiteral
    child_node: "AnySpatialNodeType"


class Intersection(SpatialNode):
    """多个子区域的交集。"""

    node_type: Literal["Intersection"] = "Intersection"

    child_nodes: list["AnySpatialNodeType"] = Field(min_length=2)


class Union(SpatialNode):
    """多个子区域的并集。"""

    node_type: Literal["Union"] = "Union"

    child_nodes: list["AnySpatialNodeType"] = Field(min_length=2)


class Difference(SpatialNode):
    """差集：child_node_1 减去 child_node_2。"""

    node_type: Literal["Difference"] = "Difference"

    child_node_1: "AnySpatialNodeType"
    child_node_2: "AnySpatialNodeType"


class Between(SpatialNode):
    """两个地点之间的区域。"""

    node_type: Literal["Between"] = "Between"

    child_node_1: "AnySpatialNodeType"
    child_node_2: "AnySpatialNodeType"


class BorderBetween(SpatialNode):
    """两个相邻区域的交界带。"""

    node_type: Literal["BorderBetween"] = "BorderBetween"

    child_node_1: "AnySpatialNodeType"
    child_node_2: "AnySpatialNodeType"


class BorderOf(SpatialNode):
    """某个区域的边界线。"""

    node_type: Literal["BorderOf"] = "BorderOf"

    child_node: "AnySpatialNodeType"


class CoastOf(SpatialNode):
    """某个区域的海岸线。"""

    node_type: Literal["CoastOf"] = "CoastOf"

    child_node: "AnySpatialNodeType"


class OffTheCoastOf(SpatialNode):
    """离岸海域：海岸线向外 distance_km 公里、且不属于该区域的带状海域。"""

    node_type: Literal["OffTheCoastOf"] = "OffTheCoastOf"

    distance_km: float = 10.0
    child_node: "AnySpatialNodeType"


class LandUseArea(SpatialNode):
    """某个范围内、某一类用地（OSM `landuse=*`）的真实多边形并集。

    解决的报障句："南山外国语学校两三公里附近四周的居民区域"——句中的"居民区域"
    是**用地类型**，不是地名：没有专名可查，也不能按老规则退成"以学校为圆心的
    3 公里圆盘"（圆盘覆盖范围内的一切：马路、水体、公园，恰恰不是用户要的那类地）。

    与 NamedPlace 的根本区别：NamedPlace 查的是**一个名字**的边界（唯一、连续），
    本节点查的是**一类用地**的边界（可以几十上百块，也可以一块都没有）。

    分工是刻意的：`child_node` 只负责圈定**范围**（通常包着 Buffer 或
    DirectionalSubset），本节点负责给出**内容**。所以范围本身是从一个点撑出来的
    圆盘也不影响结果的真实性——结果几何是 OSM 里真实存在的用地地块，不是圆盘。
    """

    node_type: Literal["LandUseArea"] = "LandUseArea"

    # 用地类别，取值受 Literal 约束（见 LandUseCategory 与 LANDUSE_LABELS）。
    landuse: LandUseCategory
    # 范围。通常包着 Buffer（"X 附近 N 公里的居民区"）或 DirectionalSubset
    # （"X 南部的工业区"），也可以直接是自带边界的 NamedPlace。
    child_node: "AnySpatialNodeType"

    # ---- 运行期状态（不进 JSON、不参与校验，由查找阶段写入）----
    # 裁剪到 child 范围内、并集后的用地多边形（Shapely 几何）
    _landuse_geometry: object | None = PrivateAttr(default=None)
    # 查找阶段失败时留下的原始 GeocodeError，构建阶段原样抛出（与 EventAffected
    # 同一个模式：失败原因只在一处构造，不在两处各写一份口径）
    _landuse_error: object | None = PrivateAttr(default=None)
    # 并入并集的地块数（给回显与 S25 的证据用）
    _landuse_pieces: int = PrivateAttr(default=0)


class EventAffected(SpatialNode):
    """某类事件在某个时间窗内影响过的、child_node 范围内的一片或多片区域。

    几何来源既不是行政边界也不是中心点，而是**事件点聚合**（见
    `event_aggregate.aggregate_events`）：child_node 先完成地点查找拿到真实边界，
    再把该边界内的事件位置点聚成凸包、与边界求交，得到"真的被烧过/涝过的那些片"。

    类别与时间都由提示词侧解析成**绝对值**后填进来（相对时间如"过去两三年"换算成
    绝对区间，当前日期注入提示词，见 prompts.RULE_EVENT_TIME），所以模型层不做任何
    时间语义推断——只校验格式与先后顺序。

    `categories` 直接存 EONET 类别 id（如 "wildfires"），不存中文：别名表随数据源
    变动，属于提示词侧的知识，模型层不该再维护一份。
    """

    node_type: Literal["EventAffected"] = "EventAffected"

    # 基础地点。通常（但不是必须）是 NamedPlace——事件也可以限定在交集、缓冲区等
    # 已有节点之上，例如"长江三角洲的洪涝"。
    child_node: "AnySpatialNodeType"
    # EONET 类别 id，至少一个。可多选："被火灾和洪水影响过的区域"。
    categories: list[str] = Field(min_length=1)
    # ISO 日期（YYYY-MM-DD）。必填：没有时间窗就无从筛选事件。
    time_start: str
    time_end: str

    # ---- 运行期状态（不进 JSON、不参与校验，由查找阶段写入）----
    # 聚合结果 EventAggregate（event_aggregate.EventAggregate）
    _event_aggregate: object | None = PrivateAttr(default=None)
    # 查找阶段失败时留下的原始 GeocodeError；由其区分"数据源没问到"与
    # "问到但没有事件"，构建阶段原样抛出（见 _build_event_affected）
    _event_error: object | None = PrivateAttr(default=None)
    # 命中的事件条数（不是位置点数）。S25 的"事件命中 0 条就淘汰"这条硬判据
    # 依赖它，所以必须与"没查事件"（事件节点根本不存在）区分开。
    _event_count: int = PrivateAttr(default=0)
    # 这次到底有没有把查询发出去。**必须单独记**：基础地点查不到时整条链会在
    # `event_base_geometry` 就断掉（`_event_error` 非空、但数据源压根没被问到），
    # 只看 `_event_error` 会把"没问到"误判成"数据源答了 0 条"，于是候选被按
    # "该时间窗内命中 0 条事件"淘汰——理由完全错。
    _event_queried: bool = PrivateAttr(default=False)
    # 数据源是否覆盖了整个时间窗。False = 早期那段没有记录（EONET 的存档边界），
    # 属于数据源限制、不是候选的错，只提示不扣分。
    _event_window_covered: bool = PrivateAttr(default=True)

    @field_validator("time_start", "time_end")
    @classmethod
    def _validate_iso_date(cls, value: str) -> str:
        """校验 YYYY-MM-DD 的形状与真实存在性。

        用 `date.fromisoformat` 而不是正则：正则挡不住 2025-02-30，而一个不存在的
        日期在查询缓存 key、时间分片、与事件的日期比较里都会静默产生错误结果。
        """
        try:
            date.fromisoformat(value)
        except ValueError:
            raise ValueError(f"时间必须是合法的 YYYY-MM-DD 日期，收到 {value!r}")
        return value

    @field_validator("time_end")
    @classmethod
    def _validate_order(cls, value: str, info) -> str:
        start = info.data.get("time_start")
        if start is not None and value < start:
            raise ValueError(f"time_end（{value}）不能早于 time_start（{start}）")
        return value


# =============================================================================
# 联合类型与根模型
# =============================================================================
# 判别联合：靠 node_type 直接选中具体模型，不做"逐个尝试"。
# 递归字段用字符串前向引用指向它，定义完在模块末尾统一 rebuild。
AnySpatialNodeType = Annotated[
    NamedPlace
    | Buffer
    | DirectionalConstraint
    | DirectionalSubset
    | Intersection
    | Union
    | Difference
    | Between
    | BorderBetween
    | BorderOf
    | CoastOf
    | OffTheCoastOf
    | LandUseArea
    | EventAffected,
    Field(discriminator="node_type"),
]

# 节点类型名 → 模型。顺序即"提示词/文档中列举节点"的顺序，把 NamedPlace 放首位。
# 这是节点名清单的唯一来源：提示词的字段说明、调试输出都从这里取，避免多处各写一份。
NODE_MODELS: dict[str, type[SpatialNode]] = {
    "NamedPlace": NamedPlace,
    "Buffer": Buffer,
    "DirectionalConstraint": DirectionalConstraint,
    "DirectionalSubset": DirectionalSubset,
    "Intersection": Intersection,
    "Union": Union,
    "Difference": Difference,
    "Between": Between,
    "BorderBetween": BorderBetween,
    "BorderOf": BorderOf,
    "CoastOf": CoastOf,
    "OffTheCoastOf": OffTheCoastOf,
    "LandUseArea": LandUseArea,
    "EventAffected": EventAffected,
}

NODE_TYPES = tuple(NODE_MODELS)


class SpatialNodeTree(RootModel[AnySpatialNodeType]):
    """整棵空间节点树的根模型：用 `SpatialNodeTree.model_validate(json_dict)` 校验 LLM 输出。"""


# 前向引用在此统一解析：此时 AnySpatialNodeType 已在模块命名空间中
for _model in (
    Buffer,
    DirectionalConstraint,
    DirectionalSubset,
    Intersection,
    Union,
    Difference,
    Between,
    BorderBetween,
    BorderOf,
    CoastOf,
    OffTheCoastOf,
    LandUseArea,
    EventAffected,
):
    _model.model_rebuild()

# 用地类别名单与 Literal 必须同键：加了一个 Literal 值却忘了补中文说法（或反之），
# 提示词里就会漏掉一个类别、或者写一个模型拒收的类别，两边都很难从现象反查。
assert set(LANDUSE_LABELS) == set(LANDUSE_CATEGORIES), (
    f"LANDUSE_LABELS 与 LandUseCategory 不一致："
    f"字典多出 {set(LANDUSE_LABELS) - set(LANDUSE_CATEGORIES)}，"
    f"Literal 多出 {set(LANDUSE_CATEGORIES) - set(LANDUSE_LABELS)}"
)
