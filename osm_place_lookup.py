"""
OSM Polygon Place Lookup — 使用 OpenStreetMap Overpass API 获取地点的多边形边界。

与源项目 (natural-language-geocoding) 的 PlaceLookup 架构对齐：
  - 源项目用 Nominatim/OpenSearch 返回真实多边形
  - 本模块用 Overpass API + 高德 District API 返回真实多边形
  - 所有 place lookup 都返回 Polygon/MultiPolygon，永不返回 Point 或矩形
  - 对外接口是 place_lookup.PlaceLookup（ABC），可被替换/注入

Overpass API 端点（按优先级尝试）：
  1. https://overpass-api.de/api/interpreter
  2. https://overpass.kumi.systems/api/interpreter
  3. https://maps.mail.ru/osm/tools/overpass/api/interpreter

响应格式说明（`[out:json]` + `out geom`）：
  该模式返回的是 OSM 原始坐标系（WGS-84），但**不是 GeoJSON**：
    - way  的坐标在 `geometry: [{lat, lon}, ...]`，且是线（未保证闭合）
    - relation 的坐标在 `members[].geometry: [{lat, lon}, ...]`，按 role 区分
      outer / inner，一个环常被拆成多个 way
  因此必须手工组装环并缝合，不能直接丢给 shapely 的 shape()。
  之所以不用 `[out:geojson]`：它只在较新版本的 Overpass 上可用，各镜像支持
  不一致，而本项目要轮询多个镜像。手工解析在所有端点上都成立。


耗时治理（S26）——实测数据与调参依据
------------------------------------
改造前的问题：三个端点是**串行**试的，每个都给 30s 读超时，最坏情况一次查询
要白等 90s+。而实测（2026-09-20，本机、同一时段反复探测）三个公共端点里
**只有 overpass-api.de 真的会回话**：

  端点                              现象
  ---------------------------------  ----------------------------------------
  overpass-api.de                    TCP+TLS 通，正常返回；密集探测时会 429/504
  overpass.kumi.systems              能建连，但一个字节都不发（挂死）
  maps.mail.ru/osm/tools/overpass    同上

即"挂死"是常态而非异常，所以治理方向不是"加大超时"，而是**尽早放弃**。

改善后的实测（同上环境）：

  场景                                          改前        改后
  --------------------------------------------  ----------  ----------
  "深圳大学" 冷启动（单次 Overpass）            ~90s 最坏   1.64s
  同一查询第二次                                重新联网    0.000s，0 次网络请求
  3/3 端点人为不可达                            180s        6.2s
  2/3 端点人为不可达                            60s+        ~1.6s

四项机制分别对应上表的一行：端点健康记忆 → 排序把已知可用的排前面；
并行竞速 → 不再串行等，最先回内容的赢；超时分级 → 只有确认好用的端点才给
长超时；查询级磁盘缓存 → 第二次不再联网。

三个实测出来的语义陷阱（都写进了对应常量的注释，这里列个索引）：

  1. `[bbox:...]` 只过滤**选择**，不裁剪**输出**。用只有深圳大小的框查"广东省"
     relation，返回的几何与无框查询逐字节相同（940268B）。所以 item 6 对
     "区域里同名的歧义点"有效（杭州框查"西湖"：304 要素/2.28MB/24.6s →
     1 要素/0.15MB/1.5s），但**解决不了巨型 relation 的载荷问题**——那要靠
     别的办法，不是 bbox。
  2. requests 的读超时是"两次 socket 读之间"的间隔，不是整个响应的总预算。
     实测 `France` 的 relation 是一个 6.1MB、服务端算 46s 才开吐的响应，
     `timeout=(5, 30)` 也能收全（总耗时 67.8s）——中间没有一次静默超过 30s。
     这条直接决定了 `_RACE_MAX_WAIT_S` 必须显著大于 `_READ_TIMEOUT_KNOWN`：
     曾经把它设成 40s，结果慢查询（在**正常流式传输**）被硬砍成"查无此地"，
     "法国和西班牙的边界"基线从比利牛斯窄带退化成中心点。
  3. 响应里的 `remark` 表示服务端截断（如 `Query timed out in "print"`），
     这类响应是**残缺**的：可以拿来出结果，但不能写进缓存，否则残缺会被
     当成正确答案长期复用。

超时触发失败记录的判定也被这条修正过：冷档（8s）的读超时**不记为失败**。
不这样区分的话，一个只是慢、但健康的端点会先被判 bad，再被排除在补轮之外
（补轮只挑非 bad 的），于是永久失联——正是"永久黑名单"这个反面目标。
"""

import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    shape as shapely_shape,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, polygonize, unary_union
from shapely import make_valid

import requests

from errors import GeocodeError
from place_lookup import (
    HIERARCHY_IN_REGION,
    HIERARCHY_NEAR_REFERENCE,
    NAME_MATCH_EXACT,
    SOURCE_PRIORITY,
    PlaceCandidate,
    PlaceLookup,
    PlaceSearchRequest,
    area_score,
    has_latin,
    name_match_score,
    rank_candidates,
)

# 查询时匹配的名字标签键（覆盖 OSM 表达"别名/多语言名"的常见写法）。
# 与 _score_group 里 tag_values(...) 的列表保持一致：查得到、也要认得出。
_QUERY_NAME_TAGS = (
    "name",
    "name:zh",
    "name:en",
    "int_name",
    "alt_name",
    "official_name",
    "old_name",
)
# 评分时视为"别名"的标签（主名称另算）
_ALIAS_NAME_TAGS = ("name:zh", "name:en", "int_name", "alt_name", "official_name", "old_name")
# "查询语言下的名称"标签：拉丁文查询看 name:en/int_name，中文查询看 name:zh。
# 单独拎出来是因为主名称常是当地语言（"Spain" 对应 name='España' 的国界关系），
# 这种对应关系必须算满分，否则会被碰巧同名的小地物抢走（见 _score_group）。
_LATIN_NAME_TAGS = ("name:en", "int_name")
_CJK_NAME_TAGS = ("name:zh",)


# =============================================================================
# 端点健康记忆 / 超时分级 / 查询缓存（S26）
#
# 实测（2026-09-20，本机直连，详见模块顶部注释的实测表）：
#   只有 overpass-api.de 真的会回数据；kumi.systems 与 maps.mail.ru
#   **TCP 能连上、但一个字节都不回**，直到客户端读超时。
#   旧实现是"逐个串行 × 每端点重试 2 轮"，于是"第一个端点没直接命中"
#   就等于固定白等 60–120 秒。下面这几组常量的作用就是把这段等待去掉：
#     - 健康度评分用来**排序**（谁先发第一枪）与**分级超时**，
#       失败是乘法衰减 + 时间半衰期回升，不做永久黑名单
#     - 竞速让等待时间等于"最快那个端点的耗时"，而不是"所有端点耗时之和"
#     - 只有"近期成功过"的端点拿长超时；未知端点先按 8s 探一次，
#       谁都没答时才用补轮的长超时重新投资
# =============================================================================

# 健康度参数：初值 0.5（未知端点不预设好坏），成功后 +0.5 封顶 1.0，
# 失败乘 0.4：0.5 → 0.2，当场低于下面的 bad 阈值，所以"刚失败过的端点"
# 立刻降级，不必等它失败很多次。
#
# **衰减的目标是中性值 _HEALTH_NEUTRAL，不是地板**：记录随时间朝 0.5 回走
# （半衰期 _HEALTH_HALF_LIFE_HOURS）。两个方向的"忘记"都靠它：
#   - 失败态 0.2 → 约 9 小时后回到 bad 阈值之上，重获被使用的资格
#   - 成功态 1.0 → 约 2.3 天后掉出 good 阈值，不再独享长超时
# 若朝地板衰减就反了——失败端点的分数只会越来越低，等于永久黑名单，
# 正是本节开头要避免的东西。_HEALTH_FLOOR 只是存储值的下限。
_HEALTH_NEUTRAL = 0.5
_HEALTH_INITIAL = _HEALTH_NEUTRAL
_HEALTH_SUCCESS_GAIN = 0.5
_HEALTH_FAIL_PENALTY = 0.4
_HEALTH_FLOOR = 0.05
_HEALTH_HALF_LIFE_HOURS = 24.0
# bad 阈值低于中性值，good 阈值高于它：中间那条带就是"unknown"。
# good 取 0.6 而不是 0.25，是为了让"很久没成功过"的端点真的会掉档——
# 若取 0.25，成功态的衰减永远到不了阈值以下，good 就成了永久的。
_HEALTH_BAD_THRESHOLD = 0.25
_HEALTH_GOOD_THRESHOLD = 0.6
_HEALTH_FILE = "endpoints.json"

# 分级超时。服务端 `[out:json][timeout:25]` 里的 25s 是**合法的计算时间**，
# 客户端读超时必须高于它，否则会把"正当的慢查询"砍成失败——所以健康端点是 30s。
# 其余（未知、近期失败过）一律 8s：这一档的用途是快速淘汰死镜像，不是等它算完。
_READ_TIMEOUT_KNOWN = 30.0
_READ_TIMEOUT_COLD = 8.0
_CONNECT_TIMEOUT = 5.0
# 竞速的错峰间隔：先只发最健康的端点，这么久内没拿到"有内容的结果"就把其余端点
# 一起发出去。避免在一次查找里把唯一可用的镜像连打三遍（公共镜像会限流）。
_RACE_STAGGER_S = 1.2
# 竞速总等待的硬上限：兜底用的，正常路径都由"所有已发端点都报了结果"提前收工。
#
# **这个值必须显著大于 _READ_TIMEOUT_KNOWN，不能只看单次超时**：requests 的读
# 超时是"两次 socket 读之间"的间隔，不是整个响应的总时限。实测 `France` 的
# relation 是一个 6.1MB、服务端算 46s 才开吐的响应，客户端 timeout=(5,30) 也能
# 收全（总耗时 67.8s）——中间没有一次静默超过 30s，所以读超时根本没触发。
# 曾经把这个上限设成 40s，结果法国那条基线被硬砍成"查无此地"：慢查询是**在
# 正常流式传输**，不是在挂死，拿错峰/兜底的时间去砍它会砍掉合法结果。
_RACE_MAX_WAIT_S = 150.0
# "有人回了 200 但 elements 为空"之后，还给其余端点多久。空响应本身不封盘（见
# `_overpass_query` 的 docstring：限流也会产生 200 空，得让别的端点有机会纠正），
# 但等待**必须封顶**——S27 实测 3 候选并发时，一个候选的 27s 全花在这里：唯一会
# 回话的端点 6.3s 就答了"空"，剩下两个镜像（一个能建连但永不发字节、一个要 20–33s
# 才答）把候选一路拖到 33.3s，而那个空答案并没有变。
#
# 顶用的是冷档超时的那个数：它正是本模块给"未知端点"的预算，语义上刚好——
# 数据源已经就这次查询给了实质回答，未知端点只配再拿一档冷超时的机会。
# 不适用于"第一枪失败/超时"的路径（那没有空响应，仍走 _RACE_MAX_WAIT_S）。
_RACE_EMPTY_GRACE_S = _READ_TIMEOUT_COLD

# 查询级缓存 TTL。空结果也缓存，但只存 6 小时：200 + 空可能是镜像限流造成的，
# 存 7 天会把一次限流固化成一周的"查无此地"。
_CACHE_TTL_FULL_S = 7 * 24 * 3600
_CACHE_TTL_EMPTY_S = 6 * 3600
_CACHE_VERSION = 1

# requests.Session 不是线程安全的，按线程各持一个（见 OsmPolygonLookup._session）
_THREAD_LOCAL = threading.local()

# 全局在飞请求上限。Overpass 的瓦片使用政策限制并发，而本机实测唯一会回话的
# 镜像（overpass-api.de）在三个请求同时打过去时直接返回 504（限流史见 S04/S20）。
# 放模块级而不是实例级：S27 的 agent 里多个候选共享一个 lookup 实例，但将来也可能
# 各持一个，而端点是同一台公共服务器——限制必须对进程内所有请求生效，按实例隔离
# 等于没限制。
_OVERPASS_SLOTS = threading.Semaphore(2)


def _cache_dir() -> str:
    """OSM 查询缓存与端点健康表的落盘目录（打包后与 exe 同级，开发时与模块同级）。"""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "_osm_cache")


def _atomic_write_json(path: str, payload: dict) -> None:
    """先写临时文件再替换，避免中途失败留下半个 JSON。写不进去就算了。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        pass


@dataclass
class _EndpointHealth:
    """一个端点的健康记录。"""

    score: float = _HEALTH_INITIAL
    updated: float = 0.0
    ever_succeeded: bool = False
    last_latency: float | None = None


class _HealthStore:
    """端点健康度：衰减式评分，落盘跨运行复用。

    **为什么不是永久黑名单**：kumi/mail.ru 只是"当前网络下不可达"，换网络、
    换时段可能就好了。永久拉黑会在 overpass-api.de 也出问题时把回退路径丢光。
    所以失败只做乘法衰减，并随时间半衰期向地板回升——放几天不用，坏端点会自己
    爬回可试状态。
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, _EndpointHealth] = self._load()

    def _load(self) -> dict[str, _EndpointHealth]:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception:
            return {}
        out: dict[str, _EndpointHealth] = {}
        for ep, rec in (raw.get("endpoints") or {}).items():
            try:
                out[ep] = _EndpointHealth(
                    score=float(rec["score"]),
                    updated=float(rec.get("updated", 0.0)),
                    ever_succeeded=bool(rec.get("ever_succeeded", False)),
                    last_latency=(None if rec.get("last_latency") is None
                                  else float(rec["last_latency"])),
                )
            except Exception:
                continue
        return out

    def _save(self) -> None:
        payload = {"endpoints": {
            ep: {"score": round(r.score, 6), "updated": r.updated,
                 "ever_succeeded": r.ever_succeeded, "last_latency": r.last_latency}
            for ep, r in self._data.items()
        }}
        _atomic_write_json(self.path, payload)

    def _decayed(self, ep: str, now: float) -> _EndpointHealth:
        """把记录按存放时长衰减到"现在"的值（不改动存储）。

        衰减目标是中性值：记录越旧越接近"未知"，而不是越旧越坏。
        """
        rec = self._data.get(ep)
        if rec is None:
            return _EndpointHealth()
        age_hours = max(0.0, (now - rec.updated) / 3600.0)
        factor = 0.5 ** (age_hours / _HEALTH_HALF_LIFE_HOURS)
        return _EndpointHealth(
            score=_HEALTH_NEUTRAL + (rec.score - _HEALTH_NEUTRAL) * factor,
            updated=rec.updated,
            ever_succeeded=rec.ever_succeeded,
            last_latency=rec.last_latency,
        )

    def scored(self, endpoints: Sequence[str]) -> list[float]:
        now = time.time()
        with self._lock:
            return [self._decayed(ep, now).score for ep in endpoints]

    def tier(self, ep: str) -> str:
        """端点分级：`good`（近期成功过）/ `bad`（近期失败过）/ `unknown`。

        三种状态的用途不同：
          - 长超时只给 `good`（慢查询在它身上是正常的）
          - 补轮只发给非 `bad`（对近期失败过的端点重试没有意义）
          - `unknown` 介于两者之间：它不配长超时（免得替死镜像把等待付掉），
            但值得在"谁都没答"时进补轮拿一次长超时
        """
        now = time.time()
        with self._lock:
            rec = self._data.get(ep)
            if rec is None:
                return "unknown"
            decayed = self._decayed(ep, now)
        if decayed.score < _HEALTH_BAD_THRESHOLD:
            return "bad"
        if rec.ever_succeeded and decayed.score >= _HEALTH_GOOD_THRESHOLD:
            return "good"
        return "unknown"

    def is_good(self, ep: str) -> bool:
        return self.tier(ep) == "good"

    def not_recently_failed(self, ep: str) -> bool:
        return self.tier(ep) != "bad"

    def record_success(self, ep: str, latency: float) -> None:
        with self._lock:
            rec = self._data.get(ep) or _EndpointHealth()
            rec.score = min(1.0, rec.score + _HEALTH_SUCCESS_GAIN)
            rec.updated = time.time()
            rec.ever_succeeded = True
            rec.last_latency = latency
            self._data[ep] = rec
            self._save()

    def record_failure(self, ep: str) -> None:
        with self._lock:
            rec = self._data.get(ep) or _EndpointHealth()
            rec.score = max(_HEALTH_FLOOR, rec.score * _HEALTH_FAIL_PENALTY)
            rec.updated = time.time()
            self._data[ep] = rec
            self._save()


class _Inflight:
    """一个正在执行的查询：后来者等 `gate`，然后直接读 `result`。"""

    __slots__ = ("gate", "result")

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.result: dict | None = None


@dataclass
class _RaceOutcome:
    """一轮竞速的结果。"""

    data: dict | None = None          # 有 elements 且完整（无 remark）的响应
    partial: dict | None = None       # 有 elements 但带 remark 的响应（服务端截断）
    empty: dict | None = None         # 200 但 elements 为空的响应
    answered: set = None              # 返回过 200 的端点
    failed: set = None                # 超时/异常/非 200 的端点

    def __post_init__(self):
        self.answered = self.answered if self.answered is not None else set()
        self.failed = self.failed if self.failed is not None else set()

    @property
    def reached_server(self) -> bool:
        return bool(self.answered)

    def best(self) -> dict | None:
        """按质量取最好的响应：完整 > 截断 > 空。"""
        if self.data is not None:
            return self.data
        if self.partial is not None:
            return self.partial
        return self.empty


@dataclass
class _OsmGroup:
    """一个 OSM element 解析出的一组多边形（同一地物的多个部件）。"""

    polygons: list[Polygon]
    tags: dict
    element_type: str
    element_id: int | None

    @property
    def geometry(self):
        """把组内部件合并成一个几何。"""
        if len(self.polygons) == 1:
            return self.polygons[0]
        return _repair(unary_union(self.polygons))

    def tag_values(self, *keys: str) -> list[str]:
        return [self.tags[k] for k in keys if self.tags.get(k)]


# 请求带 place_type 时用于判断候选类型是否相符（OSM 标签 → 地点类型的对应）。
# 注意：当前管线还没有把 place_type 传进来（LLM 节点里没有这个字段），
# 所以这一维度暂时恒为 0，等节点补上 place_type 后自动生效。
_TYPE_TAG_HINTS: dict[str, tuple[tuple[str, frozenset[str]], ...]] = {
    "country":     (("boundary", frozenset({"administrative"})), ("admin_level", frozenset({"2"}))),
    "region":      (("boundary", frozenset({"administrative"})), ("admin_level", frozenset({"4"}))),
    "macroregion": (("boundary", frozenset({"administrative"})), ("admin_level", frozenset({"4", "5"}))),
    "locality":    (("boundary", frozenset({"administrative"})), ("admin_level", frozenset({"6", "7", "8"}))),
    "lake":        (("natural", frozenset({"water"})), ("water", frozenset({"lake", "reservoir"}))),
    "river":       (("waterway", frozenset({"river", "riverbank"})),),
    "sea":         (("natural", frozenset({"sea"})), ("place", frozenset({"sea"}))),
    "bay":         (("natural", frozenset({"bay"})),),
    "strait":      (("natural", frozenset({"strait"})),),
    "island":      (("place", frozenset({"island"})),),
    "peninsula":   (("place", frozenset({"peninsula"})),),
    "desert":      (("natural", frozenset({"desert", "sand"})),),
    "port":        (("harbour", frozenset({"yes"})), ("landuse", frozenset({"harbour"}))),
}

# 类型相符的加分
TYPE_MATCH_BONUS = 1.0


def _coords_from_geometry(geom_data: list | None) -> list[tuple[float, float]]:
    """把 Overpass 的 `geometry: [{lat, lon}, ...]` 转成 Shapely 需要的 (lon, lat) 列表。

    `out geom` 在被裁剪（bbox 查询）时可能返回只有 lat 的残缺点，这类点直接跳过。
    """
    coords = []
    for p in geom_data or []:
        if "lat" in p and "lon" in p:
            coords.append((float(p["lon"]), float(p["lat"])))
    return coords


def _close_ring(coords: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    """把折线闭合成环；无法可靠闭合时返回 None。

    闭合 way 的 `geometry` 首尾点相同，直接用即可。少数情况下首尾差一个点
    （数据切分导致），当缺口小于周长千分之一时补上；缺口过大说明这本来就是
    一条开放的线（道路、未闭合的河流），不能臆造出一个多边形，返回 None。
    """
    if len(coords) < 3:
        return None
    if coords[0] == coords[-1]:
        return coords if len(coords) >= 4 else None

    ring = coords + [coords[0]]
    perimeter = LineString(ring).length
    if perimeter <= 0:
        return None
    gap = Point(coords[0]).distance(Point(coords[-1]))
    if gap / perimeter < 0.001:
        return ring
    return None


def _polygonize_lines(lines: list[LineString]) -> object:
    """把若干线段缝合成面。

    Overpass 里一条环常由多个 way 拼成，先 linemerge 拼接首尾相接的线段，
    再用 polygonize 把闭合线转成面。
    """
    if not lines:
        return GeometryCollection()
    merged = linemerge(MultiLineString(lines))
    polygons = list(polygonize(unary_union(merged)))
    if not polygons:
        return GeometryCollection()
    return unary_union(polygons)


def _extract_polygons(geom) -> list[Polygon]:
    """从任意几何中取出所有 Polygon（递归展开 GeometryCollection）。"""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        out = []
        for part in geom.geoms:
            out.extend(_extract_polygons(part))
        return out
    return []


def _repair(geom) -> object | None:
    """修复自相交等非法几何；无法修复时返回 None。

    用 make_valid 而非 .buffer(0)：后者对"蝴蝶结"自相交只保留一个瓣（面积偏小），
    make_valid 会把两个瓣都保留下来。
    """
    if geom.is_valid:
        return geom
    try:
        fixed = make_valid(geom)
    except Exception:
        fixed = geom.buffer(0)
    if fixed is None or fixed.is_empty:
        return None
    return fixed


def _relation_to_polygons(elem: dict) -> list[Polygon]:
    """把 relation 的成员按 role 组装成带孔的 Polygon。

    outer 成员缝合成外环，inner 成员缝合成孔洞，最后做差集。
    这种"先分别求面再相减"的做法能正确处理一个 outer 内含多个 inner 的常见情形。
    """
    outer_lines, inner_lines = [], []
    for member in elem.get("members", []):
        coords = _coords_from_geometry(member.get("geometry"))
        if len(coords) < 2:
            continue
        role = member.get("role") or "outer"
        (inner_lines if role == "inner" else outer_lines).append(LineString(coords))

    outers = _polygonize_lines(outer_lines)
    if outers.is_empty:
        return []

    inners = _polygonize_lines(inner_lines)
    geom = outers.difference(inners) if not inners.is_empty else outers
    return _extract_polygons(geom)


class OsmPolygonLookup(PlaceLookup):
    """
    从 OSM Overpass API 获取地点的真实多边形边界。

    实现 place_lookup.PlaceLookup：search(request) 返回 BaseGeometry。
    内部使用多数据源策略：
      1. 高德 District API → 中国行政区域真实多边形
      2. OSM Overpass API → 全球范围的多边形数据
    两个数据源都失败时抛出异常，不返回任何合成几何。
    """

    # Overpass 要求带可识别的 User-Agent，缺省 UA 会被 406/429 拒绝
    HEADERS = {"User-Agent": "natural-language-geocoding/1.0 (polygon lookup)"}

    # Overpass API 端点列表
    OVERPASS_ENDPOINTS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    # 区域弱校验容差（度）：约 30km。小于此跨度的小地物用此值兜底，
    # 大区域（省、大湖）则按自身尺度放宽，避免被误拒。
    _VALIDATE_MIN_TOLERANCE_DEG = 0.3

    def __init__(self, amap_geocoder=None, cache_dir: str | None = None,
                 session=None):
        """
        Args:
            amap_geocoder: 高德客户端（层级核验与参考点用）。
            cache_dir: 查询缓存与端点健康表的目录；None 用默认位置。
            session: 可注入的 requests 会话（测试用）。为 None 时按线程各建一个
                并复用，省掉每次 POST 的 TCP+TLS 握手。
        """
        self.amap = amap_geocoder
        self.cache_dir = cache_dir or _cache_dir()
        self.session = session
        self.health = _HealthStore(os.path.join(self.cache_dir, _HEALTH_FILE))
        self._cache_lock = threading.Lock()
        # 在飞查询表（见 _overpass_query 的合并逻辑）
        self._inflight_lock = threading.Lock()
        self._inflight: dict[str, _Inflight] = {}

    # -------------------------------------------------------------------------
    # 连接与缓存
    # -------------------------------------------------------------------------
    def _session(self) -> requests.Session:
        """取当前线程的 Session（requests.Session 不是线程安全的，按线程隔离）。"""
        if self.session is not None:
            return self.session
        sess = getattr(_THREAD_LOCAL, "osm_session", None)
        if sess is None:
            sess = requests.Session()
            sess.headers.update(self.HEADERS)
            _THREAD_LOCAL.osm_session = sess
        return sess

    def _cache_path(self, query: str) -> str:
        key = hashlib.sha1(query.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"q_{key}.json")

    def _cache_read(self, query: str) -> dict | None:
        """读查询缓存；命中且未过期返回响应体，否则 None。"""
        path = self._cache_path(query)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if raw.get("version") != _CACHE_VERSION or raw.get("query") != query:
                return None
            elements = raw.get("elements")
            if not isinstance(elements, list):
                return None
            ttl = _CACHE_TTL_FULL_S if elements else _CACHE_TTL_EMPTY_S
            age = time.time() - float(raw.get("fetched_at", 0.0))
            if age > ttl:
                return None
        except Exception:
            return None
        age_h = age / 3600.0
        print(f"[OSM] 命中查询缓存（{len(elements)} 个要素，"
              f"写入于 {age_h:.1f} 小时前）")
        return {"elements": elements}

    def _cache_write(self, query: str, data: dict) -> None:
        elements = data.get("elements") or []
        _atomic_write_json(self._cache_path(query), {
            "version": _CACHE_VERSION,
            "query": query,
            "fetched_at": time.time(),
            "elements": elements,
        })

    # -------------------------------------------------------------------------
    # 竞速
    # -------------------------------------------------------------------------
    def _ordered_endpoints(self) -> list[str]:
        """按健康度降序排出端点（同分保持声明顺序，耗时短的优先）。"""
        scores = self.health.scored(self.OVERPASS_ENDPOINTS)
        indexed = list(enumerate(zip(self.OVERPASS_ENDPOINTS, scores)))
        indexed.sort(key=lambda item: (-item[1][1], item[0]))
        return [ep for _, (ep, _) in indexed]

    def _read_timeout_for(self, ep: str) -> float:
        """该给这个端点多长的读超时：只有"近期成功过"的端点配长超时。

        长超时 30s 高于服务端 `timeout:25`，不会砍掉合法的慢查询；短超时 8s
        用于快速淘汰——公共镜像里"TCP 连得上、一个字节不回"是常态
        （kumi/mail.ru 实测如此），给它 30s 只是替死镜像把等待提前付掉。
        全新安装时所有端点都是 unknown → 一律 8s，慢查询靠补轮的长超时兜住。
        """
        return _READ_TIMEOUT_KNOWN if self.health.is_good(ep) else _READ_TIMEOUT_COLD

    @staticmethod
    def _is_partial(data: dict) -> bool:
        """Overpass 用 `remark` 标明"服务端超时、只返回了已算完的部分"。

        这种响应不能当胜者（换个端点可能给出完整答案），更不能进 7 天缓存——
        把一次服务端截断固化一周，比不缓存糟得多。
        """
        return bool(data.get("remark"))

    def _probe(self, ep: str, query: str, box: dict, lock: threading.Lock,
               done: threading.Event, read_timeout: float) -> None:
        """在独立线程里请求一个端点，把结果并进 box（胜者先到先得）。

        读超时在短超时档（8s 探路）上不算"端点失败"：8s 只够淘汰死镜像，不够
        判断一个慢查询的端点坏掉了。把它记成失败会让一个健康的慢端点被打成
        bad、从补轮里被踢出去，之后再也翻不了身。真正的失败（非 200、连不上、
        长超时档上还没答）才记。
        """
        start = time.monotonic()
        try:
            # 槽位在**发请求之前**就占住：限的是"在飞请求数"，等到响应回来再占
            # 就变成限"同时处理的响应数"，毫无意义。
            with _OVERPASS_SLOTS:
                resp = self._session().post(
                    ep, data={"data": query},
                    timeout=(_CONNECT_TIMEOUT, read_timeout),
                )
            latency = time.monotonic() - start
            if resp.status_code != 200:
                with lock:
                    box["failed"].add(ep)
                self.health.record_failure(ep)
                return
            data = resp.json()
        except requests.exceptions.ReadTimeout:
            with lock:
                box["failed"].add(ep)
            if read_timeout >= _READ_TIMEOUT_KNOWN:
                self.health.record_failure(ep)
            return
        except Exception:
            with lock:
                box["failed"].add(ep)
            self.health.record_failure(ep)
            return

        with lock:
            box["answered"].add(ep)
            if data.get("elements"):
                if self._is_partial(data):
                    if box["partial"] is None:
                        box["partial"] = data
                else:
                    if box["data"] is None:
                        box["data"] = data
                    done.set()
            elif box["empty"] is None:
                box["empty"] = data
                box["empty_at"] = time.monotonic()
        self.health.record_success(ep, latency)

    def _race(self, endpoints: Sequence[str], query: str,
              escalation: bool = False) -> _RaceOutcome:
        """并发发起，首个"有内容"的结果胜出。

        错峰：先只发最健康的端点，_RACE_STAGGER_S 内它没给出有内容的结果
        （失败、超时、或 200 但为空）就把其余端点一起发出去。这样"第一个端点
        直接命中"的常见情形不会把公共镜像连打三遍，而"第一个端点不行"的最坏
        路径也只是一个错峰间隔，不再是串行的等待之和。

        `escalation=True` 是"谁都没答"之后的补轮：一律给长超时。此时已经没有
        更快的结果可等，把等待时间投在一个慢端点上是唯一还能赢的选择。

        **已知坏的端点（tier == bad）整轮不参与**，第一轮和补轮都不发。理由是
        S27 实测出来的：并发跑 3 个候选时，每个候选错峰失败后都会补发其余端点，
        于是 3 候选 × 3 端点 = 9 个请求去挤全局 2 个槽位；而那两个镜像"能建连但
        不发字节"，每个白占槽位 8 秒，把唯一会回话的端点挤到 429/504——**自己把
        自己打到限流**，实测一次并发查询总耗时 94.6s（其中 55.6s 是在等一个慢镜像）。

        这不是永久拉黑，而是"只在没有更好选择时才用"：评分朝中性值 0.5 衰减，
        几小时后自己回到 unknown；而且**端点里一个好端点都没有时会全放回来**
        （下面的 `or endpoints`，以及 `_run_query` 里补轮的 `or ordered`）——
        "全都不可靠"不等于"放弃"。

        线程是 daemon：已经拿到结果后，还在等死镜像的线程不会拖住进程退出。
        """
        box = {"data": None, "partial": None, "empty": None, "empty_at": None,
               "answered": set(), "failed": set(), "reported": 0}
        lock = threading.Lock()
        done = threading.Event()
        first_done = threading.Event()

        def worker(ep: str, is_first: bool = False) -> None:
            timeout = (_READ_TIMEOUT_KNOWN if escalation
                       else self._read_timeout_for(ep))
            try:
                self._probe(ep, query, box, lock, done, timeout)
            finally:
                with lock:
                    box["reported"] += 1
                if is_first:
                    first_done.set()

        endpoints = list(endpoints)
        if not endpoints:
            return _RaceOutcome()

        if not escalation:
            # 已知坏的端点不参加第一轮（见 docstring）。全都坏时不挑，照发——
            # "没有好端点"不等于"放弃"。
            preferred = [ep for ep in endpoints if self.health.tier(ep) != "bad"]
            if preferred:
                endpoints = preferred

        threading.Thread(target=worker, args=(endpoints[0], True), daemon=True).start()
        # 等第一枪回信，或错峰时间到。done 一定先于 first_done 生效
        # （前者在 _probe 内、后者在 finally），所以这里看到 first_done
        # 就可以信任 done 的状态。
        first_done.wait(_RACE_STAGGER_S)
        fired = [endpoints[0]]
        if not done.is_set():
            # 第一枪没给出有内容的结果（失败/超时/200 但空），把其余端点补齐
            for ep in endpoints[1:]:
                threading.Thread(target=worker, args=(ep,), daemon=True).start()
                fired.append(ep)

        deadline = time.monotonic() + _RACE_MAX_WAIT_S
        while not done.is_set():
            now = time.monotonic()
            with lock:
                if box["reported"] >= len(fired):
                    break
                empty_at = box["empty_at"]
            if now > deadline:
                break
            # 已经有人回了"200 但空"：其余端点的机会按一档冷超时封顶，
            # 而不是等到 _RACE_MAX_WAIT_S（见 _RACE_EMPTY_GRACE_S）。
            if empty_at is not None and now > empty_at + _RACE_EMPTY_GRACE_S:
                break
            done.wait(0.05)

        with lock:
            return _RaceOutcome(data=box["data"], partial=box["partial"],
                                empty=box["empty"],
                                answered=set(box["answered"]),
                                failed=set(box["failed"]))

    def _overpass_query(self, query: str) -> dict | None:
        """执行 Overpass QL 查询，返回 JSON 结果；全部端点都不行时返回 None。

        流程：查询缓存 → 一轮短超时竞速 → （一个 200 都没拿到时才）对非近期
        失败的端点补一轮长超时。

        这里不做"可用性预检"：Overpass 的状态端点 `/api/status` 对缺少
        User-Agent 的请求会返回 406/429，预检会在服务其实可用时误判为不可用，
        直接放弃 OSM 这条路。查询失败与端点不可达在这里是同一件事。

        200 但 elements 为空的镜像（限流时会出现）不算命中，要让其余端点也有
        机会答；只有所有端点都空（或都超时）才退回空结果——否则会把"镜像限流"
        误判成"查无此地物"。

        补那一轮的条件是**一个 200 都没有**：只要有人答了 200，那就是数据源
        对这次查询的答复，再试只是白等。实测会遇到"同一个端点上一次 200、下一次
        429/504"（公共镜像限流），补轮正是为它准备的。补轮发给非近期失败的端点
        （含未知端点）：全新安装时所有端点都还是 unknown，第一轮只拿到 8s，
        正是靠补轮的长超时才不至于把"合法的慢查询"整轮砍掉。

        带 `remark` 的响应（服务端超时截断）可以用，但**不入缓存**。

        **同一查询并发到达时只发一次**：S27 的 agent 里多个候选常共享同一个基础
        地点（"亚马孙雨林"在三个候选里都出现），并发跑就会打出三条一模一样的
        Overpass 请求——而唯一可用镜像正是不许并发的那个。后来者在这里等先到者
        的结果，直接复用。等的是**内存里的结果**而不是缓存条目：截断的响应不入
        缓存，靠缓存复现的话后来者会莫名其妙拿到 None。
        """
        cached = self._cache_read(query)
        if cached is not None:
            return cached

        with self._inflight_lock:
            slot = self._inflight.get(query)
            leader = slot is None
            if leader:
                slot = _Inflight()
                self._inflight[query] = slot

        if not leader:
            slot.gate.wait(_RACE_MAX_WAIT_S + _CONNECT_TIMEOUT)
            return slot.result

        try:
            result = self._run_query(query)
            slot.result = result
            return result
        finally:
            with self._inflight_lock:
                self._inflight.pop(query, None)
            slot.gate.set()

    def _run_query(self, query: str) -> dict | None:
        """真正打网络的那一段（缓存已在上层查过）。"""
        ordered = self._ordered_endpoints()
        outcome = self._race(ordered, query)

        if outcome.data is None and not outcome.reached_server:
            # 一个 200 都没拿到。此时 outcome 里必然没有可用结果，直接换一轮。
            # 优先补"最近没失败过"的端点；**全都 recent bad 时仍然补全部**——
            # 那说明这一轮只是撞上了一次集体抖动（实测并发时唯一可用镜像会被自己
            # 打到 429，两次失败就把评分压到 bad），此时不补等于把一次抖动直接
            # 判成"服务不可用"。
            retry = [ep for ep in ordered if self.health.not_recently_failed(ep)] or ordered
            print(f"[OSM] 全部端点本轮均未响应，对 {len(retry)} 个端点补一轮长超时")
            outcome = self._race(retry, query, escalation=True)

        result = outcome.best()
        if result is None:
            return None
        if self._is_partial(result):
            print(f"[OSM] 数据源返回被截断（{result.get('remark')}）："
                  f"{len(result.get('elements') or [])} 个要素，本次不入缓存")
            return result
        self._cache_write(query, result)
        return result

    def _element_polygon_groups(self, elements: list[dict]) -> list["_OsmGroup"]:
        """
        将 Overpass API 返回的 elements 解析成"按 element 分组"的候选列表。

        每个 element 一组，组内是该地物的所有部件。这个分组很关键：
          - 一个 relation 的多个部件属于同一地物（深圳大学的东西两个校区）
          - 不同 element 之间则是同名地物（"西湖"在 OSM 里多达 61 个，
            从南极洲到马来西亚都有）
        只有保住分组，调用方才能在合并前把同名地物甄别掉。

        两类 element 分开处理：
          - relation：成员按 role 组装外环/内孔，缝合成带孔多边形（信息最完整）
          - way：独立的闭合 way 直接转成多边形；但作为 relation 成员出现过的 way
            要跳过，否则它作为孔洞的内环会被当成实心面填回去

        每组的 tags 一并带出来：评分要用它判断名称到底是主名称还是别名、
        以及地物类型是否与请求相符。
        """
        relation_member_way_ids = set()
        for elem in elements:
            if elem.get("type") == "relation":
                for member in elem.get("members", []):
                    if member.get("type") == "way":
                        relation_member_way_ids.add(member.get("ref"))

        groups = []
        for elem in elements:
            if elem.get("type") == "relation":
                polygons = _relation_to_polygons(elem)
            elif elem.get("type") == "way":
                if elem.get("id") in relation_member_way_ids:
                    continue
                ring = _close_ring(_coords_from_geometry(elem.get("geometry")))
                polygons = [Polygon(ring)] if ring else []
            else:
                continue  # node 没有面几何

            # 修复非法几何（make_valid 可能返回复合类型，需再拆回 Polygon）
            parts = []
            for poly in polygons:
                fixed = _repair(poly)
                if fixed is None:
                    continue
                for part in _extract_polygons(fixed):
                    if not part.is_empty and part.area > 0:
                        parts.append(part)
            if parts:
                groups.append(_OsmGroup(polygons=parts, tags=elem.get("tags", {}) or {},
                                        element_type=elem.get("type", ""),
                                        element_id=elem.get("id")))

        return groups

    @staticmethod
    def _bbox_of(geom, pad_deg: float = 0.05):
        """从一个几何取出 Overpass `[bbox:南,西,北,东]` 用的矩形。

        留一点外扩：`[bbox]` 只选**与框相交**的元素，外扩是防高德区划多边形
        与真实边界之间有几十米级误差时，正好压在边上的目标被漏掉。
        """
        minx, miny, maxx, maxy = geom.bounds
        return (miny - pad_deg, minx - pad_deg, maxy + pad_deg, maxx + pad_deg)

    def _build_query(self, names: Sequence[str], bbox=None) -> str:
        """
        构建 Overpass QL 查询，搜索地名（及其别名变体）对应的多边形。

        只需按名字匹配：原先那一长串 `way[...]["amenity"]`、`relation[...]
        ["leisure"]` 之类的子句，全都是 `nwr["name"=X]` 的子集，反而漏掉了
        relation 型地物——大学校区、大型湖泊这类地物在 OSM 里是纯 multipolygon
        关系，不带 amenity/leisure 标签（深圳大学就因此查不到）。

        匹配的标签键覆盖 OSM 里表达别名的几种写法（相当于 WOF 的 alternate_names）：
          name        主名称（当地语言）
          name:zh     中文名        name:en   英文名
          int_name    拉丁转写      alt_name  别名（可能用 ; 分隔多个）
          official_name 官方名      old_name  历史名
        对每个变体逐个生成等值子句，而不是用正则：
        `nwr["name"~"^(甲|乙)$"]` 看着简洁，但名字里的括号、点号要按 Overpass 的
        正则方言转义，转错就静默查不到；等值子句只需处理引号，行为可预期。

        Args:
            names: 名字及其变体（同义词），由 place_lookup.expand_name_variants 生成。
            bbox: `(南, 西, 北, 东)`，给定时把查询限制在该矩形内。用来在已知
                预期区域时少取同名地物（"西湖"在 OSM 里 61 个）。
                **它只过滤"选哪些元素"，不裁剪返回的几何**——实测把一个只有深圳
                大小的框套在"广东省"这个 relation 上，返回的字节数与无框时
                完全相同（940268B），bounds 仍是全省 (20.12,109.39,25.52,117.53)。
                所以它治不了"大 relation 载荷太大"，只能治同名候选太多。

        取到的是 node/way/relation 的混合结果，node 没有面几何，由解析环节跳过。
        """
        clauses = []
        for tag in _QUERY_NAME_TAGS:
            for name in names:
                # 转义 Overpass QL 中的特殊字符（引号与反斜杠）
                escaped = name.replace("\\", "\\\\").replace('"', '\\"')
                clauses.append(f'  nwr["{tag}"="{escaped}"];')

        header = "[out:json][timeout:25]"
        if bbox is not None:
            south, west, north, east = bbox
            header += f"[bbox:{south:.6f},{west:.6f},{north:.6f},{east:.6f}]"

        return header + ";\n(\n" + "\n".join(clauses) + "\n);\nout geom;\n"

    # -------------------------------------------------------------------------
    # 用地类型查询（对应 models.LandUseArea）
    # -------------------------------------------------------------------------
    def _build_landuse_query(self, landuse: str, bbox: tuple) -> str:
        """构建"某一类用地的多边形"查询：`landuse=<类别>` 的 way / relation。

        与 `_build_query` 的分工是**名字 vs 类别**：那个查 `nwr["name"=X]`（一个
        地物、通常一块边界），这个查 `nwr["landuse"=X]`（成百上千个地块）。只取
        way/relation——用地类型是面状标注，不会挂在裸 node 上，带上 node 只会多
        返回一批没有面几何的结果。

        **bbox 必填**，与按名字查不同：`landuse=residential` 在全地球有数百万个，
        没有范围约束的查询必然被服务端砍掉（多余的元素只会拖慢、不改变结果，因为
        调用方随后还要与 scope 精确求交）。名字本身够窄，所以那边允许无界。

        取值来自 models.LandUseCategory 的 Literal，不含引号与反斜杠，无需转义；
        这里仍按"只有受控取值才不转义"的前提注释清楚，避免以后有人把用户输入接进来。
        """
        south, west, north, east = bbox
        header = (f"[out:json][timeout:25]"
                  f"[bbox:{south:.6f},{west:.6f},{north:.6f},{east:.6f}]")
        clauses = [f'  way["landuse"="{landuse}"];',
                   f'  relation["landuse"="{landuse}"];']
        return header + ";\n(\n" + "\n".join(clauses) + "\n);\nout geom;\n"

    def search_landuse(self, landuse: str, scope, label: str = ""):
        """查 scope 范围内某一类用地的真实多边形，返回与 scope 求交后的并集。

        Args:
            landuse: OSM `landuse` 取值（已由 models.LandUseCategory 约束）。
            scope: Shapely **面**几何，圈定"哪一带"。
            label: 回显用的中文说法（如"居民区"）；空则用 landuse 原值。

        Returns:
            与 scope 求交后的 Shapely 面几何（可能是 Polygon 或 MultiPolygon）。

        Raises:
            GeocodeError: scope 没有面积；或 OSM 全部端点不可达
                （service_unavailable=True）；或该范围内没有标注这类用地
                （数据源答复了，确实没有——与"服务不可用"是两件不同的事）。
        """
        name = label or landuse
        if getattr(scope, "area", 0.0) <= 0.0:
            raise GeocodeError(
                f"「{name}」需要一个有面积的查询范围，但只拿到了一个点。",
                detail=f"search_landuse 的 scope 是 "
                       f"{getattr(scope, 'geom_type', type(scope).__name__)}，面积为 0",
            )

        query = self._build_landuse_query(landuse, self._bbox_of(scope))
        data = self._overpass_query(query)
        if data is None:
            raise GeocodeError(
                f"查询「{name}」时 OpenStreetMap 服务不可用，请稍后重试。",
                detail="Overpass 三个端点两轮均未返回可解析结果（search_landuse）。",
                service_unavailable=True,
            )

        elements = data.get("elements") or []
        groups = self._element_polygon_groups(elements)
        parts = [poly for group in groups for poly in group.polygons]
        if not parts:
            raise GeocodeError(
                f"在指定范围内没有查到标注为「{name}」的用地，无法给出它的边界。",
                detail=(f"Overpass 对本次查询返回 {len(elements)} 个要素，"
                        f"其中没有可用的 landuse={landuse} 多边形。"
                        f"OSM 的用地类型是众包标注，缺标注的片区查不到属正常。"),
            )

        merged = unary_union(parts)
        clipped = unary_union(_extract_polygons(merged.intersection(scope)))
        if clipped.is_empty or getattr(clipped, "area", 0.0) <= 0.0:
            raise GeocodeError(
                f"「{name}」的多边形都落在指定范围之外，无法给出边界。",
                detail=f"landuse={landuse} 命中 {len(parts)} 个地块，与 scope 求交后为空。",
            )

        pieces = len(clipped.geoms) if isinstance(clipped, MultiPolygon) else 1
        print(f"[OSM] landuse={landuse}：命中 {len(parts)} 个地块，"
              f"与范围求交后 {pieces} 块")
        return clipped

    def _get_reference_point(self, name: str,
                             in_region: str = None, in_country: str = None):
        """用高德地理编码取一个参考点，用于甄别同名地物；拿不到返回 None。"""
        if not self.amap:
            return None
        hint = self.amap.geocode_place(name, in_region, in_country)
        if not hint:
            return None
        return Point(hint["lng"], hint["lat"])

    def _within_hint(self, geom, reference) -> bool:
        """弱校验：单个候选多边形是否落在参考点附近。

        同名地物可能遍布各地（"西湖""太湖""中山公园"），仅按 name 精确匹配
        无法避免取错。容差取"多边形自身尺度"与 30km 的较大者：小地物统一
        放 30km，大区域（省、大湖）按自身尺度放宽，避免被误拒。
        """
        if reference is None:
            return True
        if geom.contains(reference):
            return True
        minx, miny, maxx, maxy = geom.bounds
        span = max(maxx - minx, maxy - miny)
        tolerance = max(span, self._VALIDATE_MIN_TOLERANCE_DEG)
        return geom.distance(reference) <= tolerance

    def _get_region_geometry(self, in_region: str = None, in_country: str = None):
        """取预期区域（省/国家）的多边形，用于把候选限制在该区域内。

        比参考点更可靠：参考点本身可能就取错了（高德把"西湖"解析成台湾苗栗
        县的西湖乡），而行政区的多边形是明确的。
        """
        if not self.amap:
            return None
        for region_name in (in_region, in_country):
            if not region_name:
                continue
            poly = self.amap.get_district_polygon(region_name)
            if not poly:
                continue
            try:
                geom = shapely_shape(poly)
            except Exception:
                continue
            if not geom.is_empty:
                return geom
        return None

    def _apply_hierarchy_gate(self, groups: list["_OsmGroup"], request: PlaceSearchRequest,
                              region_geom, reference) -> list["_OsmGroup"]:
        """用层级信息（预期区域 / 高德参考点）把明显不对的同名地物挡在外面。

        这里只做"硬约束"，不负责排序：
          1. 能取到预期省/国家的行政区多边形 → 组内任一部件与之相交才保留。
             有明确区域约束却一个都不落在里面时全部丢弃——拿别的同名地物顶替
             比直接失败更糟。
          2. 退一步用高德对该地名的参考点 → 组内部件离参考点足够近才保留。
             全被否掉时不硬拒：参考点本身可能就指向别的同名地物，交给评分去定。
          3. 两条线索都拿不到（例如无高德 Key）→ 不做筛选，全部交给评分。

        region_geom / reference 由调用方解析一次后传入，避免为了打分再查一遍高德。
        """
        if not groups:
            return groups

        name = request.name
        if region_geom is not None:
            kept = [g for g in groups if any(p.intersects(region_geom) for p in g.polygons)]
            if not kept:
                print(f"[丢弃] OSM 返回的 '{name}' 候选均不在 "
                      f"{request.in_region or request.in_country} 内")
            elif len(kept) < len(groups):
                print(f"[筛选] '{name}' 剔除 {len(groups) - len(kept)} 个不落在预期区域内的同名地物")
            return kept

        if reference is None:
            return groups
        kept = [g for g in groups if any(self._within_hint(p, reference) for p in g.polygons)]
        if not kept:
            print(f"[回退] '{name}' 的 OSM 候选均偏离高德参考点，改由评分排序决定")
            return groups
        if len(kept) < len(groups):
            print(f"[筛选] '{name}' 剔除 {len(groups) - len(kept)} 个偏离参考点的同名地物")
        return kept

    def _score_group(self, group: "_OsmGroup", request: PlaceSearchRequest,
                     region_geom, reference) -> PlaceCandidate:
        """给一个候选打分，并记下每一维度的依据（供打印与排查）。"""
        geometry = group.geometry
        score = SOURCE_PRIORITY["osm"]
        reasons = []

        # 维度 1：名称匹配（主名称 = 查询语言名称 > 别名 > 部分包含）
        # 查询名与地物名都可能语言不同（"Shenzhen University" vs 深圳大学），
        # 所以对"原名 + 别名变体"逐个比，取最好的一次，并注明是靠哪个变体认出来的。
        tag_aliases = group.tag_values(*_ALIAS_NAME_TAGS)
        name_score, name_reason = 0.0, "名称不匹配"
        matched_form = request.name
        for form in request.variant_names():
            # 该变体语言下的名称标签：查 "Spain" 要认 name:en，查"广东省"要认 name:zh
            same_lang_tags = _LATIN_NAME_TAGS if has_latin(form) else _CJK_NAME_TAGS
            form_score, form_reason = name_match_score(
                form, group.tags.get("name"), tag_aliases,
                group.tag_values(*same_lang_tags),
            )
            if form_score > name_score:
                name_score, name_reason, matched_form = form_score, form_reason, form
        score += name_score
        if name_score > 0 and matched_form != request.name:
            name_reason += f"（经别名扩展：{matched_form}）"
        reasons.append(name_reason)

        # 维度 2：层级一致性
        # hierarchy_confirmed 是同一判断的机器可读版本（True/False/None），
        # 供 S25 打分使用，不用去解析上面的 reasons 字符串。
        if region_geom is not None:
            hierarchy_confirmed = any(p.intersects(region_geom) for p in group.polygons)
            if hierarchy_confirmed:
                score += HIERARCHY_IN_REGION
                reasons.append("落在预期区域内")
            else:
                reasons.append("不在预期区域内")
        elif reference is not None:
            hierarchy_confirmed = any(self._within_hint(p, reference)
                                      for p in group.polygons)
            if hierarchy_confirmed:
                score += HIERARCHY_NEAR_REFERENCE
                reasons.append("靠近参考点")
            else:
                reasons.append("远离参考点")
        else:
            # 既没有区域边界也没有参考点：本次查找根本没做层级核验。
            # 这与"核验了但不一致"是两件事，所以是 None 而不是 False。
            hierarchy_confirmed = None

        # 维度 3：地点类型（请求未带 place_type 时恒为 0）
        if request.place_type:
            for key, allowed in _TYPE_TAG_HINTS.get(request.place_type, ()):
                if group.tags.get(key) in allowed:
                    score += TYPE_MATCH_BONUS
                    reasons.append(f"类型相符({request.place_type})")
                    break

        # 维度 4：规模（权重很小，只在上面几项同分时起作用）
        score += area_score(geometry)

        return PlaceCandidate(
            geometry=geometry,
            name=group.tags.get("name") or request.name,
            source="osm",
            score=score,
            reasons=tuple(reasons),
            hierarchy_confirmed=hierarchy_confirmed,
        )

    def search_for_places(self, request: PlaceSearchRequest,
                          limit: int = 5) -> list[PlaceCandidate]:
        """
        搜索地名的多边形候选，按相关度降序返回（实现 PlaceLookup.search_for_places）。

        数据源优先级：
          1. 高德 District API → 中国行政区划的官方边界，命中了就是最优候选
          2. OSM Overpass API → 全球多边形数据，先过层级硬约束，再按评分排序

        Returns:
            PlaceCandidate 列表（降序，最多 limit 个）；找不到返回空列表。
        """
        name = request.name
        candidates: list[PlaceCandidate] = []

        # ── 数据源 1：高德行政区域多边形（source_type="osm" 时跳过）──
        if self.amap and request.source_type != "osm":
            district_poly = self.amap.get_district_polygon(name, request.in_region,
                                                           request.in_country)
            if district_poly:
                try:
                    geom = shapely_shape(district_poly)
                    if geom.geom_type in ("Polygon", "MultiPolygon") and not geom.is_empty:
                        candidates.append(PlaceCandidate(
                            geometry=geom,
                            name=district_poly.get("name") or name,
                            source="amap",
                            score=SOURCE_PRIORITY["amap"] + NAME_MATCH_EXACT + area_score(geom),
                            reasons=(f"高德区划命中 {district_poly.get('level', '')}".strip(),),
                        ))
                except Exception:
                    pass

        # ── 数据源 2：OSM Overpass API（source_type="amap" 时跳过）──
        osm_unreachable = False
        if request.source_type != "amap":
            # 层级线索先解析：有预期区域时可以用它的 bbox 把查询限制在区域内，
            # 避免为评分再查一遍高德（下面打分要用同一个 region_geom）。
            region_geom = self._get_region_geometry(request.in_region, request.in_country)
            bbox = self._bbox_of(region_geom) if region_geom is not None else None
            # 一次查询带上全部别名变体：中英两个方向的写法都能命中同一地物，
            # 从而保证"深圳大学"与 "Shenzhen University" 拿到同一几何。
            data = self._overpass_query(self._build_query(request.variant_names(), bbox))

            if data is not None and bbox is not None and not data.get("elements"):
                # 区域线索本身可能就是错的（高德把地名解析到别的同名地物）。
                # 限制后一无所获时退回无界查询：宁可多查一次，不可漏掉目标。
                where = request.in_region or request.in_country
                print(f"[回退] 限制在 {where} 范围内的 OSM 查询无结果，改用无界查询重试")
                data = self._overpass_query(self._build_query(request.variant_names()))

            if data is None:
                # 所有镜像都没响应。这与"镜像答了但没有这个地物"是两回事，
                # 分别对应"服务不可用"和"查无此地"，不能混成同一句提示。
                osm_unreachable = True
            elif data.get("elements"):
                groups = self._element_polygon_groups(data["elements"])
                reference = None if region_geom is not None else self._get_reference_point(
                    name, request.in_region, request.in_country)
                groups = self._apply_hierarchy_gate(groups, request, region_geom, reference)
                for group in groups:
                    candidate = self._score_group(group, request, region_geom, reference)
                    if candidate.geometry is not None and not candidate.geometry.is_empty:
                        candidates.append(candidate)

        if not candidates and osm_unreachable:
            raise GeocodeError(
                "OSM 地名服务当前不可用（多个镜像均超时或被拒绝），请稍后重试。",
                detail="Overpass：全部端点两轮均未返回可解析结果。",
                service_unavailable=True,
            )

        return rank_candidates(candidates, limit, name)


# =============================================================================
# 离线自检：python osm_place_lookup.py
#
# 全部用假会话 / 假高德，不发一个网络请求。覆盖三件容易改坏的事：
#   - 分级（谁配长超时、谁进补轮、失败态会不会自己恢复）
#   - 竞速（第一枪命中就该只发一个请求；错峰后等待由最快的端点决定）
#   - 缓存（同查询零网络；空结果短 TTL；截断结果不入缓存）
# =============================================================================
class _FakeResponse:
    """假响应。"""

    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class _FakeSession:
    """按 URL 分派的假 Overpass 会话。

    behaviour: `{url: [行为, ...]}`，行为按调用顺序取用，用完后重复最后一个。
    未配置的 URL 一律当作"连得上但不回话"（读超时），也就是 kumi/mail.ru 的形态。
    行为：
      ("ok", elements)       200 + 有要素            ("empty",)          200 + 空
      ("partial", elements)  200 + remark 截断       ("http", code)      非 200
      ("timeout",)           读超时                  ("error",)          连接错误
      ("hang", seconds)      睡够时间再读超时（真实耗时，用于验证"不白等"）
    """

    def __init__(self, behaviour: dict | None = None):
        self.behaviour = behaviour or {}
        self.calls: list[tuple[str, object]] = []
        self.queries: list[str] = []
        self._used: dict[str, int] = {}

    def _next(self, url: str):
        seq = self.behaviour.get(url)
        if not seq:
            return ("timeout",)
        i = self._used.get(url, 0)
        self._used[url] = i + 1
        return seq[min(i, len(seq) - 1)]

    def post(self, url: str, data=None, timeout=None, **kwargs):
        self.calls.append((url, timeout))
        self.queries.append((data or {}).get("data", ""))
        act = self._next(url)
        kind = act[0]
        if kind == "ok":
            return _FakeResponse(200, {"elements": act[1]})
        if kind == "empty":
            return _FakeResponse(200, {"elements": []})
        if kind == "partial":
            return _FakeResponse(200, {"elements": act[1],
                                       "remark": "runtime error: Query timed out"})
        if kind == "http":
            return _FakeResponse(act[1], {})
        if kind == "timeout":
            raise requests.exceptions.ReadTimeout("read timed out")
        if kind == "hang":
            time.sleep(act[1])
            raise requests.exceptions.ReadTimeout("read timed out")
        if kind == "error":
            raise requests.exceptions.ConnectionError("connect failed")
        raise AssertionError(f"未知行为 {kind!r}")


class _SelfCheck:
    """极简断言器：打印每一条的结果，最后汇报总数。"""

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, label: str, ok: bool) -> bool:
        if ok:
            self.passed += 1
            print(f"  ok   {label}")
        else:
            self.failed += 1
            print(f"  FAIL {label}")
        return bool(ok)

    def eq(self, label: str, got, want) -> bool:
        return self.check(f"{label}（得到 {got!r}，期望 {want!r}）", got == want)


def _self_check() -> int:
    import itertools
    import shutil
    import tempfile

    ck = _SelfCheck()
    tmp = tempfile.mkdtemp(prefix="osm_selfcheck_")
    counter = itertools.count()
    eps = OsmPolygonLookup.OVERPASS_ENDPOINTS

    def fresh(session=None) -> OsmPolygonLookup:
        return OsmPolygonLookup(cache_dir=os.path.join(tmp, f"c{next(counter)}"),
                                session=session)

    def age(lk: OsmPolygonLookup, ep: str, hours: float) -> None:
        """把内存里某端点的记录时间往前推（模拟时间流逝，不动磁盘）。"""
        lk.health._data[ep].updated -= hours * 3600

    ELEMENTS_A = [{"type": "way", "id": 1, "geometry": []}]
    ELEMENTS_B = [{"type": "relation", "id": 2, "members": []}]

    try:
        # ── A. 端点分级与衰减 ────────────────────────────────────────────
        print("[分级]")
        lk = fresh()
        ck.eq("未知端点 → unknown", lk.health.tier(eps[0]), "unknown")
        ck.eq("unknown 不配长超时", lk._read_timeout_for(eps[0]), _READ_TIMEOUT_COLD)

        lk.health.record_success(eps[0], 1.5)
        ck.eq("成功一次 → good", lk.health.tier(eps[0]), "good")
        ck.eq("good 拿长超时", lk._read_timeout_for(eps[0]), _READ_TIMEOUT_KNOWN)
        ck.eq("good 记下耗时", lk.health._data[eps[0]].last_latency, 1.5)

        lk.health.record_failure(eps[1])
        ck.eq("失败一次 → bad", lk.health.tier(eps[1]), "bad")
        ck.eq("bad 只用短超时", lk._read_timeout_for(eps[1]), _READ_TIMEOUT_COLD)
        ck.eq("bad 不进补轮", lk.health.not_recently_failed(eps[1]), False)
        ck.eq("unknown 进补轮", lk.health.not_recently_failed(eps[2]), True)

        # 衰减方向：失败必须会恢复，否则就是永久黑名单
        age(lk, eps[1], 9)
        ck.eq("失败 9 小时后回到 unknown", lk.health.tier(eps[1]), "unknown")
        age(lk, eps[1], -9)          # 还原
        # 衰减方向：成功态也不能永久占着 good
        age(lk, eps[0], 3 * 24)
        ck.eq("成功 3 天后掉出 good", lk.health.tier(eps[0]), "unknown")
        ck.check("成功态不会掉成 bad", lk.health.tier(eps[0]) != "bad")
        age(lk, eps[0], -3 * 24)
        ck.eq("还原后仍是 good", lk.health.tier(eps[0]), "good")

        ck.check("长超时高于服务端 timeout:25", _READ_TIMEOUT_KNOWN > 25.0)
        ck.check("短超时低于服务端 timeout:25", _READ_TIMEOUT_COLD < 25.0)

        # 排序：good 在最前、bad 在最后，unknown 居中
        ck.eq("排序把 best 放第一", lk._ordered_endpoints()[0], eps[0])
        ck.eq("排序把 bad 放最后", lk._ordered_endpoints()[-1], eps[1])

        # 落盘 + 重载 + 坏文件兜底
        ck.check("健康表已落盘", os.path.exists(os.path.join(lk.cache_dir, _HEALTH_FILE)))
        reloaded = OsmPolygonLookup(cache_dir=lk.cache_dir)
        ck.eq("重载后 good 仍是 good", reloaded.health.tier(eps[0]), "good")
        broken_dir = os.path.join(tmp, "broken")
        os.makedirs(broken_dir, exist_ok=True)
        with open(os.path.join(broken_dir, _HEALTH_FILE), "w", encoding="utf-8") as fh:
            fh.write("{ 这不是 JSON")
        ck.eq("健康表损坏 → 当空表", len(_HealthStore(
            os.path.join(broken_dir, _HEALTH_FILE))._data), 0)

        # ── B. 竞速 ─────────────────────────────────────────────────────
        print("[竞速]")
        sess = _FakeSession({eps[0]: [("ok", ELEMENTS_A)]})
        out = fresh(sess)._race(eps, "q")
        ck.eq("第一枪命中 → 只发 1 个请求", len(sess.calls), 1)
        ck.eq("第一枪命中 → 胜者有要素", len(out.data["elements"]), 1)

        sess = _FakeSession({eps[0]: [("http", 504)], eps[1]: [("error",)],
                             eps[2]: [("ok", ELEMENTS_B)]})
        out = fresh(sess)._race(eps, "q")
        ck.eq("第一枪失败 → 其余端点全部发出", len(sess.calls), 3)
        ck.check("第一枪失败 → 后面的端点能胜出",
                 out.data is not None and out.data["elements"][0]["id"] == 2)

        sess = _FakeSession({eps[0]: [("empty",)], eps[1]: [("ok", ELEMENTS_A)]})
        out = fresh(sess)._race(eps, "q")
        # 后面端点一命中就立刻返回，第三个端点可能还没被调度到，所以断言"发了不止一个"
        ck.check("第一枪 200 空 → 不会就此收手（继续发后续端点）", len(sess.calls) >= 2)
        ck.check("200 空 与 有内容并存时取有内容", out.data is not None)
        ck.check("空结果被留作备选", out.empty is not None)

        sess = _FakeSession({ep: [("empty",)] for ep in eps})
        out = fresh(sess)._race(eps, "q")
        ck.check("全员 200 空 → best() 是空结果",
                 out.best() is not None and not out.best()["elements"])
        ck.check("全员 200 空 → 算「问到了服务器」", out.reached_server)

        sess = _FakeSession({ep: [("timeout",)] for ep in eps})
        out = fresh(sess)._race(eps, "q")
        ck.eq("全员无响应 → 没有结果", out.best(), None)
        ck.eq("全员无响应 → 没有 200", out.reached_server, False)

        sess = _FakeSession({eps[0]: [("partial", ELEMENTS_A)],
                             eps[1]: [("ok", ELEMENTS_B)]})
        out = fresh(sess)._race(eps, "q")
        ck.check("截断不封盘 → 完整结果仍能到达", out.data is not None)
        ck.check("截断结果留作备选", out.partial is not None)
        ck.check("有完整时 best() 取完整", out.best() is out.data)

        sess = _FakeSession({ep: [("partial", ELEMENTS_A)] for ep in eps})
        out = fresh(sess)._race(eps, "q")
        ck.check("只有截断时 best() 取截断", out.best() is out.partial)

        # 等待时间由最快的端点决定，而不是各端点之和
        sess = _FakeSession({eps[0]: [("hang", 3.0)], eps[1]: [("ok", ELEMENTS_A)]})
        started = time.monotonic()
        out = fresh(sess)._race(eps, "q")
        elapsed = time.monotonic() - started
        ck.check(f"慢端点不拖住结果（{elapsed:.1f}s < 2.5s）", elapsed < 2.5)
        ck.check("慢端点在前也能被后面的端点救回", out.data is not None)

        # ── C. 补轮 ─────────────────────────────────────────────────────
        print("[补轮]")
        # 先单看短超时档的读超时：它只淘汰、不判决，端点不能被记成 bad
        sess = _FakeSession({ep: [("timeout",)] for ep in eps})
        lk = fresh(sess)
        lk._race(eps, "q")
        ck.eq("短超时档的读超时不算失败（端点没被打成 bad）",
               lk.health.tier(eps[0]), "unknown")

        sess = _FakeSession({eps[0]: [("timeout",), ("ok", ELEMENTS_A)]})
        lk = fresh(sess)
        data = lk._overpass_query("q-escalate")
        ck.check("一个 200 都没有 → 对 unknown 端点补轮拿到结果", data is not None)
        ck.eq("补轮给的是长超时", sess.calls[-1][1], (_CONNECT_TIMEOUT, _READ_TIMEOUT_KNOWN))

        sess = _FakeSession({ep: [("empty",)] for ep in eps})
        lk = fresh(sess)
        lk._overpass_query("q-empty")
        ck.eq("有 200（空）→ 不补轮", len(sess.calls), 3)

        sess = _FakeSession({ep: [("timeout",)] for ep in eps})
        lk = fresh(sess)
        for ep in eps:
            lk.health.record_failure(ep)
        lk._overpass_query("q-allbad")
        # 全都 recent bad 时仍要补轮：这是"一轮集体抖动"（并发时唯一可用镜像被
        # 自己打到 429，两次失败就压到 bad），不补等于把抖动直接判成服务不可用。
        # 3 个端点先跑第一轮，都超时 → 全部再补一轮，共 6 次。
        ck.eq("全部 bad → 仍然补轮（免得把集体抖动判成失败）", len(sess.calls), 6)

        # 半好半坏：只要有一个非 bad 的端点，bad 的那两个整轮都不发
        # （第一轮不发、补轮也不发——补轮的挑法同样是"最近没失败过"）
        sess = _FakeSession({eps[0]: [("timeout",)], eps[1]: [("timeout",)],
                             eps[2]: [("timeout",)]})
        lk = fresh(sess)
        lk.health.record_success(eps[0], 1.0)      # good
        for ep in eps[1:]:
            lk.health.record_failure(ep)           # bad
        lk._overpass_query("q-mixed")
        ck.eq("有非 bad 端点时，第一轮只打它", [c[0] for c in sess.calls], [eps[0], eps[0]])
        ck.eq("bad 端点整轮一次都不发（没它们也还有得试）",
               sum(1 for c in sess.calls if c[0] in eps[1:]), 0)
        ck.eq("补轮补的还是那个好端点", len(sess.calls), 2)

        # ── D. 缓存 ─────────────────────────────────────────────────────
        print("[缓存]")
        sess = _FakeSession({eps[0]: [("ok", ELEMENTS_A)]})
        lk = fresh(sess)
        lk._overpass_query("q-cache")
        first = len(sess.calls)
        again = lk._overpass_query("q-cache")
        ck.eq("同一查询第二次零网络请求", len(sess.calls), first)
        ck.eq("缓存命中的要素数一致", len(again["elements"]), 1)
        ck.check("不同查询串 → 不同缓存文件",
                 lk._cache_path("甲") != lk._cache_path("乙"))

        def cache_age(query: str, hours: float) -> None:
            path = lk._cache_path(query)
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            raw["fetched_at"] -= hours * 3600
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(raw, fh)

        lk._cache_write("q-empty-ttl", {"elements": []})
        cache_age("q-empty-ttl", 5)
        ck.check("空结果 5 小时仍命中", lk._cache_read("q-empty-ttl") is not None)
        cache_age("q-empty-ttl", 2)      # 累计 7 小时
        ck.check("空结果 7 小时已过期", lk._cache_read("q-empty-ttl") is None)

        lk._cache_write("q-some-ttl", {"elements": ELEMENTS_A})
        cache_age("q-some-ttl", 6 * 24)
        ck.check("有内容 6 天仍命中", lk._cache_read("q-some-ttl") is not None)
        cache_age("q-some-ttl", 2 * 24)  # 累计 8 天
        ck.check("有内容 8 天已过期", lk._cache_read("q-some-ttl") is None)

        sess = _FakeSession({ep: [("partial", ELEMENTS_A)] for ep in eps})
        lk = fresh(sess)
        partial_data = lk._overpass_query("q-partial")
        ck.check("截断结果照常返回", partial_data is not None)
        ck.check("截断结果不入缓存", lk._cache_read("q-partial") is None)

        lk._cache_write("q-tampered", {"elements": ELEMENTS_A})
        with open(lk._cache_path("q-tampered"), "w", encoding="utf-8") as fh:
            fh.write("{坏掉的缓存")
        ck.check("缓存文件损坏 → 当未命中", lk._cache_read("q-tampered") is None)

        lk._cache_write("q-key", {"elements": ELEMENTS_A})
        with open(lk._cache_path("q-key"), "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        raw["query"] = "别的查询"
        with open(lk._cache_path("q-key"), "w", encoding="utf-8") as fh:
            json.dump(raw, fh)
        ck.check("缓存里的查询串对不上 → 当未命中", lk._cache_read("q-key") is None)

        # ── E. 失败口径 ─────────────────────────────────────────────────
        print("[失败口径]")
        sess = _FakeSession({eps[0]: [("http", 429)]})
        lk = fresh(sess)
        lk._race([eps[0]], "(q)")
        ck.eq("非 200 记为失败", lk.health.tier(eps[0]), "bad")

        sess = _FakeSession({eps[0]: [("error",)]})
        lk = fresh(sess)
        lk._race([eps[0]], "(q)")
        ck.eq("连接错误记为失败", lk.health.tier(eps[0]), "bad")

        sess = _FakeSession({eps[0]: [("timeout",)]})
        lk = fresh(sess)
        lk._race([eps[0]], "(q)", escalation=True)
        ck.eq("长超时档的读超时记为失败", lk.health.tier(eps[0]), "bad")

        lk.health.record_success(eps[0], 1.0)
        ck.eq("成功一次即可从 bad 爬回 good", lk.health.tier(eps[0]), "good")

        # ── F. 全端点不可达的语义 ───────────────────────────────────────
        print("[不可达语义]")
        sess = _FakeSession({ep: [("timeout",)] for ep in eps})
        lk = fresh(sess)
        try:
            lk.search_for_places(PlaceSearchRequest(name="某个不存在的地方"))
            ck.check("全端点无响应应抛 GeocodeError", False)
        except GeocodeError as exc:
            ck.check("全端点无响应 → service_unavailable=True", exc.service_unavailable)

        # ── G. 区域 bbox 接线 ───────────────────────────────────────────
        print("[区域 bbox]")
        hz = {"type": "Polygon", "level": "city", "name": "杭州市",
              "coordinates": [[[119.9, 29.9], [120.5, 29.9], [120.5, 30.5],
                               [119.9, 30.5], [119.9, 29.9]]]}

        class _FakeAmap:
            def get_district_polygon(self, name, in_region=None, in_country=None):
                return hz if name.startswith("浙江省") or name == "杭州市" else None

            def geocode_place(self, name, in_region=None, in_country=None):
                return None

        sess = _FakeSession({eps[0]: [("ok", ELEMENTS_A)]})
        lk = fresh(sess)
        lk.amap = _FakeAmap()
        lk.search_for_places(PlaceSearchRequest(name="西湖", in_region="浙江省杭州市"))
        ck.check("有预期区域 → 首个查询串带 bbox", "[bbox:" in sess.queries[0])
        ck.check("bbox 用的是区域几何的 bounds",
                 "[bbox:29.850000,119.850000,30.550000,120.550000]" in sess.queries[0])

        sess = _FakeSession({eps[0]: [("empty",), ("ok", ELEMENTS_A)]})
        lk = fresh(sess)
        lk.amap = _FakeAmap()
        lk.search_for_places(PlaceSearchRequest(name="西湖", in_region="浙江省杭州市"))
        # 限定那次会同时打三个端点（都返回空），回退那次只有命中的端点
        ck.eq("限定查询无结果 → 追加恰一次无界查询",
               sum(1 for q in sess.queries if "[bbox:" not in q), 1)
        ck.eq("限定查询确实发出去了",
               sum(1 for q in sess.queries if "[bbox:" in q), 3)
        ck.check("回退的那次不带 bbox", "[bbox:" not in sess.queries[-1])

        sess = _FakeSession({eps[0]: [("ok", ELEMENTS_A)]})
        lk = fresh(sess)
        lk.search_for_places(PlaceSearchRequest(name="深圳大学"))
        ck.check("没有预期区域 → 查询串不带 bbox（缓存 key 不变）",
                 "[bbox:" not in sess.queries[0])
        ck.check("无 bbox 时查询头与旧格式一致",
                 sess.queries[0].startswith("[out:json][timeout:25];"))

        padded = OsmPolygonLookup._bbox_of(
            Polygon([(119.9, 29.9), (120.5, 29.9), (120.5, 30.5), (119.9, 30.5)]))
        ck.check("_bbox_of 带外扩 pad",
                 all(abs(got - want) < 1e-9
                     for got, want in zip(padded, (29.85, 119.85, 30.55, 120.55))))

        # ── H. 全局并发上限与在飞查询合并（S27 的 agent 依赖这两条） ────
        print("[并发上限]")
        # 4 个线程同时打 _probe，数"同时在飞"的最大值。上限是给公共镜像留的，
        # 破了这条就等于没限制（限流史见 S04/S20）。
        in_flight = {"now": 0, "max": 0}
        counter_lock = threading.Lock()
        gate = threading.Event()

        class _BlockingSession:
            """post 里把"同时在飞"数一遍，然后卡住直到主线程放行。"""

            def post(self, url, data=None, timeout=None, **kwargs):
                with counter_lock:
                    in_flight["now"] += 1
                    in_flight["max"] = max(in_flight["max"], in_flight["now"])
                gate.wait(5.0)
                with counter_lock:
                    in_flight["now"] -= 1
                return _FakeResponse(200, {"elements": ELEMENTS_A})

        lk = fresh(_BlockingSession())
        box = {"data": None, "partial": None, "empty": None, "answered": set(),
               "failed": set(), "reported": 0}
        box_lock = threading.Lock()
        threads = [threading.Thread(
            target=lk._probe,
            args=(f"https://ep{i}.test/api/interpreter", "q", box, box_lock,
                  threading.Event(), _READ_TIMEOUT_COLD))
            for i in range(4)]
        for t in threads:
            t.start()
        # 等到"该卡的都卡住了"再放行：先睡到 in_flight 不再增长
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            with counter_lock:
                if in_flight["now"] >= 2:
                    break
            time.sleep(0.01)
        time.sleep(0.2)          # 给第 3、4 个线程机会去抢那个不该存在的槽位
        with counter_lock:
            peak = in_flight["max"]
        gate.set()
        for t in threads:
            t.join(5.0)
        ck.eq("同时在飞的 Overpass 请求不超过 2", peak, 2)
        ck.check("四个线程都跑完了（没有互相卡死）",
                 all(not t.is_alive() for t in threads))

        print("[在飞查询合并]")
        # 同一查询并发到达时应只发一次：S27 的多候选常共享基础地点，
        # 而唯一可用镜像正是不许并发的那个
        class _SlowOkSession:
            def __init__(self):
                self.calls = []
                self.lock = threading.Lock()

            def post(self, url, data=None, timeout=None, **kwargs):
                with self.lock:
                    self.calls.append((url, (data or {}).get("data", "")))
                time.sleep(0.3)
                return _FakeResponse(200, {"elements": ELEMENTS_A})

        ck_sess = _SlowOkSession()
        lk = fresh(ck_sess)
        shared = lk._build_query(["西湖"])
        got: list = [None, None]

        def run(which: int) -> None:
            got[which] = lk._overpass_query(shared)

        racers = [threading.Thread(target=run, args=(i,)) for i in (0, 1)]
        for t in racers:
            t.start()
        for t in racers:
            t.join(10.0)
        ck.eq("同一查询并发只发一次网络请求", len(ck_sess.calls), 1)
        ck.check("两个调用方都拿到了结果",
                 got[0] is not None and got[1] is not None)
        ck.check("两个调用方拿到的是同一份结果", got[0] is got[1])
        ck.check("合并后 _inflight 已清空", lk._inflight == {})

        # 不同查询不该被合并（合并表按查询串区分）
        lk2 = fresh(_SlowOkSession())
        q1 = lk2._build_query(["西湖"])
        q2 = lk2._build_query(["太湖"])
        ck.check("不同查询的 key 不同", q1 != q2)
        lk2._overpass_query(q1)
        lk2._overpass_query(q2)
        ck.eq("两次不同查询各自发出请求", len(lk2.session.calls), 2)

        print("[200 空的等待封顶]")
        # 空响应不封盘（其余端点仍有机会），但等待要封顶——见 _RACE_EMPTY_GRACE_S。
        # 这里把宽限期临时压到 0.3s，好让自检不必真的等 8 秒。
        grace_backup = globals()["_RACE_EMPTY_GRACE_S"]

        class _EmptyThenSlowSession:
            """首个端点立刻回 200 空；其余端点先睡 `sleep_s` 再回有内容。"""

            def __init__(self, sleep_s: float):
                self.sleep_s = sleep_s
                self.calls = []
                self.lock = threading.Lock()

            def post(self, url, data=None, timeout=None, **kwargs):
                with self.lock:
                    self.calls.append(url)
                    first = len(self.calls) == 1
                if first:
                    return _FakeResponse(200, {"elements": []})
                time.sleep(self.sleep_s)
                return _FakeResponse(200, {"elements": ELEMENTS_A})

        try:
            globals()["_RACE_EMPTY_GRACE_S"] = 0.3

            # 其余端点慢到远超宽限期 → 拿空收工，不去等它们
            sess = _EmptyThenSlowSession(sleep_s=4.0)
            lk3 = fresh(sess)
            t0 = time.monotonic()
            out = lk3._race(list(lk3.OVERPASS_ENDPOINTS), lk3._build_query(["西湖"]))
            elapsed = time.monotonic() - t0
            ck.check("其余端点很慢 → 按宽限期收工，不等到它们答",
                     elapsed < 3.0)
            ck.check("收工时拿到的仍是那个空响应", out.data is None and out.best() is not None)
            ck.eq("三个端点仍然都被发出去过（空不封盘的原意）", len(sess.calls), 3)

            # 其余端点在宽限期内答了有内容 → 仍旧取有内容（这是"空不封盘"的防线）
            sess = _EmptyThenSlowSession(sleep_s=0.1)
            lk4 = fresh(sess)
            out = lk4._race(list(lk4.OVERPASS_ENDPOINTS), lk4._build_query(["西湖"]))
            ck.check("宽限期内有端点答出内容 → 取内容而不是空",
                     out.data is not None and out.empty is not None)
        finally:
            globals()["_RACE_EMPTY_GRACE_S"] = grace_backup

        # ── K. 用地类型查询（LandUseArea）───────────────────────────────
        print("[用地查询]")
        # 查询串：way/relation 两类子句 + bbox + out geom。
        # 早先手写过一版把 relation 子句写成 `relation["landuse"]="residential"];`
        # （多一个 ']'），这个断言就是钉住那个回归的。
        q = lk._build_landuse_query("residential", (22.5, 113.9, 22.55, 113.95))
        ck.check("用地查询带 bbox",
                 "[bbox:22.500000,113.900000,22.550000,113.950000]" in q)
        ck.check("用地查询取 way 的 landuse", 'way["landuse"="residential"];' in q)
        ck.check("用地查询取 relation 的 landuse",
                 'relation["landuse"="residential"];' in q)
        ck.check("用地查询输出几何", q.rstrip().endswith("out geom;"))
        ck.check("relation 子句没有多出来的 ']'", 'landuse"]=' not in q)

        # scope：一个把居民区整个包住的面（Polygon 按 (lon, lat)）
        scope = Polygon([(113.90, 22.50), (113.95, 22.50),
                         (113.95, 22.55), (113.90, 22.55)])
        # 一块落在 scope 里的居民区 way（Overpass 的 geometry 是 lat/lon 字段）
        res_way = {
            "type": "way", "id": 101, "tags": {"landuse": "residential"},
            "geometry": [
                {"lat": 22.510, "lon": 113.910}, {"lat": 22.510, "lon": 113.930},
                {"lat": 22.530, "lon": 113.930}, {"lat": 22.530, "lon": 113.910},
                {"lat": 22.510, "lon": 113.910},
            ],
        }

        # 命中：返回该地块，且被裁剪在 scope 之内
        lk5 = fresh(_FakeSession({eps[0]: [("ok", [res_way])]}))
        geom = lk5.search_landuse("residential", scope, label="居民区")
        ck.check("命中用地 → 得到有面积的几何", geom.area > 0)
        ck.check("结果落在 scope 之内", geom.within(scope))

        # 点状 scope：没有面积，直接拒（而不是把整个地球当范围）
        try:
            lk5.search_landuse("residential", Point(113.92, 22.52))
            ck.check("点状 scope 被拒", False)
        except GeocodeError as e:
            ck.check("点状 scope 被拒并说明需要面积", "有面积" in e.user_message)

        # 范围内没有该用地：数据源答复了"没有"，不算服务故障（全部端点回空）
        lk6 = fresh(_FakeSession({ep: [("empty",)] for ep in eps}))
        try:
            lk6.search_landuse("residential", scope, label="居民区")
            ck.check("范围内无该用地被拒", False)
        except GeocodeError as e:
            ck.check("范围内无该用地：提示没有查到", "没有查到" in e.user_message)
            ck.check("无地块不算服务故障", e.service_unavailable is False)

        # 端点全不可达：service_unavailable=True（用 "error" 立即失败，不吃超时）
        lk7 = fresh(_FakeSession({ep: [("error",)] for ep in eps}))
        try:
            lk7.search_landuse("residential", scope)
            ck.check("端点全不可达被拒", False)
        except GeocodeError as e:
            ck.check("端点全不可达 → service_unavailable=True",
                     e.service_unavailable is True)

        print(f"\n{ck.passed} 项通过，{ck.failed} 项失败")
        return ck.failed
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(_self_check())
