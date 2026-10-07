"""
事件数据源：NASA EONET（Earth Observatory Natural Event Tracker）。

回答"某个时间窗内、某个范围内发生过某类事件"的查询（如"过去两三年内被火灾影响
的亚马逊雨林"）。本模块只负责**取回事件记录**，不负责聚合成区域（见
event_aggregate.py）、也不负责接进节点树（见 geocoding 的事件查找阶段）。

数据源选型（2026-09-20 实测，均从本机直连）：

| 数据源 | 结果 | 结论 |
|---|---|---|
| NASA EONET | `/events` 与 `/events/geojson` 均可用；存档回溯至 2016 年；13 个类别；| **采用** |
| NASA FIRMS 火点 | 可达，但 area API 时间窗上限 5 天 | 弃用："过去两三年"要拆 200+ 次请求 |
| EU GWIS / EFFIS | 前端页可达，WFS 地图服务超时 | 弃用 |
| GDELT / Wikipedia | 全部超时 | 弃用（与 S04 记录的 GitHub 不可达一致） |

用 `/events` 而不是 `/events/geojson`：后者把每个位置点拍平成一条 Feature，
事件与类别的归属关系只在 properties 里，多类别事件的类别会丢失；前者一条事件
带 `categories` 与 `geometry` 数组，信息完整。

数据源的局限（不是本模块的缺陷，写在这里避免以后被当成 bug）：
  - **只给点，不给区域**。要得到"受影响区域"必须由调用方聚合（event_aggregate.py）
  - **存档密度随时间递减且分源**：2016 年全球野火仅 219 条，2024 年 1000+（GDACS
    订阅较晚才并入）。GDACS 的野火存档最早到 2024-05-27，更早的只有
    IRWIN/InciWeb（几乎全在美国）。所以亚马逊流域的"过去两三年"实际只有约
    1.3 年证据，**2023 年的空白是数据源如此，不是查询写错了**
  - `earthquakes` 类别记录稀疏，权威源是 USGS 的 FDSNWS 接口（本轮不接，
    见 开发规划.md 第四节）
  - **服务端 limit 从最新一端截断**（2026-09-20 实测）：`limit=300` 查 2024 全年
    只返回 2024-11-14~12-31，年初的记录被静默丢掉。本模块用**自适应时间分片**
    解决（见 `_fetch_complete`）：触顶就把窗口对半拆开重查，直到每片都不触顶
  - **接口偶发降级**：曾观测到 7 个参数完全不同的请求返回字节相同的响应，且
    日期全在请求窗口之外。这种响应由 `_guard_window` 识别并抛错，而不是让它
    变成"该范围内没有事件"这个结论
  - 野火的 `magnitudeValue` 单位是公顷（`magnitudeUnit == "hectare"`，来自 GDACS
    的过火面积估计），其余类别是各自的强度量纲。**不要假定统一单位**，
    `magnitude` 只在同类事件之间可比

失败语义（沿用 S20 建立的那条判据）：
  EONET 不可达（超时 / 5xx / 响应无法解析）一律抛
  `GeocodeError(service_unavailable=True)`，**绝不返回空列表**——那会把"没问到
  数据源"静默地变成"该范围内没有事件"这个结论，而后者是一个会被采信的断言。
  唯一的例外是本地存有过期缓存：此时改用过期数据并**大声打印它的时间**，
  因为一份标明了时间的历史缓存，比一次彻底的失败更接近用户要的答案。
"""

import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

import requests

from errors import GeocodeError

# =============================================================================
# 数据源常量
# =============================================================================
EONET_EVENTS_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"

# EONET 的全部类别 id（2026-09-20 取自 /api/v3/categories，共 13 个）。
# 这张名单是"合法类别"的唯一来源：EventQuery 校验、提示词的类别表都从它取。
EONET_CATEGORIES = (
    "drought",
    "dustHaze",
    "earthquakes",
    "floods",
    "landslides",
    "manmade",
    "seaLakeIce",
    "severeStorms",
    "snow",
    "tempExtremes",
    "volcanoes",
    "waterColor",
    "wildfires",
)

# 类别 id → 中文名。用于回显与提示词（"命中 wildfires"不如"命中 野火"好读）。
CATEGORY_LABELS = {
    "drought": "干旱",
    "dustHaze": "沙尘/雾霾",
    "earthquakes": "地震",
    "floods": "洪水",
    "landslides": "滑坡/泥石流",
    "manmade": "人为事件",
    "seaLakeIce": "海冰/湖冰",
    "severeStorms": "强风暴",
    "snow": "雪灾",
    "tempExtremes": "极端气温",
    "volcanoes": "火山",
    "waterColor": "水色异常",
    "wildfires": "野火",
}

# 别名 → 类别 id。中文与英文两套写法都收，供提示词渲染与（未来的）文本匹配使用。
# 有意不收过于宽泛的词（如单个"火""风""雪"）："火"会同时命中"火灾"与"火山"，
# "风"会命中"风暴"与"台风"。多字词之间的包含关系无害（"森林火灾"含"火灾"，
# 两者同属 wildfires）。
EVENT_CATEGORY_ALIASES = {
    # wildfires
    "火灾": "wildfires", "火情": "wildfires", "林火": "wildfires",
    "森林火灾": "wildfires", "山火": "wildfires", "野火": "wildfires",
    "大火": "wildfires", "wildfire": "wildfires", "wildfires": "wildfires",
    "fire": "wildfires", "fires": "wildfires", "bushfire": "wildfires",
    # floods
    "洪水": "floods", "水灾": "floods", "洪涝": "floods", "内涝": "floods",
    "泛滥": "floods", "flood": "floods", "floods": "floods", "flooding": "floods",
    # severeStorms
    "风暴": "severeStorms", "台风": "severeStorms", "飓风": "severeStorms",
    "气旋": "severeStorms", "龙卷风": "severeStorms", "storm": "severeStorms",
    "storms": "severeStorms", "cyclone": "severeStorms", "hurricane": "severeStorms",
    "typhoon": "severeStorms", "tornado": "severeStorms",
    # volcanoes
    "火山": "volcanoes", "火山喷发": "volcanoes", "volcano": "volcanoes",
    "volcanoes": "volcanoes", "volcanic eruption": "volcanoes",
    # drought
    "干旱": "drought", "旱灾": "drought", "drought": "drought", "droughts": "drought",
    # dustHaze
    "沙尘": "dustHaze", "沙尘暴": "dustHaze", "扬沙": "dustHaze",
    "雾霾": "dustHaze", "烟霾": "dustHaze", "dust": "dustHaze", "haze": "dustHaze",
    "dustHaze": "dustHaze", "dust storm": "dustHaze",
    # landslides
    "滑坡": "landslides", "泥石流": "landslides", "塌方": "landslides",
    "山体滑坡": "landslides", "landslide": "landslides", "landslides": "landslides",
    "mudslide": "landslides",
    # earthquakes
    "地震": "earthquakes", "earthquake": "earthquakes", "earthquakes": "earthquakes",
    "quake": "earthquakes",
    # snow
    "雪灾": "snow", "暴雪": "snow", "降雪": "snow", "snow": "snow",
    "blizzard": "snow",
    # tempExtremes
    "极端高温": "tempExtremes", "高温": "tempExtremes", "寒潮": "tempExtremes",
    "极端低温": "tempExtremes", "heatwave": "tempExtremes",
    "heat wave": "tempExtremes", "cold wave": "tempExtremes",
    "temperature extremes": "tempExtremes",
    # seaLakeIce
    "海冰": "seaLakeIce", "湖冰": "seaLakeIce", "冰情": "seaLakeIce",
    "sea ice": "seaLakeIce", "lake ice": "seaLakeIce",
    # manmade
    "人为事件": "manmade", "人为灾害": "manmade", "溢油": "manmade",
    "漏油": "manmade", "manmade": "manmade", "oil spill": "manmade",
    # waterColor
    "水色异常": "waterColor", "藻华": "waterColor", "赤潮": "waterColor",
    "水华": "waterColor", "algae": "waterColor", "algal bloom": "waterColor",
    "waterColor": "waterColor",
}

# 单次请求的条数上限。实测全球野火 2023-01 至今超 10000 条（7MB），
# 设一个上限避免把整段历史拖下来。触顶不是警告了事——会触发时间分片重查
# （见 _fetch_complete），所以这个值同时也是分片的判据。
DEFAULT_SERVER_LIMIT = 5000
# 交给上层聚合的记录数上限。超出时按日期均匀降采样（见 _downsample）。
DEFAULT_MAX_EVENTS = 2000

# 时间分片：某片触顶就把该片对半拆开重查，直到每片都不触顶。
# 小于 MIN_CHUNK_DAYS 的窗口不再拆——再拆下去请求数会爆炸，而收益只是
# "多拿回几条"，此时改为明确警告。
MIN_CHUNK_DAYS = 7
# 分片数上限，防御性上限：正常查询（全球野火 3.7 年）分片数是个位数。
_MAX_CHUNKS = 64
# 判断"响应日期是否落在请求窗口内"时允许的天数余量。EONET 的窗口过滤按事件的
# 几何日期做，边界上的事件可能因时区/取整落在窗口外一天，给一点余量避免误判。
_WINDOW_TOLERANCE_DAYS = 7

REQUEST_TIMEOUT = 20
# 右端开口（含到今天）的时间窗仍在增长，缓存 24 小时后过期；封闭的历史区间的
# 内容不再变化，永久有效。
OPEN_WINDOW_CACHE_HOURS = 24

# bbox 缓存对齐的粒度（度）：约 100m。用于让相近但不相等的查询共用一份缓存，
# 对齐方向是**向外取整**，保证缓存覆盖的范围永远包含请求的范围。
_BBOX_SNAP_DEG = 0.01


def _cache_dir() -> str:
    """事件缓存的落盘目录（打包后与 exe 同级，开发时与模块同级）。"""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "_event_cache")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# =============================================================================
# 数据模型
# =============================================================================
@dataclass(frozen=True)
class EventRecord:
    """一条事件记录。

    一个 EONET 事件可能带多个位置（同一场火灾被多次上报的位置轨迹），
    全部收进 `track`，按日期升序。`lon` / `lat` / `date` 是其中**最新**的那个
    位置，供只需要一个代表点的调用方使用；`track` 保留全部位置是因为上层
    聚合要按位置撒点，而且**时间窗切分后需要判断"这个事件到底落在哪一段"**
    （见 `_guard_window`）——只看代表点会把跨窗口的长事件判错。

    事件计数以"事件"为单位（`len(records)`），不是以位置点为单位——否则一场
    被上报 20 次的火灾会被算成 20 起事件。
    """

    id: str
    title: str
    category: str            # EONET 类别 id（多类别事件取命中请求的那个）
    date: str                # 最新位置的 ISO 日期
    closed: str | None       # 事件关闭时间；None 表示仍在进行
    lon: float
    lat: float
    magnitude: float | None  # 单位随类别而变（野火是公顷），见模块开头说明
    track: tuple[tuple[str, float, float], ...] = ()  # (日期, lon, lat)，按日期升序

    def points(self) -> tuple[tuple[float, float], ...]:
        """全部追踪位置；没有轨迹时退化为代表点。"""
        if not self.track:
            return ((self.lon, self.lat),)
        return tuple((lon, lat) for _, lon, lat in self.track)

    def dates(self) -> tuple[str, ...]:
        """全部追踪位置的日期（"YYYY-MM-DD"）；没有轨迹时退化为代表点日期。"""
        if not self.track:
            return (self.date[:10],)
        return tuple(d[:10] for d, _, _ in self.track)


@dataclass(frozen=True)
class EventQuery:
    """一次事件查询：类别 + 时间区间 + 空间范围。

    时间区间用 ISO 日期字符串（"YYYY-MM-DD"）而非 date 对象：它的直接来源是
    LLM 输出的 JSON 字段，保持字符串可以让"提示词写下的形状"与"模型校验的形状"
    完全一致，避免在解析层多做一次格式转换（转换失败时报错位置也更明确）。

    `time_end=None` 表示"到现在"，即右端开口的区间。
    """

    categories: tuple[str, ...]
    time_start: str
    time_end: str | None
    bbox: tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    limit: int = DEFAULT_SERVER_LIMIT
    max_events: int = DEFAULT_MAX_EVENTS

    def __post_init__(self):
        """校验并归一化入参。

        这里的输入最终来自 LLM（S24 的节点字段），所以类别是**边界输入**：
        拼错一个类别 id 会让服务端静默返回 0 条，看起来就像"该区域没有事件"。
        宁可在这里报错，也不要把它变成一个假的空结果。
        """
        # 列表 → 元组（frozen dataclass 要绕过 __setattr__ 直接写）
        object.__setattr__(self, "categories", tuple(self.categories))
        object.__setattr__(self, "bbox", tuple(self.bbox))

        if not self.categories:
            raise ValueError("EventQuery 至少需要一个事件类别")
        unknown = [c for c in self.categories if c not in EONET_CATEGORIES]
        if unknown:
            raise ValueError(
                f"未知的事件类别 {unknown}，可选：{list(EONET_CATEGORIES)}"
            )

        start = _parse_date(self.time_start, "time_start")
        if self.time_end is not None:
            end = _parse_date(self.time_end, "time_end")
            if end < start:
                raise ValueError(
                    f"time_end ({self.time_end}) 早于 time_start ({self.time_start})"
                )

        if len(self.bbox) != 4:
            raise ValueError("bbox 必须是 (min_lon, min_lat, max_lon, max_lat)")
        min_lon, min_lat, max_lon, max_lat = self.bbox
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError(f"bbox 的 min 必须小于 max，收到 {self.bbox}")

    # -------------------------------------------------------------------------
    def is_closed(self) -> bool:
        """时间窗是否已经封口（右端早于今天）。

        注意用 `< today` 而不是 `<= today`：结束日期写"今天"的窗口还在增长，
        缓存它 24 小时以上会漏掉新事件。
        """
        if self.time_end is None:
            return False
        return _parse_date(self.time_end, "time_end") < _now_utc().date()

    def cache_key(self) -> str:
        """缓存键：类别 + 时间窗 + 向外对齐后的 bbox。

        bbox 向外取整是为了让"候选之间的细微差别"不至于每次重查（agent 里
        同一批候选的 bbox 常常差在小数点后第三位），而向外取整保证命中缓存时
        拿到的是**超集**，再由 _filter_bbox 精确裁回请求范围，结果不会偏。
        """
        payload = {
            "categories": sorted(self.categories),
            "start": self.time_start,
            "end": self.time_end,
            "bbox": [round(v, 6) for v in self._snapped_bbox()],
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _snapped_bbox(self) -> tuple[float, float, float, float]:
        """向外对齐到 _BBOX_SNAP_DEG 的整数倍。"""
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return (
            _floor_to(min_lon, _BBOX_SNAP_DEG),
            _floor_to(min_lat, _BBOX_SNAP_DEG),
            _ceil_to(max_lon, _BBOX_SNAP_DEG),
            _ceil_to(max_lat, _BBOX_SNAP_DEG),
        )

    def contains(self, lon: float, lat: float) -> bool:
        """点是否落在请求的（未对齐的）范围内。"""
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def _parse_date(value: str, field_name: str):
    """把 "YYYY-MM-DD" 解析成 date；不合法时给出带字段名的错误。"""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{field_name} 必须是 'YYYY-MM-DD' 格式的日期，收到 {value!r}"
        ) from e


def _floor_to(value: float, step: float) -> float:
    """向下取整到 step 的整数倍（负值同样向下，即向西/向南扩张）。"""
    return math.floor(value / step) * step


def _ceil_to(value: float, step: float) -> float:
    """向上取整到 step 的整数倍。"""
    return math.ceil(value / step) * step


def _shift_days(iso_date: str, days: int) -> str:
    """日期字符串加减天数。"""
    return (_parse_date(iso_date, "date") + timedelta(days=days)).isoformat()


def _span_days(start: str, end: str) -> int:
    """两个日期间隔的天数（end - start）。"""
    return (_parse_date(end, "end") - _parse_date(start, "start")).days


def _merge_records(a: EventRecord, b: EventRecord) -> EventRecord:
    """合并同 id 的两条记录（时间分片后同一个长事件会出现多次）。

    EONET 对 start/end 的过滤是按几何日期做的，但返回的事件未必只带窗口内的
    几何——同一个跨窗口长事件在两个分片里各带一部分位置。取轨迹的并集，
    代表点取其中日期最晚的位置，这样合并结果与"不分片地查一次"一致。
    """
    track = tuple(sorted(set(a.track) | set(b.track), key=lambda p: p[0]))
    if not track:
        return a
    date, lon, lat = track[-1][0], track[-1][1], track[-1][2]
    # magnitude 是位置的属性，取日期最晚的那条记录的值；都缺就留 None
    later, earlier = (a, b) if a.date >= b.date else (b, a)
    magnitude = later.magnitude if later.magnitude is not None else earlier.magnitude
    return EventRecord(
        id=a.id, title=a.title or b.title, category=a.category or b.category,
        date=date, closed=a.closed or b.closed, lon=lon, lat=lat,
        magnitude=magnitude, track=track,
    )


# =============================================================================
# 客户端
# =============================================================================
class EonetClient:
    """NASA EONET 客户端（带磁盘缓存）。"""

    def __init__(self, cache_dir: str | None = None, session=None):
        """
        Args:
            cache_dir: 缓存目录；None 表示用默认位置（模块同级或 exe 同级）。
            session: 可注入的 requests 会话（测试用）。为 None 时自建一个，
                复用连接以免每次请求都走一遍 TCP+TLS 握手。
        """
        self.cache_dir = cache_dir or _cache_dir()
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "natural-language-geocoding/1.0 (event lookup)",
        })

    # -------------------------------------------------------------------------
    def fetch(self, query: EventQuery, use_cache: bool = True) -> list[EventRecord]:
        """取回事件记录（按日期升序）。

        Args:
            query: 查询条件。
            use_cache: 是否读写磁盘缓存。Agent 里多个候选共享基础地点时，
                第二个候选起就靠这里免掉一次网络请求。

        Returns:
            EventRecord 列表；**空列表表示"查到了，这个范围内确实没有事件"**，
            不表示失败——失败一律抛 GeocodeError。

        Raises:
            GeocodeError: 数据源不可用（service_unavailable=True）或响应无法解析。
        """
        cache_path = os.path.join(self.cache_dir, f"{query.cache_key()}.json")
        cached, fetched_at = self._read_cache(cache_path)

        records = None
        if use_cache and cached is not None and self._is_fresh(fetched_at, query):
            print(f"[事件] 命中缓存（{len(cached)} 条，"
                  f"写入于 {fetched_at:%Y-%m-%d %H:%M} UTC）"
                  f"：{query.time_start}~{query.time_end or '今天'}")
            records = cached

        if records is None:
            from_network = True
            try:
                records = self._fetch_complete(query)
            except GeocodeError:
                # 过期缓存兜底：标明了时间的历史数据，好过彻底失败
                if not (use_cache and cached is not None):
                    raise
                from_network = False
                age = _now_utc() - fetched_at
                print(f"[事件] 数据源不可用，改用 {age.total_seconds() / 3600:.1f} 小时"
                      f"前的缓存（{len(cached)} 条）。数据可能不完整。")
                records = cached
            if use_cache and from_network:
                # 缓存里存的是**降采样前**的完整集合：max_events 是调用方的展示
                # 偏好，不是数据的属性。带着它写缓存会让"先按 2000 条查一次"把
                # 后续按 5000 条查的缓存也削成 2000 条。
                self._write_cache(cache_path, records, query)

        return _downsample(self._filter_bbox(records, query), query.max_events)

    # -------------------------------------------------------------------------
    def _fetch_complete(self, query: EventQuery) -> list[EventRecord]:
        """把整个时间窗取全：触顶的一段对半拆开重查，最后合并去重。

        为什么需要这一步：EONET 的 `limit` 是**从最新一端截断**的，返回条数
        触顶时被丢掉的是窗口**最早**的那一段。对"过去两三年内被火灾影响的
        亚马逊雨林"这种查询，被丢掉的正好是用户关心的那部分历史，而结果看起来
        完全正常（有数据、有日期、有地图），是最难发现的一类缺陷。

        做法：从一个请求开始；若某段返回的原始事件数触顶且该段跨度大于
        MIN_CHUNK_DAYS，就把这段对半拆成两段分别重查；重复到所有段都不触顶。
        分片之间不重叠（mid / mid+1），但**跨分片的长事件会出现多次**，
        所以按事件 id 去重并合并轨迹。
        """
        window_end = query.time_end or _now_utc().date().isoformat()
        pending: list[tuple[str, str]] = [(query.time_start, window_end)]
        merged: dict[str, EventRecord] = {}
        request_count = 0

        while pending:
            start, end = pending.pop(0)
            if request_count >= _MAX_CHUNKS:
                raise GeocodeError(
                    "事件查询的范围过大，请缩小时间范围或空间范围后重试。",
                    detail=f"时间分片数达到上限 {_MAX_CHUNKS}（窗口 "
                           f"{query.time_start}~{window_end}），仍有 {len(pending)} "
                           f"段未取完",
                    service_unavailable=False,
                )

            records, hit_limit = self._request(query, start, end)
            request_count += 1

            if hit_limit:
                if _span_days(start, end) > MIN_CHUNK_DAYS:
                    mid = _shift_days(start, _span_days(start, end) // 2)
                    # 插到队首并按时间顺序排列，让请求顺序与时间顺序一致，
                    # 日志读起来是从早到晚的
                    pending.insert(0, (_shift_days(mid, 1), end))
                    pending.insert(0, (start, mid))
                    continue
                print(f"[事件] 警告：{start}~{end} 触顶（已达最小分片 "
                      f"{MIN_CHUNK_DAYS} 天），这一段可能仍不完整。")

            self._guard_window(records, start, end)
            for record in records:
                existing = merged.get(record.id)
                merged[record.id] = (
                    record if existing is None else _merge_records(existing, record)
                )

        records = sorted(merged.values(), key=lambda r: r.date)
        span_note = f"，{request_count} 次请求合并" if request_count > 1 else ""
        print(f"[事件] {len(records)} 条事件（{'/'.join(query.categories)}，"
              f"{query.time_start}~{query.time_end or '今天'}{span_note}）")
        # 不在这里降采样：fetch() 要先把完整集合写进缓存，降采样是展示偏好
        return records

    # -------------------------------------------------------------------------
    def _guard_window(self, records: list[EventRecord],
                      start: str, end: str) -> None:
        """确认响应确实属于请求的时间窗，否则抛错。

        2026-09-20 实测到接口降级：参数不同的请求返回了字节相同的响应，
        日期全在请求窗口之外。若直接采用，就会得出"该范围内没有事件"的结论，
        而这是一个会被采信的断言。判据用**任意位置**的日期落在窗口内即可，
        不用事件代表点：跨窗口的长事件（野火可以烧几个月）的最晚位置可能
        已经在窗口外，只看代表点会把它误判成降级响应。
        """
        if not records:
            return
        low = _shift_days(start, -_WINDOW_TOLERANCE_DAYS)
        high = _shift_days(end, _WINDOW_TOLERANCE_DAYS)
        for record in records:
            for date in record.dates():
                if low <= date <= high:
                    return
        raise GeocodeError(
            "事件数据服务返回的数据与请求的时间范围不符，无法确认结果完整。",
            detail=f"请求 {start}~{end}，返回 {len(records)} 条事件但日期全在窗口外"
                   f"（{records[0].date[:10]} ~ {records[-1].date[:10]}）",
            service_unavailable=True,
        )

    # -------------------------------------------------------------------------
    def _request(self, query: EventQuery, start: str,
                 end: str) -> tuple[list[EventRecord], bool]:
        """发一次请求并解析。返回 (记录, 是否触顶)。

        `start` / `end` 由 `_fetch_complete` 给出（它是分片后的某一段），
        而不是直接取 `query` 的窗口。任何异常都转成 GeocodeError（不可用）。
        """
        snapped = query._snapped_bbox()
        params = {
            # status=all 是必须的：默认只返回"正在进行"的事件，历史火灾一条都取不到
            "status": "all",
            "category": ",".join(query.categories),
            "start": start,
            "end": end,
            "bbox": ",".join(str(v) for v in snapped),
            "limit": query.limit,
        }

        try:
            response = self.session.get(
                EONET_EVENTS_URL, params=params, timeout=REQUEST_TIMEOUT
            )
            if response.status_code != 200:
                raise GeocodeError(
                    "事件数据服务当前不可用，请稍后重试。",
                    detail=f"EONET 返回 HTTP {response.status_code}："
                           f"{response.text[:200]}",
                    service_unavailable=True,
                )
            payload = response.json()
        except GeocodeError:
            raise
        except Exception as e:
            raise GeocodeError(
                "事件数据服务当前不可用，请稍后重试。",
                detail=f"请求 EONET 失败：{type(e).__name__}: {e}\n参数：{params}",
                service_unavailable=True,
            ) from e

        raw_events = payload.get("events")
        # 触顶判据用**原始事件数**而不是解析后的记录数：解析会丢掉没有位置的
        # 事件，用解析后的数会漏判"服务端其实已经截断了"
        hit_limit = isinstance(raw_events, list) and len(raw_events) >= query.limit
        records = self._parse_events(payload, query)
        print(f"[事件] EONET 返回 {len(raw_events or [])} 条原始事件"
              f"（{start}~{end}）→ 解析出 {len(records)} 条")
        return records, hit_limit

    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_events(payload: dict, query: EventQuery) -> list[EventRecord]:
        """把 /events 的响应拍平成 EventRecord 列表（按日期升序）。

        丢东西的地方都要计数并打印：静默丢弃会让"这个区域没有火灾"这类结论
        建立在不完整的数据上，而且事后无从判断是数据源如此还是解析漏了。
        """
        events = payload.get("events")
        if not isinstance(events, list):
            raise GeocodeError(
                "事件数据服务的响应格式无法识别。",
                detail=f"EONET 响应里没有 events 数组，顶层键：{list(payload)[:10]}",
                service_unavailable=True,
            )

        records: list[EventRecord] = []
        skipped_no_geometry = 0
        skipped_bad_coords = 0
        polygon_events = 0

        for event in events:
            geometries = event.get("geometry") or []
            # (日期, lon, lat, magnitude)：magnitude 是位置的属性而非事件的，
            # 取最新位置那个值，所以先带着再排掉
            positions: list[tuple[str, float, float, float | None]] = []
            has_polygon = False

            for geom in geometries:
                geom_type = geom.get("type")
                coords = geom.get("coordinates")
                point = None
                if geom_type == "Point":
                    point = _as_point(coords)
                elif geom_type == "Polygon":
                    # EONET 里少数事件给的是面（多为人为事件/冰情）。这里取环上
                    # 顶点的平均位置当作代表点——够用于聚类，且不引入几何库依赖。
                    has_polygon = True
                    point = _polygon_vertex_mean(coords)
                if point is None:
                    continue
                positions.append((geom.get("date") or "", point[0], point[1],
                                  _as_float(geom.get("magnitudeValue"))))

            if not positions:
                if geometries:
                    skipped_bad_coords += 1
                else:
                    skipped_no_geometry += 1
                continue
            if has_polygon:
                polygon_events += 1

            positions.sort(key=lambda p: p[0])
            date, lon, lat, magnitude = positions[-1]
            records.append(EventRecord(
                id=str(event.get("id") or ""),
                title=event.get("title") or str(event.get("id") or ""),
                category=_pick_category(event, query),
                date=date,
                closed=event.get("closed"),
                lon=lon,
                lat=lat,
                magnitude=magnitude,
                track=tuple((d, x, y) for d, x, y, _ in positions),
            ))

        dropped = skipped_no_geometry + skipped_bad_coords
        if dropped:
            print(f"[事件] 跳过 {dropped} 条无可解析位置的事件"
                  f"（无 geometry {skipped_no_geometry} 条，"
                  f"坐标不合法 {skipped_bad_coords} 条）")
        if polygon_events:
            print(f"[事件] {polygon_events} 条事件只有面几何，已取环上顶点平均位置作代表点")

        records.sort(key=lambda r: r.date)
        return records

    # -------------------------------------------------------------------------
    def _is_fresh(self, fetched_at: datetime | None, query: EventQuery) -> bool:
        """缓存是否还在有效期内。

        封闭的历史区间永久有效（内容不会变）；右端开口的区间 24 小时后过期，
        因为新事件还在产生。
        """
        if fetched_at is None:
            return False
        if query.is_closed():
            return True
        return _now_utc() - fetched_at < timedelta(hours=OPEN_WINDOW_CACHE_HOURS)

    def _filter_bbox(self, records: list[EventRecord],
                     query: EventQuery) -> list[EventRecord]:
        """按请求的精确 bbox 裁剪（缓存/服务端用的是向外取整后的 bbox）。

        以事件代表点判断归属：点在范围内就保留整条事件（含它的全部轨迹点）。
        按点逐条裁剪会把同一场火灾劈成两条，也会让 `len(records)` 这个
        "事件数"失去意义。
        """
        return [r for r in records if query.contains(r.lon, r.lat)]

    def _read_cache(self, path: str):
        """读缓存，返回 (records, fetched_at)；不存在或损坏时返回 (None, None)。"""
        if not os.path.exists(path):
            return None, None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            fetched_at = datetime.fromisoformat(payload["fetched_at"])
            records = [
                EventRecord(
                    **{**item,
                       "track": tuple(tuple(p) for p in item.get("track", ()))}
                )
                for item in payload["records"]
            ]
            return records, fetched_at
        except Exception as e:
            # 缓存损坏不该让查询失败，重新取一次即可
            print(f"[事件] 缓存文件不可读，忽略：{type(e).__name__}: {e}")
            return None, None

    def _write_cache(self, path: str, records: list[EventRecord],
                     query: EventQuery) -> None:
        """写缓存。写失败只警告——缓存是优化，不是功能。"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({
                    "fetched_at": _now_utc().isoformat(),
                    "query": {
                        "categories": list(query.categories),
                        "time_start": query.time_start,
                        "time_end": query.time_end,
                        "bbox": list(query._snapped_bbox()),
                    },
                    "records": [asdict(r) for r in records],
                }, f, ensure_ascii=False)
            os.replace(tmp_path, path)
        except Exception as e:
            print(f"[事件] 缓存写入失败（不影响本次结果）：{type(e).__name__}: {e}")


# =============================================================================
# 辅助函数
# =============================================================================
def _as_point(coords) -> tuple[float, float] | None:
    """把 GeoJSON 坐标转成 (lon, lat)；形状不对返回 None。"""
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None
    lon, lat = coords[0], coords[1]
    if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
        return None
    return float(lon), float(lat)


def _polygon_vertex_mean(coords) -> tuple[float, float] | None:
    """面的环上顶点平均位置（不引入几何库的粗略代表点）。"""
    if not isinstance(coords, (list, tuple)) or not coords:
        return None
    ring = coords[0]
    points = [p for p in (_as_point(c) for c in ring) if p is not None]
    if not points:
        return None
    return (sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points))


def _pick_category(event: dict, query: EventQuery) -> str:
    """事件归属的类别：优先命中本次请求的类别集合。"""
    ids = [(c or {}).get("id") for c in (event.get("categories") or [])]
    for category_id in ids:
        if category_id in query.categories:
            return category_id
    return ids[0] if ids and ids[0] else ""


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _downsample(records: list[EventRecord], max_events: int) -> list[EventRecord]:
    """超过上限时按日期均匀降采样。

    均匀取样而不是"取最新 N 条"：截断一端会让"某个时间段完全没有事件"成为
    假象（S20 记的正是这类"静默改变结论"的缺陷）。首尾两条固定保留，
    这样时间窗的边界仍然是真实的事件日期。
    """
    if max_events <= 0 or len(records) <= max_events:
        return records

    step = len(records) / max_events
    picked = [records[min(int(i * step), len(records) - 1)] for i in range(max_events)]
    picked[-1] = records[-1]
    print(f"[事件] 记录数 {len(records)} 超过上限 {max_events}，"
          f"已按日期均匀降采样（保留首尾，覆盖 {records[0].date[:10]} ~ "
          f"{records[-1].date[:10]}）")
    return picked


# =============================================================================
# 离线自检（python event_data.py）
# =============================================================================
def _self_check() -> int:
    """不联网的自检：类别表、入参校验、解析、降采样、缓存有效期、失败语义。

    刻意不引入 pytest——本项目按当前决策暂缓单元测试框架（见 开发规划.md 第四节），
    所以自检跟 geocoding.py 的简易测试入口一样放在模块内，一条命令可复现。
    """
    failures: list[str] = []

    def check(name: str, condition: bool, extra: str = ""):
        if condition:
            print(f"  [通过] {name}")
        else:
            print(f"  [失败] {name} {extra}")
            failures.append(name)

    print("== 类别表 ==")
    check("别名表的值全是合法类别",
          set(EVENT_CATEGORY_ALIASES.values()) == set(EONET_CATEGORIES),
          f"多出 {set(EVENT_CATEGORY_ALIASES.values()) - set(EONET_CATEGORIES)}，"
          f"缺少 {set(EONET_CATEGORIES) - set(EVENT_CATEGORY_ALIASES.values())}")
    check("每个类别都有中文标签",
          set(CATEGORY_LABELS) == set(EONET_CATEGORIES))
    check("类别数为 13", len(EONET_CATEGORIES) == 13)

    print("== 入参校验 ==")
    def query_error(**kwargs):
        """构造 EventQuery，返回错误信息；不报错则返回 None。"""
        base = dict(categories=("wildfires",), time_start="2023-01-01",
                    time_end="2026-09-20", bbox=(-75.0, -20.0, -45.0, 5.0))
        base.update(kwargs)
        try:
            EventQuery(**base)
            return None
        except ValueError as e:
            return str(e)

    check("未知类别被拒", query_error(categories=("fire",)) is not None)
    check("空类别被拒", query_error(categories=()) is not None)
    check("日期格式错误被拒", query_error(time_start="2023/01/01") is not None)
    check("end 早于 start 被拒", query_error(time_end="2020-01-01") is not None)
    check("bbox 反向被拒", query_error(bbox=(-45.0, -20.0, -75.0, 5.0)) is not None)
    check("bbox 长度错误被拒", query_error(bbox=(-75.0, -20.0, -45.0)) is not None)
    check("合法入参通过", query_error() is None)

    print("== 时间窗与缓存有效期 ==")
    closed = EventQuery(categories=("wildfires",), time_start="2023-01-01",
                        time_end="2025-12-31", bbox=(-75.0, -20.0, -45.0, 5.0))
    open_window = EventQuery(categories=("wildfires",), time_start="2023-01-01",
                             time_end=None, bbox=(-75.0, -20.0, -45.0, 5.0))
    today_window = EventQuery(categories=("wildfires",), time_start="2023-01-01",
                              time_end=_now_utc().date().isoformat(),
                              bbox=(-75.0, -20.0, -45.0, 5.0))
    check("过去的区间算封闭", closed.is_closed())
    check("time_end=None 算开口", not open_window.is_closed())
    check("time_end=今天算开口（窗口还在增长）", not today_window.is_closed())

    client = EonetClient(cache_dir="__never_used__")
    long_ago = _now_utc() - timedelta(days=30)
    recent = _now_utc() - timedelta(hours=1)
    check("封闭区间 30 天前的缓存仍新鲜", client._is_fresh(long_ago, closed))
    check("开口区间 1 小时前的缓存新鲜", client._is_fresh(recent, open_window))
    check("开口区间 30 天前的缓存已过期", not client._is_fresh(long_ago, open_window))
    check("无缓存记录时不新鲜", not client._is_fresh(None, closed))

    print("== bbox 对齐与裁剪 ==")
    snapped = closed._snapped_bbox()
    check("对齐后包含原范围",
          snapped[0] <= -75.0 and snapped[1] <= -20.0
          and snapped[2] >= -45.0 and snapped[3] >= 5.0, str(snapped))
    check("范围内点被保留", closed.contains(-50.0, -12.0))
    check("范围外点被剔除（东侧越界）", not closed.contains(-44.0, -12.0))
    check("范围外点被剔除（南侧越界）", not closed.contains(-50.0, -21.0))
    check("对齐后的 key 对同一查询稳定",
          closed.cache_key() == EventQuery(
              categories=("wildfires",), time_start="2023-01-01",
              time_end="2025-12-31", bbox=(-75.0, -20.0, -45.0, 5.0)).cache_key())

    def key_of(categories):
        return EventQuery(categories=categories, time_start="2023-01-01",
                          time_end="2025-12-31",
                          bbox=(-75.0, -20.0, -45.0, 5.0)).cache_key()

    check("类别顺序不影响 key", key_of(("wildfires", "floods")) == key_of(("floods", "wildfires")))
    check("类别集合不同则 key 不同", key_of(("wildfires",)) != key_of(("wildfires", "floods")))
    check("时间窗不同则 key 不同",
          key_of(("wildfires",)) != EventQuery(
              categories=("wildfires",), time_start="2023-01-01",
              time_end="2024-12-31", bbox=(-75.0, -20.0, -45.0, 5.0)).cache_key())

    print("== 响应解析 ==")
    payload = {
        "events": [
            {   # 正常：3 个位置，日期递增，多类别里只有第二个命中请求
                "id": "E1", "title": "Wildfire in Brazil", "closed": None,
                "categories": [{"id": "manmade"}, {"id": "wildfires"}],
                "geometry": [
                    {"type": "Point", "date": "2026-09-10T00:00:00Z",
                     "coordinates": [-50.0, -10.0], "magnitudeValue": 100.0,
                     "magnitudeUnit": "hectare"},
                    {"type": "Point", "date": "2026-09-12T00:00:00Z",
                     "coordinates": [-50.5, -10.5], "magnitudeValue": 800.0,
                     "magnitudeUnit": "hectare"},
                    {"type": "Point", "date": "2026-09-15T00:00:00Z",
                     "coordinates": [-51.0, -11.0], "magnitudeValue": 1500.0,
                     "magnitudeUnit": "hectare"},
                ]},
            {   # 无 geometry：跳过并计数
                "id": "E2", "title": "empty", "categories": [{"id": "wildfires"}],
                "geometry": []},
            {   # 坐标非法：跳过并计数
                "id": "E3", "title": "bad coords",
                "categories": [{"id": "wildfires"}],
                "geometry": [{"type": "Point", "date": "2026-09-01T00:00:00Z",
                              "coordinates": [None, None]}]},
            {   # 面几何：取环上顶点平均
                "id": "E4", "title": "polygon event", "closed": "2026-09-02T00:00:00Z",
                "categories": [{"id": "wildfires"}],
                "geometry": [{"type": "Polygon", "date": "2026-09-02T00:00:00Z",
                              "coordinates": [[[10.0, 0.0], [12.0, 0.0],
                                               [12.0, 2.0], [10.0, 2.0]]]}]},
        ]
    }
    records = EonetClient._parse_events(payload, closed)
    check("解析出 2 条事件（丢弃 2 条）", len(records) == 2, f"实得 {len(records)}")
    check("按日期升序（面事件 09-02 在前）", records[0].date < records[1].date)
    polygon_rec, wildfire = records[0], records[1]
    check("取最新位置作代表点", (wildfire.lon, wildfire.lat) == (-51.0, -11.0),
          f"实得 {(wildfire.lon, wildfire.lat)}")
    check("保留全部轨迹位置", len(wildfire.points()) == 3,
          f"实得 {len(wildfire.points())}")
    check("轨迹带日期且按日期升序",
          wildfire.dates() == ("2026-09-10", "2026-09-12", "2026-09-15"),
          str(wildfire.dates()))
    check("轨迹坐标与日期对齐",
          wildfire.points() == ((-50.0, -10.0), (-50.5, -10.5), (-51.0, -11.0)),
          str(wildfire.points()))
    check("类别取命中请求的那个（多类别事件取 wildfires 而非 manmade）",
          wildfire.category == "wildfires", wildfire.category)
    check("magnitude 取最新值", wildfire.magnitude == 1500.0, str(wildfire.magnitude))
    check("closed 为 None 表示仍在进行", wildfire.closed is None)
    check("面几何取顶点均值",
          abs(polygon_rec.lon - 11.0) < 1e-9 and abs(polygon_rec.lat - 1.0) < 1e-9,
          f"实得 {(polygon_rec.lon, polygon_rec.lat)}")
    check("closed 字段保留", polygon_rec.closed == "2026-09-02T00:00:00Z")

    print("== 降采样 ==")
    many = [EventRecord(id=f"E{i}", title=f"t{i}", category="wildfires",
                        date=f"2026-01-{i + 1:02d}T00:00:00Z", closed=None,
                        lon=float(i), lat=0.0, magnitude=None)
            for i in range(20)]
    sampled = _downsample(many, 5)
    check("降采样到上限条数", len(sampled) == 5, f"实得 {len(sampled)}")
    check("保留首条", sampled[0].id == "E0", sampled[0].id)
    check("保留末条", sampled[-1].id == "E19", sampled[-1].id)
    check("未超限时原样返回", _downsample(many[:3], 5) == many[:3])

    print("== 时间分片（复现 limit 从最新一端截断）==")

    class _FakeResponse:
        def __init__(self, payload, status_code=200):
            self._payload = payload
            self.status_code = status_code
            self.text = json.dumps(payload)[:500]

        def json(self):
            return self._payload

    class _FakeEonetSession:
        """模拟 EONET 服务端，重点是复现两个真实行为：

        1. `limit` 从**最新**一端截断（实测：limit=300 查 2024 全年只回 12 月）
        2. 窗口过滤按几何日期做，跨窗口的长事件在多个分片里都会出现
        """

        headers: dict = {}

        def __init__(self, events, ignore_window=False):
            self.events = events
            # ignore_window=True 复现"接口降级"：无视窗口参数，照旧返回自己的数据
            self.ignore_window = ignore_window
            # (窗口起, 窗口止, limit, 实际返回条数)——最后一项用来识别"叶子分片"
            self.requests: list[tuple[str, str, int, int]] = []

        @staticmethod
        def _dates(event):
            return [g.get("date", "")[:10] for g in event.get("geometry") or []]

        def get(self, url, params=None, timeout=None):
            params = params or {}
            start, end = params["start"], params["end"]
            limit = int(params.get("limit", 10 ** 9))
            if self.ignore_window:
                matched = list(self.events)
            else:
                matched = [e for e in self.events
                           if any(start <= d <= end for d in self._dates(e))]
            matched.sort(key=lambda e: max(self._dates(e)), reverse=True)
            kept = matched[:limit]
            self.requests.append((start, end, limit, len(kept)))
            return _FakeResponse({"events": kept})

    def point_event(eid, dates, lon_base=-60.0):
        return {
            "id": eid, "title": eid, "closed": None,
            "categories": [{"id": "wildfires"}],
            "geometry": [
                {"type": "Point", "date": f"{d}T00:00:00Z",
                 "coordinates": [lon_base + i * 0.01, -10.0],
                 "magnitudeValue": float(i), "magnitudeUnit": "hectare"}
                for i, d in enumerate(dates)
            ],
        }

    # 200 个每天一条的事件，跨 2024-01-01 ~ 2024-07-18；limit 只有 40，
    # 不分片的话只能拿到最新 40 条（2024-06-09 之后）
    daily = [point_event(f"D{i:03d}",
                         [(_parse_date("2024-01-01", "x") + timedelta(days=i)).isoformat()])
             for i in range(200)]
    # 一个跨整个窗口的长事件：三个位置分别落在窗口的早/中/晚段，
    # 会在多个分片里都出现，用来验证按 id 去重 + 轨迹合并
    long_event = point_event("LONG", ["2024-01-05", "2024-03-15", "2024-07-10"])
    fake = _FakeEonetSession(daily + [long_event])

    chunked = EventQuery(categories=("wildfires",), time_start="2024-01-01",
                         time_end="2024-07-18", bbox=(-75.0, -20.0, -45.0, 5.0),
                         limit=40)
    chunked_client = EonetClient(cache_dir="__never_used__", session=fake)
    got = chunked_client.fetch(chunked, use_cache=False)

    check("分片后取回全部 201 条（200 条日事件 + 1 条长事件）",
          len(got) == 201, f"实得 {len(got)}")
    check("窗口最早的那条没有被丢掉",
          got[0].date[:10] == "2024-01-01", got[0].date)
    check("窗口最晚的那条也在",
          got[-1].date[:10] == "2024-07-18", got[-1].date)
    check("确实发生了分片（请求数 > 1）", len(fake.requests) > 1,
          f"实得 {len(fake.requests)} 次请求")
    check("每次请求都带 limit=40",
          all(r[2] == 40 for r in fake.requests), str(fake.requests[:3]))
    check("所有请求都落在查询窗口内",
          all(chunked.time_start <= s and e <= chunked.time_end
              for s, e, _, _ in fake.requests), str(fake.requests))
    # 未触顶的那些分片就是最终的"叶子"，它们必须恰好覆盖整个窗口
    leaves = [(s, e) for s, e, _, kept in fake.requests if kept < 40]
    check("叶子分片恰好铺满整个窗口（不重叠、不跳天）",
          leaves and leaves[0][0] == chunked.time_start
          and leaves[-1][1] == chunked.time_end
          and all(_shift_days(leaves[i][1], 1) == leaves[i + 1][0]
                  for i in range(len(leaves) - 1)),
          str(leaves))
    long_recs = [r for r in got if r.id == "LONG"]
    check("跨分片的长事件只算一条", len(long_recs) == 1, f"实得 {len(long_recs)}")
    check("长事件的轨迹被合并回来（3 个位置）",
          len(long_recs[0].track) == 3, str(long_recs[0].track))
    check("合并后代表点取最晚位置（2024-07-10）",
          long_recs[0].dates()[-1] == "2024-07-10", str(long_recs[0].dates()))

    # 反例：响应完全不属于请求窗口（接口降级，无视窗口参数照旧返回自己的数据）
    degraded = _FakeEonetSession([
        point_event("X1", ["2026-08-01"]), point_event("X2", ["2026-08-02"])],
        ignore_window=True)
    degraded_client = EonetClient(cache_dir="__never_used__", session=degraded)
    try:
        degraded_client.fetch(chunked, use_cache=False)
        check("窗口外的响应被拒", False, "居然正常返回了")
    except GeocodeError as e:
        check("窗口外的响应被拒（不变成'没有事件'）", True)
        check("并标记 service_unavailable", e.service_unavailable)

    # 反例：窗口内确实没有事件时，空结果不该被误判成失败
    empty_client = EonetClient(cache_dir="__never_used__", session=_FakeEonetSession([]))
    check("窗口内没有事件时返回空列表（不是抛错）",
          empty_client.fetch(chunked, use_cache=False) == [])

    class _FailingSession:
        headers = {}
        def get(self, *args, **kwargs):
            raise requests.exceptions.Timeout("模拟超时")

    print("== 缓存：降采样上限不影响缓存内容 ==")
    with tempfile.TemporaryDirectory() as tmp:
        def window(max_events):
            return EventQuery(categories=("wildfires",), time_start="2024-01-01",
                              time_end="2024-07-18",
                              bbox=(-75.0, -20.0, -45.0, 5.0),
                              limit=40, max_events=max_events)

        first = EonetClient(cache_dir=tmp, session=_FakeEonetSession(daily))
        n_small = len(first.fetch(window(20), use_cache=True))
        check("max_events=20 时本次返回被降采样到 20 条", n_small == 20,
              f"实得 {n_small}")
        # 同一个窗口换一个更大的上限再查：必须命中缓存，且拿到完整集合
        # （会话是必然失败的，能返回数据就证明没走网络）
        second = EonetClient(cache_dir=tmp, session=_FailingSession())
        n_big = len(second.fetch(window(1000), use_cache=True))
        check("换更大的 max_events 后命中缓存且拿回完整集合", n_big == 200,
              f"实得 {n_big}")

    print("== 失败语义 ==")
    failing = EonetClient(cache_dir="__never_used__", session=_FailingSession())
    try:
        failing.fetch(open_window, use_cache=False)
        check("不可达时抛错而不是返回空列表", False, "居然正常返回了")
    except GeocodeError as e:
        check("不可达时抛 GeocodeError", True)
        check("并标记 service_unavailable", e.service_unavailable)
        check("提示是中文面向用户的", "不可用" in e.user_message, e.user_message)

    print()
    if failures:
        print(f"自检失败 {len(failures)} 项：{failures}")
        return 1
    print("自检全部通过（离线，未发网络请求）")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(_self_check())
