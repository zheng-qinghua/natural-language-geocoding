"""
地名查找的抽象与请求模型。

对齐源项目 natural_language_geocoding.geocode_index.place_lookup：
把"按名字查真实几何"抽象成一个接口，数据源各自实现，调用方（geocoding.py）
只依赖这个接口。

为什么值得多一层抽象：
  - 可替换：高德 District API 与 OSM Overpass 现在是同一个类的两条内部分支，
    换数据源要动类内部；抽出接口后，新增数据源只需实现一个 search()。
  - 可注入：测试与离线场景可以塞一个假的 lookup 进来，不必联网。
  - 请求模型固定了"查一个地名需要哪些上下文"（国家/省/大陆），换数据源时
    不会因为参数签名不一致而漏传层级信息。

本模块只放接口、请求模型与可选的第二实现；当前默认实现（高德 + Overpass）
在 osm_place_lookup.py。
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict
from shapely.geometry import shape as shapely_shape
from shapely.geometry.base import BaseGeometry

from errors import GeocodeError

# 已知数据源标识。新增后端时在这里登记，避免各处传错字符串。
SourceType = Literal["amap", "osm"]

# ── 候选评分维度（对齐源项目：相关度 → 类型 → 数据源 → 规模）──
# 数据源优先级：高德 District API 给的是官方行政区划边界，精度高于 OSM 众包数据
SOURCE_PRIORITY: dict[str, float] = {"amap": 1.0, "osm": 0.0}
# 名称匹配
NAME_MATCH_EXACT = 3.0
NAME_MATCH_ALIAS = 2.0
NAME_MATCH_PARTIAL = 1.0
# 层级一致性
HIERARCHY_IN_REGION = 2.0
HIERARCHY_NEAR_REFERENCE = 1.0
# 规模权重：取 log10(面积)，权重压得很小——只在其它维度同分的候选之间起作用，
# 避免一个大省把完全匹配的小地物压下去
AREA_WEIGHT = 0.1


@dataclass
class PlaceCandidate:
    """一个候选结果及其评分依据。"""

    geometry: BaseGeometry
    name: str
    source: str
    score: float = 0.0
    # 各维度得分明细，打印出来供人工核对排序是否合理
    reasons: tuple[str, ...] = ()
    # 层级是否已核验且一致：True=一致，False=冲突，None=没能核验（**不等于冲突**）。
    # 这是给 S25 打分用的机器可读信号——同一件事在 reasons 里已经有一句话了，
    # 但那是给人看的字符串，让打分去解析它等于"解析 stdout 回读证据"，
    # 换个说法就会静默失效。默认 None 表示"这个后端不提供这个判断"。
    hierarchy_confirmed: bool | None = None

    def describe(self) -> str:
        detail = "；".join(r for r in self.reasons if r) or "无依据"
        return (f"score={self.score:.2f} source={self.source} name={self.name} "
                f"area={self.geometry.area:.6g} [{detail}]")


def name_match_score(query: str, primary_name: str | None,
                     alias_names: Sequence[str | None] = (),
                     same_language_names: Sequence[str | None] = ()) -> tuple[float, str]:
    """名称匹配度：主名称一致 / 查询语言名称一致 > 别名一致 > 主名称部分包含。

    same_language_names 是"数据源里该地物在查询语言下的名字"（OSM 的 name:en、
    name:zh、int_name）。它单独占一档满分，而不是按普通别名算：主名称通常是当地
    语言（"Spain" 的主名称是 España），字面不同不代表没对上；不这么算，一个碰巧
    真叫 "Spain" 的小地物会仅凭字面完全一致压过真正的国家。
    """
    if primary_name and primary_name == query:
        return NAME_MATCH_EXACT, "主名称完全一致"
    if any(name and name == query for name in same_language_names):
        return NAME_MATCH_EXACT, "查询语言名称一致"
    if any(alias and alias == query for alias in alias_names):
        return NAME_MATCH_ALIAS, "别名匹配"
    if primary_name and (query in primary_name or primary_name in query):
        return NAME_MATCH_PARTIAL, "主名称部分匹配"
    return 0.0, "名称不匹配"


def area_score(geometry: BaseGeometry) -> float:
    """规模得分：同分候选里取更大者（用户说的通常就是显眼的那个）。"""
    return AREA_WEIGHT * math.log10(max(geometry.area, 1e-12))


# ── 地名别名扩展（S18）──
# 为什么需要：用户可能写"深圳大学"，也可能写 "Shenzhen University"；而数据源里
# 一个地物只带其中一种名字。原项目靠 Who's On First 的 alternate_names 解决，
# 国内两个数据源都没有这种现成数据，只能在查询侧补一层名字扩展。
#
# 覆盖范围要说清：这不是通用翻译。只有"每个词都在表里"的组合才能还原
# （"Shenzhen University" → "深圳大学"），表外组合（"Lujiazui Park"）只能靠
# OSM 自带的多语言标签（name:en / alt_name / official_name / old_name）命中。
# 宁可少生成变体，也不要拿半截翻译去查——半截变体查出来的候选往往是错的。
# amap_geocoder.EN_TO_CN 直接引用这张表（原先是各写一份，容易改一处漏一处）。
# 键统一用小写，查表方负责 lower()，这样才能同时命中 "Beijing" 与 "beijing"。
GEO_EN_TO_CN: dict[str, str] = {
    # 国家/大区（高德 API 的 city 限定也会用到）
    "china": "中国", "asia": "亚洲",
    # 直辖市与主要城市
    "beijing": "北京", "shanghai": "上海", "tianjin": "天津", "chongqing": "重庆",
    "shenzhen": "深圳", "guangzhou": "广州", "hangzhou": "杭州", "nanjing": "南京",
    "chengdu": "成都", "wuhan": "武汉", "xian": "西安", "suzhou": "苏州",
    "xiamen": "厦门", "qingdao": "青岛", "dalian": "大连", "ningbo": "宁波",
    "kunming": "昆明", "changsha": "长沙", "zhengzhou": "郑州", "hefei": "合肥",
    "fuzhou": "福州", "jinan": "济南", "shenyang": "沈阳", "harbin": "哈尔滨",
    "hong kong": "香港", "macau": "澳门", "taipei": "台北",
    # 省 / 自治区 / 特别行政区
    "guangdong": "广东", "zhejiang": "浙江", "jiangsu": "江苏", "fujian": "福建",
    "hunan": "湖南", "hubei": "湖北", "sichuan": "四川", "yunnan": "云南",
    "shandong": "山东", "henan": "河南", "hebei": "河北", "shaanxi": "陕西",
    "shanxi": "山西", "anhui": "安徽", "jiangxi": "江西", "guizhou": "贵州",
    "gansu": "甘肃", "qinghai": "青海", "liaoning": "辽宁", "jilin": "吉林",
    "hainan": "海南", "taiwan": "台湾", "guangxi": "广西", "xinjiang": "新疆",
    "tibet": "西藏", "xizang": "西藏", "ningxia": "宁夏",
    "inner mongolia": "内蒙古", "neimenggu": "内蒙古",
}

# 地理通名：英文 → 中文
_WORD_EN_TO_CN: dict[str, str] = {
    "university": "大学", "college": "学院", "institute": "研究所",
    "park": "公园", "garden": "花园", "square": "广场", "plaza": "广场",
    "lake": "湖", "river": "河", "creek": "溪", "sea": "海", "bay": "湾",
    "mountain": "山", "hill": "山", "peak": "峰", "island": "岛", "peninsula": "半岛",
    "province": "省", "city": "市", "district": "区", "county": "县",
    "town": "镇", "village": "村", "street": "街", "road": "路", "avenue": "大道",
    "station": "站", "airport": "机场", "port": "港", "harbour": "港",
    "temple": "寺", "museum": "博物馆", "library": "图书馆", "bridge": "桥",
    "stadium": "体育场", "hospital": "医院", "school": "学校",
    "west": "西", "east": "东", "north": "北", "south": "南",
    "new": "新", "old": "老", "central": "中心",
}

# 反向表由上面两张表推导，避免同一组映射写两遍后走样
_GEO_CN_TO_EN: dict[str, str] = {cn: en for en, cn in GEO_EN_TO_CN.items()}
_WORD_CN_TO_EN: dict[str, str] = {cn: en for en, cn in _WORD_EN_TO_CN.items()}
# 反向查词按长度降序，保证"内蒙古"这种多字词不会被"蒙古"之类的短词抢先切分
_CN_KEYS_BY_LENGTH = sorted(
    list(_GEO_CN_TO_EN) + list(_WORD_CN_TO_EN), key=len, reverse=True
)

def has_latin(text: str) -> bool:
    """含拉丁字母即视为西文输入（"深圳 University" 这种混写也按西文试一遍）。"""
    return any(ch.isascii() and ch.isalpha() for ch in text)


def _english_to_chinese(name: str) -> str | None:
    """把西文地名逐词替换成中文；有任何一个词不认识就放弃（返回 None）。

    城市名优先于通名，且允许两词城市（"hong kong"）：先试两词，再试单词。
    已经是中文的词照抄（"深圳 University" 这种中英混写）：这类词进不了英文表，
    但必须算作"认识"，否则整串都拼不出来。
    """
    tokens = name.replace(",", " ").lower().split()
    if not tokens:
        return None

    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not has_latin(token) and any(not ch.isascii() for ch in token):
            out.append(token)
            i += 1
            continue
        # 先试两词组合（Hong Kong / Inner Mongolia 这类）
        if i + 1 < len(tokens):
            two = f"{tokens[i]} {tokens[i + 1]}"
            if two in GEO_EN_TO_CN:
                out.append(GEO_EN_TO_CN[two])
                i += 2
                continue
        if token in GEO_EN_TO_CN:
            out.append(GEO_EN_TO_CN[token])
        elif token in _WORD_EN_TO_CN:
            out.append(_WORD_EN_TO_CN[token])
        else:
            return None
        i += 1
    return "".join(out)


def _chinese_to_english(name: str) -> str | None:
    """把中文地名按词表切分后逐段换成英文；切不干净就放弃（返回 None）。

    中文地名没有词间空格，只能按词表做最长匹配切分。切不干净说明这个名字
    里有表外成分（"陆家嘴公园"），拼出来的是半截英文，查不到东西还占一次请求，
    所以直接放弃，交给 OSM 自己的多语言标签去命中。

    已经是西文的词照抄（"深圳 University" 的反方向）；但中英粘在一个词里
    （"深圳University"）没法切，照样放弃。
    """
    if not name:
        return None

    pieces: list[str] = []
    for token in name.split():
        if has_latin(token):
            if not token.isascii():
                return None     # 中英粘在一起，切不开
            pieces.append(token)
            continue
        remaining = token
        while remaining:
            for key in _CN_KEYS_BY_LENGTH:
                if remaining.startswith(key):
                    pieces.append(_GEO_CN_TO_EN.get(key) or _WORD_CN_TO_EN[key])
                    remaining = remaining[len(key):]
                    break
            else:
                return None    # 有切不开的字，放弃

    return " ".join(piece.title() for piece in pieces)


def expand_name_variants(name: str) -> tuple[str, ...]:
    """把地名扩展成若干等价写法，原名排在第一个。

    两个方向都做：西文 → 中文（"Shenzhen University" → "深圳大学"）、
    中文 → 西文（"深圳大学" → "Shenzhen University"）。这样无论数据源里
    存的是哪种名字，同一处地点都能从两个方向的查询命中同一个地物。

    Args:
        name: 用户/LLM 给出的地名。

    Returns:
        变体元组，至少含原名本身；无法可靠扩展时长度为 1。
    """
    variants = [name]
    for candidate in (_english_to_chinese(name), _chinese_to_english(name)):
        if candidate and candidate != name and candidate not in variants:
            variants.append(candidate)
    return tuple(variants)


def rank_candidates(candidates: list[PlaceCandidate], limit: int,
                    query: str) -> list[PlaceCandidate]:
    """按得分降序排序并截断，同时打印排序依据。

    排序依据必须打出来：错地点是这类系统最难排查的问题，只留一个结果、
    不留过程，出问题时无从判断是数据源错了还是打分错了。
    """
    candidates.sort(key=lambda c: (c.score, c.geometry.area), reverse=True)
    if len(candidates) > 1:
        print(f"[排序] '{query}' 共 {len(candidates)} 个候选：")
        for i, candidate in enumerate(candidates[:limit], 1):
            print(f"   {i}. {candidate.describe()}")
    return candidates[:limit]


class PlaceSearchRequest(BaseModel):
    """一次地名查找请求：名字 + 用来消歧的层级上下文。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    # 地点类型（continent/country/region/locality/...，见 prompts.PLACE_TYPES）。
    # 当前管线从 LLM 节点里没有取这个字段，留给相关度排序（S13）使用。
    place_type: str | None = None
    in_continent: str | None = None
    in_country: str | None = None
    in_region: str | None = None
    # 指定只查某个数据源；None 表示按实现自己的优先级依次尝试
    source_type: SourceType | None = None

    def variant_names(self) -> tuple[str, ...]:
        """本请求的名字及其别名扩展（原名在前，见 expand_name_variants）。"""
        return expand_name_variants(self.name)


class PlaceLookup(ABC):
    """地名 → 真实几何。实现方不得返回合成的矩形/圆形近似。"""

    @abstractmethod
    def search_for_places(self, request: PlaceSearchRequest,
                          limit: int = 5) -> list[PlaceCandidate]:
        """返回按相关度降序排列的候选，最多 limit 个。

        同名地物遍布各地（"西湖"在 OSM 里有 60 多个），实现方要先打分排序，
        再截断；不能把"数据源返回顺序"当相关度。
        """

    def search_candidate(self, request: PlaceSearchRequest) -> PlaceCandidate:
        """查找地名，返回相关度最高的**整个候选**（几何 + 打分依据 + 层级核验结果）。

        与 `search()` 的区别只在返回值：需要几何的调用方用 `search()`，需要
        证据的调用方（S25 的证据采集）用这个。分成两个方法而不是"给 search 加
        参数"：`search()` 的调用点多且只想要几何，多一个返回形状的分支没有收益。

        Raises:
            GeocodeError: 找不到真实边界——必须明确失败，不能返回近似几何。
        """
        candidates = self.search_for_places(request, limit=1)
        if not candidates:
            raise GeocodeError(
                f"找不到地点「{request.name}」的边界或范围信息。",
                detail=(
                    f"Unable to find polygon geometry for place [{request.name}] "
                    f"in_region [{request.in_region}] in_country [{request.in_country}]"
                ),
            )
        return candidates[0]

    def search(self, request: PlaceSearchRequest) -> BaseGeometry:
        """查找地名的真实边界（取相关度最高的候选），只要几何。

        Args:
            request: 查找请求（名字与层级上下文）。

        Returns:
            Polygon 或 MultiPolygon（真实边界）。

        Raises:
            GeocodeError: 找不到真实边界——必须明确失败，不能返回近似几何。
        """
        return self.search_candidate(request).geometry


class NominatimPlaceLookup(PlaceLookup):
    """Nominatim（OSM 官方地理编码）后端。

    在源项目里这是主力数据源。国内网络访问 nominatim.openstreetmap.org 基本不通，
    所以本项目默认不用它，改用 Overpass（见 osm_place_lookup.OsmPolygonLookup）。
    保留这个实现的用途是提供一个可替换后端的实证：同一个 PlaceLookup 接口下，
    换后端不需要改 geocoding.py 一行代码；网络可达时也可用它做对照验证。
    """

    ENDPOINT = "https://nominatim.openstreetmap.org/search"
    HEADERS = {"User-Agent": "natural-language-geocoding/1.0 (place lookup)"}

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    @staticmethod
    def _build_query(request: PlaceSearchRequest) -> str:
        """把请求拼成 Nominatim 的自由文本查询，层级从粗到细。"""
        parts = [request.name, request.in_region, request.in_country, request.in_continent]
        return ", ".join(p for p in parts if p)

    def search_for_places(self, request: PlaceSearchRequest,
                          limit: int = 5) -> list[PlaceCandidate]:
        params = {
            "q": self._build_query(request),
            "format": "jsonv2",
            "polygon_geojson": 1,   # 要真实多边形，不要只有中心点
            "limit": limit,
        }
        try:
            resp = requests.get(
                self.ENDPOINT, params=params, headers=self.HEADERS, timeout=self.timeout
            )
            resp.raise_for_status()
            results = resp.json()
        except Exception as e:
            # 网络不通时 Nominatim 在国内基本必然走到这里；把它归一成
            # GeocodeError，调用方按失败处理即可，不必认识 requests 的异常类型
            raise GeocodeError(
                "无法访问 Nominatim 地名服务，请检查网络。",
                detail=f"Nominatim 请求失败：{e}",
            ) from e

        candidates = []
        for result in results:
            geojson = result.get("geojson")
            if not geojson:
                continue        # 只有中心点、没有轮廓的结果一律跳过
            geom = shapely_shape(geojson)
            if geom.is_empty:
                continue
            score, reason = name_match_score(
                request.name, result.get("name"), [result.get("display_name")]
            )
            candidates.append(PlaceCandidate(
                geometry=geom,
                name=result.get("name") or request.name,
                source="osm",
                score=SOURCE_PRIORITY["osm"] + score + area_score(geom),
                reasons=(reason,),
            ))
        return rank_candidates(candidates, limit, request.name)
