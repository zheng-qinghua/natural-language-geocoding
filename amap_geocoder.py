"""
高德地图 (Amap) 地理编码客户端：将地名精确解析为坐标。

高德地图 API 是国内最常用的地图服务之一，无需 VPN，免费注册即可使用。
免费额度：5000次/天 地理编码请求。

高德返回 GCJ-02（国测局）坐标，本模块在出口处统一转换为 WGS-84（见
coord_transform.py），使调用方只需处理单一坐标系。

注册流程：
  1. 访问 https://lbs.amap.com 注册账号
  2. 进入"应用管理 → 我的应用"创建应用
  3. 获取 Key（Web服务类型）

使用方式：
  from amap_geocoder import AmapGeocoder
  gc = AmapGeocoder(api_key="your_key")
  result = gc.geocode("深圳人才公园")
  # {"lng": 113.945, "lat": 22.512, "level": "兴趣点", "address": "..."}
"""

import requests
from difflib import SequenceMatcher
from functools import lru_cache
import math
from collections.abc import Sequence

from coord_transform import gcj02_to_wgs84, transform_geojson
# 中英映射与别名扩展与其它数据源共用一套表（见 place_lookup）。
# 这里不再自己维护 EN_TO_CN：各写一份的后果是改一处漏一处，
# 而且高德侧漏掉的名字会表现为"这个地名突然查不到了"。
from place_lookup import GEO_EN_TO_CN, expand_name_variants, has_latin


# 中文名与返回地址的部分匹配门槛：
#   - 绝对下限：至少 2 个字重合（1 个字等于不设门槛）
#   - 相对下限：重合字数要覆盖名字的 3/4，防止只认住一半就放行
# 两个都要满足。相对下限是一步步收紧来的：
#   只有绝对下限 2 → 实测查"火星大裂谷"时，高德返回了一个地址里含"火星"
#     （山东某地，与火星无关）的村庄，两个字重合就放行了，于是"火星上的大裂谷"
#     变成了山东境内的一个点。改成覆盖 2/3（对 5 字名要求 4 字）。
#   2/3 对 3 字名只要求 2 字 → 查"清华园"时返回"清华大学"会因"清华"两字通过，
#     而这是相邻的两个不同地方。提到 3/4 后 3 字名要求 3 字，正好堵住；
#     受影响的只有 3 字名（2→3）、6 字名（4→5）、7 字名（5→6），
#     "上海外滩"（4 字全中"上海市浦东新区外滩"）这类拆开写的情况照常通过。
_MIN_CJK_NAME_OVERLAP = 2
_MIN_CJK_NAME_COVERAGE = 3 / 4


def _name_evidence(name_forms: Sequence[str], address: str) -> str | None:
    """在返回地址里找名字证据，命中则返回命中的那个名字，否则返回 None。

    为什么需要这道门槛：高德地理编码对查不到的名字不会返回空，而是做模糊匹配，
    悄悄丢掉不认识的那部分（实测 "Nowhereland University" → 福建省厦门市同安区
    "大学"，落到一个凭空出现的村庄上）。调用方据此降级成的中心点看似正常，
    实际是错地点，而错地点正是这类系统最难排查的问题。宁可判为查不到。

    判定分两档：
      - 名字整体出现在地址里（西文按小写比，中文直接比）
      - 中文名与地址的重合字数达标（应对地址把名字拆开写的情况，见上面两个门槛）

    西文名不做这档部分匹配：高德对西文查询做的是转写/模糊匹配，地址里写的是
    中文，字面重合的两个字符不代表同一个地方。西文名靠 expand_name_variants
    先换成中文变体再来命中（"Shenzhen University" 会带出"深圳大学"）。
    """
    if not address:
        return None
    lowered = address.lower()
    for form in name_forms:
        if not form:
            continue
        if form.lower() in lowered:
            return form
        if has_latin(form):
            continue
        # 统计所有匹配片段的总字数，而不是只看最长的一段：
        # "上海外滩" 对 "上海市浦东新区外滩" 是两个各自两字的片段，总覆盖 4/4。
        # 单字的片段一律不计：地址里的通名（"站""村""路"）太容易撞上，
        # 实测查"深圳北站"时靠地址尾巴"(公交站)"的"站"凑够数，就放行了深圳大学。
        matched = sum(block.size for block in
                      SequenceMatcher(None, form, address,
                                      autojunk=False).get_matching_blocks()
                      if block.size >= _MIN_CJK_NAME_OVERLAP)
        needed = max(_MIN_CJK_NAME_OVERLAP, math.ceil(len(form) * _MIN_CJK_NAME_COVERAGE))
        if matched >= needed:
            return form
    return None


class AmapGeocoder:
    """
    高德地图地理编码封装。

    使用高德 Web API 将地名（中文）解析为精确坐标。高德返回 GCJ-02，
    本类在出口处统一转换为 WGS-84，调用方拿到的始终是 WGS-84 坐标。
    """

    # 高德地理编码 API 地址
    GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"

    # 精度等级 → 建议半径（公里）
    LEVEL_RADIUS_MAP = {
        "兴趣点": 0.15,     # POI级别，150米半径
        "门牌号": 0.1,      # 门牌号，100米半径
        "村庄": 0.5,        # 村庄，500米
        "乡镇": 2.0,         # 乡镇，2公里
        "区县": 5.0,         # 区县，5公里
        "市": 20.0,          # 城市，20公里
        "省": 50.0,          # 省份，50公里
        "国家": 100.0,       # 国家，100公里
    }

    def __init__(self, api_key: str):
        """
        Args:
            api_key: 高德地图 Web 服务 API Key
        """
        self.api_key = api_key
        self._region_adcode_cache = {}  # region name → adcode

    def _get_region_adcode(self, region_name: str) -> str | None:
        """Look up a region's adcode via District API (with cache)."""
        if region_name in self._region_adcode_cache:
            return self._region_adcode_cache[region_name]

        cn_name = self._to_cn_city(region_name)
        names_to_try = [region_name]
        if cn_name and cn_name != region_name:
            names_to_try.append(cn_name)

        for name in names_to_try:
            if not name:
                continue
            # Try with and without administrative suffixes
            for suffix in ["", "省", "市", "区", "县"]:
                keyword = name if not suffix else (name + suffix if suffix not in name else name)
                try:
                    params = {
                        "key": self.api_key,
                        "keywords": keyword,
                        "subdistrict": 0,
                        "extensions": "base",
                    }
                    resp = requests.get("https://restapi.amap.com/v3/config/district",
                                       params=params, timeout=10)
                    data = resp.json()
                    if data.get("status") == "1" and data.get("districts"):
                        adcode = data["districts"][0].get("adcode", "")
                        if adcode:
                            self._region_adcode_cache[region_name] = adcode
                            return adcode
                except Exception:
                    pass

        self._region_adcode_cache[region_name] = None
        return None

    @lru_cache(maxsize=500)
    def geocode(self, address: str, city: str = None) -> dict | None:
        """
        地理编码：地名 → 坐标。

        Args:
            address: 地名，如"深圳人才公园""天安门广场"
            city: 可选的城市名，用于缩小搜索范围提高精确度

        Returns:
            {"lng": 经度, "lat": 纬度, "level": 精度等级, "address": 完整地址}
            经纬度为 WGS-84。失败返回 None
        """
        params = {
            "key": self.api_key,
            "address": address,
            "output": "JSON",
        }
        if city:
            params["city"] = city

        try:
            resp = requests.get(self.GEOCODE_URL, params=params, timeout=10)
            data = resp.json()

            if data.get("status") == "1" and data.get("geocodes"):
                geo = data["geocodes"][0]
                raw_lng, raw_lat = geo["location"].split(",")
                # 高德返回 GCJ-02，统一转换为 WGS-84 后再交给调用方
                lng, lat = gcj02_to_wgs84(float(raw_lng), float(raw_lat))
                level = geo.get("level", "兴趣点")
                return {
                    "lng": lng,
                    "lat": lat,
                    "level": level,
                    "address": geo.get("formatted_address", address),
                    "suggested_radius_km": self.LEVEL_RADIUS_MAP.get(level, 0.5),
                }

            return None
        except Exception:
            return None

    # 英文名 → 中文名映射（用于高德 API 的城市限定）；表本身在 place_lookup，
    # 键统一为小写，查表方负责 lower()。
    EN_TO_CN = GEO_EN_TO_CN

    def _to_cn_city(self, name: str) -> str | None:
        """将英文地名转换为中文城市名（用于高德API限定）。"""
        if not name:
            return None
        return self.EN_TO_CN.get(name.lower()) or None

    def _validate_result(self, result: dict, in_region: str = None, in_country: str = None) -> bool:
        """校验高德返回的地址是否在预期的区域内。"""
        if not result:
            return False
        address = result.get("address", "")
        cn_region = self._to_cn_city(in_region)
        cn_country = self._to_cn_city(in_country)
        if cn_region and cn_region in address:
            return True
        if cn_country and cn_country in address:
            return True
        if in_region and in_region.lower() in address.lower():
            return True
        if in_country and in_country.lower() in address.lower():
            return True
        # 无校验信息时默认通过
        if not in_region and not in_country:
            return True
        return False

    # 等级优先级：数值越小越精确
    _LEVEL_PRIORITY = {"门牌号": 1, "兴趣点": 2, "村庄": 3, "乡镇": 4, "道路": 4, "住宅小区": 4,
                       "区县": 5, "高等院校": 5, "市": 6, "省": 7, "国家": 8}

    def _level_score(self, level: str) -> int:
        """返回精度等级分数，数值越小越精确。"""
        return self._LEVEL_PRIORITY.get(level, 5)

    def geocode_place(self, name: str, in_region: str = None, in_country: str = None) -> dict | None:
        """
        对 NamedPlace 节点进行地理编码。

        多策略查询，收集所有有效结果后选择精度最高的那个。
        对返回结果进行地址校验，过滤掉不在预期区域的结果。

        地名及其别名变体（见 place_lookup.expand_name_variants，"Shenzhen
        University" 与 "深圳大学" 互为变体）都走一遍策略：高德的 POI 库里
        通常只存中文名，只拿英文原名去查往往一个结果都没有。

        Args:
            name: 地名
            in_region: 所在省份/州（来自 LLM 解析，可能为英文如"Guangdong"）
            in_country: 所在国家（来自 LLM 解析）

        Returns:
            同 geocode()
        """
        cn_region = self._to_cn_city(in_region)
        cn_country = self._to_cn_city(in_country)
        candidates = []  # (result, level_score)
        # 名字的全部等价写法（含中英互转），用来在返回地址里找名字证据。
        # 一次算好、闭包固定引用：早先按循环变量 variant 判断会随迭代变化，
        # 而证据只需要"这个名字族里任一写法出现过"即可。
        variant_forms = expand_name_variants(name)

        def _try_add(result):
            if not result or not self._validate_result(result, in_region, in_country):
                return
            address = result.get("address", "")
            if _name_evidence(variant_forms, address) is None:
                print(
                    f"[高德] 丢弃「{name}」的无名证据结果："
                    f"addr='{address}' level={result.get('level', '')}"
                )
                return
            candidates.append((result, self._level_score(result.get("level", ""))))

        for variant in variant_forms:
            # 策略1：中文省/城市限定
            if cn_region:
                _try_add(self.geocode(variant, city=cn_region))

            # 策略2：中文国家限定
            if cn_country:
                _try_add(self.geocode(variant, city=cn_country))

            # 策略3：英文省名
            if in_region and in_region != cn_region:
                _try_add(self.geocode(variant, city=in_region))

            # 策略4：英文国家名
            if in_country and in_country != cn_country:
                _try_add(self.geocode(variant, city=in_country))

            # 策略5：无城市限定
            _try_add(self.geocode(variant))

            # 策略6：地名+省份拼接
            if cn_region and cn_region not in variant:
                _try_add(self.geocode(cn_region + variant))

            # 策略7：地名+国家拼接
            if cn_country and cn_country not in variant:
                _try_add(self.geocode(cn_country + variant))

            # 策略8：尝试添加具体化后缀（带城市限定更精确）。
            # 只对纯中文变体做：中文后缀拼在西文名后面只会拼出无意义的查询串。
            if not has_latin(variant):
                suffixes = ["风景名胜区", "风景区", "景区", "公园", "湖"]
                for suffix in suffixes:
                    if suffix not in variant:
                        if cn_region:
                            _try_add(self.geocode(variant + suffix, city=cn_region))
                        _try_add(self.geocode(variant + suffix))

        # 选择精度最高的结果
        if candidates:
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

        return None

    def get_district_polygon(self, name: str, in_region: str = None, in_country: str = None) -> dict | None:
        """
        通过高德行政区划API获取地名的精确多边形边界。

        适用于行政区域（省/市/区县），对POI（公园/大学/商场）无效。
        返回 Shapely Polygon/MultiPolygon 的 __geo_interface__ 兼容字典。

        Args:
            name: 地名（中文，如"深圳市""广东省""南山区"）
            in_region: 所在省份（可选，用于多策略回退）
            in_country: 所在国家（可选）

        Returns:
            {
                "type": "Polygon" | "MultiPolygon",
                "coordinates": [...],
                "level": 行政级别,
                "adcode": 行政区划代码,
            }
            坐标为 WGS-84（已从高德的 GCJ-02 转换）。失败返回 None
        """
        DISTRICT_URL = "https://restapi.amap.com/v3/config/district"
        candidates = []
        names = expand_name_variants(name)
        # 命中"名字完全一致"的候选可优先，别名也要算数：
        # 查 "Shenzhen" 命中名为"深圳市"的区划时，不能因为字面不同就丢掉这个偏好
        accepted_names = set(names)
        for variant in names:
            cn_variant = self._to_cn_city(variant)
            if cn_variant:
                accepted_names.add(cn_variant)

        def _try(name_to_try):
            params = {
                "key": self.api_key,
                "keywords": name_to_try,
                "subdistrict": 0,
                "extensions": "all",
            }
            try:
                resp = requests.get(DISTRICT_URL, params=params, timeout=10)
                data = resp.json()
                if data.get("status") == "1" and data.get("districts"):
                    for dist in data["districts"]:
                        # 与 geocode_place 同一道门槛：区划 API 也会对不认识的名字
                        # 做模糊匹配返回一个不相干的区划，名字对不上就整条丢掉
                        if _name_evidence(names, dist.get("name", "")) is None:
                            continue
                        pl = dist.get("polyline", "")
                        if pl:
                            candidates.append(dist)
            except Exception:
                pass

        def _polyline_to_coords(pl_str):
            """Parse polyline string into GeoJSON coordinate arrays."""
            rings = pl_str.split("|")
            polygons = []
            for ring in rings:
                pairs = ring.strip().split(";")
                if not pairs:
                    continue
                coords = []
                # Amap format: lng,lat (GCJ-02); GeoJSON format: [lng, lat]
                for p in pairs:
                    parts = p.strip().split(",")
                    if len(parts) == 2:
                        coords.append([float(parts[0]), float(parts[1])])
                if coords:
                    polygons.append([coords])
            if not polygons:
                return None
            if len(polygons) == 1:
                return {"type": "Polygon", "coordinates": polygons[0]}
            return {"type": "MultiPolygon", "coordinates": polygons}

        # Try multiple strategies (same pattern as geocode_place)
        cn_region = self._to_cn_city(in_region)
        cn_country = self._to_cn_city(in_country)

        for variant in names:
            _try(variant)

            # 英文名换中文再试一轮（高德区划库里存的是中文名）
            cn_variant = self._to_cn_city(variant)
            if cn_variant and cn_variant != variant:
                _try(cn_variant)
                for suffix in ["省", "市", "区", "县"]:
                    if suffix not in cn_variant:
                        _try(cn_variant + suffix)

            if cn_region and cn_region not in variant:
                _try(cn_region + variant)

            if cn_country and cn_country not in variant:
                _try(cn_country + variant)

            # Also try with/without suffix variations for district API
            if not has_latin(variant):
                for suffix in ["市", "区", "县", "省"]:
                    if suffix not in variant:
                        _try(variant + suffix)

        # Select best result: filter by region adcode, then by level priority
        if candidates:
            # Filter by region: if in_region is specified, match by adcode prefix
            region_adcode = None
            if in_region or in_country:
                region_key = in_region or in_country
                region_adcode = self._get_region_adcode(region_key)

            filtered = candidates
            if region_adcode:
                # Adcode hierarchy: XX0000=province, XXXX00=city, XXXXXX=district
                # Match first 2 digits for province, first 4 for city
                prefix_len = 2 if region_adcode.endswith("0000") else 4 if region_adcode.endswith("00") else None
                if prefix_len:
                    prefix = region_adcode[:prefix_len]
                    region_matches = [d for d in filtered
                                     if d.get("adcode", "")[:prefix_len] == prefix]
                    if region_matches:
                        filtered = region_matches

            # Prefer exact name match
            exact = [d for d in filtered if d.get("name") in accepted_names]
            if exact:
                filtered = exact

            # Sort by administrative level (prefer more specific: district > city > province > country)
            level_order = {"district": 1, "city": 2, "province": 3, "country": 4}
            filtered.sort(key=lambda d: level_order.get(d.get("level", ""), 5))

            dist = filtered[0]
            coords = _polyline_to_coords(dist["polyline"])
            if coords:
                # 高德 polyline 是 GCJ-02，统一转换为 WGS-84 后再返回
                return {
                    **transform_geojson(coords),
                    "level": dist.get("level", ""),
                    "adcode": dist.get("adcode", ""),
                    "name": dist.get("name", name),
                }

        return None

    def get_precise_bounds(self, name: str, in_region: str = None, in_country: str = None) -> dict | None:
        """
        获取地名的精确坐标和推荐边界框。

        返回格式与 LLM JSON 中 NamedPlace 的 bounds/center_lon/center_lat 兼容。

        Args:
            name: 地名
            in_region: 所在省份
            in_country: 所在国家

        Returns:
            {
                "center_lon": 经度,
                "center_lat": 纬度,
                "radius_km": 推荐半径(公里),
                "level": 精度等级,
                "address": 完整地址
            }
            经纬度为 WGS-84。失败返回 None
        """
        result = self.geocode_place(name, in_region, in_country)
        if result is None:
            return None
        return {
            "center_lon": result["lng"],
            "center_lat": result["lat"],
            "radius_km": result["suggested_radius_km"],
            "level": result["level"],
            "address": result["address"],
        }
