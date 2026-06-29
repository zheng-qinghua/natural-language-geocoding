"""
高德地图 (Amap) 地理编码客户端：将地名精确解析为坐标。

高德地图 API 是国内最常用的地图服务之一，无需 VPN，免费注册即可使用。
免费额度：5000次/天 地理编码请求。

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
import time
from functools import lru_cache


class AmapGeocoder:
    """
    高德地图地理编码封装。

    使用高德 Web API 将地名（中文）转换为精确的 GCJ-02 坐标。
    GCJ-02 是中国国测局坐标系，在中国地图上显示更准确。
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
            失败返回 None
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
                lng, lat = geo["location"].split(",")
                level = geo.get("level", "兴趣点")
                return {
                    "lng": float(lng),
                    "lat": float(lat),
                    "level": level,
                    "address": geo.get("formatted_address", address),
                    "suggested_radius_km": self.LEVEL_RADIUS_MAP.get(level, 0.5),
                }

            return None
        except Exception:
            return None

    # 英文省名 → 中文省名映射（用于高德API城市限定）
    EN_TO_CN = {
        "beijing": "北京", "Beijing": "北京",
        "shanghai": "上海", "Shanghai": "上海",
        "guangdong": "广东", "Guangdong": "广东",
        "zhejiang": "浙江", "Zhejiang": "浙江",
        "sichuan": "四川", "Sichuan": "四川",
        "jiangsu": "江苏", "Jiangsu": "江苏",
        "hubei": "湖北", "Hubei": "湖北",
        "hunan": "湖南", "Hunan": "湖南",
        "fujian": "福建", "Fujian": "福建",
        "shandong": "山东", "Shandong": "山东",
        "henan": "河南", "Henan": "河南",
        "hebei": "河北", "Hebei": "河北",
        "liaoning": "辽宁", "Liaoning": "辽宁",
        "yunnan": "云南", "Yunnan": "云南",
        "guizhou": "贵州", "Guizhou": "贵州",
        "shanxi": "山西", "Shanxi": "山西",
        "shaanxi": "陕西", "Shaanxi": "陕西",
        "gansu": "甘肃", "Gansu": "甘肃",
        "qinghai": "青海", "Qinghai": "青海",
        "hainan": "海南", "Hainan": "海南",
        "jilin": "吉林", "Jilin": "吉林",
        "anhui": "安徽", "Anhui": "安徽",
        "jiangxi": "江西", "Jiangxi": "江西",
        "taiwan": "台湾", "Taiwan": "台湾",
        "guangxi": "广西", "Guangxi": "广西",
        "neimenggu": "内蒙古", "Inner Mongolia": "内蒙古",
        "xinjiang": "新疆", "Xinjiang": "新疆",
        "xizang": "西藏", "Tibet": "西藏",
        "ningxia": "宁夏", "Ningxia": "宁夏",
        "hong kong": "香港", "Hong Kong": "香港",
        "macau": "澳门", "Macau": "澳门",
        "china": "中国", "China": "中国",
        "asia": "亚洲",
    }

    def _to_cn_city(self, name: str) -> str | None:
        """将英文地名转换为中文城市名（用于高德API限定）。"""
        if not name:
            return None
        return self.EN_TO_CN.get(name) or self.EN_TO_CN.get(name.lower(), None)

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

        def _try_add(result):
            if result and self._validate_result(result, in_region, in_country):
                candidates.append((result, self._level_score(result.get("level", ""))))

        # 策略1：中文省/城市限定
        if cn_region:
            _try_add(self.geocode(name, city=cn_region))

        # 策略2：中文国家限定
        if cn_country:
            _try_add(self.geocode(name, city=cn_country))

        # 策略3：英文省名
        if in_region and in_region != cn_region:
            _try_add(self.geocode(name, city=in_region))

        # 策略4：英文国家名
        if in_country and in_country != cn_country:
            _try_add(self.geocode(name, city=in_country))

        # 策略5：无城市限定
        _try_add(self.geocode(name))

        # 策略6：地名+省份拼接
        if cn_region and cn_region not in name:
            _try_add(self.geocode(cn_region + name))

        # 策略7：地名+国家拼接
        if cn_country and cn_country not in name:
            _try_add(self.geocode(cn_country + name))

        # 策略8：尝试添加具体化后缀（带城市限定更精确）
        suffixes = ["风景名胜区", "风景区", "景区", "公园", "湖"]
        for suffix in suffixes:
            if suffix not in name:
                if cn_region:
                    _try_add(self.geocode(name + suffix, city=cn_region))
                _try_add(self.geocode(name + suffix))

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
            失败返回 None
        """
        DISTRICT_URL = "https://restapi.amap.com/v3/config/district"
        candidates = []

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
        cn_name = self._to_cn_city(name)

        _try(name)

        # If name is English, try Chinese translation
        if cn_name and cn_name != name:
            _try(cn_name)
            # Also try with suffix on Chinese name
            for suffix in ["省", "市", "区", "县"]:
                if suffix not in cn_name:
                    _try(cn_name + suffix)

        if cn_region and cn_region not in name:
            _try(cn_region + name)

        if cn_country and cn_country not in name:
            _try(cn_country + name)

        # Also try with/without suffix variations for district API
        suffixes = ["市", "区", "县", "省"]
        for suffix in suffixes:
            if suffix not in name:
                _try(name + suffix)

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
            exact = [d for d in filtered if d.get("name") == name or d.get("name") == cn_name]
            if exact:
                filtered = exact

            # Sort by administrative level (prefer more specific: district > city > province > country)
            level_order = {"district": 1, "city": 2, "province": 3, "country": 4}
            filtered.sort(key=lambda d: level_order.get(d.get("level", ""), 5))

            dist = filtered[0]
            coords = _polyline_to_coords(dist["polyline"])
            if coords:
                return {
                    **coords,
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
            失败返回 None
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
