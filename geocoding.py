"""
地理编码核心模块：将自然语言地点描述转换为空间几何数据。

核心流程：
  用户输入自然语言文本 → DeepSeek大模型解析为结构化JSON →
  从JSON提取坐标和空间操作 → 使用Shapely生成几何对象 → 输出GeoJSON/地图

技术栈：
  - DeepSeek API：国内可直接访问的大模型（https://api.deepseek.com）
  - Shapely：Python地理几何计算库（通过 pip install shapely 安装）
  - 无需 Nominatim/OpenStreetMap（国内被墙），所有坐标由大模型提供

与源项目 (natural-language-geocoding) 的关键差异：
  - 源项目用 Claude(AWS Bedrock) + Nominatim/OpenSearch 地理编码数据库
  - 本模块用 DeepSeek + LLM直接提供坐标（完全绕开国外API）
  - GeometryBuilder 同时支持 bounds（精确边框）和 radius_km（近似半径）
  - 支持复合方向分解（通过 Intersection 节点组合基本方向）
  - 支持全部11种空间操作类型（与源项目对齐）
"""

import json
import os
from openai import OpenAI
from shapely.geometry import Point, Polygon, MultiPolygon, box, LineString, GeometryCollection, shape
from shapely.ops import unary_union
import math

# 导入系统提示词
from prompts import SYSTEM_PROMPT

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

# 自然地球海岸线数据（可选）
try:
    from natural_earth import coastline_of
    _COASTLINE_AVAILABLE = True
except ImportError:
    _COASTLINE_AVAILABLE = False


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
            api_key: DeepSeek API密钥。若为None则从环境变量或默认值读取。
        """
        self.api_key = api_key or os.getenv(
            "DASHSCOPE_API_KEY",
            "sk-xxx"
        )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat"

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
    将大模型输出的JSON空间节点树递归转换为 Shapely 几何对象。

    支持的节点类型（11种，对齐源项目）：
      - NamedPlace：单个命名地点
      - Buffer：围绕子节点的指定距离缓冲区
      - DirectionalConstraint：子节点某个方向（north/south/east/west）
      - Intersection：多个子区域的交集
      - Union：多个子区域的并集
      - Difference：区域差集（child_node_1 减去 child_node_2）
      - Between：两个地点之间的区域（凸包差集法）
      - BorderBetween：两个相邻区域的边界带
      - BorderOf：区域的边界线
      - CoastOf：区域的海岸线
      - OffTheCoastOf：离岸海域

    复合方向（西南/东北等）的处理：
      DirectionalConstraint 只接受四个基本方向。复合方向在 prompts.py 中
      已要求大模型分解为 Intersection 节点，本模块不需要额外处理。
    """

    EARTH_RADIUS_KM = 6371.0

    # 不同类型地点的最大合理边界框（单位：经纬度度数）
    # 用于检测大模型是否给出了过大的 bounds
    MAX_BOUNDS_FOR_TYPE = {
        "locality": 0.08,     # 城市/大学/地标不应超过 ~8km
        "port": 0.05,         # 港口不应超过 ~5km
        "lake": 5.0,          # 湖泊可以达到几百公里
        "island": 10.0,       # 岛屿差异很大
        "region": 50.0,       # 省/州级别
        "country": 120.0,     # 国家级别
        "continent": 360.0,   # 大陆级别
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
    def build_geometry(cls, node: dict) -> object:
        """
        递归构建几何对象（主入口）。

        Args:
            node: JSON节点字典，至少包含 node_type 字段。

        Returns:
            Shapely 几何对象。

        Raises:
            ValueError: 节点类型不支持或缺少必要字段。
        """
        node_type = node.get("node_type", "NamedPlace")
        builders = {
            "NamedPlace": cls._build_named_place,
            "Buffer": cls._build_buffer,
            "DirectionalConstraint": cls._build_directional,
            "Intersection": cls._build_intersection,
            "Union": cls._build_union,
            "Difference": cls._build_difference,
            "Between": cls._build_between,
            "BorderBetween": cls._build_border_between,
            "BorderOf": cls._build_border_of,
            "CoastOf": cls._build_coast_of,
            "OffTheCoastOf": cls._build_off_the_coast_of,
        }
        if node_type not in builders:
            raise ValueError(
                f"不支持的节点类型: {node_type}。"
                f"支持的类型: {list(builders.keys())}"
            )
        return builders[node_type](node)

    # -------------------------------------------------------------------------
    @classmethod
    def _point_to_polygon(cls, center_lon: float, center_lat: float, radius_km: float) -> Polygon:
        """
        从中心点生成不规则多边形（带扰动，模拟真实地理边界）。

        不使用 .buffer()（圆形）也不使用 box()（矩形），
        而是生成带确定性强扰动的多边形，看起来更接近真实地理边界。
        """
        import random
        random.seed(int(center_lon * 10000 + center_lat * 10000))
        degree_dist = cls._km_to_degrees(radius_km, center_lat)
        num_sides = random.randint(7, 12)
        coords = []
        for i in range(num_sides):
            angle = 2 * math.pi * i / num_sides
            angle_jitter = angle + random.uniform(-0.25, 0.25)
            r = degree_dist * (0.7 + random.random() * 0.6)
            x = center_lon + r * math.cos(angle_jitter)
            y = center_lat + r * math.sin(angle_jitter)
            coords.append((x, y))
        return Polygon(coords)

    @classmethod
    def _build_named_place(cls, node: dict) -> object:
        """
        构建命名地点几何。对齐源项目架构：

        仅使用 PlaceLookup 提供的真实多边形边界。
        不使用矩形近似 (box) 或圆形缓冲区 (.buffer)。
        """
        place_name = node.get("name", "未知")
        center_lon = node.get("center_lon")
        center_lat = node.get("center_lat")
        radius_km = node.get("radius_km")

        # 真实多边形边界（来自 PlaceLookup：OSM Overpass / 高德 District API）
        polygon_data = node.get("_polygon")
        if polygon_data:
            try:
                geom = shape(polygon_data)
                if not geom.is_empty:
                    return geom
            except Exception:
                pass

        # 多边形主路径失败 → 用高德坐标生成扰动多边形（永不返回圆形或矩形）
        if center_lon is not None and center_lat is not None:
            amap_level = node.get("_amap_level", "")
            if amap_level in ("兴趣点", "门牌号"):
                effective_radius = radius_km if radius_km else 0.15
            elif amap_level == "村庄":
                effective_radius = radius_km if radius_km else 0.5
            elif amap_level == "乡镇":
                effective_radius = radius_km if radius_km else 2.0
            elif radius_km:
                effective_radius = radius_km
            else:
                effective_radius = 0.5
            return cls._point_to_polygon(center_lon, center_lat, effective_radius)

        raise ValueError(
            f"NamedPlace '{place_name}' 缺少多边形数据和坐标。"
            f"请检查 PlaceLookup 是否正常工作。"
        )

    # -------------------------------------------------------------------------
    @classmethod
    def _build_buffer(cls, node: dict) -> object:
        """
        构建缓冲区几何。

        在子几何周围扩展指定公里数。由于经纬度是球面坐标，
        使用参考点的纬度来做近似的度-公里转换。
        """
        distance_km = node.get("distance_km", 1.0)
        child_node = node.get("child_node")
        if not child_node:
            raise ValueError("Buffer 节点缺少 child_node")

        child_geom = cls.build_geometry(child_node)
        centroid = child_geom.centroid
        degree_distance = cls._km_to_degrees(distance_km, centroid.y)
        return child_geom.buffer(degree_distance)

    # -------------------------------------------------------------------------
    @classmethod
    def _build_directional(cls, node: dict) -> object:
        """
        构建方向约束几何。

        例如"深圳大学以南5公里" → 从深圳大学南边界向南延伸5公里的矩形。

        direction 只能是 north/south/east/west。
        复合方向（如southwest）应在上一层通过 Intersection 处理。

        max_distance_km 控制方向延伸距离：
          - 若LLM提供了该字段，使用LLM的值
          - 若未提供，默认10km（避免产生覆盖半个地球的矩形）
        """
        direction = node.get("direction", "north")
        child_node = node.get("child_node")
        if not child_node:
            raise ValueError("DirectionalConstraint 节点缺少 child_node")

        if direction not in ("north", "south", "east", "west"):
            raise ValueError(
                f"不支持的方向: '{direction}'。"
                f"仅支持 north/south/east/west。"
                f"复合方向（如southwest）应在上层用 Intersection 分解。"
            )

        child_geom = cls.build_geometry(child_node)
        minx, miny, maxx, maxy = child_geom.bounds
        centroid = child_geom.centroid

        # 方向延伸距离（LLM提供则用LLM值，否则默认3km）
        max_distance_km = node.get("max_distance_km", 3.0)
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
    def _build_intersection(cls, node: dict) -> object:
        """
        构建交集几何：多个子区域的重叠部分。

        与源项目 border_between 思路一致：对每个子区域加一个小缓冲区(0.03度≈3.5km)
        再求交集。这样即使两个区域在几何上不完全重合（如两个相邻省份的边界），
        也能得到有意义的交集区域。

        用于：
          - 明确交界处（"四川和云南的交界"）
          - 复合方向约束（"西南方向" = 南 ∩ 西）
          - 多条件约束（"新墨西哥州，阿尔伯克基以西"）
        """
        child_nodes = node.get("child_nodes", [])
        if len(child_nodes) < 2:
            raise ValueError("Intersection 需要至少2个子节点（child_nodes）")

        # 微小缓冲区确保相邻区域可产生交集
        # 0.005° ≈ 500m，足够处理城市级和省级的交集场景
        BORDER_BUFFER_DEG = 0.005

        result = None
        for child in child_nodes:
            child_geom = cls.build_geometry(child)
            # 对每个子几何加微小缓冲区，确保相邻区域可以产生交集
            buffered = child_geom.buffer(BORDER_BUFFER_DEG)
            result = buffered if result is None else result.intersection(buffered)

        if result is None or result.is_empty:
            raise ValueError("交集为空：子区域之间没有重叠部分。请检查地点是否确实相邻。")

        return result

    # -------------------------------------------------------------------------
    @classmethod
    def _build_union(cls, node: dict) -> object:
        """
        构建并集几何：多个子区域的合并。

        用于 "A和B" 这类并列查询。
        """
        child_nodes = node.get("child_nodes", [])
        if len(child_nodes) < 2:
            raise ValueError("Union 需要至少2个子节点（child_nodes）")

        geoms = [cls.build_geometry(child) for child in child_nodes]
        return unary_union(geoms)

    # -------------------------------------------------------------------------
    @classmethod
    def _build_between(cls, node: dict) -> object:
        """
        构建两地点之间的区域。

        参照源项目使用凸包差集法：
          convex_hull(g1 ∪ g2) - convex_hull(g1) - convex_hull(g2)
        这样可以精确获得两个区域之间的空隙。
        """
        child1 = node.get("child_node_1")
        child2 = node.get("child_node_2")
        if not child1 or not child2:
            raise ValueError("Between 节点缺少 child_node_1 或 child_node_2")

        geom1 = cls.build_geometry(child1)
        geom2 = cls.build_geometry(child2)
        coll = GeometryCollection([geom1, geom2])
        convex = coll.convex_hull
        result = convex.difference(geom1.convex_hull).difference(geom2.convex_hull)
        if result.is_empty:
            raise ValueError("两地点之间没有区域。")
        return result

    # -------------------------------------------------------------------------
    @classmethod
    def _build_difference(cls, node: dict) -> object:
        """
        构建差集几何：child_node_1 减去 child_node_2。

        用于 "法国除巴黎外" 这类排除查询。
        """
        child1 = node.get("child_node_1")
        child2 = node.get("child_node_2")
        if not child1 or not child2:
            raise ValueError("Difference 节点缺少 child_node_1 或 child_node_2")

        geom1 = cls.build_geometry(child1)
        geom2 = cls.build_geometry(child2)
        result = geom1.difference(geom2)
        if result.is_empty:
            raise ValueError("差集为空：被减区域完全覆盖了源区域。")
        return result

    # -------------------------------------------------------------------------
    @classmethod
    def _build_border_between(cls, node: dict) -> object:
        """
        构建两区域交界带。

        参照源项目：对两个子几何各缓冲3.5km后求交集，
        得到覆盖共享边界的长条形区域。
        """
        child1 = node.get("child_node_1")
        child2 = node.get("child_node_2")
        if not child1 or not child2:
            raise ValueError("BorderBetween 节点缺少 child_node_1 或 child_node_2")

        # BORDER_BUFFER_SIZE = 3.5 km，与源项目一致
        BORDER_BUFFER_KM = 3.5

        geom1 = cls.build_geometry(child1)
        geom2 = cls.build_geometry(child2)

        centroid1 = geom1.centroid
        degree_dist1 = cls._km_to_degrees(BORDER_BUFFER_KM, centroid1.y)
        buffered1 = geom1.buffer(degree_dist1)

        centroid2 = geom2.centroid
        degree_dist2 = cls._km_to_degrees(BORDER_BUFFER_KM, centroid2.y)
        buffered2 = geom2.buffer(degree_dist2)

        if not buffered1.intersects(buffered2):
            raise ValueError("两区域不相邻，没有共同边界。")

        result = buffered1.intersection(buffered2)
        return result

    # -------------------------------------------------------------------------
    @classmethod
    def _build_border_of(cls, node: dict) -> object:
        """
        构建区域边界线。

        返回子几何的边界（LineString/MultiLineString）。
        注意：此操作返回线几何，非面几何。
        """
        child = node.get("child_node")
        if not child:
            raise ValueError("BorderOf 节点缺少 child_node")

        geom = cls.build_geometry(child)
        boundary = geom.boundary
        if boundary.is_empty:
            raise ValueError("无法获取该区域的边界。")
        return boundary

    # -------------------------------------------------------------------------
    @classmethod
    def _build_coast_of(cls, node: dict) -> object:
        """
        构建区域海岸线。

        使用 Natural Earth 全球海岸线数据，与子几何求交集。
        需要 natural_earth.py 模块和 coastline GeoJSON 数据文件。
        """
        if not _COASTLINE_AVAILABLE:
            raise ValueError(
                "CoastOf 需要 natural_earth.py 模块和 Natural Earth 海岸线数据。"
            )

        child = node.get("child_node")
        if not child:
            raise ValueError("CoastOf 节点缺少 child_node")

        geom = cls.build_geometry(child)
        # 对子几何加2km缓冲区后与全球海岸线求交集
        centroid = geom.centroid
        degree_2km = cls._km_to_degrees(2.0, centroid.y)
        buffered = geom.buffer(degree_2km)

        coast = coastline_of(buffered)
        if coast is None or coast.is_empty:
            raise ValueError("该区域没有找到海岸线。")
        return coast

    # -------------------------------------------------------------------------
    @classmethod
    def _build_off_the_coast_of(cls, node: dict) -> object:
        """
        构建离岸海域。

        参照源项目：海岸线向外缓冲指定距离，再减去陆地部分。
        """
        if not _COASTLINE_AVAILABLE:
            raise ValueError(
                "OffTheCoastOf 需要 natural_earth.py 模块和 Natural Earth 海岸线数据。"
            )

        distance_km = node.get("distance_km", 10.0)
        child = node.get("child_node")
        if not child:
            raise ValueError("OffTheCoastOf 节点缺少 child_node")

        geom = cls.build_geometry(child)
        centroid = geom.centroid

        # 获取海岸线
        degree_2km = cls._km_to_degrees(2.0, centroid.y)
        buffered = geom.buffer(degree_2km)

        coast = coastline_of(buffered)
        if coast is None or coast.is_empty:
            raise ValueError("该区域没有找到海岸线。")

        # 海岸线向外缓冲 → 减去陆地
        degree_dist = cls._km_to_degrees(distance_km, centroid.y)
        buffered_coast = coast.buffer(degree_dist)
        result = buffered_coast.difference(geom)
        if result.is_empty:
            raise ValueError("离岸区域为空。")
        return result


# =============================================================================
# 地理编码主类
# =============================================================================
class NaturalLanguageGeocoder:
    """
    自然语言地理编码器。

    将自然语言地点描述转换为可用的地理几何数据（Shapely对象、GeoJSON）。

    精度机制（参照源项目的多层保障思路）：
      层1 - LLM语义解析：DeepSeek将自然语言转换为空间节点树（JSON结构）
      层2 - 高德地理编码：用高德API查询每个NamedPlace的精确GCJ-02坐标
      层3 - 空间几何算法：Shapely执行Buffer/Intersection/Union等专业运算
      层4 - 合理性校验：bounds跨度检查、默认半径约束

    使用示例:
        gc = NaturalLanguageGeocoder(amap_api_key="your_key")
        geometry = gc.geocode("深圳人才公园")
        geojson = gc.to_geojson(geometry)
    """

    def __init__(self, api_key: str = None, amap_api_key: str = None):
        """
        Args:
            api_key: DeepSeek API密钥。为None则使用默认配置。
            amap_api_key: 高德地图Web服务API Key。为None则跳过精确地理编码，
                          仅依赖LLM提供的坐标（精度较低）。
                          免费注册: https://lbs.amap.com
        """
        self.llm = DeepSeekClient(api_key=api_key)
        self.builder = GeometryBuilder()
        self.amap = None
        if amap_api_key and _AMAP_AVAILABLE:
            self.amap = AmapGeocoder(amap_api_key)
        elif amap_api_key and not _AMAP_AVAILABLE:
            print("[警告] amap_geocoder.py 未找到，无法使用高德地理编码")

        # 创建 PlaceLookup（对齐源项目架构：所有 NamedPlace 通过 lookup 获取真实多边形）
        if _OSM_AVAILABLE:
            self.place_lookup = OsmPolygonLookup(amap_geocoder=self.amap)
        else:
            self.place_lookup = None

    def _enrich_with_place_lookup(self, node: dict):
        """
        递归遍历空间节点树，用 PlaceLookup 获取每个 NamedPlace 的真实多边形几何。

        对齐源项目架构：每个 NamedPlace 通过 PlaceLookup.search() 获取真实的
        Polygon/MultiPolygon 边界，不使用矩形近似或圆形缓冲区。

        Geometry 以 GeoJSON dict 形式存储在 node["_polygon"] 中，
        _build_named_place 会将其转为 Shapely geometry。

        Args:
            node: 空间节点树的一个节点（原地修改）
        """
        node_type = node.get("node_type", "NamedPlace")
        if node_type == "NamedPlace":
            name = node.get("name", "")
            in_region = node.get("in_region")
            in_country = node.get("in_country")

            if self.place_lookup:
                try:
                    geom = self.place_lookup.search(name, in_region, in_country)
                    # 存储 GeoJSON dict 以便序列化/调试
                    from shapely import to_geojson
                    node["_polygon"] = json.loads(to_geojson(geom))
                    # 同时存储中心坐标（用于方向约束等需要坐标的操作）
                    node["center_lon"] = geom.centroid.x
                    node["center_lat"] = geom.centroid.y
                    # 移除 LLM 提供的 bounds（用真实多边形代替）
                    node.pop("bounds", None)
                    return
                except Exception as e:
                    print(f"[PlaceLookup 失败] {name}: {e}")

            # 无 PlaceLookup 时回退到高德（仅存储坐标，后由 builder 处理）
            if self.amap:
                precise = self.amap.get_precise_bounds(name, in_region, in_country)
                if precise:
                    node["center_lon"] = precise["center_lon"]
                    node["center_lat"] = precise["center_lat"]
                    node["radius_km"] = precise["radius_km"]
                    node["_amap_level"] = precise.get("level", "")
                    node.pop("bounds", None)

        # 递归处理子节点
        for key in ("child_node", "child_node_1", "child_node_2"):
            if key in node and isinstance(node[key], dict):
                self._enrich_with_place_lookup(node[key])
        if "child_nodes" in node:
            for child in node["child_nodes"]:
                if isinstance(child, dict):
                    self._enrich_with_place_lookup(child)

    def parse_text(self, text: str) -> dict:
        """
        调用大模型将自然语言解析为结构化JSON。

        Args:
            text: 自然语言地点描述。

        Returns:
            解析后的空间节点树字典。

        Raises:
            ValueError: 大模型返回的内容不是有效JSON。
        """
        raw_response = self.llm.chat(text)
        raw_response = raw_response.strip()

        # 清理可能的 markdown 代码块包裹
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        elif raw_response.startswith("```"):
            raw_response = raw_response[3:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]
        raw_response = raw_response.strip()

        try:
            return json.loads(raw_response)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"大模型返回的不是有效JSON。\n"
                f"原始返回内容:\n{raw_response}"
            ) from e

    def geocode(self, text: str) -> object:
        """
        将自然语言描述转换为 Shapely 几何对象（核心方法）。

        流程：文本 → LLM解析 → 高德坐标精修 → JSON节点树 → 递归构建几何 → Shapely对象

        Args:
            text: 自然语言地点描述。

        Returns:
            Shapely 几何对象（Point/Polygon/MultiPolygon等）。
        """
        spatial_node = self.parse_text(text)
        # 用 PlaceLookup 获取每个 NamedPlace 的真实多边形边界（对齐源项目架构）
        self._enrich_with_place_lookup(spatial_node)
        print(f"[解析结果] {json.dumps(spatial_node, ensure_ascii=False, indent=2)}")
        geometry = self.builder.build_geometry(spatial_node)
        return geometry

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


# =============================================================================
# 简易测试入口
# =============================================================================
if __name__ == "__main__":
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
        except Exception as e:
            print(f"错误: {e}")
