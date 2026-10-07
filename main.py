"""
主程序入口：提供命令行、图形界面、环境检查三种方式使用自然语言地理编码。

pip install -e . 之后可直接用 nlgeocoding 命令；未安装时把 nlgeocoding
换成 python main.py 效果相同。

命令行模式：
    nlgeocoding --cli "北京天安门广场附近5公里"
    （也可省掉 --cli，直接把文本当参数）
批量模式：
    nlgeocoding --batch queries.txt    每行一条查询，导出 FeatureCollection
图形界面模式：
    nlgeocoding --gui
    或直接双击运行（启动Tkinter GUI）
环境检查：
    nlgeocoding init    检查海岸线数据与 API Key 配置

输出：
    - 生成包含编码区域的 globalMap.html 地图文件
    - 在浏览器中自动打开地图
    - 支持导出 GeoJSON 文件
"""

import sys
import os
import json
import webbrowser
import tkinter as tk
from pathlib import Path
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading

# 导入核心模块
from agent import solve
from geocoding import NaturalLanguageGeocoder
from coord_transform import transform_geometry
from errors import GeocodeError

# GUI 里"扩展查询"勾选框旁的说明。放在模块级是为了和 Web 端（app.py）用同一段话，
# 两处说法不一致会让用户以为是两个不同的功能。
AGENT_HINT = ("把描述扩展成多个候选解读、各自完整查一遍再择优。"
              "抽象描述（如「过去两三年内被火灾影响的亚马逊雨林」）才需要它；"
              "具体地名请勿勾选——会慢数倍，结果未必更好。")


def _use_utf8_console() -> None:
    """把标准输出/错误切成 UTF-8。

    中文 Windows 控制台与重定向文件的默认编码是 GBK，打印编不出来的字符
    （地名 "España" 的 ñ）会抛 UnicodeEncodeError。而这些打印都在诊断路径上，
    一崩就把整个查找拖成失败——排序本身没问题，用户看到的却是"查无此地"
    （"法国和西班牙的边界"就是这么挂的）。errors="replace" 保证宁可少印几个
    字符，也不让诊断输出反过来决定主流程的成败。
    """
    for stream in (sys.stdout, sys.stderr):
        # --noconsole 打包后标准流可能为 None，hasattr 一并挡掉
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_use_utf8_console()

# =============================================================================
# 高德地图 API Key（可选，显著提升坐标精度）
# 免费注册: https://lbs.amap.com → 应用管理 → 创建应用 → 获取Key
# 不配置则仅用LLM坐标（精度较低）
# =============================================================================
_AMAP_API_KEY = os.environ.get("AMAP_API_KEY", "")

# =============================================================================
# 默认输出路径
# 在PyInstaller打包后 sys.frozen=True，__file__ 指向临时目录
# 此时应使用 exe 所在目录作为输出目录
# =============================================================================
if getattr(sys, 'frozen', False):
    # 打包后的 exe 环境
    OUTPUT_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    # 开发环境
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_FILE = os.path.join(OUTPUT_DIR, "globalMap.html")
GEOJSON_FILE = os.path.join(OUTPUT_DIR, "output.geojson")
BATCH_GEOJSON_FILE = os.path.join(OUTPUT_DIR, "output_batch.geojson")

# 从配置文件读取 API Key（持久化存储，无需每次设置环境变量）
# 搜索路径：exe同级目录 → exe上级目录 → 开发目录（去重并保持优先级顺序）
_CONFIG_PATHS = list(dict.fromkeys([
    OUTPUT_DIR,
    *([os.path.dirname(OUTPUT_DIR)] if getattr(sys, 'frozen', False) else []),
    os.path.dirname(os.path.abspath(__file__)),
]))

if not _AMAP_API_KEY:
    for _dir in _CONFIG_PATHS:
        _cfg_path = os.path.join(_dir, "config.json")
        if os.path.exists(_cfg_path):
            try:
                with open(_cfg_path, "r", encoding="utf-8") as f:
                    _config = json.load(f)
                    _AMAP_API_KEY = _config.get("amap_api_key", "")
                    if _AMAP_API_KEY:
                        break
            except Exception:
                pass


def amap_api_key() -> str:
    """解析出的高德 Key（环境变量优先，其次 config.json）；未配置返回空串。

    app.py（Web Demo）与 GUI/CLI 共用同一处解析结果，避免各自实现一套
    config.json 搜索逻辑后行为不一致。
    """
    return _AMAP_API_KEY


# =============================================================================
# 地图生成函数
# =============================================================================
# 底图与坐标系的配对关系（错配会导致中国境内约 300–600m 的显示偏移）：
#   - "osm" ：OSM 瓦片为 WGS-84，几何直接按 WGS-84 绘制
#   - "amap"：高德瓦片为 GCJ-02，绘制前必须先把几何转成 GCJ-02
BASEMAPS = {
    "osm": {
        # 不用官方 tile.openstreetmap.org：它按瓦片使用政策拒绝"不带 Referer"的请求，
        # 而 globalMap.html 是以 file:// 打开的本地文件，浏览器不发 Referer，
        # 于是每一块瓦片都被换成写着 "403 Access blocked" 的图——HTTP 状态码仍是 200，
        # 所以页面上看不出报错，只看到满屏 Access blocked。
        # FOSSGIS 的德国镜像提供同一套 OSM 瓦片（同为 WGS-84），不要求 Referer。
        "tiles": "https://tile.openstreetmap.de/{z}/{x}/{y}.png",
        "attr": "&copy; OpenStreetMap contributors",
        "subdomains": None,
        "to_wgs84": True,  # 几何已是 WGS-84，无需转换
    },
    "amap": {
        "tiles": (
            "https://webrd0{s}.is.autonavi.com/appmaptile"
            "?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}"
        ),
        "attr": "高德地图",
        "subdomains": "1234",
        "to_wgs84": False,  # 需要把 WGS-84 几何加偏为 GCJ-02
    },
}
DEFAULT_BASEMAP = "osm"


def build_map(geometry, basemap: str = DEFAULT_BASEMAP):
    """
    使用 Folium 构建交互式地图对象（不落盘）。

    项目内部几何统一为 WGS-84（见 coord_transform.py）。底图决定了绘制坐标系：
    OSM 底图直接用 WGS-84 几何；高德底图是 GCJ-02，必须先把几何加偏，
    否则中国境内会出现数百米偏移。

    与 create_map 拆开是为了给 Web Demo（app.py）复用同一套底图/坐标系逻辑：
    它只需要拿到地图对象在页面里内嵌，不需要写文件。底图与坐标变换的配对关系
    只在这一个函数里维护，避免第二处实现漏掉加偏。

    Args:
        geometry: Shapely 几何对象（WGS-84）
        basemap: 底图类型，"osm" 或 "amap"

    Returns:
        folium.Map 对象
    """
    import folium

    if basemap not in BASEMAPS:
        raise ValueError(f"未知底图 '{basemap}'，可选：{list(BASEMAPS)}")
    cfg = BASEMAPS[basemap]

    # 底图坐标系与几何坐标系对齐：唯一一次转换，之后不再动坐标
    render_geom = geometry if cfg["to_wgs84"] else transform_geometry(geometry, to_wgs84=False)

    # 获取几何中心作为地图中心
    centroid = render_geom.centroid
    # 创建地图，以几何中心为焦点；tiles=None 表示底图图层由下面自行添加
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=6, tiles=None)
    tile_kwargs = {"attr": cfg["attr"], "name": basemap, "max_zoom": 18}
    if cfg["subdomains"]:
        # 只有瓦片地址里带 {s} 的底图（高德）才需要子域轮询
        tile_kwargs["subdomains"] = cfg["subdomains"]
    folium.TileLayer(cfg["tiles"], **tile_kwargs).add_to(m)

    # 将几何对象添加到地图上
    # 包装为完整的 GeoJSON Feature（包含 properties 字段）
    from shapely import to_geojson
    raw_geojson = json.loads(to_geojson(render_geom))
    feature = {
        "type": "Feature",
        "geometry": raw_geojson,
        "properties": {"name": "查询区域"}
    }

    # 根据几何类型选择合适的渲染方式
    geom_type = render_geom.geom_type

    if geom_type == "Point":
        folium.Marker(
            location=[centroid.y, centroid.x],
            popup="目标位置",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
    elif geom_type in ("LineString", "MultiLineString"):
        # 线几何（BorderOf 等操作返回边界线）
        folium.GeoJson(
            feature,
            style_function=lambda x: {
                "color": "#ff4400",
                "weight": 4,
                "opacity": 0.8,
            }
        ).add_to(m)
    else:
        # 多边形/矩形等面状几何，用高亮多边形显示
        folium.GeoJson(
            feature,
            style_function=lambda x: {
                "fillColor": "#ff7800",
                "color": "#ff4400",
                "weight": 2,
                "fillOpacity": 0.4,
            }
        ).add_to(m)

    # 自动缩放至几何范围
    bounds = render_geom.bounds  # (minx, miny, maxx, maxy)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    return m


def create_map(geometry, output_path: str = MAP_FILE, basemap: str = DEFAULT_BASEMAP):
    """
    生成地图并保存为 HTML 文件。

    Args:
        geometry: Shapely 几何对象（WGS-84）
        output_path: 地图HTML保存路径
        basemap: 底图类型，"osm" 或 "amap"

    Returns:
        地图HTML的保存路径
    """
    m = build_map(geometry, basemap=basemap)
    m.save(output_path)
    return output_path


# =============================================================================
# 初始化模式（nlgeocoding init）
# =============================================================================
def run_init():
    """检查运行环境：海岸线数据是否就绪，配置是否齐全。

    海岸线数据（19MB）不进分发包，装机后由本命令检查/下载，与 GUI、CLI 并存。
    """
    print("=" * 60)
    print("自然语言地理编码 — 环境检查")
    print("=" * 60)

    # ── 1. 海岸线数据（CoastOf / OffTheCoastOf 依赖）──
    try:
        from natural_earth import coastline_file_path, download_coastline_file
    except ImportError as e:
        print(f"[海岸线] 无法导入 natural_earth.py：{e}")
        print("         CoastOf / OffTheCoastOf 将不可用，其余功能不受影响。")
    else:
        path = coastline_file_path()
        print(f"[海岸线] 数据路径: {path}")
        if _coastline_ready(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"[海岸线] 状态: 就绪（{size_mb:.1f} MB）")
        else:
            print("[海岸线] 状态: 缺失或不可读，尝试下载...")
            try:
                download_coastline_file()
            except Exception as e:
                print(f"[海岸线] 下载失败：{type(e).__name__}: {e}")
                print("[海岸线] 请手动下载后放到上述路径：")
                print("  https://raw.githubusercontent.com/martynafford/"
                      "natural-earth-geojson/master/10m/physical/ne_10m_coastline.json")
                print("  国内网络访问 raw.githubusercontent.com 常不通，"
                      "如无法下载，CoastOf / OffTheCoastOf 暂不可用。")
            else:
                print("[海岸线] 下载完成")

    # ── 2. API Key 配置（只报告有无，不回显任何 Key 内容）──
    print(f"[配置] 查找路径: {[os.path.join(d, 'config.json') for d in _CONFIG_PATHS]}")
    print(f"[配置] DeepSeek Key: {_describe_key('deepseek_api_key')}"
          "（也可用环境变量 DASHSCOPE_API_KEY）")
    print(f"[配置] 高德 Key: {_describe_key('amap_api_key')}"
          "（也可用环境变量 AMAP_API_KEY）")

    print("=" * 60)
    print("检查完成。下一步：")
    print('  图形界面：nlgeocoding --gui')
    print('  命令行：  nlgeocoding --cli "北京天安门广场附近5公里范围内"')
    print("=" * 60)


def _coastline_ready(path: str) -> bool:
    """文件存在、非空、且能解析出 features（防止半截下载被当成就绪）。"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    return bool(data.get("features"))


def _describe_key(name: str) -> str:
    """报告配置里某个 Key 是否已填写——只返回"已配置/未配置"，不返回 Key 本身。"""
    for _dir in _CONFIG_PATHS:
        cfg_path = os.path.join(_dir, "config.json")
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                value = json.load(f).get(name, "")
        except Exception:
            continue
        if value:
            return "已配置"
    return "未配置"


# =============================================================================
# 命令行模式
# =============================================================================
def run_cli(text: str, use_agent: bool = False):
    """
    命令行模式：处理单个查询文本并输出地图和GeoJSON。
    Args:
        text: 自然语言地点描述
        use_agent: 走 agent.py 的"扩展 → 并行 → 打分 → 裁判"，并把候选清单、
            各自的分数与耗时打到终端。不带开关时本函数的行为与 S28 之前一致。
    """
    print(f"[输入] {text}")
    print("-" * 50)

    try:
        # 构造也可能失败（未配置 DeepSeek Key），一并放进 try，避免吐堆栈
        geocoder = NaturalLanguageGeocoder(amap_api_key=_AMAP_API_KEY or None)
        # 调用核心编码逻辑
        if use_agent:
            # 候选内部已经各自 print 过一轮（多线程下会交错，见 agent.py 顶部），
            # 这里再打一遍汇总：带序号、带分数、带耗时，让日志能对上人。
            result = solve(text, geocoder=geocoder)
            print()
            print(result.describe())
            geometry = result.geometry
        else:
            geometry = geocoder.geocode(text)
        print(f"[几何类型] {geometry.geom_type}")
        print(f"[几何中心] ({geometry.centroid.x:.4f}, {geometry.centroid.y:.4f})")
        print(f"[几何范围] {geometry.bounds}")

        # 导出 GeoJSON
        geojson_str = geocoder.to_geojson(geometry)
        with open(GEOJSON_FILE, "w", encoding="utf-8") as f:
            f.write(geojson_str)
        print(f"[GeoJSON已保存] {GEOJSON_FILE}")

        # 生成地图
        create_map(geometry, MAP_FILE)
        print(f"[地图已保存] {MAP_FILE}")

        # 自动在浏览器打开地图
        # 用 as_uri() 而非手工拼 "file://"：Windows 盘符路径需转成 file:///C:/...
        webbrowser.open(Path(MAP_FILE).as_uri())

    except GeocodeError as e:
        print(f"[错误] {e.user_message}")
        print(f"[详情] {e.detail}")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 发生未预期的内部错误：{type(e).__name__}: {e}")
        sys.exit(1)


# =============================================================================
# 批量模式
# =============================================================================
def run_batch(file_path: str):
    """批量模式：从文本文件逐行读取查询，一次性导出 FeatureCollection。

    文件格式：每行一条查询，空行与 `#` 开头的行跳过（便于在文件里分组注释）。
    单条失败不中断整批；失败条目在结尾集中列出，并让进程以非 0 退出。
    """
    if not os.path.exists(file_path):
        print(f"[错误] 找不到查询文件：{file_path}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f
                 if line.strip() and not line.strip().startswith("#")]
    if not texts:
        print(f"[错误] 文件中没有可用的查询行（空行与 # 注释会被跳过）：{file_path}")
        sys.exit(1)

    print(f"[批量] 文件: {file_path}，共 {len(texts)} 条查询")
    print("-" * 50)

    try:
        geocoder = NaturalLanguageGeocoder(amap_api_key=_AMAP_API_KEY or None)
    except GeocodeError as e:
        print(f"[错误] {e.user_message}")
        print(f"[详情] {e.detail}")
        sys.exit(1)

    items = geocoder.geocode_batch(texts)
    failed = [item for item in items if not item.ok]

    with open(BATCH_GEOJSON_FILE, "w", encoding="utf-8") as f:
        f.write(geocoder.to_feature_collection(items))
    print(f"[GeoJSON已保存] {BATCH_GEOJSON_FILE}"
          f"（{len(items) - len(failed)} 个 Feature）")

    if failed:
        print(f"[失败条目] 共 {len(failed)} 条：")
        for item in failed:
            print(f"   第 {item.index} 条「{item.text}」: {item.error.user_message}")
        sys.exit(1)


# =============================================================================
# 图形界面模式（Tkinter）
# =============================================================================
class GeocodingGUI:
    """自然语言地理编码工具的图形界面。"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("自然语言地理编码工具")
        self.root.geometry("700x550")
        self.root.resizable(True, True)

        # 设置样式
        self._setup_style()
        # 构建UI组件
        self._build_ui()

        self.geocoder = NaturalLanguageGeocoder(amap_api_key=_AMAP_API_KEY or None)
        self.current_geometry = None
        # 上一次 agent 运行的全部候选（None 表示没跑过 agent）。切换下拉框要用它，
        # 而不是从输出框的文本里反解——文本是给人看的，不是数据源。
        self.current_candidates = []

    def _setup_style(self):
        """配置 Tkinter 样式。"""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Microsoft YaHei", 10))
        style.configure("TLabel", font=("Microsoft YaHei", 10))
        style.configure("TEntry", font=("Microsoft YaHei", 10))

    def _build_ui(self):
        """构建界面组件。"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === 输入区域 ===
        input_frame = ttk.LabelFrame(main_frame, text="输入地点描述", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(input_frame, text="请输入自然语言地点描述，例如:").pack(anchor=tk.W)
        ttk.Label(input_frame, text='"北京天安门广场附近5公里"', foreground="gray").pack(anchor=tk.W)

        self.text_input = ttk.Entry(input_frame, font=("Microsoft YaHei", 11))
        self.text_input.pack(fill=tk.X, pady=(5, 5))
        self.text_input.bind("<Return>", lambda e: self._start_geocode())
        # 设置默认文本
        self.text_input.insert(0, "北京天安门广场附近5公里范围内")

        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X)

        self.btn_run = ttk.Button(btn_frame, text="开始编码", command=self._start_geocode)
        self.btn_run.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_clear = ttk.Button(btn_frame, text="清空", command=lambda: self.text_input.delete(0, tk.END))
        self.btn_clear.pack(side=tk.LEFT)

        # === 扩展查询开关（默认不勾：它比直连慢数倍，只对抽象描述有价值）===
        self.use_agent = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            input_frame, text="扩展查询（多候选择优）", variable=self.use_agent
        ).pack(anchor=tk.W, pady=(8, 0))
        ttk.Label(input_frame, text=AGENT_HINT, foreground="gray",
                  wraplength=640, justify=tk.LEFT).pack(anchor=tk.W)

        # === 输出区域 ===
        output_frame = ttk.LabelFrame(main_frame, text="输出结果", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(
            output_frame, font=("Consolas", 10), height=15, wrap=tk.WORD
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # === 操作按钮区域 ===
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))

        self.btn_map = ttk.Button(action_frame, text="打开地图", command=self._open_map, state=tk.DISABLED)
        self.btn_map.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_geojson = ttk.Button(action_frame, text="导出GeoJSON", command=self._export_geojson, state=tk.DISABLED)
        self.btn_geojson.pack(side=tk.LEFT)

        # 候选切换（只在跑过扩展查询后可用）。选谁就把谁画到地图上、
        # 并成为"导出GeoJSON"的对象——所见即所导，避免图和文件对不上。
        ttk.Label(action_frame, text="查看候选:").pack(side=tk.LEFT, padx=(12, 4))
        self.candidate_box = ttk.Combobox(action_frame, state="disabled", width=34)
        self.candidate_box.pack(side=tk.LEFT)
        self.candidate_box.bind("<<ComboboxSelected>>", self._on_candidate_selected)

        self.status_label = ttk.Label(action_frame, text="就绪", foreground="gray")
        self.status_label.pack(side=tk.RIGHT)

    def _start_geocode(self):
        """在后台线程中启动地理编码。"""
        text = self.text_input.get().strip()
        if not text:
            messagebox.showwarning("输入为空", "请输入地点描述文本。")
            return

        # 禁用按钮，防止重复点击
        self.btn_run.config(state=tk.DISABLED, text="处理中...")
        self.status_label.config(text="正在调用大模型解析...")
        self.output_text.delete(1.0, tk.END)

        # 上一轮的候选在这里作废：不清掉的话，直连一次之后下拉框里还挂着旧候选，
        # 切过去画出来的是一个与本次输入无关的几何。
        self.current_candidates = []
        self.candidate_box.set("")
        self.candidate_box.config(state="disabled")

        # 在后台线程运行（避免阻塞GUI）
        thread = threading.Thread(
            target=self._do_geocode, args=(text, self.use_agent.get()), daemon=True
        )
        thread.start()

    def _do_geocode(self, text: str, use_agent: bool = False):
        """执行地理编码（在后台线程中运行）。

        use_agent 在主线程就取好再传进来：`tk.BooleanVar.get()` 不是线程安全的，
        在后台线程读控件状态属于跨线程访问。
        """
        try:
            if use_agent:
                # 候选的 stdout 是交错的（见 agent.py 顶部），但 GUI 本来就把过程
                # 信息留在控制台，不截获，免得为日志再引一处全局 stdout 重定向。
                result = solve(text, geocoder=self.geocoder)
                self.root.after(0, self._on_agent_success, text, result)
            else:
                self.current_geometry = self.geocoder.geocode(text)
                self.root.after(0, self._on_success, text)
        except GeocodeError as e:
            # 界面只显示 user_message；detail 打到控制台供排查
            print(f"[错误详情] {e.detail}")
            self.root.after(0, self._on_error, e.user_message, e.detail)
        except Exception as e:
            print(f"[未预期错误] {type(e).__name__}: {e}")
            self.root.after(
                0, self._on_error, f"发生未预期的内部错误：{type(e).__name__}: {e}"
            )

    def _on_success(self, text: str):
        """直连编码成功后的UI更新（在主线程执行）。"""
        self.output_text.insert(tk.END, f"[输入] {text}\n")
        self._show_geometry(self.current_geometry)

    def _on_agent_success(self, text: str, result):
        """扩展查询成功后的UI更新：候选清单 → 胜出候选的几何 → 填充切换下拉框。"""
        self.current_geometry = result.geometry
        self.current_candidates = result.candidates
        self.output_text.insert(tk.END, f"[输入] {text}\n")
        self.output_text.insert(tk.END, result.describe() + "\n")
        self.output_text.insert(tk.END, "-" * 40 + "\n")
        self._show_geometry(result.geometry)

        self.candidate_box["values"] = (
            [f"胜出：候选 {result.winner.index}"]
            + [f"候选 {c.index}：{c.text}" for c in result.candidates]
        )
        self.candidate_box.current(0)
        self.candidate_box.config(state="readonly")

    def _on_candidate_selected(self, _event=None):
        """把下拉框选中的候选画到地图上（并成为"导出GeoJSON"的对象）。"""
        picked = self.candidate_box.current()
        if picked <= 0 or picked > len(self.current_candidates):
            return
        cand = self.current_candidates[picked - 1]
        if cand.geometry is None:
            reason = cand.error.user_message if cand.error else "未知原因"
            self.output_text.insert(tk.END, f"[候选 {cand.index}] 没有几何可画：{reason}\n")
            return
        self.current_geometry = cand.geometry
        self.output_text.insert(tk.END, f"[查看候选 {cand.index}] {cand.summary}\n")
        self._show_geometry(cand.geometry)

    def _show_geometry(self, geom):
        """画一个几何：指标 → GeoJSON → 地图，并解锁操作按钮。

        直连与扩展查询两条路共用，避免各写一遍后行为漂移。
        """
        self.output_text.insert(tk.END, f"[几何类型] {geom.geom_type}\n")
        self.output_text.insert(tk.END, f"[几何中心] ({geom.centroid.x:.4f}, {geom.centroid.y:.4f})\n")
        self.output_text.insert(tk.END, f"[几何范围] {geom.bounds}\n")
        self.output_text.insert(tk.END, "-" * 40 + "\n")

        # 导出 GeoJSON
        geojson_str = self.geocoder.to_geojson(geom)
        self.output_text.insert(tk.END, f"[GeoJSON]\n{geojson_str}\n")

        # 生成地图
        create_map(geom, MAP_FILE)
        self.output_text.insert(tk.END, f"[地图已保存] {MAP_FILE}\n")

        # 恢复按钮状态
        self.btn_run.config(state=tk.NORMAL, text="开始编码")
        self.btn_map.config(state=tk.NORMAL)
        self.btn_geojson.config(state=tk.NORMAL)
        self.status_label.config(text="编码完成")

    def _on_error(self, user_message: str, detail: str = ""):
        """编码失败后的UI更新：先给用户能看懂的一句话，再附技术细节。"""
        self.output_text.insert(tk.END, f"[错误] {user_message}\n")
        if detail and detail != user_message:
            self.output_text.insert(tk.END, f"[详情] {detail}\n")
        self.btn_run.config(state=tk.NORMAL, text="开始编码")
        self.status_label.config(text="编码失败")

    def _open_map(self):
        """在浏览器中打开生成的地图。"""
        if os.path.exists(MAP_FILE):
            webbrowser.open(Path(MAP_FILE).as_uri())
        else:
            messagebox.showwarning("文件不存在", "请先完成一次编码后再打开地图。")

    def _export_geojson(self):
        """导出GeoJSON到用户指定路径。"""
        if self.current_geometry is None:
            messagebox.showwarning("无数据", "请先完成一次编码后再导出。")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".geojson",
            filetypes=[("GeoJSON文件", "*.geojson"), ("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        if file_path:
            geojson_str = self.geocoder.to_geojson(self.current_geometry)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(geojson_str)
            self.status_label.config(text=f"已导出: {file_path}")

    def run(self):
        """启动GUI主循环。"""
        self.root.mainloop()


# =============================================================================
# 程序入口
# =============================================================================
def _start_gui():
    """启动 GUI；构造阶段失败（如未配置 API Key）时用系统对话框提示，不吐堆栈。

    构造失败时窗口还不存在，无法把提示写进输出框，只能借一个临时隐藏窗口
    弹 messagebox。
    """
    try:
        app = GeocodingGUI()
    except GeocodeError as e:
        print(f"[错误] {e.user_message}")
        print(f"[详情] {e.detail}")
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("启动失败", e.user_message)
        root.destroy()
        return
    app.run()


def _run_agent_cli(args: list) -> None:
    """`--agent` 专属的参数检查与执行（args 里已经去掉 `--agent` 本身）。

    组合限制是有意的，不是没来得及做：
      - `--batch`：批量 × N 个候选会把耗时与外部请求量乘 N，而本机唯一会回话的
        Overpass 镜像连 3 个并发都吃不消（见 agent.py 顶部的实测记录）。这里明确
        拒绝而不是静默退化成单候选——静默降级会让人以为跑的是 agent 的结果。
      - `--gui` / `init`：GUI 里有等价的"扩展查询"勾选框（S29），不必再叠一个命令行
        开关；`init` 是环境检查，跟查询无关。
    """
    if args and args[0] in ("--batch", "--gui", "init"):
        print(f"[错误] --agent 目前不支持与 {args[0]} 组合。")
        if args[0] == "--batch":
            print("       批量 × N 个候选会让耗时与外部请求量乘 N，请逐条用 --cli 跑。")
        else:
            print("       GUI 里有等价的\"扩展查询\"勾选框；init 是环境检查。")
        sys.exit(1)
    if args and args[0] in ("--help", "-h"):
        print("--agent：多 query 扩展与择优（抽象描述用它，具体地名不必）")
        print('  用法：nlgeocoding --agent "过去两三年内被火灾影响的亚马逊雨林"')
        print("  会先扩展出多个候选解读，各自跑一遍完整查询，再择优输出。")
        print("  代价：耗时与外部请求量约为单次查询的数倍。")
        return
    if args and args[0] == "--cli":
        args = args[1:]
    text = " ".join(args)
    if not text:
        print("[错误] --agent 需要一个查询文本，例如：")
        print('       nlgeocoding --agent "过去两三年内被火灾影响的亚马逊雨林"')
        sys.exit(1)
    run_cli(text, use_agent=True)


def main():
    """程序主入口：根据命令行参数选择运行模式。"""
    # --agent 是可以与文本参数并用的开关，先摘出来再按原有分支分发；
    # 不带 --agent 时 argv 原样进入下面的分支，行为与之前完全一致。
    argv = sys.argv[1:]
    if "--agent" in argv:
        _run_agent_cli([a for a in argv if a != "--agent"])
        return

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--gui":
            # 图形界面模式
            _start_gui()
        elif arg == "--cli" and len(sys.argv) > 2:
            # 命令行模式（带参数）
            run_cli(sys.argv[2])
        elif arg == "init":
            # 环境检查（海岸线数据、API Key 配置）
            run_init()
        elif arg == "--batch" and len(sys.argv) > 2:
            # 批量模式（从文件读查询）
            run_batch(sys.argv[2])
        elif arg == "--batch":
            print("[错误] --batch 需要一个查询文件路径，例如："
                  "nlgeocoding --batch queries.txt")
            sys.exit(1)
        elif arg in ("--help", "-h"):
            print("自然语言地理编码工具")
            print("用法:")
            print("  nlgeocoding                启动图形界面")
            print("  nlgeocoding --gui          启动图形界面")
            print("  nlgeocoding --cli <文本>    命令行模式")
            print("  nlgeocoding --batch <文件>  批量模式（每行一条查询，# 开头为注释）")
            print("  nlgeocoding --agent <文本>  扩展与择优模式（抽象描述用它，见下）")
            print("  nlgeocoding init           检查运行环境（海岸线数据 / API Key）")
            print("示例:")
            print('  nlgeocoding --cli "北京天安门广场附近5公里范围内"')
            print('  nlgeocoding --agent "过去两三年内被火灾影响的亚马逊雨林"')
            print("  nlgeocoding --batch queries.txt")
            print("--agent 会先把描述扩展成多个候选解读、各自跑一遍完整查询，再择优输出；"
                  "具体地名不必用它（更慢），代价约为单次查询的数倍。不能与 --batch 组合。")
            print(f"批量模式的输出: {BATCH_GEOJSON_FILE}")
            print("未安装为命令时，把上面的 nlgeocoding 换成 python main.py 同样可用。")
        else:
            # 未指定模式时，直接作为查询文本处理
            run_cli(" ".join(sys.argv[1:]))
    else:
        # 默认启动图形界面
        _start_gui()


if __name__ == "__main__":
    main()
