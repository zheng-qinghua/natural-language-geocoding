"""
主程序入口：提供命令行和图形界面两种方式使用自然语言地理编码。

命令行模式：
    python main.py "北京天安门广场附近5公里"
图形界面模式：
    python main.py --gui
    或直接双击运行（启动Tkinter GUI）

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
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading

# 导入核心模块
from geocoding import NaturalLanguageGeocoder

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

# 从配置文件读取 API Key（持久化存储，无需每次设置环境变量）
# 搜索路径：exe同级目录 → exe上级目录 → 开发目录
_CONFIG_PATHS = [OUTPUT_DIR]
if getattr(sys, 'frozen', False):
    _CONFIG_PATHS.append(os.path.dirname(OUTPUT_DIR))  # exe 上级目录
_CONFIG_PATHS.append(os.path.dirname(os.path.abspath(__file__)))  # 脚本目录

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


# =============================================================================
# 地图生成函数
# =============================================================================
def create_map(geometry, output_path: str = MAP_FILE):
    """
    使用 Folium 生成交互式地图。
    Args:
        geometry: Shapely 几何对象
        output_path: 地图HTML保存路径
    """
    import folium

    # 获取几何中心作为地图中心
    centroid = geometry.centroid
    # 创建地图，以几何中心为焦点
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=6)

    # 将几何对象添加到地图上
    # 包装为完整的 GeoJSON Feature（包含 properties 字段）
    from shapely import to_geojson
    raw_geojson = json.loads(to_geojson(geometry))
    feature = {
        "type": "Feature",
        "geometry": raw_geojson,
        "properties": {"name": "查询区域"}
    }

    # 根据几何类型选择合适的渲染方式
    geom_type = geometry.geom_type

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
    bounds = geometry.bounds  # (minx, miny, maxx, maxy)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    m.save(output_path)
    return output_path


# =============================================================================
# 命令行模式
# =============================================================================
def run_cli(text: str):
    """
    命令行模式：处理单个查询文本并输出地图和GeoJSON。
    Args:
        text: 自然语言地点描述
    """
    print(f"[输入] {text}")
    print("-" * 50)

    geocoder = NaturalLanguageGeocoder(amap_api_key=_AMAP_API_KEY or None)
    try:
        # 调用核心编码逻辑
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
        webbrowser.open("file://" + MAP_FILE)

    except Exception as e:
        print(f"[错误] {e}")
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

        # 在后台线程运行（避免阻塞GUI）
        thread = threading.Thread(target=self._do_geocode, args=(text,), daemon=True)
        thread.start()

    def _do_geocode(self, text: str):
        """执行地理编码（在后台线程中运行）。"""
        try:
            self.current_geometry = self.geocoder.geocode(text)
            self.root.after(0, self._on_success, text)
        except Exception as e:
            self.root.after(0, self._on_error, str(e))

    def _on_success(self, text: str):
        """编码成功后的UI更新（在主线程执行）。"""
        geom = self.current_geometry
        self.output_text.insert(tk.END, f"[输入] {text}\n")
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

    def _on_error(self, error_msg: str):
        """编码失败后的UI更新。"""
        self.output_text.insert(tk.END, f"[错误] {error_msg}\n")
        self.btn_run.config(state=tk.NORMAL, text="开始编码")
        self.status_label.config(text="编码失败")

    def _open_map(self):
        """在浏览器中打开生成的地图。"""
        if os.path.exists(MAP_FILE):
            webbrowser.open("file://" + MAP_FILE)
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
def main():
    """程序主入口：根据命令行参数选择运行模式。"""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--gui":
            # 图形界面模式
            app = GeocodingGUI()
            app.run()
        elif arg == "--cli" and len(sys.argv) > 2:
            # 命令行模式（带参数）
            run_cli(sys.argv[2])
        elif arg in ("--help", "-h"):
            print("自然语言地理编码工具")
            print("用法:")
            print("  python main.py             启动图形界面")
            print("  python main.py --gui       启动图形界面")
            print("  python main.py --cli <文本> 命令行模式")
            print("示例:")
            print('  python main.py --cli "北京天安门广场附近5公里范围内"')
        else:
            # 未指定模式时，直接作为查询文本处理
            run_cli(" ".join(sys.argv[1:]))
    else:
        # 默认启动图形界面
        app = GeocodingGUI()
        app.run()


if __name__ == "__main__":
    main()
