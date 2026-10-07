"""
PyInstaller 打包脚本：将自然语言地理编码工具打包为独立的 .exe 文件。

使用方法：
    python build_exe.py

打包后的文件 `GeocodingTool.exe` 位于项目根目录（与 main.py 同级）。

依赖：
    pip install pyinstaller
"""

import os
import sys
import subprocess


def build_exe():
    """使用 PyInstaller 打包为单个 exe 文件。"""
    # 确认 pyinstaller 已安装
    try:
        import PyInstaller
    except ImportError:
        print("正在安装 PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 主入口文件
    main_script = os.path.join(script_dir, "main.py")

    # 输出目录 = 项目根目录（与 main.py 同级）
    dist_dir = script_dir
    work_dir = os.path.join(script_dir, "build_temp")

    # PyInstaller 命令参数
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",                      # 打包为单文件
        "--name", "GeocodingTool",        # 输出文件名
        "--clean",                        # 清理临时文件
        "--noconsole",                    # 不显示控制台(GUI模式)
        "--distpath", dist_dir,           # 输出到项目根目录
        "--workpath", work_dir,           # 临时文件目录
        "--specpath", work_dir,           # .spec 文件放临时目录
        # 添加隐式导入
        "--hidden-import", "shapely",
        "--hidden-import", "shapely.geometry",
        "--hidden-import", "shapely.ops",
        "--hidden-import", "openai",
        "--hidden-import", "folium",
        "--hidden-import", "json",
        "--hidden-import", "amap_geocoder",
        "--hidden-import", "osm_place_lookup",
        "--hidden-import", "place_lookup",
        "--hidden-import", "errors",
        "--hidden-import", "prompts",
        "--hidden-import", "natural_earth",
        "--hidden-import", "event_data",
        "--hidden-import", "event_aggregate",
        "--hidden-import", "candidate_score",
        "--hidden-import", "agent",
        "--hidden-import", "coord_transform",
        "--hidden-import", "geometry_utils",
        "--hidden-import", "models",
        "--hidden-import", "splitter",
        # 打包海岸线数据文件（CoastOf/OffTheCoastOf 依赖）。
        # 源路径必须写绝对路径：--add-data 的相对路径是按 --specpath 解析的
        # （这里是 build_temp/），不是当前工作目录，写成"ne_10m_coastline.json"
        # 会让打包在"找不到文件"处直接失败。
        "--add-data", f"{os.path.join(script_dir, 'ne_10m_coastline.json')}{os.pathsep}.",

        # 收集 shapely 的 DLL
        "--collect-binaries", "shapely",
        # 收集 openai 的数据文件
        "--collect-all", "openai",
        # 入口脚本
        main_script,
    ]

    print("=" * 60)
    print("开始打包自然语言地理编码工具...")
    print(f"入口文件: {main_script}")
    print(f"输出目录: {dist_dir}")
    print("=" * 60)

    # 执行打包
    result = subprocess.run(cmd, cwd=script_dir)

    if result.returncode == 0:
        exe_path = os.path.join(dist_dir, "GeocodingTool.exe")
        # 清理临时文件
        import shutil
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        print("=" * 60)
        print(f"打包成功! 可执行文件位于: {exe_path}")
        print("=" * 60)
    else:
        print("打包失败，请检查错误信息。")
        sys.exit(1)


if __name__ == "__main__":
    build_exe()
