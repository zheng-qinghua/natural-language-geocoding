"""
PyInstaller 打包脚本：将自然语言地理编码工具打包为独立的 .exe 文件。

使用方法：
    python build_exe.py

打包后的文件位于 ./dist/ 目录下。

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

    # 输出目录
    dist_dir = os.path.join(script_dir, "dist")

    # PyInstaller 命令参数
    # --onefile：打包为单个 exe
    # --windowed：GUI模式不显示控制台（若需要控制台调试可删除此参数）
    # --name：输出文件名
    # --add-data：打包数据文件（提示词模块）
    # --hidden-import：添加隐式依赖
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",                      # 打包为单文件
        "--name", "GeocodingTool",        # 输出文件名
        "--clean",                        # 清理临时文件
        "--noconsole",                    # 不显示控制台(GUI模式)
        "--distpath", dist_dir,           # 输出目录
        "--workpath", os.path.join(script_dir, "build_temp"),  # 临时文件目录
        "--specpath", script_dir,         # .spec 文件目录
        # 添加隐式导入
        "--hidden-import", "shapely",
        "--hidden-import", "shapely.geometry",
        "--hidden-import", "shapely.ops",
        "--hidden-import", "openai",
        "--hidden-import", "folium",
        "--hidden-import", "json",
        "--hidden-import", "amap_geocoder",
        "--hidden-import", "osm_place_lookup",
        "--hidden-import", "prompts",
        "--hidden-import", "natural_earth",
        # 打包海岸线数据文件
        "--add-data", f"ne_10m_coastline.json{os.pathsep}.",

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
        print("=" * 60)
        print(f"打包成功! 可执行文件位于: {exe_path}")
        print("=" * 60)
    else:
        print("打包失败，请检查错误信息。")
        sys.exit(1)


if __name__ == "__main__":
    build_exe()
