"""
Web Demo：在浏览器里做自然语言地理编码。

运行：
    pip install -e ".[web]"     # 安装 streamlit
    streamlit run app.py

与桌面 GUI 的关系：跑的是同一套 geocoding 管线（DeepSeek 解析 → PlaceLookup
查真实边界 → Shapely 运算），只是把结果渲染到网页。地图复用 main.build_map，
底图与坐标加偏的配对关系只在那一处维护，这里不重复实现。

为什么不用 streamlit-folium：folium 已在依赖里，`_repr_html_()` 直接产出可内嵌的
HTML，没必要为一个底图组件再引一个包（国内装包成本偏高）。
"""

import contextlib
import io

import streamlit as st
import streamlit.components.v1 as components

import agent
import main
from errors import GeocodeError
from geocoding import NaturalLanguageGeocoder

# 底图选项 → main.BASEMAPS 的键。中文标签写清坐标系，避免选错底图后
# 中国境内出现数百米偏移却不知道原因。
BASEMAP_OPTIONS = {
    "OSM（WGS-84）": "osm",
    "高德（GCJ-02）": "amap",
}


@st.cache_resource
def get_geocoder() -> NaturalLanguageGeocoder:
    """整个服务共用一个编码器：构造要读配置、建 HTTP 客户端，每次重跑都重建没必要。"""
    return NaturalLanguageGeocoder(amap_api_key=main.amap_api_key() or None)


def geocode_with_log(text: str):
    """跑一次编码，并回收管线打到 stdout 的过程信息。

    解析结果、候选排序依据、降级/兜底提示都只在 stdout 里。在网页里看不到，
    出问题时只能去翻服务端日志，所以顺手截获，页面上给一个可展开的"处理日志"。

    Returns:
        (geometry, error, log)：成功时 error 为 None，失败时 geometry 为 None。
    """
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            geometry = get_geocoder().geocode(text)
    except GeocodeError as e:
        return None, e, buffer.getvalue()
    return geometry, None, buffer.getvalue()


def solve_with_log(text: str):
    """跑一次扩展查询，并回收管线打到 stdout 的过程信息。

    与 `agent.py` 顶部那段"刻意不重定向 stdout"不矛盾：那里要避免的是**在候选
    线程内部**换掉全局 `sys.stdout`，会让各候选的输出互相污染；这里要的恰恰是把
    交错的所有输出收进同一个桶，所以从外面重定向。代价是多会话并发时可能串味，
    与已有的 `geocode_with_log` 相同（也是一个全局缓冲）。
    """
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            result = agent.solve(text, geocoder=get_geocoder())
    except GeocodeError as e:
        return None, e, buffer.getvalue()
    return result, None, buffer.getvalue()


def render_result(geometry, basemap_key: str):
    """把成功的结果渲染成：指标 → 地图 → GeoJSON 下载。"""
    col1, col2, col3 = st.columns(3)
    col1.metric("几何类型", geometry.geom_type)
    col2.metric("中心坐标", f"{geometry.centroid.x:.4f}, {geometry.centroid.y:.4f}")
    col3.metric("面积（平方度）", f"{geometry.area:.6f}")
    st.caption(f"范围 (minx, miny, maxx, maxy): {geometry.bounds}")

    folium_map = main.build_map(geometry, basemap=basemap_key)
    components.html(folium_map._repr_html_(), height=520)

    st.download_button(
        "下载 GeoJSON",
        get_geocoder().to_geojson(geometry),
        file_name="result.geojson",
        mime="application/geo+json",
    )


def render_agent_result(result, basemap_key: str):
    """扩展查询的结果：先给胜出者（复用直连那套渲染），再列全部候选。

    候选的几何摘要与评分直接取自 `AgentCandidate`，不从 `describe()` 的文本里
    反解——那段文本是给人看的，格式一改这里就静默错位。
    """
    st.subheader("胜出结果")
    st.caption(f"扩展 {result.expand_seconds:.1f}s / 计算 {result.solve_seconds:.1f}s / "
               f"裁判 {result.judge_seconds:.1f}s，合计 {result.total_seconds:.1f}s")
    if result.expand_error:
        st.caption(f"（扩展降级：{result.expand_error}）")
    tag = "裁判选出" if result.judged else "本地分数选出"
    st.info(f"**候选 {result.winner.index}**（{tag}）：{result.winner.text}\n\n"
            f"理由：{result.reason}")
    render_result(result.geometry, basemap_key)

    st.subheader(f"候选清单（{len(result.candidates)} 个）")
    for cand in result.candidates:
        if cand is result.winner:
            mark = "胜出"
        elif cand.geometry is None:
            mark = "失败"
        else:
            mark = "备选"
        with st.expander(f"[{mark}] 候选 {cand.index}｜{cand.elapsed:.1f}s｜{cand.text}"):
            st.write(f"**几何**：{cand.summary}")
            st.write(f"**评分**：{cand.score.describe() if cand.score else '未打分'}")
            if cand.geometry is None:
                st.caption("这个候选没有几何可画。")
            elif cand is not result.winner:
                # 胜出者上面已经画过一张大图，这里不重复渲染。
                candidate_map = main.build_map(cand.geometry, basemap=basemap_key)
                components.html(candidate_map._repr_html_(), height=380)


def main_page():
    st.title("自然语言地理编码")
    st.caption(
        "把地点描述解析为真实边界（高德行政区划 / OSM），"
        "支持 12 种空间操作：方位子集、缓冲、交集、差集、边界带……"
    )

    with st.form("query_form"):
        text = st.text_input(
            "地点描述",
            value="深圳大学西南方向的公园",
            help="例如：广东省的南半部分、四川和云南的交界、太湖以西 10 公里",
        )
        basemap_label = st.radio("底图", list(BASEMAP_OPTIONS), horizontal=True)
        # 默认不勾：它比直连慢数倍，只对抽象描述有价值（说明文字与 GUI 共用一处）。
        use_agent = st.checkbox("扩展查询（多候选择优）", help=main.AGENT_HINT)
        submitted = st.form_submit_button("开始编码")

    if submitted:
        if not text.strip():
            st.warning("请输入地点描述。")
            return

        if use_agent:
            with st.spinner("正在扩展查询并并行计算候选（比直连慢数倍）..."):
                result, error, log = solve_with_log(text.strip())
            if error is not None:
                st.error(error.user_message)
                with st.expander("技术详情"):
                    st.code(error.detail)
            else:
                render_agent_result(result, BASEMAP_OPTIONS[basemap_label])
        else:
            with st.spinner("正在解析并查找真实边界..."):
                geometry, error, log = geocode_with_log(text.strip())
            if error is not None:
                st.error(error.user_message)
                with st.expander("技术详情"):
                    st.code(error.detail)
            else:
                render_result(geometry, BASEMAP_OPTIONS[basemap_label])

        with st.expander("处理日志（解析结果与候选排序依据）"):
            st.code(log or "(无输出)")


main_page()
