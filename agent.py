"""
多 query 扩展与择优 agent：让抽象文本也能落到真实几何上。

要解决的问题（S20/S21 暴露、S24 定位清楚）：抽象文本只解析一次，必然掉进
"模型编个名字 → 数据源查不到 → 降级成中心点"的坑。用户报障句
"过去两三年内被火灾影响的亚马逊雨林"就是标准形态——"Amazon rainforest" 在
OSM 里没有多边形（自然地理实体，不是行政区），于是整条链在**第一次解读**上就
断了，而节点解析、事件查询、几何构建其实全是正确的（S24/S25 都验证过）。

本模块不提升任何一环的能力，而是承认"一次解读可能选错落点"，然后：

  1. **扩展**：一次大模型调用，把同一句话按不同的解读维度改写成 N 个具体描述
  2. **并行计算**：N 个候选各自跑一遍**现有的**管线（不改 geocoding.py 的语义）
  3. **本地打分**：用 S25 的 `candidate_score` 排名（它不联网、不调模型）
  4. **裁判复核**：取前 K 个交给大模型挑一个，并给一句话理由

裁判只做"在已经算出来的几何里挑一个"，不参与几何构建——让模型参与算几何会
把可复现的 Shapely 运算变成不可复现的文本生成。

分工的边界是有意的：**扩展与裁判都要调大模型，所以都可能翻车，翻车时必须有
确定性的退路。**

    扩展失败（输出不合法、候选全被过滤）  → 退化成"就用原文"，单候选
    裁判失败/输出不合法/指数越界        → 退化成本地分数第一名
    某个候选抛错                        → 只丢这一个候选，其余照常

agent 是**增强项**：它可以把"查不到"变成"查得到"，但不允许把本来能跑的查询
弄成失败。凡是"模型没听话"的分支，走向都是一条能给出结果的旧路径。

不使用 agent 时（不带开关）行为与之前完全一致——`solve` 是本模块唯一的入口，
geocoding.py 一行未改。

已知局限（留给 S30 评估，不在这里掩盖）：
  - 扩展出的候选被要求"与原文不同"，所以对**本来就解析正确**的查询（"深圳大学"），
    agent 的结果不会包含"原文解读"这一支，理论上可能选出一个比直连更差的答案。
    因此界面上它必须是**用户自己按的开关**（S29），不默认开启。
  - 候选的 stdout 是交错的：N 个候选在各自线程里跑同一条会大量 print 的管线，
    多行输出会互相插队。这里刻意不去重定向 stdout（`redirect_stdout` 换的是全局
    `sys.stdout`，多线程下会互相污染，比交错更糟），改为每个候选跑完后补一行
    带序号的汇总，让日志至少能对上人。

实测记录（2026-09-22，本机联网，3 候选）：

    报障句  "过去两三年内被火灾影响的亚马逊雨林"  合计 95.1s（扩展 1.4 / 计算 92.4 / 裁判 1.3）
            胜出 MultiPolygon 8.16835 度²（巴西马托格罗索州，451 条野火），三个候选各 90 分
    具体句  "深圳大学"                            合计 103.5s（扩展 0.7 / 计算 102.8 / 裁判 0.0）
            胜出 MultiPolygon 6.56241e-05 度²；两个校区候选均降级出局，不影响整体

**S27 计划的"端到端 30s 内"没达到，如实记在这里**：agent 自己的开销只有约 3s
（扩展 + 裁判各 1s 上下），95–103s 里 92–103s 全在 Overpass。原因是端点现实，
不是并发结构：本机唯一稳定回话的 overpass-api.de 在并发请求下会返回 429/504
（自己把自己打到限流，实测一次 3 候选并发让它的评分从 1.0 掉到 0.4），
overpass.kumi.systems 能建连但一个字节都不发，maps.mail.ru 要 10–33s 才答。
S26 记的"深圳大学冷启动 1.64s"是端点健康时；端点抖动时同一条查询要几十秒，
且会看到同一个候选两次运行一次成功一次降级（丽湖校区/粤海校区都出现过）。
`osm_place_lookup` 一侧为此加了四条针对性的减损措施（全局信号量 2、在飞查询
合并、坏端点整轮不发、200 空的等待封顶），每条都能单独测出来，但都变不出第二台
会回话的镜像——这是本机环境的硬上限，不是本模块能解的。
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, Field, ValidationError

from candidate_score import (
    SOURCE_EVENT,
    SOURCE_LANDUSE,
    SOURCE_POLYGON,
    CandidateScore,
    GeometryEvidence,
    score_candidates,
)
from errors import GeocodeError
from geocoding import DeepSeekClient, NaturalLanguageGeocoder


# =============================================================================
# 可调参数
# =============================================================================
# 默认候选数与交给裁判的数量。3 个候选是"够分叉"与"别把公共镜像打爆"之间的
# 折中：实测唯一可用的 Overpass 镜像在 3 个并发请求下会 504（限流史见 S04/S20），
# 所以候选数不是越多越好，真正的并发限制在 osm_place_lookup 的全局信号量里。
DEFAULT_CANDIDATE_COUNT = 3
DEFAULT_TOP_K = 2

# 候选级并发。比候选数小也无所谓：Overpass 在飞请求数由 osm_place_lookup 的
# 全局信号量（上限 2）兜住，这里的并发主要省的是大模型解析与高德的等待时间。
DEFAULT_CONCURRENCY = 3

# 扩展输出的重试次数（与 geocoding.MAX_PARSE_RETRIES 同一思路：把上次的
# 输出和错误回喂，让模型自己修）。裁判只给 1 次重试：它的退路完全可用
# （本地第一名），不值得为它多等一轮。
MAX_EXPAND_RETRIES = 1
MAX_JUDGE_RETRIES = 1

# 本地早停：第一名是真实边界（不是降级）且领先第二名这么多分时，直接判它赢，
# 不再花一次大模型调用。20 分是"至少扣满一个层级冲突（-25）级别"的量级——
# 差距小于它说明两个候选在打分看来难分伯仲，那正是需要模型判断的情形。
EARLY_STOP_MARGIN = 20.0


# =============================================================================
# 提示词
# =============================================================================
def _category_table() -> str:
    """事件类别 id 清单，从 prompts 渲染（不在这里手写第二份）。

    `prompts.render_event_categories` 是那份清单的唯一来源：类别 id 随数据源
    变动，两处各写一份必然漂移。prompts 缺失时退化成一个空表——扩展提示词不完整
    也好过整个模块 import 不进来。
    """
    try:
        from prompts import render_event_categories
        return render_event_categories()
    except Exception:
        return "（事件数据模块未安装，不要写事件类别）"


EXPAND_SYSTEM_PROMPT = f"""
你是一个地理空间查询改写器。用户会给出一句**抽象、模糊**的地点描述，你要把它
改写成若干条**具体、可直接交给下游地理编码引擎解析**的描述。

下游引擎的强项与弱项（决定什么改写是有用的）：
  - 它能把"有真实边界的地方"（行政区、国家公园、保护区、流域管理单元）变成多边形；
  - 它能把"某一类用地"变成多边形，但**只认这些类别**（写成"某范围内 + 用地类别"
    的句式）：居民区、商业区、商铺集中区、工业区、农田、林地、草地、果园、墓地、
    采石场、军事区、在建工地、水库。除此之外的类别词（"商圈""大学城""科创走廊"
    "粮食主产区"）它认不出，会退成整块范围，要避免。
  - 它**查不到**自然地理泛称的多边形："亚马孙雨林""撒哈拉沙漠""火星上的大裂谷"
    这类没有行政边界的大范围泛称，在数据源里根本没有边界，只会失败或退回一个点。
    所以每个候选都要落到**真实存在的行政/管理区划**上。
  - 它支持"某类事件 + 绝对时间区间"影响某地的写法（事件维度）。

改写要求：
1. 每个候选都是一个**完整、可独立看懂**的描述，不依赖原始问题，不要用"那里"
   "该地区"这类指代。
2. 候选之间要在**不同的解读维度**上分叉，不要只是换同义词。可用的分叉方向：
   - **落点维度**：换一个真实存在的行政/管理区划来代表同一片区域。例如抽象
     区域落在不同的国家、州省、流域管理区上（"亚马孙雨林"可以落到巴西的亚马孙州、
     帕拉州、马托格罗索州，或"亚马孙河流域"）。
   - **事件维度**：把"被某事件影响"写成"在某个行政子区里被该事件影响"，
     让事件过滤发生在有边界的区域上。
   - **时间维度**：把模糊的时间说法换成不同的**绝对区间**（见下）。
3. 时间必须换算成绝对日期。当前日期（程序注入的参照系）：{date.today().isoformat()}
   范围表述取区间上界（"过去两三年"→ 3 年）；"2023 年以来"→ 2023-01-01 ~ 今天。
   输出里**不要**出现"过去两三年""最近"这类相对说法。
4. 事件类别只能用这些 id（NASA EONET 分类），不要自创：
{_category_table()}
5. 不要产出与原始意图无关的候选。宁可保守，也不要跑到另一个话题上。
6. 尽最大努力保留原始问题里的空间约束（国家/地区）与事件+时间约束。

只输出 JSON，不要任何其他文字：
{{"candidates": ["候选描述 1", "候选描述 2", "候选描述 3"]}}
"""

JUDGE_SYSTEM_PROMPT = f"""
你是一个地理空间查询的评审。用户的原始问题很抽象，下游系统把它改写成多条
具体查询，各自算出了一个几何。请判断**哪一个候选最接近用户原始问题的真实意图**。

当前日期：{date.today().isoformat()}

评判依据，按重要性排序：
1. 有没有回答原始问题想问的那片范围。落到了相近但不同的地方，比范围不精确更糟。
2. 几何是不是**真实边界**。降级成的中心点（只知道位置、不知道范围）或大模型
   自己编的矩形，都不如一个真实的行政区/保护区多边形。
3. 事件类问题：有没有真的命中事件，范围是否覆盖了事件实际发生的区域。
4. 时间区间是否与原始问题的说法一致。

只输出 JSON，不要任何其他文字：
{{"index": 选中候选的序号（从 1 开始，按下面给出的顺序）, "reason": "一句话中文理由"}}
"""


# =============================================================================
# 大模型输出的结构（pydantic 校验）
# =============================================================================
class _ExpandedQueries(BaseModel):
    """扩展阶段的输出。数量上限给得宽松，真正的取舍在 _clean_candidates 里。"""

    candidates: list[str] = Field(min_length=1, max_length=8)


class _Verdict(BaseModel):
    """裁判阶段的输出。index 的下界由 pydantic 卡住，上界要跟候选数比才知道。"""

    index: int = Field(ge=1)
    reason: str = Field(min_length=1)


# =============================================================================
# 结果结构
# =============================================================================
# 计算失败的候选拿不到证据（geocode_with_evidence 在返回证据之前就抛了）。
# 给它一个"面积为 0、来源为空"的证据：必然被判出局，且不会被误标成某个具体来源。
# 曾担心"来源空"会不会被当成 polygon——不会，出局候选的明细不展示，这里只需要
# "淘汰"这一个语义是对的。
_FAILED_EVIDENCE = GeometryEvidence(source="", area_deg2=0.0)


@dataclass
class AgentCandidate:
    """一个候选解读及其结局。字段与 BatchItem 对齐（几何或错误，二选一）。"""

    index: int                      # 在候选列表里的序号（从 1 开始，与打印一致）
    text: str                       # 扩展出来的重写文本
    geometry: object | None = None
    evidence: GeometryEvidence | None = None
    score: CandidateScore | None = None
    error: GeocodeError | None = None
    elapsed: float = 0.0            # 这个候选自身的耗时（秒）

    @property
    def ok(self) -> bool:
        return self.geometry is not None

    @property
    def summary(self) -> str:
        """一行摘要：几何类型/面积/中心，或失败原因。给人看，也给裁判看。"""
        if not self.ok:
            return f"失败：{self.error.user_message if self.error else '未知原因'}"
        return _geometry_summary(self.geometry)


@dataclass
class AgentResult:
    """一次 agent 求解的全部产出。"""

    text: str                        # 用户原始输入
    candidates: list[AgentCandidate]  # **按提交顺序**，不是排名顺序
    ranking: list[AgentCandidate]     # 按本地分数降序（出局者垫底）
    winner: AgentCandidate | None
    reason: str                       # 胜出理由（裁判给的，或退路的说明）
    judged: bool                      # 是否真的用了大模型裁判
    expand_error: str | None          # 扩展失败的原因；成功则为 None
    expand_seconds: float = 0.0
    solve_seconds: float = 0.0        # 候选并行计算那一段的墙钟耗时
    judge_seconds: float = 0.0
    total_seconds: float = 0.0

    @property
    def geometry(self) -> object | None:
        return self.winner.geometry if self.winner else None

    def describe(self) -> str:
        """多行摘要：候选清单 → 排名 → 胜出。CLI/GUI 直接用这一段。"""
        lines = [f"[扩展查询] 共 {len(self.candidates)} 个候选，"
                 f"扩展 {self.expand_seconds:.1f}s / 计算 {self.solve_seconds:.1f}s / "
                 f"裁判 {self.judge_seconds:.1f}s，合计 {self.total_seconds:.1f}s"]
        if self.expand_error:
            lines.append(f"  （扩展降级：{self.expand_error}）")
        for cand in self.candidates:
            score = cand.score.describe() if cand.score else "未打分"
            lines.append(f"  候选 {cand.index}｜{cand.elapsed:.1f}s｜{cand.text}")
            lines.append(f"       {cand.summary}")
            lines.append(f"       评分：{score}")
        if self.winner:
            tag = "裁判选出" if self.judged else "本地分数选出"
            lines.append(f"[胜出] 候选 {self.winner.index}（{tag}）：{self.reason}")
        return "\n".join(lines)


def _geometry_summary(geometry) -> str:
    """几何的一句话摘要。空几何（面积为 0）显式说出来，不要让裁判以为它是个小地方。"""
    kind = geometry.geom_type
    if geometry.is_empty or geometry.area <= 0:
        return f"{kind}（空几何）"
    center = geometry.centroid
    return (f"{kind}，面积 {geometry.area:.6g} 平方度，"
            f"中心 ({center.x:.3f}, {center.y:.3f})")


# =============================================================================
# 大模型调用的公共部分
# =============================================================================
def _strip_fences(text: str) -> str:
    """去掉 markdown 围栏。与 geocoding._parse_response 同一处理，理由也一样：
    模型经常把 JSON 包在 ```json 里，围栏不是"输出不合法"，不该触发重试。"""
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _normalize(text: str) -> str:
    """比较用的归一化：折叠空白。中文标点差异不处理——那已经不是"同一条描述"了。"""
    return " ".join((text or "").split())


def _clean_candidates(raw: list[str], original: str, limit: int) -> list[str]:
    """过滤扩展结果：去掉空串、与原文相同的、以及彼此重复的，保留前 limit 条。

    与原文相同的候选被**丢掉**而不是判扩展失败：一个候选不合格不等于这次扩展
    没用（另外两条可能正好是我们要的分叉），为它把整轮重试掉是浪费一次调用。
    三条都不合格时上层会退化成"用原文"，那条退路才是最终兜底。
    """
    original_key = _normalize(original)
    seen = {original_key}
    kept: list[str] = []
    for candidate in raw or []:
        candidate = (candidate or "").strip()
        key = _normalize(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(candidate)
        if len(kept) >= limit:
            break
    return kept


# =============================================================================
# 第一步：扩展
# =============================================================================
def _expand(text: str, n: int, llm) -> tuple[list[str], str | None]:
    """扩展的完整实现，连失败原因一起返回（solve 需要知道它降级了）。

    Returns:
        (候选重写文本列表, 失败原因或 None)。失败时返回空列表，由调用方决定退路。
    """
    prompt = (f"原始描述：{text}\n\n"
              f"请改写出 {n} 条候选描述。")
    raw_response = ""
    last_error = ""
    for attempt in range(MAX_EXPAND_RETRIES + 1):
        try:
            raw_response = llm.chat(prompt, system_prompt=EXPAND_SYSTEM_PROMPT)
        except Exception as e:
            # 网络/鉴权问题重试同样的请求不会好转，直接降级
            return [], f"调用大模型失败：{type(e).__name__}: {e}"

        try:
            payload = json.loads(_strip_fences(raw_response))
            parsed = _ExpandedQueries.model_validate(payload)
            candidates = _clean_candidates(parsed.candidates, text, n)
            if candidates:
                return candidates, None
            last_error = "模型给出的候选全部为空、与原文相同或彼此重复"
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = f"输出不合法：{e}"

        if attempt < MAX_EXPAND_RETRIES:
            print(f"[扩展重试 {attempt + 1}/{MAX_EXPAND_RETRIES}] {last_error}")
            prompt = (f"原始描述：{text}\n\n"
                      f"你上一次的输出不合法，请修正后重新输出完整 JSON。\n"
                      f"上一次输出：\n{raw_response}\n\n"
                      f"错误信息：\n{last_error}\n\n"
                      f"请改写出 {n} 条候选描述。")

    return [], last_error


def expand_queries(text: str, n: int = DEFAULT_CANDIDATE_COUNT, llm=None) -> list[str]:
    """把一句抽象描述改写成 n 条具体描述。

    这是给外部直接调用的简单契约（失败返回空列表，不抛异常）：调用方想自己
    决定退路时用它，`solve` 用的是内部那条能拿到失败原因的版本。

    Args:
        text: 用户原始描述。
        n: 期望的候选数量。
        llm: 大模型客户端（需有 `chat(user_text, system_prompt=...)`）；
             None 时新建一个 DeepSeekClient。

    Returns:
        候选描述列表（可能少于 n 条，也可能为空——空表示扩展整体失败）。
    """
    if llm is None:
        llm = DeepSeekClient()
    return _expand(text, n, llm)[0]


# =============================================================================
# 第二步：并行跑候选
# =============================================================================
def _run_one(geocoder, index: int, text: str) -> AgentCandidate:
    """跑一个候选。**这里绝不往外抛异常**——线程池里的异常会被吞进 Future，
    整个候选会静默消失，而"少了一个候选"与"这个候选失败"对上层是完全不同的两件事。
    """
    started = time.monotonic()
    try:
        geometry, evidence = geocoder.geocode_with_evidence(text)
    except GeocodeError as e:
        return AgentCandidate(index=index, text=text, error=e,
                              elapsed=time.monotonic() - started)
    except Exception as e:
        wrapped = GeocodeError(
            "这个候选在解析过程中发生了未预期的内部错误。",
            detail=f"{type(e).__name__}: {e}",
        )
        return AgentCandidate(index=index, text=text, error=wrapped,
                              elapsed=time.monotonic() - started)
    return AgentCandidate(index=index, text=text, geometry=geometry,
                          evidence=evidence, elapsed=time.monotonic() - started)


def _run_candidates(texts: list[str], geocoder,
                    concurrency: int) -> list[AgentCandidate]:
    """把 N 个候选并行丢进线程池，返回**与输入同序**的候选结果。

    顺序必须由调用方按 index 回收，不能依赖"谁先跑完"：打分的面积中位数、
    打印的序号、裁判看到的序号全靠这个顺序对齐（`executor.map` 保证同序）。
    """
    workers = max(1, min(concurrency, len(texts)))
    if workers == 1:
        return [_run_one(geocoder, i, t) for i, t in enumerate(texts, 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda pair: _run_one(geocoder, *pair),
                             list(enumerate(texts, 1))))


# =============================================================================
# 第三步：打分与早停
# =============================================================================
def _rank(candidates: list[AgentCandidate]) -> tuple[list[AgentCandidate], bool]:
    """给候选打分并排名，同时判断能否本地早停。

    Returns:
        (按分数降序的排名列表, 是否可以跳过裁判)。
        排名的顺序与 `AgentCandidate.score` 已经写回候选对象，调用方不必自己排。
    """
    evidences = [c.evidence if c.evidence is not None else _FAILED_EVIDENCE
                 for c in candidates]
    for cand, score in zip(candidates, score_candidates(evidences)):
        cand.score = score

    ranked = sorted(candidates, key=lambda c: c.score.sort_key, reverse=True)
    alive = [c for c in ranked if not c.score.eliminated]

    # 只有 0 个或 1 个候选活着时没有"挑"可言，裁判的输入本身就不成立。
    if len(alive) < 2:
        return ranked, True
    best, second = alive[0], alive[1]
    if best.evidence is None:
        return ranked, True
    # 早停只认"真实边界 + 明显领先"：来源是降级的（点/矩形）时，即使分数领先
    # 也说明两个候选都不好，那种情形更需要模型来判断哪个更贴近原始意图。
    clearly_ahead = (
        best.evidence.source in (SOURCE_POLYGON, SOURCE_EVENT, SOURCE_LANDUSE)
        and best.score.total - second.score.total >= EARLY_STOP_MARGIN
    )
    return ranked, clearly_ahead


# =============================================================================
# 第四步：裁判
# =============================================================================
def _judge_prompt(text: str, pool: list[AgentCandidate]) -> str:
    lines = [f"用户的原始问题：{text}", "", "改写得出的候选（按序号排列）："]
    for order, cand in enumerate(pool, 1):
        lines.append(f"{order}. 重写文本：{cand.text}")
        lines.append(f"   几何：{cand.summary}")
        if cand.evidence is not None and cand.evidence.event_categories:
            lines.append(f"   事件命中：{cand.evidence.event_count} 条")
    lines.append("")
    lines.append("请选出一个最符合原始问题真实意图的候选。")
    return "\n".join(lines)


def _judge(text: str, pool: list[AgentCandidate],
           llm) -> tuple[int, str] | None:
    """让大模型在 pool 里挑一个。

    Returns:
        (pool 里的 0 基下标, 理由)；模型没听话时返回 None，由调用方退到本地第一名。
        **返回的是 pool 的下标，不是候选的 index**：裁判看到的序号是给它自己看的
        （从 1 开始、只覆盖 pool），把两套编号混起来是最容易出的错。
    """
    prompt = _judge_prompt(text, pool)
    last_error = ""
    for attempt in range(MAX_JUDGE_RETRIES + 1):
        try:
            raw_response = llm.chat(prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
        except Exception as e:
            print(f"[裁判] 调用失败，改用本地分数第一名：{type(e).__name__}: {e}")
            return None

        try:
            verdict = _Verdict.model_validate(json.loads(_strip_fences(raw_response)))
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
        else:
            if 1 <= verdict.index <= len(pool):
                return verdict.index - 1, verdict.reason
            last_error = f"序号 {verdict.index} 不在 1~{len(pool)} 之间"

        if attempt < MAX_JUDGE_RETRIES:
            print(f"[裁判重试 {attempt + 1}/{MAX_JUDGE_RETRIES}] {last_error}")
            prompt = (f"{_judge_prompt(text, pool)}\n\n"
                      f"你上一次的输出不合法，请修正后重新输出完整 JSON。\n"
                      f"上一次输出：\n{raw_response}\n\n"
                      f"错误信息：\n{last_error}")

    print(f"[裁判] 连续输出不合法（{last_error}），改用本地分数第一名")
    return None


def judge(text: str, scored: list[AgentCandidate],
          llm=None) -> tuple[AgentCandidate, str] | None:
    """在已排名的候选里挑一个，返回 (胜出候选, 理由)；模型没听话时返回 None。

    对外契约：调用方拿到 None 就意味着"该用本地分数第一名了"，不需要知道
    模型为什么不听话（调用被拒、JSON 坏了、序号越界，处理方式都一样）。
    """
    if llm is None:
        llm = DeepSeekClient()
    verdict = _judge(text, scored, llm)
    if verdict is None:
        return None
    index, reason = verdict
    return scored[index], reason


# =============================================================================
# 入口：solve
# =============================================================================
def _default_geocoder() -> NaturalLanguageGeocoder:
    """按 main.py 的口径构造默认 geocoder（含高德 Key）。

    不从 main 导入 amap_api_key()：main 在模块级 `import tkinter`，为了一个 Key
    把 GUI 依赖拖进库模块不值得。
    """
    return NaturalLanguageGeocoder(amap_api_key=_config_amap_key() or None)


def _config_amap_key() -> str:
    """读高德 Key：环境变量优先，其次 config.json（与 main._CONFIG_PATHS 同序）。"""
    key = os.environ.get("AMAP_API_KEY", "")
    if key:
        return key
    dirs = [os.path.dirname(os.path.abspath(__file__))]
    if getattr(sys, "frozen", False):
        dirs.insert(0, os.path.dirname(os.path.abspath(sys.executable)))
    for directory in dirs:
        path = os.path.join(directory, "config.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                key = json.load(fh).get("amap_api_key", "")
        except Exception:
            continue
        if key:
            return key
    return ""


def solve(text: str, n: int = DEFAULT_CANDIDATE_COUNT, top_k: int = DEFAULT_TOP_K,
          concurrency: int = DEFAULT_CONCURRENCY, geocoder=None, llm=None,
          use_judge: bool = True) -> AgentResult:
    """对一句抽象描述做"扩展 → 并行计算 → 打分 → 裁判"。

    Args:
        text: 用户原始描述。
        n: 扩展出的候选数量。
        top_k: 交给裁判的候选数量（按本地分数取前 K）。
        concurrency: 候选级并发数。
        geocoder: 注入的编码器（需有 `geocode_with_evidence(text)`）；None 则新建。
        llm: 注入的大模型客户端；None 则用 geocoder.llm（省一次构造）。
        use_judge: 关掉就纯靠本地分数选（自检与排查用）。

    Returns:
        AgentResult。**只要有一个候选算出几何就正常返回**，绝不因为某个候选
        失败而整体失败。

    Raises:
        GeocodeError: 所有候选都没能算出几何。理由取信息量最大的那一条
            （优先"数据源已答复查无此地"，其次"服务不可用"，再次第一条）。
    """
    total_started = time.monotonic()
    if geocoder is None:
        geocoder = _default_geocoder()
    if llm is None:
        llm = getattr(geocoder, "llm", None) or DeepSeekClient()

    # ---- 扩展 ----
    expand_started = time.monotonic()
    texts, expand_error = _expand(text, n, llm)
    expand_seconds = time.monotonic() - expand_started
    if not texts:
        # 退路：单候选 = 只用原文，行为与不带 agent 时一致
        texts = [text]
        expand_error = expand_error or "扩展未产出可用候选"
        print(f"[扩展] 降级为只解析原文：{expand_error}")

    # ---- 并行计算 ----
    solve_started = time.monotonic()
    candidates = _run_candidates(texts, geocoder, concurrency)
    solve_seconds = time.monotonic() - solve_started
    for cand in candidates:
        print(f"[候选 {cand.index}] {cand.elapsed:.1f}s｜{cand.text}")
        print(f"    {cand.summary}")

    # ---- 打分与排名 ----
    ranked, can_early_stop = _rank(candidates)
    alive = [c for c in ranked if not c.score.eliminated]
    if not alive:
        raise GeocodeError(
            "扩展查询后，所有候选解释都没能算出几何。",
            detail=_all_failed_detail(candidates),
        )

    pool = alive[:max(1, top_k)]

    # ---- 裁判 ----
    judge_seconds = 0.0
    winner, reason, judged = pool[0], _local_reason(pool), False
    if len(pool) > 1 and use_judge and not can_early_stop:
        judge_started = time.monotonic()
        verdict = _judge(text, pool, llm)
        judge_seconds = time.monotonic() - judge_started
        if verdict is not None:
            index, reason = verdict
            winner, judged = pool[index], True
    elif len(pool) > 1 and use_judge and can_early_stop:
        reason = (f"本地分数明显领先（{pool[0].score.total:.0f} 分），"
                  f"跳过裁判直接采用")

    return AgentResult(
        text=text,
        candidates=candidates,
        ranking=ranked,
        winner=winner,
        reason=reason,
        judged=judged,
        expand_error=expand_error,
        expand_seconds=expand_seconds,
        solve_seconds=solve_seconds,
        judge_seconds=judge_seconds,
        total_seconds=time.monotonic() - total_started,
    )


def _local_reason(pool: list[AgentCandidate]) -> str:
    """本地选出的理由。不编话：只说"本地分最高"，明细在候选的评分里。"""
    return f"本地分数最高（{pool[0].score.total:.0f} 分）：{pool[0].score.describe()}"


def _all_failed_detail(candidates: list[AgentCandidate]) -> str:
    """所有候选都失败时，挑一条信息量最大的错误作为对外的失败原因。

    优先级与 `GeocodeError.service_unavailable` 的语义一致：**"数据源答复了
    查无此地"比"服务不可用"更值得报**——前者说明这条路真的走不通，后者只说明
    这一轮没问到，用户据此该做的动作完全不同（换描述 vs 稍后重试）。
    """
    errors = [c.error for c in candidates if c.error is not None]
    if not errors:
        return "所有候选都未产出几何，且没有记录到具体原因。"
    definitive = [e for e in errors if not e.service_unavailable]
    chosen = (definitive or errors)[0]
    lines = [f"共 {len(candidates)} 个候选，全部失败。第一条失败原因："
             f"{chosen.user_message}"]
    for cand in candidates:
        if cand.error is not None:
            lines.append(f"  候选 {cand.index}「{cand.text}」：{cand.error.user_message}")
    return "\n".join(lines)


# =============================================================================
# 离线自检（python agent.py）
# =============================================================================
class _FakeLLM:
    """按脚本回复的假大模型：能数出被调用了几次（早停要看这个数）。"""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def chat(self, user_text: str, system_prompt: str = None) -> str:
        self.calls.append((system_prompt or "", user_text))
        if not self.replies:
            raise AssertionError("假大模型没有预备的回复了（多调了一次？）")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class _FakeGeocoder:
    """按文本查表的假编码器。表里的值要么是 (几何, 证据)，要么是一个异常。"""

    def __init__(self, table: dict):
        self.table = table
        self.seen: list[str] = []

    def geocode_with_evidence(self, text: str):
        self.seen.append(text)
        if text not in self.table:
            raise GeocodeError(f"假编码器不认识「{text}」")
        outcome = self.table[text]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _self_check() -> int:
    """不联网、不调大模型的自检：把"模型不听话"的每条分支都走一遍。"""
    from shapely.geometry import box

    failures: list[str] = []

    def check(name: str, condition: bool, extra: str = ""):
        if condition:
            print(f"  ok   {name}")
        else:
            print(f"  FAIL {name} {extra}")
            failures.append(name)

    def geom(side: float):
        """以 (0,0) 为左下角、边长 side 的正方形（面积 = side²）。"""
        return box(0.0, 0.0, side, side)

    def entry(side: float, source=SOURCE_POLYGON, hierarchy=True):
        g = geom(side)
        return (g, GeometryEvidence(source=source, area_deg2=g.area,
                                    hierarchy_confirmed=hierarchy))

    def expand_reply(*texts: str) -> str:
        return json.dumps({"candidates": list(texts)}, ensure_ascii=False)

    # ---------------- 扩展 ----------------
    print("== 扩展 ==")
    llm = _FakeLLM([expand_reply("巴西的亚马孙州（2023-09-20 ~ 2026-09-20）",
                                 "巴西的帕拉州（2023-09-20 ~ 2026-09-20）",
                                 "巴西的马托格罗索州（2023-09-20 ~ 2026-09-20）")])
    texts, err = _expand("亚马孙雨林被烧了", 3, llm)
    check("正常扩展出 3 条", len(texts) == 3 and err is None, f"{texts} / {err}")
    check("扩展时用的是扩展系统提示词",
          "改写" in llm.calls[0][0], llm.calls[0][0][:40])

    # 围栏不该触发重试（重试的信号是 JSON 坏了，不是包了围栏）
    fenced = _FakeLLM(["```json\n" + expand_reply("甲地", "乙地") + "\n```"])
    texts, err = _expand("原文", 3, fenced)
    check("markdown 围栏不算输出不合法",
          texts == ["甲地", "乙地"] and len(fenced.calls) == 1,
          f"{texts} / 调用 {len(fenced.calls)} 次")

    # 与原文相同 / 重复 / 空的候选要被过滤，但不该报销整轮扩展
    dirty = _FakeLLM([expand_reply("原文", "  ", "甲地", "甲地", "乙地")])
    texts, err = _expand("原文", 3, dirty)
    check("过滤掉与原文相同、空白、重复的候选",
          texts == ["甲地", "乙地"], str(texts))

    # 输出坏 JSON → 重试 → 仍坏 → 上报失败原因（由 solve 决定退回原文）
    broken = _FakeLLM(["这不是 JSON", "还是不是 JSON"])
    texts, err = _expand("原文", 3, broken)
    check("连续不合法 → 返回空列表与原因",
          texts == [] and err and "不合法" in err, f"{texts} / {err}")
    check("坏 JSON 会重试一次（共 2 次调用）", len(broken.calls) == 2,
          str(len(broken.calls)))

    # 候选全被过滤 → 也算扩展失败
    all_dup = _FakeLLM([expand_reply("原文", "原文")])
    texts, err = _expand("原文", 2, all_dup)
    check("候选全被过滤 → 扩展失败（交给退回原文的路径）",
          texts == [] and err is not None, f"{texts} / {err}")

    # 调用大模型直接抛错 → 不重试，立刻降级
    boom = _FakeLLM([RuntimeError("网络炸了")])
    texts, err = _expand("原文", 3, boom)
    check("调用失败不重试，直接降级",
          texts == [] and len(boom.calls) == 1 and "网络炸了" in err,
          f"{len(boom.calls)} 次 / {err}")

    check("expand_queries 的对外契约是失败返回空列表",
          expand_queries("原文", 3, _FakeLLM(["坏"])) == [])

    # ---------------- 候选并行：一个失败不影响其他 ----------------
    print("== 候选并行与失败隔离 ==")
    table = {
        "甲地": entry(1.0),
        "乙地": GeocodeError("乙地查无此地"),
        "丙地": entry(1.0),
    }
    geo = _FakeGeocoder(table)
    llm = _FakeLLM([expand_reply("甲地", "乙地", "丙地"),
                    json.dumps({"index": 1, "reason": "甲地最贴近"})])
    result = solve("抽象问题", geocoder=geo, llm=llm, concurrency=3)
    check("三个候选都在（失败的那个也在列表里）", len(result.candidates) == 3,
          str(len(result.candidates)))
    check("失败候选带错误、不带几何",
          result.candidates[1].error is not None and not result.candidates[1].ok)
    check("其余候选照常有几何",
          result.candidates[0].ok and result.candidates[2].ok)
    check("失败的候选被判出局且垫在排名末位",
          result.ranking[-1].index == 2 and result.ranking[-1].score.eliminated,
          f"{[c.index for c in result.ranking]}")
    check("候选顺序与输入一致（不按完成先后）",
          [c.index for c in result.candidates] == [1, 2, 3],
          str([c.index for c in result.candidates]))

    # ---------------- 裁判 ----------------
    print("== 裁判 ==")
    # 甲/丙 面积相同 → 分数并列 → 触不到早停，裁判真的会被调用
    pool = [c for c in result.ranking if not c.score.eliminated]
    check("并列时不会早停", len(pool) == 2,
          str([(c.index, c.score.total) for c in pool]))
    check("裁判选出的是它说的那一个",
          result.judged and result.winner.index == 1 and result.reason == "甲地最贴近",
          f"judged={result.judged} winner={result.winner.index} {result.reason}")

    # 裁判说要第 2 个 → 应该听它的（这一条同时钉住"index 是 pool 的序号不是候选号"）
    llm2 = _FakeLLM([expand_reply("甲地", "乙地", "丙地"),
                     json.dumps({"index": 2, "reason": "第二个更贴切"})])
    result2 = solve("抽象问题", geocoder=_FakeGeocoder(table), llm=llm2)
    check("裁判指定第 2 个时采纳第 2 个",
          result2.judged and result2.winner.index == 3,
          f"winner={result2.winner.index}（应为 3）")

    # 裁判输出不合法 → 回退本地第一名
    bad_judge = _FakeLLM([expand_reply("甲地", "乙地", "丙地"),
                          "not json", "still not json"])
    result3 = solve("抽象问题", geocoder=_FakeGeocoder(table), llm=bad_judge)
    check("裁判连续不合法 → 回退本地第一名",
          not result3.judged and result3.winner is result3.ranking[0],
          f"judged={result3.judged} winner={result3.winner.index}")
    check("回退时理由里说明是本地分数",
          "本地分数" in result3.reason, result3.reason)

    # 序号越界（模型常见的"我数错了"）→ 同样回退，不能崩
    oob = _FakeLLM([expand_reply("甲地", "乙地", "丙地"),
                    json.dumps({"index": 9, "reason": "越界"}),
                    json.dumps({"index": 9, "reason": "还是越界"})])
    result4 = solve("抽象问题", geocoder=_FakeGeocoder(table), llm=oob)
    check("裁判序号越界 → 回退而不是崩", not result4.judged, str(result4.judged))

    # 裁判调用直接抛错 → 回退（这是"不允许裁判翻车拖垮整体"的最后一档）
    judge_boom = _FakeLLM([expand_reply("甲地", "乙地", "丙地"),
                           RuntimeError("裁判超时")])
    result5 = solve("抽象问题", geocoder=_FakeGeocoder(table), llm=judge_boom)
    check("裁判调用抛错 → 回退本地第一名",
          not result5.judged and result5.winner is result5.ranking[0],
          f"judged={result5.judged}")

    # use_judge=False → 一次裁判调用都不发
    no_judge_llm = _FakeLLM([expand_reply("甲地", "乙地", "丙地")])
    result6 = solve("抽象问题", geocoder=_FakeGeocoder(table), llm=no_judge_llm,
                    use_judge=False)
    check("关掉裁判时不再多调一次大模型",
          len(no_judge_llm.calls) == 1 and not result6.judged,
          str(len(no_judge_llm.calls)))

    # ---------------- 本地早停 ----------------
    print("== 本地早停 ==")
    # 甲：真实边界且层级一致（100）；丙：真实边界但层级冲突（-25 → 75）→ 差 25 ≥ 20
    strong = {
        "甲地": entry(1.0, hierarchy=True),
        "丙地": entry(1.0, hierarchy=False),
    }
    strong_llm = _FakeLLM([expand_reply("甲地", "丙地")])
    result7 = solve("抽象问题", geocoder=_FakeGeocoder(strong), llm=strong_llm)
    check("明显领先时跳过裁判（大模型只被调了 1 次：扩展）",
          not result7.judged and len(strong_llm.calls) == 1,
          f"judged={result7.judged} 调用 {len(strong_llm.calls)} 次")
    check("早停时选的是本地第一名", result7.winner.index == 1, str(result7.winner.index))
    check("早停理由里点明是本地分数领先", "本地分数" in result7.reason, result7.reason)

    # 领先但来源是降级 → 不早停（两个都不好时该让模型判断）
    weak = {
        "甲地": entry(1.0, source="point"),
        "丙地": entry(9.0, source="point"),
    }
    weak_llm = _FakeLLM([expand_reply("甲地", "丙地"),
                         json.dumps({"index": 1, "reason": "都降级了，选第一个"})])
    result8 = solve("抽象问题", geocoder=_FakeGeocoder(weak), llm=weak_llm)
    check("降级来源不早停（即使分数领先）", result8.judged,
          f"judged={result8.judged}")

    # ---------------- 扩展失败时的退路 ----------------
    print("== 扩展失败 → 退回原文 ==")
    fallback_geo = _FakeGeocoder({"抽象问题": entry(1.0)})
    fallback_llm = _FakeLLM(["坏 JSON", "还是坏"])
    result9 = solve("抽象问题", geocoder=fallback_geo, llm=fallback_llm)
    check("扩展失败时只用原文跑一次", fallback_geo.seen == ["抽象问题"],
          str(fallback_geo.seen))
    check("记下了扩展降级的原因", result9.expand_error is not None,
          str(result9.expand_error))
    check("退路下仍然给出几何", result9.geometry is not None and result9.winner.index == 1)
    check("只有一个候选时不调裁判", not result9.judged, str(result9.judged))

    # ---------------- 全部失败 ----------------
    print("== 全部失败 ==")
    dead_geo = _FakeGeocoder({"甲地": GeocodeError("查无此地"),
                             "丙地": GeocodeError("也是查无此地")})
    dead_llm = _FakeLLM([expand_reply("甲地", "丙地")])
    try:
        solve("原问题", geocoder=dead_geo, llm=dead_llm)
    except GeocodeError as e:
        check("全部候选失败时抛 GeocodeError", True)
        check("失败详情里列出每个候选的原因",
              "候选 1" in e.detail and "候选 2" in e.detail
              and "甲地" in e.detail and "丙地" in e.detail, e.detail[:200])
    else:
        check("全部候选失败时抛 GeocodeError", False, "没有抛异常")
        check("失败详情里列出每个候选的原因", False, "没有抛异常")

    # 失败原因优先取"数据源已答复"而不是"服务不可用"：两条候选各占一种，
    # 对外的失败原因必须是"查无此地"那条
    mixed_geo = _FakeGeocoder({
        "甲地": GeocodeError("端点全不可达", service_unavailable=True),
        "丙地": GeocodeError("查无此地"),
    })
    try:
        solve("原问题", geocoder=mixed_geo, llm=_FakeLLM([expand_reply("甲地", "丙地")]))
    except GeocodeError as e:
        # 只看"第一条失败原因"那一句，不看后面的逐候选列表（那里两条原因都在，
        # 拿整段做断言等于没测优先级）
        headline = e.detail.split("第一条失败原因：")[1].splitlines()[0].strip()
        check("优先报'查无此地'而不是'服务不可用'",
              headline == "查无此地", f"{headline!r}")
    else:
        check("优先报'查无此地'而不是'服务不可用'", False, "没有抛异常")

    # ---------------- 并发与串行结果一致 ----------------
    print("== 并发与串行结果一致 ==")
    script = [expand_reply("甲地", "乙地", "丙地"),
              json.dumps({"index": 1, "reason": "甲地最贴近"})]
    serial = solve("抽象问题", geocoder=_FakeGeocoder(table),
                   llm=_FakeLLM(list(script)), concurrency=1)
    parallel = solve("抽象问题", geocoder=_FakeGeocoder(table),
                     llm=_FakeLLM(list(script)), concurrency=3)
    check("串行与并行选出的胜者相同",
          serial.winner.index == parallel.winner.index,
          f"{serial.winner.index} vs {parallel.winner.index}")
    check("串行与并行的分数完全相同",
          [round(c.score.total) for c in serial.ranking]
          == [round(c.score.total) for c in parallel.ranking],
          f"{[round(c.score.total) for c in serial.ranking]} vs "
          f"{[round(c.score.total) for c in parallel.ranking]}")
    check("串行与并行的候选顺序相同",
          [c.index for c in serial.candidates] == [c.index for c in parallel.candidates])

    # ---------------- describe ----------------
    print("== 输出摘要 ==")
    text = result.describe()
    check("摘要里有候选清单、评分与胜出",
          "扩展查询" in text and "候选 1" in text and "胜出" in text, text[:200])
    check("摘要里列出扩展与计算耗时",
          "扩展" in text and "裁判" in text and "合计" in text)

    print()
    if failures:
        print(f"agent 自检失败 {len(failures)} 项：{failures}")
        return 1
    print("agent 自检全部通过（离线：假编码器 + 假大模型，未联网）")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(_self_check())
