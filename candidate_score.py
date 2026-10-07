"""
候选答案打分：把"这个结果有多可信"从人看的日志变成一个可比较、可排序的数。

要解决的问题：S27 的 agent 会为一句抽象文本产出多个候选解读，每个解读各跑一遍
管线得到一个几何。此时管线的输出只有一个几何对象——"这是真实多边形还是降级成
了点""事件到底命中没有"全部只 print 到 stdout，程序上无从读取，也就无法比较。

本模块只做**打分**，不采集证据（采集在 `geocoding.collect_evidence`，因为那需要
知道节点树的私有属性）。分层刻意做成这个方向：

    candidate_score（纯函数，不依赖 models/geocoding）
        ↑
    geocoding（采集证据）

反过来让 candidate_score 去 import models 也能跑，但打分规则会被节点结构绑死，
S27 里想对"人工构造的证据"跑打分就得先造一棵假的节点树。

两个层次的判断，顺序不能反：
  1. **硬淘汰**：事件类别被明确要求、但命中 0 条；或者结果根本没有几何（面积为
     0）。这不是"分低"，是"答非所问"，任何加权都救不回来，先出局再说。
     **两条淘汰内部也讲顺序：更具体的理由先判。**"要求了事件但命中 0 条"比"空几何"
     更能解释候选为什么不行，而两者常常同时成立（0 条事件正是几何为空的原因）。
  2. **加权扣分**：几何退化成点/矩形、层级冲突、面积量级异常、降级次数过多。
     这些是"能答但答得糙"，扣分但不淘汰。

每一项扣分都带一句话依据（`ScoreItem.note`），CLI 与 LLM 裁判共用同一份明细——
两处各写一套理由，迟早会出现"分数说 B 好、理由说 A 好"。
"""

from dataclasses import dataclass, replace

# =============================================================================
# 几何来源等级
# =============================================================================
# 从可信到不可信。取值是字符串而不是 Enum：它要进 dataclass、进日志、将来可能
# 进 JSON，用裸字符串省掉一层 .value（与 place_lookup.SourceType 同一取舍）。
SOURCE_POLYGON = "polygon"    # 真实边界（OSM Overpass / 高德 District API）
SOURCE_EVENT = "event"        # 事件点聚合（S23）——同样是真实几何，不打折
SOURCE_LANDUSE = "landuse"    # 用地类型多边形（S31，OSM landuse=*）——真实地块
SOURCE_POINT = "point"        # 降级到高德中心点：只知道"在哪"，不知道"到哪"
SOURCE_BOUNDS = "bounds"      # LLM 自己编的矩形：兜底，可信度最低

# 来源 → (扣分, 一句话依据)。polygon / event / landuse 是"正确答案该有的样子"，不扣分。
SOURCE_PENALTY: dict[str, tuple[float, str]] = {
    SOURCE_POLYGON: (0.0, "几何是数据源返回的真实边界"),
    SOURCE_EVENT: (0.0, "几何由事件点聚合而成，基础地点边界为真"),
    SOURCE_LANDUSE: (0.0, "几何是 OSM 里真实存在的用地地块，不是合成图形"),
    SOURCE_POINT: (-35.0, "几何降级成了中心点：只知道位置，不知道范围"),
    SOURCE_BOUNDS: (-55.0, "几何是大模型自己给的矩形，不是查到的边界"),
}

# 起始分。100 分制便于人手看，且留出"多项同时出错也不会变负数"的余量。
BASE_SCORE = 100.0

# 层级一致性：候选带的国家/省份与查找实际命中不符 → 扣分。
# "未知"（数据源没能核验）扣得比"冲突"少：没核验不等于冲突，重罚未知会把
# "层级没写清楚但答案正确"的候选冤杀。
PENALTY_HIERARCHY_MISMATCH = -25.0
PENALTY_HIERARCHY_UNKNOWN = -10.0

# 面积量级异常：偏离候选中位数超过这个倍数（两个方向）就扣分。
# 取 8 倍：一个候选的面积是其余候选的 8 倍以上，几乎一定是"理解成了另一个
# 尺度"（问一个区却答成了整个省），而不是合理的差异。
AREA_ANOMALY_FACTOR = 8.0
PENALTY_AREA_ANOMALY = -20.0

# 每一处降级（某个 NamedPlace 没能拿到真实边界）扣的分，以及封顶值。
# 与"来源等级"扣分**测的不是同一件事**：来源等级看的是"结果几何本身有多可信"，
# 这里看的是"管线上回退了多少次"。三个基础地点里两个降级，比只有一个降级更差，
# 而两者的"最弱来源"是一样的。
PENALTY_PER_DEGRADATION = -5.0
MAX_DEGRADATION_PENALTY = -20.0


# =============================================================================
# 证据与分数
# =============================================================================
@dataclass(frozen=True)
class GeometryEvidence:
    """一个候选答案的可比较证据。

    这些字段都是**程序上能取到的事实**，不是推测：采集见
    `geocoding.collect_evidence`。任何需要"解析 stdout"才能拿到的信息都不算证据。
    """

    source: str
    # 结果几何的面积（平方度）。0 或负数说明几何退化，会被硬淘汰。
    area_deg2: float
    # 候选集合的面积中位数，由 score_candidates 统一填；单独打分时为 None
    # （此时不做面积合理性判断——只有一个候选就没有"相对"可言）。
    area_median_deg2: float | None = None

    # ---- 层级：候选声称的 vs 查找实际命中的 ----
    requested_country: str | None = None
    requested_region: str | None = None
    # True=已核验且一致，False=已核验且冲突，None=数据源没能核验（不等于冲突）
    hierarchy_confirmed: bool | None = None

    # ---- 事件维度 ----
    # 空元组表示"这个候选不涉及事件"，不是"要求了但没命中"。
    # 这个区分决定了会不会硬淘汰，所以必须用空元组而不是 None 之外的任何东西。
    event_categories: tuple[str, ...] = ()
    # 命中事件条数。0 且 event_categories 非空 → 硬淘汰。
    event_count: int | None = None
    event_pieces: int | None = None
    event_window: str | None = None
    # 数据源是否覆盖了整个时间窗。False 表示"早期那段没有记录"，属于数据源的
    # 覆盖边界而不是候选的错，因此**只提示不扣分**（见 score_candidate 的说明）。
    event_window_covered: bool | None = None

    # ---- 降级与兜底 ----
    degradations: int = 0


@dataclass(frozen=True)
class ScoreItem:
    """一个评分维度的一句话明细。"""

    dimension: str
    delta: float
    note: str

    def describe(self) -> str:
        sign = f"{self.delta:+.0f}" if self.delta else " 0"
        return f"{sign} {self.dimension}：{self.note}"


@dataclass(frozen=True)
class CandidateScore:
    """一个候选的分数与明细。"""

    total: float
    eliminated: bool
    elimination_reason: str | None
    items: tuple[ScoreItem, ...]

    @property
    def sort_key(self) -> tuple:
        """排序键：出局的永远排在最后，其余按分数降序。

        用属性而不是让调用方自己写 `sorted(key=...)`：淘汰位与分数的关系
        （淘汰不参与比大小）只在这里定义一次，S27/S28 各写一遍迟早不一致。
        """
        return (0 if self.eliminated else 1, self.total)

    def describe(self) -> str:
        if self.eliminated:
            return f"[出局] {self.elimination_reason}"
        return f"{self.total:.0f} 分｜" + "；".join(i.describe() for i in self.items)


# =============================================================================
# 打分
# =============================================================================
def score_candidate(evidence: GeometryEvidence) -> CandidateScore:
    """给一个候选打分。硬淘汰在前，加权扣分在后。

    Returns:
        CandidateScore。被淘汰时 total=0，明细里只有淘汰理由——淘汰的候选不再
        逐项扣分，那些维度对它已经没有意义（"面积不合理"不会让一个答非所问的
        候选变得更有用）。
    """
    # ---- 硬淘汰 ----
    # 顺序是刻意的：**更具体的理由先判**。"要求了事件类别但命中 0 条"比"结果空几何"
    # 更能解释候选为什么不行，而两者常常同时成立（0 条事件正是几何为空的原因）。
    if evidence.event_categories and evidence.event_count == 0:
        categories = "、".join(evidence.event_categories)
        window = f"（{evidence.event_window}）" if evidence.event_window else ""
        return CandidateScore(
            total=0.0, eliminated=True,
            elimination_reason=f"明确要求了事件类别 {categories}，"
                               f"但该时间窗内命中 0 条{window}",
            items=(),
        )

    # 面积不为正就出局，**不看有没有事件类别**。事件候选也可能根本没有几何：
    # 基础地点压根没查到、事件查询也就没发出去（此时 `event_count` 是 None 而不是
    # 0），上面那条不成立，但"没有可输出的范围"依然成立。早先把这条限定成
    # `and not event_categories`，结果这类候选一个扣分项都没有、拿到 90 分还不淘汰。
    if evidence.area_deg2 <= 0:
        return CandidateScore(
            total=0.0, eliminated=True,
            elimination_reason="结果是空几何，没有可输出的范围", items=(),
        )

    items: list[ScoreItem] = []

    # ---- 几何来源 ----
    penalty, note = SOURCE_PENALTY.get(
        evidence.source, (-30.0, f"未知的几何来源 {evidence.source!r}"))
    items.append(ScoreItem("几何来源", penalty, note))

    # ---- 层级一致性 ----
    if evidence.hierarchy_confirmed is True:
        items.append(ScoreItem("层级一致性", 0.0, "候选声称的国家/省份与查找命中一致"))
    elif evidence.hierarchy_confirmed is False:
        requested = "／".join(
            v for v in (evidence.requested_country, evidence.requested_region) if v
        ) or "声明的层级"
        items.append(ScoreItem("层级一致性", PENALTY_HIERARCHY_MISMATCH,
                               f"候选声称在「{requested}」，但查找命中的位置不在其中"))
    else:
        # None：数据源没能核验。可能是候选没写层级，也可能是数据源没给出可核验的
        # 信息——两种都不是"冲突"，所以扣得轻，且说明里要讲清楚是"没核验"。
        note = ("候选未声明国家/省份，无从核对"
                if not (evidence.requested_country or evidence.requested_region)
                else "候选声明了层级，但数据源没给出可核验的信息")
        items.append(ScoreItem("层级一致性", PENALTY_HIERARCHY_UNKNOWN, note))

    # ---- 面积合理性 ----
    median = evidence.area_median_deg2
    if median and median > 0 and evidence.area_deg2 > 0:
        ratio = evidence.area_deg2 / median
        if ratio > AREA_ANOMALY_FACTOR or ratio < 1.0 / AREA_ANOMALY_FACTOR:
            items.append(ScoreItem(
                "面积合理性", PENALTY_AREA_ANOMALY,
                f"面积是候选中位数的 {ratio:.1f} 倍，量级与其他候选明显不同",
            ))
        else:
            items.append(ScoreItem(
                "面积合理性", 0.0,
                f"面积是候选中位数的 {ratio:.1f} 倍，量级正常",
            ))
    else:
        items.append(ScoreItem("面积合理性", 0.0,
                               "只有一个候选或中位数不可用，不做量级比较"))

    # ---- 降级次数 ----
    if evidence.degradations > 0:
        penalty = max(PENALTY_PER_DEGRADATION * evidence.degradations,
                      MAX_DEGRADATION_PENALTY)
        items.append(ScoreItem("降级次数", penalty,
                               f"管线上有 {evidence.degradations} 处从真实边界回退"))
    else:
        items.append(ScoreItem("降级次数", 0.0, "没有发生降级"))

    # ---- 事件证据 ----
    if evidence.event_categories:
        categories = "、".join(evidence.event_categories)
        pieces = "" if evidence.event_pieces is None else f"，聚成 {evidence.event_pieces} 块"
        items.append(ScoreItem("事件证据", 0.0,
                               f"{categories} 命中 {evidence.event_count} 条{pieces}"))
        if evidence.event_window_covered is False:
            # **刻意不扣分**：这是数据源的存档覆盖边界（EONET 的亚马逊野火最早
            # 只到 2024-05-27），不是候选的错。扣分会让 agent 倾向选择"时间窗更窄
            # 所以看起来证据更完整"的候选，那恰好是更差的解读。
            items.append(ScoreItem("时间窗覆盖", 0.0,
                                   "数据源在该时间窗的早期没有记录，"
                                   "属于存档边界而非候选的问题（不扣分）"))

    total = max(0.0, BASE_SCORE + sum(item.delta for item in items))
    return CandidateScore(total=total, eliminated=False,
                          elimination_reason=None, items=tuple(items))


def score_candidates(evidences) -> list[CandidateScore]:
    """给一组候选打分，并统一填上"面积中位数"。

    面积合理性必须相对**这一组**候选来判断，所以中位数只能在这里算：单个候选
    自己不知道自己算大还是算小。

    Returns:
        与输入**顺序一致**的分数列表。刻意不在这里排序——调用方要按序号把分数
        与自己的候选对象对应起来（S27 的线程池返回顺序不保证与提交顺序相同，
        排序越早做，错位的风险越大）。排序请用 `CandidateScore.sort_key`。
    """
    evidences = list(evidences)
    positive = sorted(e.area_deg2 for e in evidences if e.area_deg2 > 0)
    median: float | None = None
    if positive:
        mid = len(positive) // 2
        median = (positive[mid] if len(positive) % 2
                  else (positive[mid - 1] + positive[mid]) / 2.0)

    return [score_candidate(replace(e, area_median_deg2=median)) for e in evidences]


# =============================================================================
# 离线自检（python candidate_score.py）
# =============================================================================
def _self_check() -> int:
    """不联网、不依赖任何其他模块的自检：打分规则本身是否按设计工作。

    用**人工构造**的证据，不造节点树——这也是把打分与采集拆成两个模块的收益：
    规则可以脱离真实管线单独验证。
    """
    import sys

    failures: list[str] = []

    def check(name: str, condition: bool, extra: str = ""):
        if condition:
            print(f"  [通过] {name}")
        else:
            print(f"  [失败] {name} {extra}")
            failures.append(name)

    def items_of(score) -> dict[str, float]:
        return {item.dimension: item.delta for item in score.items}

    # ---------------- 三组来源的证据 ----------------
    print("== 几何来源三档 ==")
    polygon = GeometryEvidence(source=SOURCE_POLYGON, area_deg2=10.0,
                               hierarchy_confirmed=True)
    point = GeometryEvidence(source=SOURCE_POINT, area_deg2=10.0,
                             hierarchy_confirmed=True, degradations=1)
    bounds = GeometryEvidence(source=SOURCE_BOUNDS, area_deg2=10.0,
                              hierarchy_confirmed=True, degradations=1)

    s_polygon, s_point, s_bounds = score_candidates([polygon, point, bounds])
    check("真实多边形得满分", s_polygon.total == BASE_SCORE, str(s_polygon.total))
    check("多边形 > 中心点 > 矩形兜底",
          s_polygon.total > s_point.total > s_bounds.total,
          f"{s_polygon.total} / {s_point.total} / {s_bounds.total}")
    check("三者都没被淘汰",
          not any(s.eliminated for s in (s_polygon, s_point, s_bounds)))
    check("中心点来源扣分落在'几何来源'维度",
          items_of(s_point)["几何来源"] == SOURCE_PENALTY[SOURCE_POINT][0],
          str(items_of(s_point)))
    check("矩形兜底比中心点扣得更多",
          items_of(s_bounds)["几何来源"] < items_of(s_point)["几何来源"],
          f"{items_of(s_bounds)['几何来源']} vs {items_of(s_point)['几何来源']}")
    check("事件聚合来源不打折",
          score_candidate(GeometryEvidence(source=SOURCE_EVENT, area_deg2=10.0,
                                           hierarchy_confirmed=True)).total
          == BASE_SCORE)
    check("明细里每一项都有非空说明",
          all(item.note for item in s_point.items), str(s_point.items))
    check("describe 输出分数与明细",
          "分｜" in s_point.describe() and "几何来源" in s_point.describe(),
          s_point.describe())

    # ---------------- 硬淘汰 ----------------
    print("== 硬淘汰：事件命中 0 条 ==")
    zero = GeometryEvidence(source=SOURCE_EVENT, area_deg2=0.0,
                            event_categories=("wildfires",), event_count=0,
                            event_pieces=0, event_window="2023-09-20 ~ 2026-09-20")
    s_zero = score_candidate(zero)
    check("事件命中 0 条 → 出局", s_zero.eliminated, str(s_zero.total))
    check("出局分数归零", s_zero.total == 0.0, str(s_zero.total))
    check("淘汰理由里含类别与'0 条'",
          "wildfires" in s_zero.elimination_reason
          and "0 条" in s_zero.elimination_reason, str(s_zero.elimination_reason))
    check("出局后不再逐项扣分（明细为空）", s_zero.items == (), str(s_zero.items))
    check("出局时 describe 只讲淘汰理由",
          s_zero.describe().startswith("[出局]"), s_zero.describe())

    hit = GeometryEvidence(source=SOURCE_EVENT, area_deg2=10.0,
                           event_categories=("wildfires",), event_count=184,
                           event_pieces=11, event_window="2024-01-01 ~ 2026-09-20")
    check("事件命中 184 条 → 不出局（只扣层级未核验那一档）",
          not score_candidate(hit).eliminated
          and score_candidate(hit).total == BASE_SCORE + PENALTY_HIERARCHY_UNKNOWN,
          str(score_candidate(hit).total))

    no_event = GeometryEvidence(source=SOURCE_POLYGON, area_deg2=10.0,
                                event_count=0, hierarchy_confirmed=True)
    check("没要求事件的候选命中 0 条也不出局",
          not score_candidate(no_event).eliminated, str(no_event.event_categories))

    empty_geom = GeometryEvidence(source=SOURCE_POLYGON, area_deg2=0.0,
                                  hierarchy_confirmed=True)
    s_empty = score_candidate(empty_geom)
    check("空几何 → 出局", s_empty.eliminated, str(s_empty.total))
    check("空几何的出局理由就是空几何",
          "空几何" in s_empty.elimination_reason, str(s_empty.elimination_reason))

    # 事件候选 + 空几何 + 条数未知（基础地点没查到，事件查询根本没发出去）：
    # 这是真实管线上"抽象查询落空"的形态，必须出局，不能因为"要求了事件"就放行
    no_base = GeometryEvidence(source=SOURCE_POLYGON, area_deg2=0.0,
                               event_categories=("wildfires",), event_count=None,
                               event_window="2023-09-20 ~ 2026-09-20")
    s_no_base = score_candidate(no_base)
    check("事件候选但基础地点没查到（条数未知）→ 仍然出局",
          s_no_base.eliminated, str(s_no_base.total))
    check("这种情况下理由是空几何，不是'命中 0 条'",
          "空几何" in s_no_base.elimination_reason
          and "0 条" not in s_no_base.elimination_reason,
          str(s_no_base.elimination_reason))

    # ---------------- 层级 ----------------
    print("== 层级一致性 ==")
    consistent = score_candidate(GeometryEvidence(
        source=SOURCE_POLYGON, area_deg2=10.0, hierarchy_confirmed=True))
    mismatch = score_candidate(GeometryEvidence(
        source=SOURCE_POLYGON, area_deg2=10.0, hierarchy_confirmed=False,
        requested_country="Brazil"))
    unknown = score_candidate(GeometryEvidence(
        source=SOURCE_POLYGON, area_deg2=10.0, hierarchy_confirmed=None))
    check("一致 > 未知 > 冲突",
          consistent.total > unknown.total > mismatch.total,
          f"{consistent.total} / {unknown.total} / {mismatch.total}")
    check("一致不扣分", items_of(consistent)["层级一致性"] == 0.0,
          str(items_of(consistent)))
    check("冲突扣 PENALTY_HIERARCHY_MISMATCH",
          items_of(mismatch)["层级一致性"] == PENALTY_HIERARCHY_MISMATCH,
          str(items_of(mismatch)))
    check("未知扣得比冲突轻",
          abs(items_of(unknown)["层级一致性"]) < abs(items_of(mismatch)["层级一致性"]),
          f"{items_of(unknown)['层级一致性']} vs {items_of(mismatch)['层级一致性']}")
    check("冲突理由里点出候选声称的层级",
          "Brazil" in [i for i in mismatch.items
                       if i.dimension == "层级一致性"][0].note,
          [i for i in mismatch.items if i.dimension == "层级一致性"][0].note)
    check("未声明层级时理由说明是'无从核对'",
          "无从核对" in [i for i in unknown.items
                         if i.dimension == "层级一致性"][0].note,
          [i for i in unknown.items if i.dimension == "层级一致性"][0].note)

    # ---------------- 面积合理性 ----------------
    print("== 面积合理性（相对中位数）==")
    def areas(*values):
        return score_candidates([
            GeometryEvidence(source=SOURCE_POLYGON, area_deg2=v,
                             hierarchy_confirmed=True) for v in values])

    normal = areas(8.0, 10.0, 12.0)
    check("三个量级相近的候选都不扣面积分",
          all(items_of(s)["面积合理性"] == 0.0 for s in normal),
          str([items_of(s)["面积合理性"] for s in normal]))

    # 用阈值两侧的两个点同时把"中位数"和"倍数门槛"钉住：
    # [10,10,10,80] 的中位数是 10，80/10 = 8.0 不超门槛；81/10 = 8.1 超。
    at_threshold = areas(10.0, 10.0, 10.0, 80.0)
    over_threshold = areas(10.0, 10.0, 10.0, 81.0)
    check("倍数正好等于门槛时不扣分（中位数=10 被钉住）",
          items_of(at_threshold[3])["面积合理性"] == 0.0,
          str(items_of(at_threshold[3])))
    check("刚过门槛就扣分",
          items_of(over_threshold[3])["面积合理性"] == PENALTY_AREA_ANOMALY,
          str(items_of(over_threshold[3])))
    check("同组里的正常候选不受影响",
          all(items_of(s)["面积合理性"] == 0.0 for s in over_threshold[:3]),
          str([items_of(s)["面积合理性"] for s in over_threshold[:3]]))
    check("异常理由里给出倍数",
          "倍" in [i for i in over_threshold[3].items
                   if i.dimension == "面积合理性"][0].note,
          [i for i in over_threshold[3].items
           if i.dimension == "面积合理性"][0].note)

    tiny = areas(10.0, 10.0, 10.0, 0.01)
    check("面积远小于中位数同样被扣分",
          items_of(tiny[3])["面积合理性"] == PENALTY_AREA_ANOMALY,
          str(items_of(tiny[3])))
    # 偶数个候选时中位数必须取"中间两个的平均值"，而不是任取其一。
    # 构造中间两个为 10 与 11 的两组，让结论在三种取法下互不相同：
    #   中位数 10.5（正确）→ A 不扣、B 扣
    #   中位数 10（错取偏小）→ A 也扣   → 被 A 的断言排除
    #   中位数 11（错取偏大）→ B 不扣   → 被 B 的断言排除
    # 两条断言合起来唯一确定"取平均"这一种实现。
    check("中位数取平均：81/10.5=7.7 未过门槛 → 不扣",
          items_of(areas(1.0, 10.0, 11.0, 81.0)[3])["面积合理性"] == 0.0,
          str(items_of(areas(1.0, 10.0, 11.0, 81.0)[3])))
    check("中位数取平均：87/10.5=8.29 过门槛 → 扣",
          items_of(areas(1.0, 10.0, 11.0, 87.0)[3])["面积合理性"]
          == PENALTY_AREA_ANOMALY,
          str(items_of(areas(1.0, 10.0, 11.0, 87.0)[3])))
    check("单个候选不做面积比较（没有'相对'可言）",
          items_of(score_candidate(
              GeometryEvidence(source=SOURCE_POLYGON, area_deg2=10.0)))["面积合理性"] == 0.0)

    # ---------------- 降级次数 ----------------
    print("== 降级次数 ==")
    def degradations(n):
        return score_candidate(GeometryEvidence(
            source=SOURCE_POLYGON, area_deg2=10.0, hierarchy_confirmed=True,
            degradations=n))

    d0, d1, d2, d9 = degradations(0), degradations(1), degradations(2), degradations(9)
    check("无降级不扣分", items_of(d0)["降级次数"] == 0.0, str(items_of(d0)))
    check("两处降级比一处扣得多",
          d2.total < d1.total < d0.total, f"{d0.total} / {d1.total} / {d2.total}")
    check("降级扣分有封顶",
          items_of(d9)["降级次数"] == MAX_DEGRADATION_PENALTY,
          str(items_of(d9)["降级次数"]))

    # ---------------- 时间窗覆盖：只提示不扣分 ----------------
    print("== 时间窗覆盖缺口只提示不扣分 ==")
    covered = score_candidate(GeometryEvidence(
        source=SOURCE_EVENT, area_deg2=10.0, hierarchy_confirmed=True,
        event_categories=("wildfires",), event_count=184, event_pieces=11,
        event_window_covered=True))
    uncovered = score_candidate(GeometryEvidence(
        source=SOURCE_EVENT, area_deg2=10.0, hierarchy_confirmed=True,
        event_categories=("wildfires",), event_count=184, event_pieces=11,
        event_window_covered=False))
    check("覆盖与不覆盖的分数相同", covered.total == uncovered.total,
          f"{covered.total} vs {uncovered.total}")
    check("不覆盖时多出一条说明",
          len(uncovered.items) == len(covered.items) + 1, str(len(uncovered.items)))
    check("该说明明确写了'不扣分'",
          "不扣分" in uncovered.describe() and "存档边界" in uncovered.describe(),
          uncovered.describe())

    # ---------------- 排序 ----------------
    print("== 排序 ==")
    check("出局的排序键排在正常候选之后",
          s_zero.sort_key < s_polygon.sort_key,
          f"{s_zero.sort_key} vs {s_polygon.sort_key}")
    check("正常候选按分数降序排列",
          sorted([s_point, s_polygon, s_bounds], key=lambda s: s.sort_key,
                 reverse=True)[0] is s_polygon)
    check("分数不会变成负数", score_candidates([
        GeometryEvidence(source=SOURCE_BOUNDS, area_deg2=10.0,
                         hierarchy_confirmed=False, degradations=99)])[0].total >= 0.0)

    print()
    if failures:
        print(f"自检失败 {len(failures)} 项：{failures}")
        return 1
    print("自检全部通过（纯函数，无网络、无外部数据）")
    return 0


if __name__ == "__main__":
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.exit(_self_check())
