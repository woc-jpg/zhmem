"""A hand-built Chinese memory corpus with gold answers.

Why hand-built rather than a public benchmark: the numbers this project reports
have to survive the question "how did you measure that?". DuReader-retrieval and
C-MTEB exist and are large, but they are paragraph retrieval over web documents —
a different shape from twenty short personal memories per person, and their gold
labels say nothing about the failure modes that actually matter here (a query that
must return *nothing*, a pronoun that needs the previous turn, two clues that must
both be used).

Every query below was written by hand against memories written by hand, so the
gold answer is not an approximation. Six personas, four query families, and the
families are scored separately: an aggregate number hides the fact that a change
helped paraphrases and broke the negatives.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CATEGORIES",
    "MEMORIES",
    "PERSONAS",
    "QUERIES",
    "Query",
    "build",
]

# (namespace, key, body, kind, salience)
MEMORIES: list[tuple[str, str, str, str, float]] = [
    # -- 妈妈：健康与口味 ---------------------------------------------------
    ("mum", "m-puer", "喜欢喝普洱，不加糖，下午三点左右", "preference", 0.7),
    ("mum", "m-knee", "膝盖不太好，阴雨天会疼", "fact", 0.9),
    ("mum", "m-exhibit", "上周去看了敦煌数字艺术展，回来念叨了好几天", "event", 0.6),
    ("mum", "m-insomnia", "最近睡不着，常常到半夜", "fact", 0.8),
    ("mum", "m-spouse", "老伴陪我去医院看过一次", "event", 0.5),
    ("mum", "m-grandson", "女儿家的孩子今年上小学二年级", "fact", 0.6),
    ("mum", "m-walk", "每天傍晚去公园走一走，大概四十分钟", "habit", 0.5),
    ("mum", "m-salt", "吃菜偏淡，太咸的会胃里难受", "preference", 0.6),
    ("mum", "m-glasses", "老花镜放在客厅电视柜第二个抽屉", "fact", 0.4),
    ("mum", "m-call", "每周日晚上固定给老家打电话", "habit", 0.5),
    # -- 爸爸：工作与手艺 ---------------------------------------------------
    ("dad", "d-retire", "明年三月退休，单位已经通知过了", "fact", 0.8),
    ("dad", "d-fish", "喜欢钓鱼，常去城东那个水库", "habit", 0.6),
    ("dad", "d-car", "那辆旧车开了十一年，空调不太制冷", "fact", 0.5),
    ("dad", "d-back", "腰椎间盘突出，不能搬重物", "fact", 0.9),
    ("dad", "d-tea", "喝浓茶会心慌，晚上只喝白开水", "preference", 0.6),
    ("dad", "d-photo", "年轻时在照相馆工作过，会自己洗照片", "event", 0.5),
    ("dad", "d-checkup", "体检报告说血脂偏高，医生让少油少盐", "fact", 0.8),
    ("dad", "d-tool", "工具箱在阳台最里面那层架子上", "fact", 0.4),
    ("dad", "d-friend", "老同事姓周，住同一个小区", "fact", 0.4),
    ("dad", "d-birthday", "生日是农历八月十六", "fact", 0.5),
    # -- 女儿小雨：学校与朋友 -------------------------------------------------
    ("kid", "k-exam", "下周三有数学期中考试", "event", 0.8),
    ("kid", "k-desk", "同桌叫李梦，两个人一起坐最后一排", "fact", 0.5),
    ("kid", "k-swim", "在学游泳，蛙泳还不太会换气", "habit", 0.6),
    ("kid", "k-piano", "钢琴课调到周六上午十点", "event", 0.7),
    ("kid", "k-milk", "不喜欢喝牛奶，喜欢酸奶", "preference", 0.5),
    ("kid", "k-cat", "家里那只猫叫煤球，是去年冬天捡的", "fact", 0.5),
    ("kid", "k-draw", "画画得过区里二等奖，画的是一只橘猫", "event", 0.6),
    ("kid", "k-bus", "坐 17 路公交到学校，四站路", "fact", 0.4),
    ("kid", "k-book", "最近在看《夏洛的网》，看到第七章", "fact", 0.4),
    ("kid", "k-sleep", "晚上九点半必须上床", "habit", 0.5),
    # -- 我自己：工作与技术 ---------------------------------------------------
    ("me", "me-db", "项目里用 sqlite3 存本地缓存，没上 postgres", "fact", 0.5),
    ("me", "me-standup", "每天早上十点站会，十分钟结束", "habit", 0.4),
    ("me", "me-review", "周五下午集中过一遍代码评审", "habit", 0.5),
    ("me", "me-deploy", "发布只能走周二和周四，避开月底", "fact", 0.7),
    ("me", "me-coffee", "喝美式不加糖，一天两杯就够", "preference", 0.4),
    ("me", "me-bug", "那个内存泄漏是连接没关，加了 with 就好了", "event", 0.6),
    ("me", "me-gym", "周三晚上去健身房，主要练背", "habit", 0.4),
    ("me", "me-trip", "十月中旬要去杭州出差三天", "event", 0.7),
    ("me", "me-landlord", "房东姓吴，房租每月十号前交", "fact", 0.5),
    ("me", "me-migrate", "数据库迁移脚本放在 scripts 目录，用 alembic 管", "fact", 0.4),
    # -- 奶奶：老事与吃食 -----------------------------------------------------
    ("gran", "g-dumpling", "包的饺子是白菜猪肉馅，一次包一大盆", "habit", 0.6),
    ("gran", "g-hometown", "老家在山东潍坊，小时候住四合院", "fact", 0.5),
    ("gran", "g-sugar", "有糖尿病，甜的基本不碰", "fact", 0.9),
    ("gran", "g-grandpa", "爷爷走了八年了，忌日在腊月", "event", 0.6),
    ("gran", "g-radish", "腌萝卜要放花椒和冰糖", "preference", 0.4),
    ("gran", "g-ear", "耳朵有点背，说话得大点声", "fact", 0.7),
    ("gran", "g-photo", "柜子里的相册有 1968 年在照相馆拍的那张", "fact", 0.4),
    ("gran", "g-walk", "住在三楼，没有电梯，上下楼要歇一会儿", "fact", 0.6),
    ("gran", "g-radio", "收音机每天下午听评书", "habit", 0.4),
    ("gran", "g-sleep", "睡得早，八点就困", "habit", 0.4),
    # -- 同事老陈：工作情况 ---------------------------------------------------
    ("chen", "c-handover", "下个月转岗，手上三个服务要交接出去", "event", 0.8),
    ("chen", "c-kid", "孩子今年高考，家里气氛很紧张", "fact", 0.7),
    ("chen", "c-algo", "算法那块是他在负责，排序和召回都归他", "fact", 0.6),
    ("chen", "c-motor", "骑摩托车上班，下雨就改地铁", "habit", 0.4),
    ("chen", "c-review", "代码评审特别仔细，一个命名能问三轮", "fact", 0.5),
    ("chen", "c-sick", "上个月请了两周病假，做了个小手术", "event", 0.6),
    ("chen", "c-lunch", "中午一般自己带饭，很少点外卖", "habit", 0.4),
    ("chen", "c-badminton", "周五晚上打羽毛球，固定那四个人", "habit", 0.5),
    ("chen", "c-cert", "在准备一个云厂商的架构师认证", "event", 0.6),
    ("chen", "c-desk", "座位在靠窗第三排，桌上有一盆绿萝", "fact", 0.3),
]

PERSONAS = ("mum", "dad", "kid", "me", "gran", "chen")

CATEGORIES = ("paraphrase", "context", "switch", "multihop", "negative")


@dataclass(frozen=True)
class Query:
    """One graded query.

    `gold` is a tuple of memory keys in rank-independent order; for `negative`
    it is empty and the only correct answer is nothing at all. `recent` carries
    the previous turns for `context` queries — the utterance alone is a pronoun
    and genuinely cannot be answered without them.
    """

    namespace: str
    query: str
    gold: tuple[str, ...]
    category: str
    recent: tuple[str, ...] = ()

    @property
    def is_negative(self) -> bool:
        return self.category == "negative"


# Paraphrase: same meaning, different words. These are what a synonym table and a
# bigram index buy you, and the reason folding must apply to the index as well.
_PARAPHRASE: list[Query] = [
    Query("mum", "老人家失眠有什么办法", ("m-insomnia",), "paraphrase"),
    Query("mum", "伴侣陪同就诊那次", ("m-spouse",), "paraphrase"),
    Query("mum", "她膝盖疼得厉害", ("m-knee",), "paraphrase"),
    Query("mum", "妈平时爱喝什么茶", ("m-puer",), "paraphrase"),
    Query("mum", "出去遛弯一般多久", ("m-walk",), "paraphrase"),
    Query("mum", "口味重的东西能吃吗", ("m-salt",), "paraphrase"),
    Query("dad", "腰不好能不能搬东西", ("d-back",), "paraphrase"),
    Query("dad", "他喝浓茶会怎么样", ("d-tea",), "paraphrase"),
    Query("dad", "体检结果有什么问题", ("d-checkup",), "paraphrase"),
    Query("dad", "什么时候退休", ("d-retire",), "paraphrase"),
    Query("kid", "牛奶她爱喝吗", ("k-milk",), "paraphrase"),
    Query("kid", "数学考试是哪天", ("k-exam",), "paraphrase"),
    Query("kid", "游泳学得怎么样", ("k-swim",), "paraphrase"),
    Query("me", "发布有什么限制", ("me-deploy",), "paraphrase"),
    Query("me", "咖啡怎么喝", ("me-coffee",), "paraphrase"),
    Query("me", "那个内存泄漏是怎么修的", ("me-bug",), "paraphrase"),
    Query("gran", "血糖高不能吃甜的吧", ("g-sugar",), "paraphrase"),
    Query("gran", "腌萝卜放什么", ("g-radish",), "paraphrase"),
    Query("gran", "她耳朵听得清吗", ("g-ear",), "paraphrase"),
    Query("chen", "他孩子今年考试", ("c-kid",), "paraphrase"),
    Query("chen", "羽毛球是周几", ("c-badminton",), "paraphrase"),
    Query("chen", "手术之后请了多久假", ("c-sick",), "paraphrase"),
]

# Context: the utterance alone is a pronoun or an ellipsis. Only the previous
# turn names the topic. Paired with the `switch` family below, this is the control
# that punishes "just concatenate the previous turn": concatenation helps here but
# drifts on a topic switch, and only a set carrying BOTH can tell the two apart.
_CONTEXT: list[Query] = [
    Query("mum", "那个展览在哪看的", ("m-exhibit",), "context", ("你还记得敦煌那个展吗",)),
    Query("mum", "那次疼了几天", ("m-knee",), "context", ("我妈膝盖怎么样",)),
    Query("mum", "他叫什么名字来着", ("m-grandson",), "context", ("女儿家孩子多大了",)),
    Query("mum", "放在哪儿了", ("m-glasses",), "context", ("老花镜找不到了",)),
    Query("dad", "那个水库远不远", ("d-fish",), "context", ("你爸现在还去钓鱼吗",)),
    Query("dad", "开了多少年了", ("d-car",), "context", ("家里那辆旧车怎么样",)),
    Query("kid", "她叫什么", ("k-desk",), "context", ("小雨同桌是谁",)),
    Query("kid", "几点上课来着", ("k-piano",), "context", ("钢琴课改时间了吗",)),
    Query("me", "在哪个目录", ("me-migrate",), "context", ("数据库迁移脚本放哪",)),
    Query("me", "练哪部分", ("me-gym",), "context", ("周三晚上去健身房吗",)),
    Query("gran", "她老家是哪的", ("g-hometown",), "context", ("奶奶小时候住什么地方",)),
    Query("gran", "要走几层楼", ("g-walk",), "context", ("奶奶家有没有电梯",)),
    Query("chen", "他负责哪块", ("c-algo",), "context", ("老陈在组里做什么",)),
    Query("chen", "他中午吃什么", ("c-lunch",), "context", ("老陈平时点外卖吗",)),
]

# Topic switch: the current utterance names a NEW topic, the previous turn is
# unrelated. This is the control that context queries alone cannot provide —
# without it, concatenating every previous turn also scores perfectly and the
# evaluation cannot see that it drifts back to the earlier subject.
_SWITCH: list[Query] = [
    Query("mum", "我膝盖疼怎么办", ("m-knee",), "switch", ("今天下午喝什么好",)),
    Query("mum", "膝盖不舒服", ("m-knee",), "switch", ("敦煌展好看吗",)),
    Query("dad", "腰又疼了", ("d-back",), "switch", ("水库钓鱼怎么样",)),
    Query("kid", "钢琴课几点", ("k-piano",), "switch", ("数学考试难不难",)),
    Query("me", "咖啡喝什么", ("me-coffee",), "switch", ("数据库迁移完了吗",)),
    Query("chen", "羽毛球周几", ("c-badminton",), "switch", ("转岗交接顺利吗",)),
]

# Multi-hop: two clues in one sentence, and the gold memory is the only one that
# contains both. A ranker that matches on one clue lands on a decoy that the
# other persona's memories provide — that is the point.
_MULTIHOP: list[Query] = [
    Query("mum", "妈膝盖疼还睡不着", ("m-knee", "m-insomnia"), "multihop"),
    Query("dad", "血脂高又腰不好，哪些不能做", ("d-checkup", "d-back"), "multihop"),
    Query("mum", "女儿家孩子上几年级了", ("m-grandson",), "multihop"),
    Query("kid", "数学考试和钢琴课哪个先", ("k-exam", "k-piano"), "multihop"),
    Query("me", "周三健身房和周五评审冲突吗", ("me-gym", "me-review"), "multihop"),
    Query("gran", "糖尿病还能吃腌萝卜吗", ("g-sugar", "g-radish"), "multihop"),
    Query("chen", "他孩子高考还要转岗，忙得过来吗", ("c-kid", "c-handover"), "multihop"),
    Query("dad", "退休以后还去钓鱼吗", ("d-retire", "d-fish"), "multihop"),
    Query("me", "杭州出差那几天有发布窗口吗", ("me-trip", "me-deploy"), "multihop"),
    Query("mum", "妈喝茶会不会影响睡觉", ("m-puer", "m-insomnia"), "multihop"),
]

# Negative: nothing in that persona's memories answers this, and the correct
# behaviour is to return nothing. Every one of these once returned a confident
# wrong answer.
_NEGATIVE: list[Query] = [
    Query("mum", "她炒股赚了多少", (), "negative"),
    Query("mum", "妈的护照什么时候到期", (), "negative"),
    Query("mum", "上次去日本买了什么", (), "negative"),
    Query("mum", "量子力学的书看完没有", (), "negative"),
    Query("dad", "他会不会弹钢琴", (), "negative"),
    Query("dad", "车险是哪家公司", (), "negative"),
    Query("kid", "她数学考了多少分", (), "negative"),
    Query("kid", "编程课报了吗", (), "negative"),
    Query("me", "k8s 集群几个节点", (), "negative"),
    Query("me", "上次面试是哪家公司", (), "negative"),
    Query("gran", "奶奶年轻时做什么工作", (), "negative"),
    Query("gran", "她养过狗吗", (), "negative"),
    Query("chen", "他老家是哪里的", (), "negative"),
    Query("chen", "他用的什么键盘", (), "negative"),
    Query("mum", "那次怎么样", (), "negative"),
    Query("me", "那个东西弄好了吗", (), "negative"),
]

QUERIES: list[Query] = [*_PARAPHRASE, *_CONTEXT, *_SWITCH, *_MULTIHOP, *_NEGATIVE]


def build(store, namespace: str | None = None) -> dict[str, dict[str, int]]:
    """Load the corpus into `store`; returns key → row id per namespace.

    Loading is separated from scoring so an ablation can reuse one populated
    store and only change the ranking configuration.
    """
    mapping: dict[str, dict[str, int]] = {p: {} for p in PERSONAS}
    for ns, key, body, kind, salience in MEMORIES:
        if namespace is not None and ns != namespace:
            continue
        mapping[ns][key] = store.add(ns, body, kind=kind, salience=salience)
    return mapping
