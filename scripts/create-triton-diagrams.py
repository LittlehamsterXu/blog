"""Generate the native SVG figures used by the Triton articles.

Run with Python 3, from any directory. Only the standard library is needed.
The element ownership and byte offsets match the checked-in Triton 3.8.0 /
sm_89 compiler dumps under static/downloads/.
"""

from html import escape
from pathlib import Path


OUT = Path(__file__).resolve().parents[1] / "static" / "images" / "triton"
BG = "#fbf7f0"
INK = "#302c2a"
MUTED = "#706760"
LINE = "#d9cec3"
CORAL = "#e18568"
CORAL_PALE = "#fae4da"
BLUE = "#427e9a"
BLUE_PALE = "#e3f0f4"
GREEN = "#4c8164"
GREEN_PALE = "#e6efe5"
PURPLE = "#8668a5"
PURPLE_PALE = "#eee7f4"


class Figure:
    def __init__(self, height, title, description):
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="680" height="{height}" '
            f'viewBox="0 0 680 {height}" role="img" aria-labelledby="title desc">',
            f"<title id=\"title\">{escape(title)}</title>",
            f"<desc id=\"desc\">{escape(description)}</desc>",
            "<defs><marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" "
            "refX=\"6\" refY=\"4\" orient=\"auto\" markerUnits=\"strokeWidth\">"
            f'<path d="M0,0 L8,4 L0,8 Z" fill="{MUTED}"/></marker></defs>',
            "<style>text{font-family:'Noto Sans SC','Microsoft YaHei','PingFang SC',"
            "sans-serif;font-weight:400} .mono{font-family:Consolas,'DejaVu Sans Mono',"
            "monospace} .bold{font-weight:650}</style>",
            f'<rect width="680" height="{height}" rx="16" fill="{BG}"/>',
        ]

    def rect(self, x, y, w, h, fill="white", stroke=LINE, radius=9, sw=1.5):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def text(self, x, y, text, size=22, fill=INK, anchor="start", bold=False, mono=False):
        cls = " ".join(c for c, enabled in (("bold", bold), ("mono", mono)) if enabled)
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" class="{cls}">{escape(str(text))}</text>'
        )

    def line(self, x1, y1, x2, y2, color=LINE, sw=1.5, arrow=False, dash=None):
        extra = (' marker-end="url(#arrow)"' if arrow else "")
        extra += f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{sw}"{extra}/>'
        )

    def cell(self, x, y, w, label, fill="white", stroke=LINE, h=38, size=22, bold=False):
        self.rect(x, y, w, h, fill, stroke, 6)
        self.text(x + w / 2, y + h / 2 + size * 0.35, label, size,
                  anchor="middle", bold=bold)

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / name
        path.write_text("\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8")
        print(f"{path.relative_to(OUT.parent.parent.parent)}: 680 x {self.height}")


def reduce_ownership():
    f = Figure(
        1080,
        "128 个位置，几条线程？",
        "reduce128 的输入始终是 x[0] 到 x[127]。一个 warp 用 32 个线程，每个 lane "
        "持有连续四个值，i=4*lane+slot。四个 warp 用 128 个线程，每个线程持有一个值，"
        "i=32*warp+lane。tl.arange 的 128 是逻辑位置的数量，不是线程数量。",
    )
    f.text(36, 48, "128 个位置，几条线程？", 29, bold=True)
    f.text(36, 82, "同一个 reduce128，只改 num_warps", 22, MUTED)
    f.rect(36, 110, 608, 65, CORAL_PALE, CORAL)
    f.text(340, 151, "逻辑 tile：x[0] … x[127]", 25, anchor="middle", bold=True)
    f.line(340, 185, 340, 218, MUTED, 2, arrow=True)

    f.rect(36, 238, 608, 342)
    f.text(58, 275, "1 warp = 32 threads", 25, BLUE, bold=True)
    f.text(58, 308, "每个 lane 拿连续 4 个值", 22)
    f.text(58, 347, "Lane", 20, MUTED)
    f.text(395, 347, "持有的元素下标 i", 20, MUTED, anchor="middle")
    for row, (lane, values) in enumerate(((0, range(4)), (1, range(4, 8)),
                                         ("…", ("…",) * 4), (31, range(124, 128)))):
        y = 359 + 42 * row
        f.text(58, y + 27, str(lane), 22)
        for col, value in enumerate(values):
            f.cell(171 + col * 110, y, 102, value, BLUE_PALE, h=36)
    f.text(58, 554, "i = 4 × lane + slot   ·   slot = 0…3", 22, BLUE)

    f.rect(36, 608, 608, 350)
    f.text(58, 645, "4 warps = 128 threads", 25, GREEN, bold=True)
    f.text(58, 678, "每个 lane 拿 1 个值", 22)
    f.text(58, 717, "Warp", 20, MUTED)
    for x, label in ((258, "lane 0"), (418, "…"), (560, "lane 31")):
        f.text(x, 717, label, 20, MUTED, anchor="middle")
    for warp in range(4):
        y = 731 + warp * 41
        f.text(58, y + 26, warp, 22)
        f.cell(201, y, 114, 32 * warp, GREEN_PALE, h=35)
        f.text(418, y + 26, "…", 22, MUTED, anchor="middle")
        f.cell(503, y, 114, 32 * warp + 31, GREEN_PALE, h=35)
    f.text(58, 933, "i = 32 × warp + lane", 22, GREEN)
    f.line(36, 991, 644, 991)
    f.text(36, 1026, "128 个逻辑位置 ≠ 必须启动 128 个线程", 23, bold=True)
    f.text(36, 1056, "两种分工都覆盖同一批输入，归约的通信随之改变。", 20, MUTED)
    f.save("reduce128-thread-ownership.svg")


def one_warp_conversion():
    f = Figure(
        1110,
        "每线程都是四个值",
        "一个 warp、128 个元素的布局转换。SRC 的 sizePerThread 为 2，基本覆盖 "
        "64 个元素，需要重复两轮；两轮指布局分组，不表示执行先后。lane 0 的 slot 0 到 3 是 0、1、64、65。"
        "DST 的 sizePerThread 为 4，一轮覆盖 128 个元素，lane 16 的 slot 0 到 3 "
        "是 64、65、66、67。x[64] 从 lane 0 slot 2 交给同一 warp 的 lane 16 slot 0，"
        "这次编译使用 Shuffle，不使用 shared memory。",
    )
    f.text(36, 48, "每线程都是四个值", 29, bold=True)
    f.text(36, 82, "1 warp · N = 128 · SRC [2] → DST [4]", 22, MUTED)
    f.rect(36, 111, 608, 370)
    f.text(58, 149, "SRC：先铺 64 个，再重复一轮", 24, BLUE, bold=True)
    f.text(58, 181, "2 × 32 × 1 = 64；128 ÷ 64 = 2 轮", 22)
    f.text(58, 220, "Lane", 20, MUTED)
    f.text(288, 220, "第一轮", 20, BLUE, anchor="middle")
    f.text(508, 220, "第二轮", 20, PURPLE, anchor="middle")
    for col in range(4):
        f.text(222 + col * 110, 251, f"slot {col}", 18, MUTED, anchor="middle")
    src_rows = ((0, (0, 1, 64, 65)), (1, (2, 3, 66, 67)),
                (16, (32, 33, 96, 97)), (31, (62, 63, 126, 127)))
    for row, (lane, values) in enumerate(src_rows):
        y = 268 + row * 44
        f.text(58, y + 27, lane, 22)
        for col, value in enumerate(values):
            highlighted = value == 64
            f.cell(171 + col * 110, y, 102, value,
                   CORAL_PALE if highlighted else (BLUE_PALE if col < 2 else PURPLE_PALE),
                   CORAL if highlighted else LINE, h=36, bold=highlighted)
    f.text(58, 460, "两轮表示布局分组，不表示执行先后。", 20, MUTED)

    f.rect(36, 510, 608, 154, CORAL_PALE, CORAL)
    f.text(58, 548, "只追 x[64]", 24, bold=True)
    f.text(58, 585, "lane 0 · slot 2", 23)
    f.line(269, 578, 366, 578, MUTED, 2, arrow=True)
    f.text(390, 585, "lane 16 · slot 0", 23)
    f.text(58, 629, "同一 warp 内换手：Shuffle", 23, BLUE, bold=True)

    f.rect(36, 695, 608, 330)
    f.text(58, 733, "DST：一轮就覆盖 128 个", 24, GREEN, bold=True)
    f.text(58, 765, "4 × 32 × 1 = 128；不需要重复", 22)
    f.text(58, 805, "Lane", 20, MUTED)
    for col in range(4):
        f.text(222 + col * 110, 805, f"slot {col}", 18, MUTED, anchor="middle")
    dst_rows = ((0, (0, 1, 2, 3)), (1, (4, 5, 6, 7)),
                (16, (64, 65, 66, 67)), (31, (124, 125, 126, 127)))
    for row, (lane, values) in enumerate(dst_rows):
        y = 821 + row * 44
        f.text(58, y + 27, lane, 22)
        for col, value in enumerate(values):
            highlighted = value == 64
            f.cell(171 + col * 110, y, 102, value,
                   CORAL_PALE if highlighted else GREEN_PALE,
                   CORAL if highlighted else LINE, h=36, bold=highlighted)
    f.text(36, 1064, "数量一样，元素归属不同；逻辑下标始终没变。", 23, bold=True)
    f.text(36, 1093, "slot 只表示线程内的逻辑值位置，不是 PTX 寄存器编号。", 19, MUTED)
    f.save("layout-one-warp-shuffle.svg")


def shared_memory_trace():
    f = Figure(
        1320,
        "两对值，怎样交给 tid 16",
        "四个 warp、512 个 FP32 元素的布局转换，静态推导自实际 PTX。tid 32 "
        "即 warp 1 lane 0，从 X+256 字节加载 x[64]、x[65]，写到 shared_base+128 字节。"
        "tid 33 即 warp 1 lane 1，从 X+264 字节加载 x[66]、x[67]，写到 "
        "shared_base+1216 字节。所有参与线程经过 bar.sync 0 后，tid 16 即 warp 0 "
        "lane 16 用两条向量化 shared load 读回两对值。最后以一条 v4 store 把 "
        "x[64] 到 x[67] 写到 Y+256 字节，其中 x[66] 位于 Y+264 字节。"
        "所有 shared 地址都是相对基址的字节偏移，图中两地址之间距离不按比例绘制。",
    )
    f.text(36, 48, "两对值，交给同一个线程", 29, bold=True)
    f.text(36, 82, "4 warps · N = 512 · FP32，每个元素 4 B", 22, MUTED)

    for x, tid, lane, values, offset, color, pale in (
        (36, 32, 0, "x[64], x[65]", 256, BLUE, BLUE_PALE),
        (354, 33, 1, "x[66], x[67]", 264, PURPLE, PURPLE_PALE),
    ):
        f.rect(x, 112, 290, 180, pale, color)
        f.text(x + 18, 150, f"源 tid {tid}", 24, color, bold=True)
        f.text(x + 18, 182, f"warp 1 / lane {lane}", 22)
        f.text(x + 18, 224, values, 24, bold=True)
        f.text(x + 18, 267, f"global：X + {offset} B", 21)
        f.line(x + 145, 305, x + 145, 356, MUTED, 2, arrow=True)
        f.text(x + 145, 389, "st.shared.v2.b32", 20, MUTED, anchor="middle", mono=True)

    f.rect(36, 418, 608, 238, "#f2eee7", LINE)
    f.text(58, 456, "Shared Memory：临时排列", 24, bold=True)
    f.text(58, 488, "下列数字都是 shared_base 后的字节偏移", 20, MUTED)
    for x, values, offset, color, pale in (
        (58, "x[64], x[65]", 128, BLUE, BLUE_PALE),
        (365, "x[66], x[67]", 1216, PURPLE, PURPLE_PALE),
    ):
        f.rect(x, 511, 257, 92, pale, color)
        f.text(x + 128.5, 548, f"+ {offset} B", 25, color, anchor="middle", bold=True)
        f.text(x + 128.5, 582, values, 23, anchor="middle")
    f.text(58, 636, "两段相距 1088 B；示意位置不按地址比例。", 20, MUTED)

    f.line(36, 705, 644, 705, MUTED, 2, dash="7 6")
    f.rect(213, 677, 254, 56, BG, BG)
    f.text(340, 713, "bar.sync 0", 25, anchor="middle", bold=True, mono=True)
    f.text(340, 758, "先让整个 CTA 完成写入，再继续读", 22, MUTED, anchor="middle")
    for x, offset in ((181, 128), (499, 1216)):
        f.line(x, 779, x, 830, MUTED, 2, arrow=True)
        f.text(x, 860, f"读 +{offset} B", 22, MUTED, anchor="middle")

    f.rect(36, 890, 608, 196, GREEN_PALE, GREEN)
    f.text(58, 928, "目标 tid 16 · warp 0 / lane 16", 24, GREEN, bold=True)
    f.text(58, 965, "两条 ld.shared.v2.b32，读回两对值", 21)
    for col, value in enumerate((64, 65, 66, 67)):
        f.text(116 + 148 * col, 1002, f"slot {col}", 19, MUTED, anchor="middle")
        f.cell(58 + col * 148, 1014, 126, f"x[{value}]",
               BLUE_PALE if col < 2 else PURPLE_PALE,
               BLUE if col < 2 else PURPLE, h=46, size=23)

    f.line(340, 1098, 340, 1145, MUTED, 2, arrow=True)
    f.text(340, 1181, "st.global.v4.b32：一起写回 Y + 256 B", 22, anchor="middle")
    f.rect(36, 1205, 608, 77)
    f.text(58, 1237, "Y+256 B → x[64]；Y+264 B → x[66]", 21, bold=True)
    f.text(58, 1266, "逻辑顺序没变；shared 中间地址由编译器安排。", 20, MUTED)
    f.text(36, 1306, "Triton 3.8.0 / sm_89 · 来自 PTX 静态地址推导", 18, MUTED)
    f.save("layout-shared-memory-trace.svg")


def cuda_tile_and_threads():
    f = Figure(
        1400,
        "线程形状与数据形状",
        "CUDA warp 在线程块内按线性线程编号分组，threadIdx.x 变化最快。16乘16的"
        "线程块中，warp 0 覆盖 y=0、x=0到15，以及 y=1、x=0到15。32乘8的线程块中，"
        "warp 0 正好覆盖 y=0、x=0到31。底部是一个独立的数据映射例子：32乘8，"
        "共256个线程，可以通过每线程处理四个元素覆盖32乘32、共1024个元素的tile。"
        "来自 matmul.cu 的实际映射为 tid=ty*32+tx，输出行=tid整除8，"
        "输出起始列=(tid模8)*4。warp 0 的前8个lane负责输出第0行，接下来8个lane"
        "负责第1行，依此类推，32个lane覆盖输出的前4行。例如线程坐标(9,0)的tid为9，"
        "负责输出第1行第4到7列。线程块坐标与输出tile坐标不同。",
    )
    f.text(36, 48, "线程形状与数据形状", 29, bold=True)
    f.text(36, 82, "warp 按线程编号分组，threadIdx.x 变化最快", 22, MUTED)
    for x, cols, rows, color, pale in (
        (36, 16, 16, CORAL, CORAL_PALE),
        (354, 32, 8, BLUE, BLUE_PALE),
    ):
        f.rect(x, 112, 290, 414)
        f.text(x + 18, 151, f"{cols} × {rows} 个线程", 25, color, bold=True)
        f.text(x + 18, 183, "两者都是 256 threads", 20, MUTED)
        gx, gy = x + 35, 228
        cw, ch = 224 / cols, 192 / rows
        f.text(gx, 215, "x = 0", 17, MUTED)
        f.text(gx + 224, 215, str(cols - 1), 17, MUTED, anchor="end")
        for row in range(rows):
            for col in range(cols):
                in_warp0 = row * cols + col < 32
                f.rect(gx + col * cw, gy + row * ch, cw, ch,
                       pale if in_warp0 else "white", color if in_warp0 else LINE,
                       radius=0, sw=0.9)
        f.text(x + 18, 458, "warp 0（着色的 32 格）", 21, color, bold=True)
        if cols == 16:
            f.text(x + 18, 489, "y=0 和 y=1，各 16 个", 20)
        else:
            f.text(x + 18, 489, "y=0，一行 32 个", 20)
    f.text(36, 564, "每个小格代表线程，不是一个输入元素。", 23, bold=True)
    f.line(36, 590, 644, 590)

    f.text(36, 633, "256 个线程，处理 1024 个元素", 27, bold=True)
    f.text(36, 669, "线程块 32 × 8     数据 tile 32 × 32", 23)
    f.text(36, 704, "matmul.cu 里实际选用的映射：", 22, MUTED)
    f.rect(36, 730, 608, 143, GREEN_PALE, GREEN)
    f.text(58, 769, "tid       = ty * 32 + tx", 23, GREEN, bold=True, mono=True)
    f.text(58, 809, "row       = tid // 8", 23, mono=True)
    f.text(58, 849, "col_start = (tid % 8) * 4", 23, mono=True)
    f.text(36, 917, "warp 0：每 8 个 lane 负责输出的一行", 23, bold=True)
    f.text(36, 950, "格内为 lane 编号；一格计算连续 4 列", 21, MUTED)
    f.text(59, 986, "输出行", 19, MUTED)
    f.text(139, 986, "col 0…3", 18, MUTED)
    f.text(603, 986, "28…31", 18, MUTED, anchor="end")
    for row in range(4):
        y = 1002 + row * 43
        f.text(78, y + 27, row, 22, anchor="middle")
        for col in range(8):
            lane = row * 8 + col
            highlighted = lane == 9
            f.cell(139 + col * 58, y, 53, lane,
                   CORAL_PALE if highlighted else GREEN_PALE,
                   CORAL if highlighted else GREEN, h=35,
                   size=21, bold=highlighted)
    f.rect(36, 1201, 608, 86, CORAL_PALE, CORAL)
    f.text(58, 1237, "线程 (tx, ty) = (9, 0) → tid 9", 23, bold=True)
    f.text(58, 1270, "输出：row 1，col 4…7", 23)
    f.text(36, 1331, "warp 在线程块的第一行，输出却覆盖了四行。", 23, bold=True)
    f.text(36, 1368, "线程块坐标与输出 tile 坐标，要靠映射连接。", 22, MUTED)
    f.save("cuda-tile-vs-threads.svg")


def bank_conflict_basics():
    f = Figure(
        850,
        "广播与 Bank Conflict",
        "独立的简化示意：同一warp中的两个lane执行32-bit标量shared读取，假定"
        "shared起点落在bank 0，32个bank，每个word为4字节。bank=(byte_addr整除4)模32。"
        "第一种情况，lane 0和lane 1都读取字节偏移0，即同一个word，可以广播给两个lane。"
        "第二种情况，lane 0读取字节偏移0，lane 1读取字节偏移128，两者是不同word，"
        "却同属bank 0，需要分批服务。这不是对本文布局转换中向量化访存的冲突测量。",
    )
    f.text(36, 48, "广播与 Bank Conflict", 29, bold=True)
    f.text(36, 86, "同一 warp · 两个 lane · 32-bit 标量读取", 23, MUTED)
    f.text(36, 123, "bank = (byte_addr // 4) % 32", 23, mono=True)

    f.rect(36, 150, 608, 269)
    f.text(58, 187, "同一个 word：广播", 25, GREEN, bold=True)
    f.cell(58, 215, 138, "lane 0", GREEN_PALE, GREEN, h=48, size=24)
    f.cell(58, 302, 138, "lane 1", GREEN_PALE, GREEN, h=48, size=24)
    f.text(267, 230, "+0 B", 23, MUTED, anchor="middle")
    f.text(267, 339, "+0 B", 23, MUTED, anchor="middle")
    f.line(206, 244, 420, 277, MUTED, 2, arrow=True)
    f.line(206, 321, 420, 289, MUTED, 2, arrow=True)
    f.text(524, 238, "bank 0", 24, GREEN, anchor="middle", bold=True)
    f.cell(434, 253, 180, "word 0", GREEN_PALE, GREEN, h=65, size=24)
    f.text(58, 390, "读出同一份值，分发给两个 lane。", 23)

    f.rect(36, 446, 608, 291)
    f.text(58, 483, "不同 word，同一 bank：冲突", 25, CORAL, bold=True)
    f.cell(58, 514, 138, "lane 0", CORAL_PALE, CORAL, h=48, size=24)
    f.cell(58, 600, 138, "lane 1", CORAL_PALE, CORAL, h=48, size=24)
    f.text(293, 525, "+0 B", 23, MUTED, anchor="middle")
    f.text(293, 612, "+128 B", 23, MUTED, anchor="middle")
    f.line(206, 546, 420, 546, MUTED, 2, arrow=True)
    f.line(206, 633, 420, 633, MUTED, 2, arrow=True)
    f.text(524, 519, "bank 0", 24, CORAL, anchor="middle", bold=True)
    f.cell(434, 531, 180, "word 0", CORAL_PALE, CORAL, h=47, size=24)
    f.cell(434, 610, 180, "word 32", CORAL_PALE, CORAL, h=47, size=24)
    f.line(524, 585, 524, 602, LINE, 2, dash="3 3")
    f.text(58, 701, "两个不同的 word，要分批服务。", 23)
    f.text(36, 779, "示意假定 shared 起点落在 bank 0。", 22, MUTED)
    f.text(36, 812, "32 个 bank，每个 word 4 B；不是本次 v2 访问分析。", 21, MUTED)
    f.save("bank-conflict-basics.svg")


if __name__ == "__main__":
    cuda_tile_and_threads()
    reduce_ownership()
    one_warp_conversion()
    shared_memory_trace()
    bank_conflict_basics()
