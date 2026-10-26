"""Draw the sinusoidal position-encoding figure (NumPy + Matplotlib)."""

from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "static/images/position-encoding"
PREVIEW = ROOT / ".local/position-plots"
OUT.mkdir(parents=True, exist_ok=True)
PREVIEW.mkdir(parents=True, exist_ok=True)

for candidate in (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
):
    if candidate.exists():
        font_manager.fontManager.addfont(str(candidate))
        family = font_manager.FontProperties(fname=str(candidate)).get_name()
        break
else:
    family = "sans-serif"

BG, INK, MUTED = "#fbf7f0", "#302c2a", "#706760"
COLORS = ["#bf633f", "#427e9a", "#4c8164", "#8668a5"]
plt.rcParams.update({
    "font.family": family,
    "font.size": 16,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "svg.fonttype": "none",
    "svg.hashsalt": "position-encoding",
    "axes.unicode_minus": False,
})

positions = np.arange(64, dtype=np.float64)
theta = 10000.0 ** (-np.arange(0, 8, 2, dtype=np.float64) / 8)
angles = positions[:, None] * theta
encoding = np.stack((np.sin(angles), np.cos(angles)), axis=-1).reshape(64, 8)

fig = plt.figure(figsize=(10, 4.6), dpi=100, facecolor=BG)
fig.text(.035, .93, "正弦位置编码：同一位置，四种变化速度", fontsize=21, weight="bold")
left = fig.add_axes([.085, .22, .39, .51], facecolor=BG)
curve_positions = np.linspace(0, 63, 2048)
for j, color in enumerate(COLORS):
    left.plot(curve_positions, np.sin(curve_positions * theta[j]), color=color, lw=1.8,
              label=f"θ = {theta[j]:g}")
left.set(xlim=(0, 63), ylim=(-1.08, 1.08), xlabel="位置 p", ylabel="sin(pθ)")
left.set_xticks([0, 16, 32, 48, 63])
left.set_yticks([-1, 0, 1])
left.grid(color="#d9cec3", lw=.8, alpha=.65)
left.spines[["top", "right"]].set_visible(False)
left.spines[["left", "bottom"]].set_color("#b7aa9d")
left.legend(loc="lower left", bbox_to_anchor=(-.03, 1.0), ncol=2,
            fontsize=14.5, frameon=False, handlelength=1.2,
            columnspacing=1.0, handletextpad=.4, borderpad=0.)

right = fig.add_axes([.62, .22, .25, .57], facecolor=BG)
cmap = LinearSegmentedColormap.from_list("position", ["#427e9a", BG, "#bf633f"])
im = right.imshow(encoding, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
                  interpolation="nearest", origin="upper", extent=(-.5, 7.5, 63.5, -.5))
right.set(xlabel="特征维度", ylabel="位置 p")
right.set_xticks([0, 2, 4, 6])
right.set_yticks([0, 16, 32, 48, 63])
right.set_title("完整 PE：sin / cos 交错", fontsize=15, pad=12)
color_axis = fig.add_axes([.89, .22, .017, .57])
bar = fig.colorbar(im, cax=color_axis, ticks=[-1, 0, 1])
bar.outline.set_edgecolor("#b7aa9d")
fig.text(.5, .035, "D = 8；θ = [1, 0.1, 0.01, 0.001]，每一对维度共用一个频率。",
         ha="center", fontsize=15)

svg = OUT / "sinusoidal-patterns.svg"
fig.savefig(svg, facecolor=BG, metadata={"Date": None, "Creator": "Matplotlib"})
fig.savefig(PREVIEW / "sinusoidal-patterns.png", facecolor=BG)
plt.close(fig)

# Give the standalone SVG explicit pixel dimensions and an accessible title.
ET.register_namespace("", "http://www.w3.org/2000/svg")
tree = ET.parse(svg)
root = tree.getroot()
root.set("width", "1000")
root.set("height", "460")
root.set("role", "img")
root.set("aria-labelledby", "title desc")
title = ET.Element("{http://www.w3.org/2000/svg}title", {"id": "title"})
title.text = "D等于8的正弦位置编码：不同频率的曲线与完整位置矩阵"
desc = ET.Element("{http://www.w3.org/2000/svg}desc", {"id": "desc"})
desc.text = "左图为位置0到63的四条sin曲线，频率为1、0.1、0.01和0.001；右图为相同位置的八维PE矩阵，偶数维为sin，奇数维为cos，色标固定为负1到1。"
root.insert(0, desc)
root.insert(0, title)
tree.write(svg, encoding="unicode", xml_declaration=False)
print(svg.name)
