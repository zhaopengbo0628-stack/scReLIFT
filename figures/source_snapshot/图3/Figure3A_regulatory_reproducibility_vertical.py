"""Vertical version of scReLIFT reproducibility DotPlot: 6 pairs x 3 metrics.

Reads the same user-supplied CSV as the approved horizontal version. Preserves
all 18 values, fixed circle area, teal scale 0..1, and Decimal HALF_UP labels.
Original horizontal figure and data remain unchanged.
"""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import csv
import hashlib
import json
import sys

try:
    import matplotlib
except ImportError:
    sys.path.insert(0, r"E:\my_research\_work\table1_python_deps")
    import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "绘图数据/图3.2_Regulatory_Reproducibility_用户指定.csv"
STEM = ROOT / "Figure3A_regulatory_reproducibility_vertical"
PAIRS = ("D1-D2", "D1-D3", "D1-D4", "D2-D3", "D2-D4", "D3-D4")
METRICS = (("interval_jaccard", "Interval\nJaccard"),
           ("sign_consistency", "Sign\nconsistency"),
           ("weight_spearman", "Weight\nSpearman"))


def main():
    with DATA.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["fold_pair"] for r in rows] == list(PAIRS)
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
                         "font.size":10, "pdf.fonttype":42, "svg.fonttype":"none",
                         "savefig.facecolor":"white", "savefig.dpi":350})
    cmap = LinearSegmentedColormap.from_list("regulatory_teal", ["#f1f7f6", "#a8ccc7", "#247a72"])
    norm = Normalize(vmin=0, vmax=1, clip=True)
    fig = plt.figure(figsize=(4.8, 5.9), facecolor="white")
    ax = fig.add_axes((.22, .16, .65, .66))
    xs, ys, values, labels = [], [], [], []
    for y, row in enumerate(rows):
        for x, (key, _) in enumerate(METRICS):
            value = float(row[key])
            assert 0 <= value <= 1
            xs.append(x)
            ys.append(y)
            values.append(value)
            labels.append(dict(fold_pair=row["fold_pair"], metric=key, input=row[key],
                               label=str(Decimal(row[key]).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP))))
    dots = ax.scatter(xs, ys, s=1300, c=values, cmap=cmap, norm=norm,
                      marker="o", edgecolors="none", linewidths=0)
    for x,y,value,entry in zip(xs,ys,values,labels):
        rgb = cmap(norm(value))[:3]
        linear = [v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb]
        luminance = sum(a*b for a,b in zip(linear,(.2126,.7152,.0722)))
        ax.text(x,y,entry["label"],ha="center",va="center",fontsize=11.3,fontweight="bold",
                color="white" if luminance<.32 else "#173a36")
    ax.set_xlim(-.5,2.5)
    ax.set_ylim(5.5,-.5)
    ax.set_xticks(range(3),[label for _,label in METRICS],fontsize=10)
    ax.set_yticks(range(6),PAIRS,fontsize=10.5)
    ax.tick_params(axis="both",length=0,pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes((.915,.22,.018,.54))
    cb = fig.colorbar(dots,cax=cax,ticks=(0,.25,.5,.75,1))
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0,labelsize=8,pad=4)
    cb.ax.set_title("Value",fontsize=9,pad=9)
    # Wrapped user-requested title fits the portrait canvas without shrinking.
    fig.text(.035,.963,"A. Cross-round regulatory\nreproducibility",ha="left",va="top",
             fontsize=13,fontweight="bold",linespacing=1.25)
    for ext in ("png","pdf","svg"):
        fig.savefig(STEM.with_suffix("."+ext),dpi=350)
    plt.close(fig)
    meta = dict(model="scReLIFT",layout="6 fold-pair rows x 3 metric columns",
                circle_area_pt2=1300,color_limits=[0,1],data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                data_source=str(DATA),number_labels=labels,rounding="Decimal ROUND_HALF_UP",
                original_horizontal_preserved=True)
    STEM.with_suffix(".json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Saved vertical PNG/PDF/SVG, 18 fixed-size circles.")


if __name__ == "__main__":
    main()
