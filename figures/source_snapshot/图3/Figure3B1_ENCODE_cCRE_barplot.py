"""Standalone ENCODE cCRE grouped horizontal bars for scReLIFT.

python Figure3B1_ENCODE_cCRE_barplot.py
Reads audited Figure3B_used_data.csv (or --data the exported 12-row CSV).
No experiments; does not modify the combined Figure 3B or Figure 3A.
"""
from pathlib import Path
import argparse
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
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FormatStrFormatter

ROOT = Path(__file__).resolve().parent
FOLDS = ("D1", "D2", "D3", "D4")
CATEGORIES = ("PLS", "pELS", "dELS")
PALETTE = ("#c3e1e6", "#f3dfb7", "#dcc6dc", "#b7d5ac")
plt.rcParams.update({
    "font.family":"sans-serif", "font.sans-serif":["Arial", "DejaVu Sans"],
    "font.size":8, "axes.labelsize":8.3, "xtick.labelsize":7.5,
    "ytick.labelsize":8.5, "axes.linewidth":.6, "axes.edgecolor":"#a4a8ab",
    "axes.spines.top":False, "axes.spines.right":False,
    "text.color":"#292929", "axes.labelcolor":"#292929",
    "svg.fonttype":"none", "pdf.fonttype":42,
    "figure.facecolor":"white", "savefig.facecolor":"white", "savefig.dpi":350,
})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"Figure3B_used_data.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    with args.data.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["panel"] == "B1"]
    index = {(r["fold"], r["item_name"]):r for r in rows}
    assert len(rows) == len(index) == 12
    assert set(index) == {(f,c) for f in FOLDS for c in CATEGORIES}
    for r in rows:
        assert float(r["fold_enrichment"]) > 0 and 0 <= float(r["q_value"]) <= 1
    fig = plt.figure(figsize=(122/25.4, 3.12))
    fig.text(.035, .963, "B1  ENCODE cCRE enrichment", fontsize=9.2,
             fontweight="bold", ha="left", va="top")
    ax = fig.add_axes([.145, .220, .815, .590])
    offsets = (-.285, -.095, .095, .285)
    for j, category in enumerate(CATEGORIES):
        for i, fold in enumerate(FOLDS):
            r = index[(fold, category)]
            value = float(r["fold_enrichment"])
            significant = float(r["q_value"]) < .05
            ax.barh(j+offsets[i], value, left=0, height=.151,
                    color=PALETTE[i], edgecolor="none", zorder=2)
            ax.text(value+.027, j+offsets[i], f"{value:.3f}" + ("*" if not significant else ""),
                    ha="left", va="center", fontsize=7.8)
    # Ordinary bars begin at zero; the dashed line marks the enrichment null.
    ax.axvline(1, color="#74797d", linestyle=(0,(3,3)), lw=.8, zorder=3)
    ax.set_xlim(0, 2.09)
    ax.set_ylim(2.56, -.56)
    ax.set_yticks(range(3), CATEGORIES)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(axis="x", length=3, pad=4)
    ax.xaxis.set_major_locator(FixedLocator((0,.5,1,1.5,2)))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel("cCRE fold enrichment", labelpad=5)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#edf0f1", linewidth=.5, zorder=0)
    handles = [Patch(facecolor=PALETTE[i], edgecolor="none", label=fold)
               for i, fold in enumerate(FOLDS)]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.965,.91),
               frameon=False, ncol=4, handlelength=1.25, handleheight=.8,
               handletextpad=.5, columnspacing=1.35, borderaxespad=0, fontsize=8)
    fig.text(.035, .034, "* D2 dELS: not significant (BH-FDR = 0.746).",
             fontsize=7, color="#62676b", ha="left", va="bottom")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir/"Figure3B1_ENCODE_cCRE_barplot"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("."+extension), dpi=350)
    plt.close(fig)
    ordered = [index[(f,c)] for c in CATEGORIES for f in FOLDS]
    with (args.output_dir/"Figure3B1_ENCODE_cCRE_used_data.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(ordered)
    metadata = {
        "source_csv":str(args.data), "source_sha256":hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "rows":12, "fold_order":list(FOLDS), "category_order":list(CATEGORIES),
        "colors":dict(zip(FOLDS,PALETTE)), "bar_baseline":0,
        "enrichment_reference":1, "labels":"true per-fold fold enrichment, three decimals",
        "legend":"Fold color mapping only", "width_mm":122, "height_mm":3.12*25.4,
        "note":"Source cCRE matched-Peak null, 200 draws, with matching relaxations; not a motif or GO test.",
    }
    (args.output_dir/"Figure3B1_ENCODE_cCRE_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("Saved standalone cCRE bars:",stem)


if __name__ == "__main__":
    main()
