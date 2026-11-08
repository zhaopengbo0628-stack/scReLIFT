"""Adapt MAGCA reference 02 / Fig3.ipynb cell 0 to the supplied D-mode data.

Dependencies: matplotlib, pandas, plottable, scipy.
Input means are frozen in the companion JSON; no recalculation is needed. Optional --deps adds an isolated package path.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--deps", type=Path, default=ROOT.parents[2] / "_work" / "table1_python_deps")
args = parser.parse_args()
if args.deps.is_dir():
    sys.path.insert(0, str(args.deps))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap


def main():
    data = json.loads((ROOT / "图2.1_PBMC-D评分_绘图数据.json").read_text(encoding="utf-8"))
    metrics = data["metrics"]
    frame = pd.DataFrame([row["values"] for row in data["records"]], columns=metrics)
    frame.index = [row["method"] for row in data["records"]]
    frame["Method"] = frame.index
    titles = {
        "ARI": "ARI", "NMI": "NMI", "ACC": "ACC",
        "Macro-F1": "Macro-F1", "Balanced ACC": "Balanced\nACC",
        "Purity@15": "Purity@15",
        "Silhouette": "Silhouette",
    }
    definitions = [ColumnDefinition("Method", width=1.55, textprops={"ha": "left", "weight": "bold"})]
    for metric in metrics:
        # Identical PRGn / median +/- 2.5 sample SD mapping to reference 02.
        finite = frame[metric].dropna()
        base_cmap = normed_cmap(finite, cmap=matplotlib.colormaps["PRGn"], num_stds=2.5)

        def cmap(value, mapper=base_cmap):
            return (1.0, 1.0, 1.0, 1.0) if pd.isna(value) else mapper(value)

        definitions.append(ColumnDefinition(
            metric, title=titles[metric], width=1.0,
            textprops={"ha": "center", "bbox": {"boxstyle": "circle", "pad": 0.25}},
            cmap=cmap, formatter=lambda value: "NA" if pd.isna(value) else f"{value:.2f}",
        ))

    with matplotlib.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    }):
        figure_width = 13.3 * (len(metrics) + 1.55) / 10.55
        fig, ax = plt.subplots(figsize=(figure_width, 4.35))
        table = Table(
            frame, column_definitions=definitions, index_col="Method", ax=ax,
            cell_kw={"linewidth": 0, "edgecolor": "k"},
            row_dividers=True, footer_divider=True,
            textprops={"fontsize": 12, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5)), "color": "#8c8c8c"},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
        ).autoset_fontcolors(colnames=metrics)
        # No aggregate score sorting: retain the requested method ordering.
        fig.subplots_adjust(left=0.012, right=0.995, top=0.955, bottom=0.025)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        figure_bounds = fig.bbox
        clipped = []
        for text in ax.texts:
            bounds = text.get_window_extent(renderer)
            if (bounds.x0 < figure_bounds.x0 or bounds.y0 < figure_bounds.y0 or
                    bounds.x1 > figure_bounds.x1 or bounds.y1 > figure_bounds.y1):
                clipped.append(text.get_text())
        if clipped:
            raise RuntimeError(f"Text outside figure bounds: {clipped}")
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(ROOT / f"图2.1_PBMC-D评分.{extension}", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)

    # Preserve display rounding and palette parameters for reproducibility.
    style = {
        "reference": "MAGCA reference 02 / Fig3.ipynb, code cell 0",
        "cmap": "PRGn", "normalization": "per-column median +/- 2.5 sample SD; missing values excluded",
        "data_normalized": False, "display_decimals": 2, "missing": "NA (white background)",
        "font": "Arial with DejaVu Sans fallback", "dpi": 300,
        "matplotlib": matplotlib.__version__,
        "columns": metrics, "methods": list(frame.index), "clipped_texts": clipped,
    }
    (ROOT / "图2.1_PBMC-D评分_绘图配置.json").write_text(json.dumps(style, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    caption = (
        "D-mode method comparison using the user-provided mean scores. Circles use the original MAGCA PRGn "
        "palette with each column independently mapped around its median +/- 2.5 sample SD; the printed values "
        "are unchanged means rounded to two decimals. NA denotes unavailable metrics. scReLIFT D-mode means are "
        "equal-weight averages over four folds, each with three seeds (20020130, 20020131, 20020132). "
        "Comparator aggregation and protocol equivalence have not been fully audited; this is a figure preview. "
        "The retained historical evidence flags official ECLARE as incompatible with strict D-mode, so its "
        "user-supplied row requires provenance verification before a final strict-protocol comparison."
    )
    (ROOT / "图2.1_PBMC-D评分_caption.txt").write_text(caption + "\n", encoding="utf-8")
    latex_rows = [r"\begin{tabular}{l" + "r" * len(metrics) + "}", r"\toprule",
                  "Method & " + " & ".join(metrics) + r" \\", r"\midrule"]
    for record in data["records"]:
        latex_rows.append(record["method"] + " & " + " & ".join(
            "NA" if record["values"][m] is None else f"{record['values'][m]:.2f}" for m in metrics) + r" \\")
    latex_rows += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "图2.1_PBMC-D评分.tex").write_text("\n".join(latex_rows) + "\n", encoding="utf-8")
    (ROOT / "图2.1_PBMC-D评分_include.tex").write_text(
        "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{图2.1_PBMC-D评分.pdf}\n"
        "\\caption{" + caption.replace("+/-", r"$\pm$") + "}\n\\label{fig:dmode-comparison}\n\\end{figure}\n",
        encoding="utf-8")
    print("Saved 图2.1_PBMC-D评分.png, .pdf, .svg and reproducible data/style files.")


if __name__ == "__main__":
    main()
