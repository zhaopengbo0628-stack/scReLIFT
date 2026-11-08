"""PBMC intra efficiency bubble chart from preserved user evidence.

X: trainable parameters (millions, decreasing to the right).
Y: cells/sec (two explicitly separated linear intervals).
Scatter area, not radius, is proportional to peak GPU memory in GiB.
CPU methods are excluded before plotting and scaling.
"""
import argparse
import base64
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = "图2.3_PBMC-imtra-效率"
AREA_PER_GIB = 3200.0  # scatter marker area in points squared / GiB
PALETTE = {
    "scReLIFT": "#E7A26B", "Multigrate": "#8A6AA6", "Cobolt": "#EBCB6A",
    "ECLARE-S": "#E57373", "scMomer-S": "#63AE8C",
}


def parse_number_pair(text):
    cleaned = text.replace("**", "").replace(",", "").replace("−", "-").strip()
    if cleaned == "NA" or not cleaned:
        return {"mean": None, "reported_spread": None}
    parts = re.split(r"\s*±\s*", cleaned)
    return {"mean": float(parts[0]), "reported_spread": float(parts[1]) if len(parts) == 2 else None}


def prepare_data():
    source = ROOT / "图2.3_PBMC-imtra-效率_输入原表.md"
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.startswith("|")]
    split = lambda line: [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
    headers = split(lines[0])
    records, excluded = [], []
    for line in lines[2:]:
        row = dict(zip(headers, split(line)))
        original = row["Method"]
        if row["GPU"].strip().upper() == "CPU":
            excluded.append({"source_method": original, "device": row["GPU"], "reason": "CPU excluded by user"})
            continue
        method = {"Full B2-S": "scReLIFT", "ECLARE": "ECLARE-S"}.get(original, original)
        params = parse_number_pair(row["Trainable params"])
        memory = parse_number_pair(row["Peak VRAM (GiB)"])
        speed = parse_number_pair(row["Cells/sec"])
        if not all(value["mean"] is not None and value["mean"] > 0 for value in [params, memory, speed]):
            raise ValueError(f"Missing/non-positive plotted value: {method}")
        records.append({
            "method": method, "source_method": original, "device": row["GPU"],
            "trainable_params": int(params["mean"]),
            "peak_gpu_memory_gib": memory, "throughput_cells_per_sec": speed,
            "inference_time_sec": parse_number_pair(row["Inference time (s)"]),
            "color": PALETTE[method],
        })
    order = ["scReLIFT", "Multigrate", "Cobolt", "ECLARE-S", "scMomer-S"]
    records.sort(key=lambda record: order.index(record["method"]))
    if [row["method"] for row in records] != order:
        raise ValueError("Expected exactly the five requested GPU methods")
    data = {
        "dataset": "PBMC", "protocol": "intra", "model_name": "scReLIFT",
        "source": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "encoding": {"x": "trainable_params / 1e6 (linear, reversed; limits 41 to 3)", "y": "throughput_cells_per_sec.mean (broken linear axis: 0-20 and 10000-500000)",
                     "area": f"peak_gpu_memory_gib.mean * {AREA_PER_GIB:g} points^2/GiB", "color": "method"},
        "records": records, "excluded": excluded,
        "aggregation_note": "Supplied aggregate means are plotted; reported spreads are preserved, not reinterpreted as confidence intervals.",
        "comparability_note": "Same stated GPU model (NVIDIA A30). Inference batch, timing scope and phase of peak VRAM were not fully verified from this summary.",
    }
    (ROOT / f"{STEM}_数据.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def plot_chart(data):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    offsets = {
        "scReLIFT": (-19, 0, "right"),
        "Multigrate": (-30, -5, "right"),
        "Cobolt": (28, 0, "left"),
        "ECLARE-S": (23, 16, "left"),
        "scMomer-S": (35, 0, "left"),
    }
    with matplotlib.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": 0.8, "axes.edgecolor": "#444444",
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    }):
        fig, (ax, low_ax) = plt.subplots(
            2, 1, sharex=True, figsize=(7.4, 4.9),
            gridspec_kw={"height_ratios": [3.4, 1.15], "hspace": 0.09})
        fig.subplots_adjust(left=0.12, right=0.975, top=0.95, bottom=0.16)
        ax.set_ylim(10000, 500000)
        low_ax.set_ylim(0, 20)
        low_ax.set_xlim(41, 3)
        low_ax.set_xticks([40, 30, 20, 10, 5])
        ax.set_yticks([100000, 200000, 300000, 400000, 500000])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        low_ax.set_yticks([10])
        for panel in (ax, low_ax):
            panel.grid(which="major", color="#c9c9c9", linestyle=":", linewidth=0.75, zorder=0)
            panel.set_axisbelow(True)
        ax.spines["bottom"].set_visible(False)
        low_ax.spines["top"].set_visible(False)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        low_ax.set_xlabel("Trainable parameters (M; fewer →)", labelpad=8)
        y_label = fig.text(0.018, 0.57, "Throughput (cells/s; higher ↑)",
                           rotation=90, ha="center", va="center", fontsize=12)
        # Diagonal marks on BOTH sides indicate the omitted throughput interval.
        for panel, y_fraction in [(ax, 0), (low_ax, 1)]:
            for x_fraction in [0, 1]:
                panel.plot([x_fraction - 0.009, x_fraction + 0.009],
                           [y_fraction - 0.025, y_fraction + 0.025],
                           transform=panel.transAxes, color="#444444", linewidth=1,
                           clip_on=False, zorder=10)
        method_labels = []
        values = []
        for record in data["records"]:
            method = record["method"]
            x = record["trainable_params"] / 1e6
            y = record["throughput_cells_per_sec"]["mean"]
            memory = record["peak_gpu_memory_gib"]["mean"]
            area = memory * AREA_PER_GIB
            panel = low_ax if y <= 20 else ax
            if not (0 <= y <= 20 or 10000 <= y <= 500000):
                raise ValueError(f"Point in omitted interval: {method}")
            panel.scatter(x, y, s=area, c=record["color"], alpha=0.88,
                       edgecolors="none", linewidths=0, zorder=3)
            dx, dy, align = offsets[method]
            annotation = panel.annotate(method, (x, y), xytext=(dx, dy), textcoords="offset points",
                                    ha=align, va="center", fontsize=11,
                                    fontweight="bold" if method == "scReLIFT" else "normal",
                                    color="#222222", zorder=5)
            method_labels.append(annotation)
            values.append({"method": method, "x_million": x, "y_cells_per_sec": y,
                           "area_pt2": area, "memory_gib": memory})

        # Same points^2/GiB mapping in the figure inset and main data axes.
        memory_ax = ax.inset_axes([0.02, 0.50, 0.37, 0.48])
        memory_ax.patch.set_alpha(0)
        memory_ax.set_xlim(0, 3)
        memory_ax.set_ylim(0, 1)
        memory_ax.axis("off")
        memory_ax.text(1.5, 0.99, "Peak GPU memory", ha="center", va="top", fontsize=10.5)
        legend_values = [0.15, 0.5, 1.0]
        for position, memory in zip([0.5, 1.5, 2.5], legend_values):
            memory_ax.scatter(position, 0.50, s=memory * AREA_PER_GIB, color="#b8b8b8",
                              alpha=0.65, edgecolors="none", linewidths=0, clip_on=False)
            memory_ax.text(position, 0.01, f"{memory:g} GiB", ha="center", va="bottom", fontsize=9)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bubble_visibility = []
        for point in values:
            panel = low_ax if point["y_cells_per_sec"] <= 20 else ax
            cx, cy = panel.transData.transform((point["x_million"], point["y_cells_per_sec"]))
            radius_px = 0.5 * math.sqrt(point["area_pt2"]) * fig.dpi / 72.0
            panel_box = panel.get_window_extent(renderer)
            fully_visible = (cx - radius_px >= panel_box.x0 and cx + radius_px <= panel_box.x1 and
                             cy - radius_px >= panel_box.y0 and cy + radius_px <= panel_box.y1)
            bubble_visibility.append({"method": point["method"], "fully_inside_panel": bool(fully_visible)})
            if not fully_visible:
                raise ValueError(f"Bubble would be clipped: {point['method']}")
        clipped = []
        for label in [*method_labels, low_ax.xaxis.label, y_label, *memory_ax.texts]:
            bounds = label.get_window_extent(renderer)
            if not (fig.bbox.x0 <= bounds.x0 and bounds.x1 <= fig.bbox.x1 and
                    fig.bbox.y0 <= bounds.y0 and bounds.y1 <= fig.bbox.y1):
                clipped.append(label.get_text())
        if clipped:
            raise ValueError(f"Clipped labels: {clipped}")
        label_overlap = []
        for i, left in enumerate(method_labels):
            for right in method_labels[i + 1:]:
                if left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer)):
                    label_overlap.append([left.get_text(), right.get_text()])
        if label_overlap:
            raise ValueError(f"Overlapping method labels: {label_overlap}")
        for extension in ["png", "pdf", "svg"]:
            fig.savefig(ROOT / f"{STEM}.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        png_buffer = io.BytesIO()
        fig.savefig(png_buffer, format="png", dpi=160, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        notebook_preview = base64.b64encode(png_buffer.getvalue()).decode("ascii")
        plt.close(fig)
    checks = {
        "included_methods": len(values), "all_devices": sorted({row["device"] for row in data["records"]}),
        "excluded_methods": data["excluded"], "marker_values": values,
        "area_per_gib_pt2": AREA_PER_GIB, "memory_legend_gib": legend_values,
        "yscale": "broken linear", "y_intervals_cells_per_sec": [[0, 20], [10000, 500000]],
        "omitted_interval_cells_per_sec": [20, 10000], "x_limits_million": [41, 3],
        "preferred_direction": "upper right; smaller bubbles indicate lower memory",
        "bubble_visibility": bubble_visibility,
        "clipped_labels": clipped, "overlapping_method_labels": label_overlap,
        "matplotlib_version": matplotlib.__version__,
    }
    (ROOT / f"{STEM}_检查记录.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return notebook_preview


def write_support_files(data, preview):
    caption = (
        "PBMC intra efficiency comparison for five GPU methods on the stated NVIDIA A30 device. "
        "The reversed linear x-axis shows trainable parameters in millions (41 to 3, fewer to the right); "
        "the y-axis shows throughput in cells/s in two linear intervals, 0-20 and 10000-500000, "
        "separated by explicit break marks. The interval 20-10000 cells/s is omitted, with no data points omitted. "
        "The two vertical intervals have different scales; distances across the break are not continuous. "
        "The upper right combines fewer trainable parameters and higher throughput; smaller bubbles denote lower memory. "
        "Bubble area is proportional to the supplied peak GPU memory in GiB, with the same area scale used "
        "for the gray reference bubbles. Colors identify methods. Palette-S (CPU) is excluded. "
        "Markers show the user-provided aggregate means; reported spreads are preserved in the linked JSON "
        "but are not drawn as confidence intervals. The exact replicate aggregation, inference batch, timing "
        "scope and phase of peak VRAM have not been fully verified, so the figure describes the supplied "
        "implementations rather than a claim of identical-batch hardware efficiency. The model formerly "
        "called Full B2-S is displayed as scReLIFT."
    )
    (ROOT / f"{STEM}_caption.txt").write_text(caption + "\n", encoding="utf-8")
    (ROOT / f"{STEM}_include.tex").write_text(
        "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.95\\linewidth]{" + STEM + ".pdf}\n"
        "\\caption{" + caption + "}\n\\label{fig:pbmc-intra-efficiency}\n\\end{figure}\n", encoding="utf-8")
    cells = [
        {"cell_type": "markdown", "id": "efficiency-notes", "metadata": {}, "source":
            "# 图2.3 PBMC-imtra-效率\n\n横轴：Trainable parameters (M)，向右递减；纵轴：Cells/sec（明确断轴）；"
            "右上角表示参数更少、吞吐量更高。纵轴显示0–20与10000–500000两个线性区间，区间刻度不同。"
            "圆面积：Peak GPU memory (GiB)；颜色：方法。排除CPU的Palette-S。"
            "模型正式名称scReLIFT，比较方法名ECLARE-S。数值直接来自保留的用户原表。\n\n" + caption},
        {"cell_type": "code", "id": "efficiency-plot", "metadata": {}, "execution_count": 1,
         "source": "from pathlib import Path\nimport importlib.util\nimport sys\n\n"
                   "root = Path.cwd()\nif not (root / '图2.3_PBMC-imtra-效率.py').exists():\n"
                   "    root = Path(r'E:\\my_research\\画图\\我们的图\\图2')\n"
                   "deps = root.parents[2] / '_work' / 'table1_python_deps'\n"
                   "if deps.is_dir():\n    sys.path.insert(0, str(deps))\n"
                   "spec = importlib.util.spec_from_file_location('efficiency_chart', root / '图2.3_PBMC-imtra-效率.py')\n"
                   "chart = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(chart)\n"
                   "data = chart.prepare_data()\nchart.plot_chart(data)\n"
                   "from IPython.display import Image, display\ndisplay(Image(filename=str(root / '图2.3_PBMC-imtra-效率.png')))\n",
         "outputs": [{"output_type": "display_data", "metadata": {}, "data": {
             "image/png": preview, "text/plain": "PBMC intra efficiency: trainable parameters, cells/sec, peak GPU memory"}}]},
    ]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / f"{STEM}.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deps", type=Path, default=ROOT.parents[2] / "_work" / "table1_python_deps")
    args = parser.parse_args()
    if args.deps.is_dir():
        sys.path.insert(0, str(args.deps))
    import matplotlib
    matplotlib.use("Agg")
    data = prepare_data()
    preview = plot_chart(data)
    write_support_files(data, preview)
    for row in data["records"]:
        print(f"{row['method']}: params={row['trainable_params']:,}; VRAM={row['peak_gpu_memory_gib']['mean']} GiB; cells/s={row['throughput_cells_per_sec']['mean']:,}")
    print("Saved PNG, vector PDF/SVG, JSON data, notebook and caption.")


if __name__ == "__main__":
    main()

