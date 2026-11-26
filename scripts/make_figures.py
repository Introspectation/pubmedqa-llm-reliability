"""Regenerate the Q4 class-recall PNG from the published held-out report."""

import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
Q4_REPORT = ROOT / "repro/qlora/optimized/q4_heldout_eval.json"
Q4_FIGURE = ROOT / "results/q4-class-recall.png"

INK = "#11243b"
MUTED = "#486174"
BASE_COLOR = "#53708b"
ADAPTER_COLOR = "#087d70"


def class_counts(matrix, label):
    """Return the correct, actual, and predicted counts for one label."""
    rows = matrix["values"]
    row = rows[matrix["row_labels"].index(label)]
    column = matrix["column_labels"].index(label)
    return row[column], sum(row), sum(values[column] for values in rows)


def add_panel(fig, bounds, *, face="#ffffff", edge="#dbe4eb"):
    x, y, width, height = bounds
    fig.add_artist(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.006,rounding_size=0.018",
        linewidth=0.9, edgecolor=edge, facecolor=face,
        transform=fig.transFigure, zorder=0,
    ))


def draw_class_recall(fig, models, labels):
    add_panel(fig, (0.060, 0.154, 0.585, 0.546))
    fig.text(0.086, 0.634, "Recall by answer class", color=INK, fontsize=17, weight="bold")

    ax = fig.add_axes([0.129, 0.277, 0.487, 0.302], zorder=2, facecolor="none")
    centers = list(range(len(labels)))
    series = (("base", -0.155, BASE_COLOR), ("B", 0.155, ADAPTER_COLOR))
    for name, offset, color in series:
        matrix = models[name]["confusion_matrix"]
        counts = [class_counts(matrix, label) for label in labels]
        recall = [correct / actual for correct, actual, _ in counts]
        bars = ax.bar(
            [center + offset for center in centers], recall,
            width=0.27, color=color, zorder=3,
        )
        ax.bar_label(
            bars, labels=[f"{value:.1%}" for value in recall],
            padding=4, fontsize=11, color=INK,
        )

    ax.set_xlim(-0.57, len(labels) - 0.43)
    ax.set_ylim(0, 1.01)
    ax.set_xticks(centers, labels, fontsize=13, color=INK)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="both", length=0, pad=9)
    ax.tick_params(axis="y", labelsize=10, labelcolor=MUTED)
    ax.grid(axis="y", color="#e7edf2", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(0.092, 0.185, "●", color=BASE_COLOR, fontsize=14)
    fig.text(0.113, 0.186, "Base", color=INK, fontsize=12)
    fig.text(0.194, 0.185, "●", color=ADAPTER_COLOR, fontsize=14)
    fig.text(0.215, 0.186, "QLoRA Adapter B", color=INK, fontsize=12)


def draw_result_summary(fig, base, adapter):
    base_correct, actual, base_predicted = class_counts(base["confusion_matrix"], "maybe")
    adapter_correct, _, adapter_predicted = class_counts(adapter["confusion_matrix"], "maybe")

    add_panel(fig, (0.669, 0.154, 0.270, 0.546))
    fig.text(0.697, 0.630, "Macro-F1", color=MUTED, fontsize=15, weight="bold")
    fig.text(
        0.697, 0.552, f"{base['macro_f1']:.3f} → {adapter['macro_f1']:.3f}",
        color=INK, fontsize=26, weight="bold",
    )
    fig.text(0.697, 0.503, "Base to QLoRA Adapter B", color=MUTED, fontsize=11.5)

    add_panel(fig, (0.695, 0.201, 0.220, 0.251), face="#fff6e7", edge="#f3e5cc")
    fig.text(0.711, 0.410, "‘Maybe’ predictions", color="#88460f", fontsize=11.5, weight="bold")
    fig.text(
        0.711, 0.359, f"{base_predicted} → {adapter_predicted}",
        color=INK, fontsize=22, weight="bold",
    )
    fig.text(0.711, 0.308, "Correct ‘maybe’ predictions", color="#88460f", fontsize=11.5, weight="bold")
    fig.text(
        0.711, 0.257, f"{base_correct} → {adapter_correct}",
        color=INK, fontsize=22, weight="bold",
    )
    fig.text(0.711, 0.225, f"Out of {actual} true ‘maybe’ cases", color=MUTED, fontsize=10.5)


def main():
    report = json.loads(Q4_REPORT.read_text(encoding="utf-8"))
    models = report["models"]
    labels = models["base"]["confusion_matrix"]["row_labels"]

    fig = Figure(figsize=(12.8, 7.2), dpi=125, facecolor="#f1f5f8")
    FigureCanvasAgg(fig)
    fig.text(0.060, 0.911, "PubMedQA · Q4 held-out evaluation", color=ADAPTER_COLOR, fontsize=16, weight="bold")
    fig.text(
        0.060, 0.823, "Adapter B predicts ‘maybe’ less and rarely catches it",
        color=INK, fontsize=28, weight="bold",
    )
    fig.text(
        0.060, 0.763, f"Qwen3-1.7B  ·  {report['n']} official test questions  ·  Shared label-scoring protocol",
        color=MUTED, fontsize=15,
    )

    draw_class_recall(fig, models, labels)
    draw_result_summary(fig, models["base"], models["B"])

    fig.text(
        0.060, 0.079, "Source · github.com/Introspectation/pubmedqa-llm-reliability",
        color=MUTED, fontsize=11,
    )
    fig.text(0.940, 0.079, report["protocol"]["name"], color=MUTED, fontsize=11, ha="right")

    Q4_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Q4_FIGURE, metadata={
        "Title": "PubMedQA Q4 class recall for Base and QLoRA Adapter B",
        "Description": "Source: repro/qlora/optimized/q4_heldout_eval.json",
    })


if __name__ == "__main__":
    main()
