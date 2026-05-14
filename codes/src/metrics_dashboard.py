"""
codes/src/metrics_dashboard.py
==============================
SEA-DP Metrics Visualization Dashboard

Plugs into the existing free-test-gui.py experiment results.
Call show_metrics_dashboard(results_dict) after run_experiment() to
display a styled Tkinter metrics window.

Alternatively, run standalone with --demo to preview with sample data.

Metrics displayed:
  TEC  - Topological Error Count
  SEC  - Shared Edge Consistency
  HD   - Hausdorff Distance
  VRR  - Vertex Reduction Rate
  T    - Execution Time
"""

import os
import sys
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS = {
    "bg":        "#f5f7fb",   # soft off-white
    "panel":     "#ffffff",   # clean white cards
    "border":    "#dfe3eb",   # subtle gray border

    "original":  "#4f8cc9",   # cleaner blue
    "std_dp":    "#e07a5f",   # warm coral
    "sea_dp":    "#4caf7d",   # fresh green

    "text":      "#1f2937",   # dark gray text
    "subtext":   "#6b7280",   # muted gray

    "good":      "#4caf7d",
    "warn":      "#d9a441",
    "bad":       "#d96b5f",

    "accent":    "#7c6cf2",   # modern soft violet
}


# ---------------------------------------------------------------------------
# Main dashboard builder
# ---------------------------------------------------------------------------

def show_metrics_dashboard(results: dict, parent=None):
    """
    Open the metrics dashboard window.

    Parameters
    ----------
    results : dict
        Must contain keys:
            feature1, feature2, title_name, tolerance,
            orig_tec, orig_vertices,
            std_tec, std_sec, std_hd, std_vrr, std_time, std_vertices,
                std_gaps, std_overlaps, std_invalid,
            sea_tec, sea_sec, sea_hd, sea_vrr, sea_time, sea_vertices,
                sea_gaps, sea_overlaps, sea_invalid,
                sea_n_edges, sea_n_arcs
    parent : tk.Tk or None
        If provided, opens as a Toplevel. Otherwise creates a new Tk root.
    """

    win = tk.Toplevel(parent) if parent else tk.Tk()
    win.title("SEA-DP Metrics Dashboard")
    win.configure(bg=COLORS["bg"])
    win.geometry("1200x740")
    win.resizable(True, True)

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------

    hdr = tk.Frame(win, bg=COLORS["bg"])
    hdr.pack(fill="x", padx=24, pady=(18, 0))

    tk.Label(
        hdr,
        text="SEA-DP  \u00b7  Metrics Dashboard",
        font=("Consolas", 17, "bold"),
        fg=COLORS["accent"],
        bg=COLORS["bg"],
    ).pack(side="left")

    subtitle = (
        f"{results['feature1']}  \u00d7  {results['feature2']}  "
        f"\u2502  \u03b5 = {results['tolerance']:,.0f} m"
    )

    tk.Label(
        hdr,
        text=subtitle,
        font=("Consolas", 10),
        fg=COLORS["subtext"],
        bg=COLORS["bg"],
    ).pack(side="left", padx=(18, 0), pady=(4, 0))

    ttk.Separator(win, orient="horizontal").pack(fill="x", padx=24, pady=10)

    # -----------------------------------------------------------------------
    # Build the matplotlib figure
    # -----------------------------------------------------------------------

    fig = plt.Figure(figsize=(14.5, 7.5), facecolor=COLORS["bg"])
    fig.subplots_adjust(
        left=0.04, right=0.98,
        top=0.88, bottom=0.12,
        wspace=0.40, hspace=0.55,
    )

    outer = gridspec.GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[3.2, 1],
        hspace=0.55,
    )

    bar_grid = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=outer[0], wspace=0.45
    )

    metrics_data = [
        ("tec",  "TEC",           results["orig_tec"],  results["std_tec"],  results["sea_tec"],  False),
        ("sec",  "SEC",           1.0,                  results["std_sec"],  results["sea_sec"],  True),
        ("hd",   "Hausdorff (m)", 0.0,                  results["std_hd"],   results["sea_hd"],   False),
        ("vrr",  "VRR",           0.0,                  results["std_vrr"],  results["sea_vrr"],  True),
        ("time", "Exec. Time (s)",0.0,                  results["std_time"], results["sea_time"], False),
    ]

    algo_labels = ["Original", "Std DP", "SEA-DP"]
    algo_colors = [COLORS["original"], COLORS["std_dp"], COLORS["sea_dp"]]

    for col_idx, (key, label, v_orig, v_std, v_sea, higher_better) in enumerate(metrics_data):
        ax = fig.add_subplot(bar_grid[0, col_idx])

        vals = [v_orig, v_std, v_sea]

        # Scale and format — use a fixed fmt_key string to avoid closure capture bug
        if key in ("vrr", "sec"):
            bar_vals = [v * 100 for v in vals]
            fmt_key = "pct"
        elif key == "hd":
            bar_vals = vals
            fmt_key = "hd"
        elif key == "time":
            bar_vals = vals
            fmt_key = "time"
        else:
            bar_vals = vals
            fmt_key = "int"

        def fmt(v, fk=fmt_key):
            if fk == "pct":  return f"{v:.1f}%"
            if fk == "hd":   return f"{v:,.0f}"
            if fk == "time": return f"{v:.3f}s"
            return str(int(v))

        x = np.arange(len(algo_labels))
        bars = ax.bar(
            x,
            bar_vals,
            color=algo_colors,
            width=0.58,
            zorder=3,
            edgecolor=COLORS["bg"],
            linewidth=0.8,
        )

        max_bar = max(bar_vals) if max(bar_vals) > 0 else 1

        for bar, bv in zip(bars, bar_vals):
            ypos = bar.get_height() + max_bar * 0.03
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                fmt(bv),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=COLORS["text"],
                fontfamily="Consolas",
            )

        ax.set_facecolor(COLORS["panel"])
        ax.spines[:].set_color(COLORS["border"])
        ax.tick_params(colors=COLORS["subtext"], labelsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(algo_labels, fontsize=7.5, color=COLORS["subtext"])
        ax.yaxis.set_tick_params(labelcolor=COLORS["subtext"])
        ax.set_title(label, color=COLORS["text"], fontsize=9.5, pad=6, fontweight="bold")
        ax.grid(axis="y", color=COLORS["border"], linestyle="--", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max_bar * 1.28)

        # Highlight winner with accent border
        compare = [bar_vals[1], bar_vals[2]]
        if higher_better:
            winner_idx = 1 + int(compare[1] > compare[0])
        else:
            winner_idx = 1 + int(compare[1] > compare[0] if key == "hd" else compare[1] < compare[0])

        bars[winner_idx].set_edgecolor(COLORS["accent"])
        bars[winner_idx].set_linewidth(2.0)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------

    ax_tbl = fig.add_subplot(outer[1])
    ax_tbl.set_facecolor(COLORS["panel"])
    ax_tbl.axis("off")

    col_labels = ["Algorithm", "TEC", "SEC", "HD (m)", "VRR", "Time (s)", "Vertices"]
    table_data = [
        [
            "Original",
            str(results["orig_tec"]),
            "1.0000",
            "0",
            "0.00%",
            "—",
            str(results["orig_vertices"]),
        ],
        [
            "Standard DP",
            str(results["std_tec"]),
            f"{results['std_sec']:.4f}",
            f"{results['std_hd']:,.1f}",
            f"{results['std_vrr']:.2%}",
            f"{results['std_time']:.4f}",
            str(results["std_vertices"]),
        ],
        [
            "SEA-DP",
            str(results["sea_tec"]),
            f"{results['sea_sec']:.4f}",
            f"{results['sea_hd']:,.1f}",
            f"{results['sea_vrr']:.2%}",
            f"{results['sea_time']:.4f}",
            str(results["sea_vertices"]),
        ],
    ]

    # softer table colors
    row_colors = [
        [COLORS["original"]] + ["#f8fafc"] * 6,
        [COLORS["std_dp"]]   + ["#f8fafc"] * 6,
        [COLORS["sea_dp"]]   + ["#f8fafc"] * 6,
    ]

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.75)

    # header styling
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#e9edf5")
        cell.set_edgecolor(COLORS["border"])
        cell.set_text_props(
            color=COLORS["accent"],
            fontweight="bold",
        )

    # body styling
    for i in range(1, 4):
        for j in range(len(col_labels)):
            cell = tbl[i, j]

            cell.set_edgecolor(COLORS["border"])

            # first column = algorithm label
            if j == 0:
                cell.set_text_props(
                    color="white",
                    fontweight="bold",
                )
            else:
                cell.set_text_props(
                    color=COLORS["text"],
                )

    ax_tbl.set_title(
        "Summary  ·  accent border = winner per metric",
        color=COLORS["subtext"],
        fontsize=8,
        pad=4,
        loc="left",
    )

    # -----------------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------------

    patches = [
        mpatches.Patch(color=COLORS["original"], label="Original"),
        mpatches.Patch(color=COLORS["std_dp"],   label="Standard DP"),
        mpatches.Patch(color=COLORS["sea_dp"],   label="SEA-DP"),
    ]
    fig.legend(
        handles=patches,
        loc="upper right",
        fontsize=8.5,
        framealpha=0.15,
        facecolor=COLORS["panel"],
        edgecolor=COLORS["border"],
        labelcolor=COLORS["text"],
    )

    # -----------------------------------------------------------------------
    # Embed in Tkinter window
    # -----------------------------------------------------------------------

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # Footer strip: TEC breakdown + SEA-DP algo stats
    footer = tk.Frame(win, bg=COLORS["panel"], pady=6)
    footer.pack(fill="x", padx=8, pady=(0, 8))

    def kv(parent, label, value, good=None):
        color = COLORS["text"]
        if good is True:
            color = COLORS["good"]
        elif good is False:
            color = COLORS["bad"]
        tk.Label(parent, text=f"{label}: ", fg=COLORS["subtext"],
                 bg=COLORS["panel"], font=("Consolas", 8)).pack(side="left")
        tk.Label(parent, text=str(value), fg=color,
                 bg=COLORS["panel"], font=("Consolas", 8, "bold")).pack(side="left", padx=(0, 16))

    kv(footer, "Std gaps",     results.get("std_gaps", "?"))
    kv(footer, "Std overlaps", results.get("std_overlaps", "?"))
    kv(footer, "Std invalid",  results.get("std_invalid", "?"))

    tk.Label(footer, text="|", fg=COLORS["border"], bg=COLORS["panel"],
             font=("Consolas", 9)).pack(side="left", padx=8)

    kv(footer, "SEA gaps",     results.get("sea_gaps", "?"),     good=results.get("sea_gaps", 1) == 0)
    kv(footer, "SEA overlaps", results.get("sea_overlaps", "?"), good=results.get("sea_overlaps", 1) == 0)
    kv(footer, "SEA invalid",  results.get("sea_invalid", "?"),  good=results.get("sea_invalid", 1) == 0)

    tk.Label(footer, text="|", fg=COLORS["border"], bg=COLORS["panel"],
             font=("Consolas", 9)).pack(side="left", padx=8)

    kv(footer, "Shared edges",   results.get("sea_n_edges", "N/A"))
    kv(footer, "Arcs assembled", results.get("sea_n_arcs", "N/A"))

    if parent is None:
        win.mainloop()

    return win


# ---------------------------------------------------------------------------
# Integration helper: pack run_experiment() locals into the results dict
# ---------------------------------------------------------------------------

def results_from_experiment(
    feature1, feature2, title_name, tolerance,
    orig_stats, orig_vertices,
    std_new_tec, std_raw_stats, std_sec, std_hd, std_vrr, std_time, std_vertices,
    sea_new_tec, sea_raw_stats, sea_sec, sea_hd, sea_vrr, sea_time, sea_vertices,
    sea_algo_stats,
) -> dict:
    """
    Pack run_experiment() local variables into the dict expected by
    show_metrics_dashboard().
    """
    return dict(
        feature1=feature1,
        feature2=feature2,
        title_name=title_name,
        tolerance=tolerance,

        orig_tec=orig_stats.get("tec", 0),
        orig_vertices=orig_vertices,

        std_tec=std_new_tec,
        std_sec=std_sec,
        std_hd=std_hd,
        std_vrr=std_vrr,
        std_time=std_time,
        std_vertices=std_vertices,
        std_gaps=std_raw_stats.get("n_gaps", 0),
        std_overlaps=std_raw_stats.get("n_overlaps", 0),
        std_invalid=std_raw_stats.get("n_invalid", 0),

        sea_tec=sea_new_tec,
        sea_sec=sea_sec,
        sea_hd=sea_hd,
        sea_vrr=sea_vrr,
        sea_time=sea_time,
        sea_vertices=sea_vertices,
        sea_gaps=sea_raw_stats.get("n_gaps", 0),
        sea_overlaps=sea_raw_stats.get("n_overlaps", 0),
        sea_invalid=sea_raw_stats.get("n_invalid", 0),
        sea_n_edges=sea_algo_stats.get("n_edges_shared", "N/A"),
        sea_n_arcs=sea_algo_stats.get("n_arcs_assembled", "N/A"),
    )


# ---------------------------------------------------------------------------
# Standalone demo (python metrics_dashboard.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = dict(
        feature1="Philippines",
        feature2="Indonesia",
        title_name="Countries",
        tolerance=5000,

        orig_tec=0,
        orig_vertices=4812,

        std_tec=3,
        std_sec=0.7241,
        std_hd=4823.18,
        std_vrr=0.7132,
        std_time=0.214,
        std_vertices=1381,
        std_gaps=1,
        std_overlaps=2,
        std_invalid=0,

        sea_tec=0,
        sea_sec=0.9887,
        sea_hd=4650.92,
        sea_vrr=0.6994,
        sea_time=0.389,
        sea_vertices=1448,
        sea_gaps=0,
        sea_overlaps=0,
        sea_invalid=0,
        sea_n_edges=218,
        sea_n_arcs=12,
    )

    show_metrics_dashboard(sample)
