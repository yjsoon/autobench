#!/usr/bin/env -S uv run --with seaborn --with pandas --no-project python
"""Generate the blog's data charts as SVG.

Run:  ./assets/make_plots.py         (uv pulls seaborn/pandas into a throwaway env)
Data: assets/data/*.csv   →   Output: assets/plots/*.svg  (gitignored, regenerate on demand)

Add a chart = add a CSV + a function + one call at the bottom. No framework.
"""
import pathlib
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import seaborn as sns

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)

# dataviz palette (light mode, validated): neutral baseline + blue/aqua categorical slots.
BASE, MTP, AQUA = "#898781", "#2a78d6", "#1baf7a"
sns.set_theme(style="whitegrid", font="sans-serif")

# ponytail: static SVG for a Jekyll post — no hover/dark-mode/table-view layer (those
# are the interactive-HTML path). Relief rule met by direct value labels on every bar.


def label_bars(ax, fmt="{:.0f}"):
    for c in ax.containers:
        ax.bar_label(c, fmt=fmt, padding=2, fontsize=8, color="#52514e")


def base_vs_mtp():
    df = pd.read_csv(DATA / "base_vs_mtp.csv")
    df["label"] = df["model"] + "\n" + df["quant"]
    long = df.melt(id_vars="label", value_vars=["base", "mtp"],
                   var_name="config", value_name="decode")
    long["config"] = long["config"].map({"base": "base", "mtp": "+ MTP"})

    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(long, x="label", y="decode", hue="config",
                palette={"base": BASE, "+ MTP": MTP}, saturation=1, ax=ax)
    label_bars(ax)
    ax.set_xlabel("")
    ax.set_ylabel("decode tok/s (aggregate, conc-32)")
    ax.set_title("Native MTP vs base — vLLM, ShareGPT V3")
    ax.legend(title="", frameon=False)
    ax.tick_params(axis="x", labelsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "base_vs_mtp.svg")
    plt.close(fig)
    print("wrote", OUT / "base_vs_mtp.svg")


def mtp_vs_dflash():
    df = pd.read_csv(DATA / "mtp_vs_dflash_35b.csv")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    order = ["base", "MTP", "DFlash"]
    sns.lineplot(df, x="concurrency", y="decode", hue="method", style="method",
                 hue_order=order, style_order=order, markers=True, dashes=False,
                 markersize=8, linewidth=2,
                 palette={"base": BASE, "MTP": MTP, "DFlash": AQUA}, ax=ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(sorted(df.concurrency.unique()))
    ax.set_yticks([100, 150, 200, 300, 400, 500])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # direct end labels instead of relying on the legend (aqua is sub-3:1 → relief rule)
    for method, color in (("base", BASE), ("MTP", MTP), ("DFlash", AQUA)):
        end = df[df.method == method].sort_values("concurrency").iloc[-1]
        ax.annotate(f"{method} {end.decode:.0f}", (end.concurrency, end.decode),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=color, fontweight="bold")

    # crossover: DFlash leads only at conc-1 (101.9 > 93.9); MTP is ahead from conc-2 on.
    ax.annotate("MTP overtakes\nby conc-2", (2, 154), xytext=(2.9, 110),
                textcoords="data", fontsize=8, color="#52514e",
                arrowprops=dict(arrowstyle="->", color="#898781", lw=1))

    ax.set_xlabel("concurrency (requests)")
    ax.set_ylabel("decode tok/s (aggregate)")
    ax.set_title("Qwen3.6-35B-A3B NVFP4, vLLM")
    ax.legend(title="", frameon=False, loc="upper left")
    ax.margins(x=0.12)
    # ponytail: DFlash conc-8/32 not measured — footnote it rather than hide the gap.
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "mtp_vs_dflash_35b.svg")
    plt.close(fig)
    print("wrote", OUT / "mtp_vs_dflash_35b.svg")


if __name__ == "__main__":
    base_vs_mtp()
    mtp_vs_dflash()
