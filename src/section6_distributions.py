import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.configuration import PRE_POST_METRICS, get_output_dir

SECTION_OUTPUT_DIR = get_output_dir("section6_distributions")
GROUP_ORDER = [
    ("Madrid", "Before"),
    ("Madrid", "After"),
    ("Segovia", "Before"),
    ("Segovia", "After"),
]

# Colour palette
C_VR_PRE    = "#85C1E9"   # VR Before    — dark green
C_VR_POST   = "#76D7B0"   # VR After     — light mint
C_NOVR_PRE  = "#F0B27A"    # No VR Before — deep purple/pink
C_NOVR_POST = "#F4A6E0"   # No VR After  — light pink


def _metric_limits(metric_name):
    if metric_name == "Test score":
        return 0, 7, range(0, 8)
    return 1, 5, range(1, 6)


def _build_distribution_df(df, metric_name, pre_col, post_col):
    """Reshape the data but only keep participants with BOTH pre and post values."""
    df = df.dropna(subset=[pre_col, post_col]).copy()
    print(f"{metric_name}: {len(df)} complete cases")

    before = df[["Participant", "site"]].copy()
    before["time"]   = "Before"
    before["value"]  = pd.to_numeric(df[pre_col], errors="coerce")
    before["metric"] = metric_name

    after = df[["Participant", "site"]].copy()
    after["time"]   = "After"
    after["value"]  = pd.to_numeric(df[post_col], errors="coerce")
    after["metric"] = metric_name

    out = pd.concat([before, after], ignore_index=True)
    return out.dropna(subset=["value"])


# ══════════════════════════════════════════════════════════════════════════════
# ORIGINAL SUBSECTION — 2×2 grid, one panel per (site × time)
# ══════════════════════════════════════════════════════════════════════════════

def plot_metric_histograms(df, metric_name, pre_col, post_col):
    """Plot histograms of scores for a specific metric by site and time, arranged in a 2×2 grid."""
    plot_df = _build_distribution_df(df, metric_name, pre_col, post_col)
    min_val, max_val, tick_values = _metric_limits(metric_name)
    bins = [x - 0.5 for x in range(min_val, max_val + 2)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (site, time) in zip(axes, GROUP_ORDER):
        sub = plot_df[(plot_df["site"] == site) & (plot_df["time"] == time)]["value"]
        ax.hist(sub, bins=bins, edgecolor="black", alpha=0.8)
        ax.set_title(f"{site} - {time}")
        ax.set_xlim(min_val - 0.5, max_val + 0.5)
        ax.set_xticks(list(tick_values))
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"{metric_name}: score distribution by site and time", fontsize=14, weight="bold")
    fig.supxlabel("Observed value")
    fig.supylabel("Number of participants")
    plt.tight_layout()

    filename = metric_name.lower().replace(" ", "_") + "_histograms.png"
    plt.savefig(SECTION_OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_metric_histograms(df):
    """Generate histogram plots for all metrics defined in PRE_POST_METRICS."""
    for metric_name, (pre_col, post_col) in PRE_POST_METRICS.items():
        plot_metric_histograms(df, metric_name, pre_col, post_col)


# ══════════════════════════════════════════════════════════════════════════════
# NEW SUBSECTION — superposed histograms  (VR vs No-VR, Before vs After)
# ══════════════════════════════════════════════════════════════════════════════

def plot_metric_superposed(df, metric_name, pre_col, post_col):
    """
    Two panels side-by-side:
      Left  — VR group    (Madrid):  Before vs After superposed
      Right — No-VR group (Segovia): Before vs After superposed

    All four series are also drawn together on a single combined panel saved
    as a separate file.
    """
    plot_df = _build_distribution_df(df, metric_name, pre_col, post_col)
    min_val, max_val, tick_values = _metric_limits(metric_name)
    bins = np.array([x - 0.5 for x in range(min_val, max_val + 2)])

    # ── slice the four series ─────────────────────────────────────────────────
    vr_pre    = plot_df[(plot_df["site"] == "Madrid")  & (plot_df["time"] == "Before")]["value"]
    vr_post   = plot_df[(plot_df["site"] == "Madrid")  & (plot_df["time"] == "After") ]["value"]
    novr_pre  = plot_df[(plot_df["site"] == "Segovia") & (plot_df["time"] == "Before")]["value"]
    novr_post = plot_df[(plot_df["site"] == "Segovia") & (plot_df["time"] == "After") ]["value"]

    hist_kw = dict(bins=bins, alpha=0.55, edgecolor="white", linewidth=0.8)

    # ── Figure 1: two side-by-side panels (VR | No-VR) ───────────────────────
    fig, (ax_vr, ax_novr) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, pre, post, c_pre, c_post, label in [
        (ax_vr,   vr_pre,   vr_post,   C_VR_PRE,   C_VR_POST,   "VR (Madrid)"),
        (ax_novr, novr_pre, novr_post, C_NOVR_PRE, C_NOVR_POST, "No VR (Segovia)"),
    ]:
        ax.hist(pre,  color=c_pre,  label=f"Before  (n={len(pre)})",  **hist_kw)
        ax.hist(post, color=c_post, label=f"After   (n={len(post)})", **hist_kw)
        ax.axvline(pre.mean(),  color=c_pre,  lw=2, ls="--",
                   label=f"Before mean = {pre.mean():.2f}")
        ax.axvline(post.mean(), color=c_post, lw=2, ls=":",
                   label=f"After  mean = {post.mean():.2f}")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlim(min_val - 0.5, max_val + 0.5)
        ax.set_xticks(list(tick_values))
        ax.set_xlabel("Observed value", fontsize=10)
        ax.set_ylabel("Number of participants", fontsize=10)
        ax.legend(fontsize=8.5)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"{metric_name}: Before vs After — VR and No-VR groups",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fname_2panel = metric_name.lower().replace(" ", "_") + "_superposed_2panel.png"
    plt.savefig(SECTION_OUTPUT_DIR / fname_2panel, dpi=300, bbox_inches="tight")
    plt.close()

    # ── Figure 2: all four series in one panel ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))

    series = [
        (vr_pre,    C_VR_PRE,    f"VR — Before   (n={len(vr_pre)})"),
        (vr_post,   C_VR_POST,   f"VR — After    (n={len(vr_post)})"),
        (novr_pre,  C_NOVR_PRE,  f"No VR — Before (n={len(novr_pre)})"),
        (novr_post, C_NOVR_POST, f"No VR — After  (n={len(novr_post)})"),
    ]
    mean_styles = ["--", ":", "--", ":"]

    for (s, color, lbl), ms in zip(series, mean_styles):
        ax.hist(s, color=color, label=lbl, **hist_kw)
        ax.axvline(s.mean(), color=color, lw=1.8, ls=ms)

    ax.set_xlim(min_val - 0.5, max_val + 0.5)
    ax.set_xticks(list(tick_values))
    ax.set_xlabel("Observed value", fontsize=11)
    ax.set_ylabel("Number of participants", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"{metric_name}: all four groups superposed\n"
        f"Blue = VR (Madrid) · Orange = No VR (Segovia) · Solid = Before · Light = After",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fname_combined = metric_name.lower().replace(" ", "_") + "_superposed_combined.png"
    plt.savefig(SECTION_OUTPUT_DIR / fname_combined, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_metric_superposed(df):
    """Generate superposed histogram plots for all metrics defined in PRE_POST_METRICS."""
    for metric_name, (pre_col, post_col) in PRE_POST_METRICS.items():
        plot_metric_superposed(df, metric_name, pre_col, post_col)
