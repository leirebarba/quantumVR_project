"""
Q2 Analysis: Does VR have an impact on attitudes beyond learning?

Groups:
  - BAM (treatment)  : Segovia BAM + Madrid BAM combined (n=40 with both pre+post)
  - Non-BAM (control): Madrid Non-BAM only               (n=9  with both pre+post)

Metrics (1-5 Likert scale, same question asked pre and post):
  - Interest, Relevance, Future work, Confidence

Filter: only students who answered BOTH pre and post for all 4 metrics.

WARNING: Non-BAM post response rate is only 9/24 (37%) - flagged in all plots.
WARNING: Groups are NOT comparable at baseline - Non-BAM started higher on all metrics.
         Score change (delta) is therefore the primary comparison.

Run:  python src/q2_attitudes.py
      -> saves 8 figures to outputs_q2/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from src.configuration import MADRID_FILE, SEGOVIA_FILE, PRE_POST_METRICS, BASE_DIR

OUT = BASE_DIR / "outputs_q2"
OUT.mkdir(parents=True, exist_ok=True)

C_BAM    = "#2E86AB"
C_NONBAM = "#E07A5F"

ATTITUDE_METRICS = {k: v for k, v in PRE_POST_METRICS.items() if k != "Test score"}
M_LABELS = list(ATTITUDE_METRICS.keys())


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_filter():
    madrid  = pd.read_excel(MADRID_FILE)
    segovia = pd.read_excel(SEGOVIA_FILE)
    madrid  = madrid[madrid["Relationship to IE"] != "Professor/Lecturer"].copy()
    segovia = segovia[segovia["Relationship to IE"] != "Professor/Lecturer"].copy()

    madrid["is_BAM"]  = madrid["What degree are you studying?"].str.contains("BAM", na=False)
    segovia["is_BAM"] = segovia["What degree are you studying?"].str.contains("BAM", na=False)

    bam     = pd.concat([madrid[madrid["is_BAM"]], segovia[segovia["is_BAM"]]],
                        ignore_index=True)
    non_bam = madrid[~madrid["is_BAM"]].copy()

    pre_cols  = [v[0] for v in ATTITUDE_METRICS.values()]
    post_cols = [v[1] for v in ATTITUDE_METRICS.values()]

    for df in [bam, non_bam]:
        for metric, (pre_col, post_col) in ATTITUDE_METRICS.items():
            df[f"pre_{metric}"]   = pd.to_numeric(df[pre_col],  errors="coerce")
            df[f"post_{metric}"]  = pd.to_numeric(df[post_col], errors="coerce")
            df[f"delta_{metric}"] = df[f"post_{metric}"] - df[f"pre_{metric}"]

    bam     = bam.dropna(subset=pre_cols + post_cols).copy()
    non_bam = non_bam.dropna(subset=pre_cols + post_cols).copy()

    return bam, non_bam


# ==============================================================================
# HELPERS
# ==============================================================================

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def _h_label(h):
    a = abs(h)
    if a < 0.2:   return "negligible"
    elif a < 0.5: return "small"
    elif a < 0.8: return "medium"
    else:         return "large"


def _cohen_h_delta(vb, vn):
    """Cohen's h on delta values normalised to [0,1] over the [-4,+4] range."""
    pb = np.clip((vb + 4) / 8, 0.0, 1.0)
    pn = np.clip((vn + 4) / 8, 0.0, 1.0)
    return 2 * np.arcsin(np.sqrt(pb)) - 2 * np.arcsin(np.sqrt(pn))


# ==============================================================================
# PART A - COMBINED 2x2 FIGURES PER LEVEL
# ==============================================================================

def plot_A1_baseline(bam, non_bam):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    bins = np.arange(0.5, 6.5, 1)

    for idx, metric in enumerate(M_LABELS):
        ax  = axes[idx]
        d_b = bam[f"pre_{metric}"].dropna()
        d_n = non_bam[f"pre_{metric}"].dropna()

        ax.hist(d_b, bins=bins, color=C_BAM,    alpha=0.65, edgecolor="white",
                label=f"BAM (n={len(d_b)})")
        ax.hist(d_n, bins=bins, color=C_NONBAM, alpha=0.65, edgecolor="white",
                label=f"Non-BAM (n={len(d_n)})")
        ax.axvline(d_b.mean(), color=C_BAM,    lw=2, ls="--",
                   label=f"BAM mean={d_b.mean():.2f}")
        ax.axvline(d_n.mean(), color=C_NONBAM, lw=2, ls="--",
                   label=f"Non-BAM mean={d_n.mean():.2f}")

        u, p = stats.mannwhitneyu(d_b, d_n, alternative="two-sided")
        verdict = "comparable" if p > 0.05 else "differ"
        ax.set_title(f"{metric}\nMWU p={p:.3f} ({verdict})",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Pre score (1-5 Likert)", fontsize=9)
        ax.set_ylabel("Number of students", fontsize=9)
        ax.set_xticks(range(1, 6))
        ax.legend(fontsize=7.5)
        _style(ax)

    fig.suptitle(
        "A1 - Baseline Check: Pre-test Attitude Distributions\n"
        "WARNING: groups are not comparable at baseline on most metrics",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "A1_baseline.png", dpi=150)
    plt.close()


def plot_A2_posttest(bam, non_bam):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    bins = np.arange(0.5, 6.5, 1)

    for idx, metric in enumerate(M_LABELS):
        ax  = axes[idx]
        d_b = bam[f"post_{metric}"].dropna()
        d_n = non_bam[f"post_{metric}"].dropna()

        ax.hist(d_b, bins=bins, color=C_BAM,    alpha=0.65, edgecolor="white",
                label=f"BAM (n={len(d_b)})")
        ax.hist(d_n, bins=bins, color=C_NONBAM, alpha=0.65, edgecolor="white",
                label=f"Non-BAM (n={len(d_n)}) [!]")
        ax.axvline(d_b.mean(), color=C_BAM,    lw=2, ls="--",
                   label=f"BAM mean={d_b.mean():.2f}")
        ax.axvline(d_n.mean(), color=C_NONBAM, lw=2, ls="--",
                   label=f"Non-BAM mean={d_n.mean():.2f}")

        u, p = stats.mannwhitneyu(d_b, d_n, alternative="two-sided")
        ax.set_title(f"{metric}\nMWU p={p:.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Post score (1-5 Likert)", fontsize=9)
        ax.set_ylabel("Number of students", fontsize=9)
        ax.set_xticks(range(1, 6))
        ax.legend(fontsize=7.5)
        _style(ax)

    fig.suptitle(
        "A2 - Post-test Attitude Distributions\n"
        "[!] Non-BAM n=9 only (37% response rate) - interpret with caution",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "A2_posttest.png", dpi=150)
    plt.close()


def plot_A3_change(bam, non_bam):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    bins = np.arange(-4.5, 5.5, 1)

    for idx, metric in enumerate(M_LABELS):
        ax  = axes[idx]
        d_b = bam[f"delta_{metric}"].dropna()
        d_n = non_bam[f"delta_{metric}"].dropna()

        ax.hist(d_b, bins=bins, color=C_BAM,    alpha=0.65, edgecolor="white",
                label=f"BAM (n={len(d_b)})")
        ax.hist(d_n, bins=bins, color=C_NONBAM, alpha=0.65, edgecolor="white",
                label=f"Non-BAM (n={len(d_n)}) [!]")
        ax.axvline(0,          color="black",  lw=1,   ls="-",  alpha=0.35)
        ax.axvline(d_b.mean(), color=C_BAM,    lw=2,   ls="--",
                   label=f"BAM mean={d_b.mean():.2f}")
        ax.axvline(d_n.mean(), color=C_NONBAM, lw=2,   ls="--",
                   label=f"Non-BAM mean={d_n.mean():.2f}")

        p_b   = stats.wilcoxon(d_b).pvalue if len(d_b) > 5 else float("nan")
        p_n   = stats.wilcoxon(d_n).pvalue if len(d_n) > 5 else float("nan")
        u, p_mwu = stats.mannwhitneyu(d_b, d_n, alternative="two-sided")
        p_b_str = f"{p_b:.3f}" if not np.isnan(p_b) else "n/a"
        p_n_str = f"{p_n:.3f}" if not np.isnan(p_n) else "n/a"

        ax.set_title(
            f"{metric}\n"
            f"Wilcoxon: BAM p={p_b_str}  |  Non-BAM p={p_n_str}\n"
            f"Between-group MWU p={p_mwu:.3f}",
            fontsize=9, fontweight="bold",
        )
        ax.set_xlabel("Delta (post - pre)", fontsize=9)
        ax.set_ylabel("Number of students", fontsize=9)
        ax.legend(fontsize=7.5)
        _style(ax)

    fig.suptitle(
        "A3 - Score Change Distributions (post - pre)\n"
        "[!] Non-BAM n=9 only - interpret with caution",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "A3_change.png", dpi=150)
    plt.close()


def plot_A4_table(bam, non_bam):
    rows = []
    for metric in M_LABELS:
        for grp_label, df in [("BAM (treatment)", bam), ("Non-BAM (control) [!]", non_bam)]:
            pre   = df[f"pre_{metric}"].dropna()
            post  = df[f"post_{metric}"].dropna()
            delta = df[f"delta_{metric}"].dropna()
            p_wil = stats.wilcoxon(delta).pvalue if len(delta) > 5 else float("nan")
            p_str = f"{p_wil:.3f}" if not np.isnan(p_wil) else "n/a"
            rows.append([
                metric, grp_label, len(delta),
                f"{pre.mean():.2f} +/- {pre.std():.2f}",
                f"{post.mean():.2f} +/- {post.std():.2f}",
                f"{delta.mean():.2f} +/- {delta.std():.2f}",
                f"{delta.median():.1f}",
                p_str,
            ])

        u_pre, p_pre = stats.mannwhitneyu(
            bam[f"pre_{metric}"].dropna(),
            non_bam[f"pre_{metric}"].dropna(),
            alternative="two-sided")
        u_d, p_d = stats.mannwhitneyu(
            bam[f"delta_{metric}"].dropna(),
            non_bam[f"delta_{metric}"].dropna(),
            alternative="two-sided")
        rows.append(["", "Between-group MWU p", "--",
                     f"p={p_pre:.3f}", "--", f"p={p_d:.3f}", "--", "--"])

    cols = ["Metric", "Group", "n",
            "Pre mean +/- SD", "Post mean +/- SD",
            "Delta mean +/- SD", "Delta median",
            "Wilcoxon p\n(Delta != 0)"]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.65)

    metric_colors = {
        "Interest":    "#e8f4f8",
        "Relevance":   "#fef9e7",
        "Future work": "#eafaf1",
        "Confidence":  "#fdf2f8",
    }
    row_idx = 1
    for metric in M_LABELS:
        c = metric_colors[metric]
        for j in range(3):
            for col_idx in range(len(cols)):
                cell = tbl[(row_idx, col_idx)]
                if j == 2:
                    cell.set_facecolor("#f0f0f0")
                    cell.set_text_props(color="#555555", style="italic")
                else:
                    cell.set_facecolor(c)
            row_idx += 1

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2E4057")
            cell.set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "A4 - Attitude Metrics: Summary Statistics\n"
        "[!] Non-BAM n=9 for post/delta  |  Scale 1-5 Likert  |  "
        "Groups NOT comparable at baseline",
        fontsize=11, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    fig.savefig(OUT / "A4_table.png", dpi=150, bbox_inches="tight")
    plt.close()


# ==============================================================================
# PART B - ALL 4 METRICS TOGETHER
# ==============================================================================

def plot_B1_pre(bam, non_bam):
    x, w = np.arange(len(M_LABELS)), 0.35
    vals_b = [bam[f"pre_{m}"].mean()    for m in M_LABELS]
    vals_n = [non_bam[f"pre_{m}"].mean() for m in M_LABELS]
    sems_b = [bam[f"pre_{m}"].sem()     for m in M_LABELS]
    sems_n = [non_bam[f"pre_{m}"].sem()  for m in M_LABELS]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bs = ax.bar(x - w/2, vals_b, w, yerr=sems_b, capsize=4,
                color=C_BAM,    alpha=0.85, edgecolor="white",
                label=f"BAM - treatment (n={len(bam)})")
    bn = ax.bar(x + w/2, vals_n, w, yerr=sems_n, capsize=4,
                color=C_NONBAM, alpha=0.85, edgecolor="white",
                label=f"Non-BAM - control (n={len(non_bam)})")

    for bars, vals in [(bs, vals_b), (bn, vals_n)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(M_LABELS, fontsize=10)
    ax.set_ylabel("Mean score (1-5 Likert)", fontsize=10)
    ax.set_ylim(1, 5.8)
    ax.set_title("B1 - Pre-test Attitude Scores by Metric\n"
                 "WARNING: groups differ at baseline on most metrics",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "B1_pre_attitudes.png", dpi=150)
    plt.close()


def plot_B2_post(bam, non_bam):
    x, w = np.arange(len(M_LABELS)), 0.35
    vals_b = [bam[f"post_{m}"].mean()    for m in M_LABELS]
    vals_n = [non_bam[f"post_{m}"].mean() for m in M_LABELS]
    sems_b = [bam[f"post_{m}"].sem()     for m in M_LABELS]
    sems_n = [non_bam[f"post_{m}"].sem()  for m in M_LABELS]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bs = ax.bar(x - w/2, vals_b, w, yerr=sems_b, capsize=4,
                color=C_BAM,    alpha=0.85, edgecolor="white",
                label=f"BAM - treatment (n={len(bam)})")
    bn = ax.bar(x + w/2, vals_n, w, yerr=sems_n, capsize=4,
                color=C_NONBAM, alpha=0.85, edgecolor="white",
                label=f"Non-BAM - control (n={len(non_bam)}) [!]")

    for bars, vals in [(bs, vals_b), (bn, vals_n)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(M_LABELS, fontsize=10)
    ax.set_ylabel("Mean score (1-5 Likert)", fontsize=10)
    ax.set_ylim(1, 5.8)
    ax.set_title("B2 - Post-test Attitude Scores by Metric\n"
                 "[!] Non-BAM n=9 only - interpret with caution",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "B2_post_attitudes.png", dpi=150)
    plt.close()


def plot_B3_change(bam, non_bam):
    x, w = np.arange(len(M_LABELS)), 0.35
    vals_b = [bam[f"delta_{m}"].mean()    for m in M_LABELS]
    vals_n = [non_bam[f"delta_{m}"].mean() for m in M_LABELS]
    sems_b = [bam[f"delta_{m}"].sem()     for m in M_LABELS]
    sems_n = [non_bam[f"delta_{m}"].sem()  for m in M_LABELS]

    fig, ax = plt.subplots(figsize=(12, 6))
    bs = ax.bar(x - w/2, vals_b, w, yerr=sems_b, capsize=4,
                color=C_BAM,    alpha=0.85, edgecolor="white",
                label=f"BAM - treatment (n={len(bam)})")
    bn = ax.bar(x + w/2, vals_n, w, yerr=sems_n, capsize=4,
                color=C_NONBAM, alpha=0.85, edgecolor="white",
                label=f"Non-BAM - control (n={len(non_bam)}) [!]")

    for bars, vals in [(bs, vals_b), (bn, vals_n)]:
        for bar, v in zip(bars, vals):
            ypos = v + 0.03 if v >= 0 else v - 0.08
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"{v:+.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color="black", lw=0.9, ls="-", alpha=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(M_LABELS, fontsize=10)
    ax.set_ylabel("Mean change (post - pre)", fontsize=10)
    ax.set_title(
        "B3 - Attitude Change (post - pre) by Metric\n"
        "[!] Non-BAM n=9 only - interpret with caution",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "B3_change_attitudes.png", dpi=150)
    plt.close()


def plot_B4_stats_table(bam, non_bam):
    rows = []
    for metric in M_LABELS:
        d_b = bam[f"delta_{metric}"].dropna()
        d_n = non_bam[f"delta_{metric}"].dropna()
        pre_b = bam[f"pre_{metric}"].dropna()
        pre_n = non_bam[f"pre_{metric}"].dropna()

        p_wil_b = stats.wilcoxon(d_b).pvalue if len(d_b) > 5 else float("nan")
        p_wil_n = stats.wilcoxon(d_n).pvalue if len(d_n) > 5 else float("nan")
        _, p_pre = stats.mannwhitneyu(pre_b, pre_n, alternative="two-sided")
        _, p_mwu = stats.mannwhitneyu(d_b,   d_n,   alternative="two-sided")

        h   = _cohen_h_delta(d_b.mean(), d_n.mean())
        p_b_str = f"{p_wil_b:.3f}" if not np.isnan(p_wil_b) else "n/a"
        p_n_str = f"{p_wil_n:.3f}" if not np.isnan(p_wil_n) else "n/a"

        rows.append([
            metric,
            f"{pre_b.mean():.2f}", f"{pre_n.mean():.2f}", f"p={p_pre:.3f}",
            f"{d_b.mean():.2f}", f"{d_n.mean():.2f}",
            p_b_str, p_n_str,
            f"p={p_mwu:.3f}",
            f"{h:+.2f} ({_h_label(h)})",
        ])

    cols = [
        "Metric",
        "Pre\nBAM", "Pre\nNon-BAM", "MWU p\n(pre baseline)",
        "Delta\nBAM", "Delta\nNon-BAM",
        "Wilcoxon p\nBAM", "Wilcoxon p\nNon-BAM [!]",
        "MWU p\n(delta)", "Cohen's h\n(delta)",
    ]

    fig, ax = plt.subplots(figsize=(18, 3.8))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.0)

    metric_colors = {
        "Interest":    "#e8f4f8",
        "Relevance":   "#fef9e7",
        "Future work": "#eafaf1",
        "Confidence":  "#fdf2f8",
    }
    for r in range(1, len(rows) + 1):
        m = rows[r-1][0]
        c = metric_colors.get(m, "#ffffff")
        for col_idx in range(len(cols)):
            tbl[(r, col_idx)].set_facecolor(c)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2E4057")
            cell.set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "B4 - Attitude Metrics: Full Statistics Table\n"
        "[!] Non-BAM n=9 for delta columns  |  Scale 1-5 Likert  |  "
        "Groups NOT comparable at baseline",
        fontsize=11, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    fig.savefig(OUT / "B4_stats_table.png", dpi=150, bbox_inches="tight")
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def run():
    bam, non_bam = load_and_filter()

    print("=" * 54)
    print("  Q2 - VR Effect on Attitudes")
    print("=" * 54)
    print(f"\n  BAM (treatment)    n = {len(bam)}")
    print(f"  Non-BAM (control)  n = {len(non_bam)}  [!] low post response rate")
    print(f"  Metrics: {', '.join(M_LABELS)}")

    print("\n-- Part A: Per metric -------------------------")
    plot_A1_baseline(bam, non_bam)
    print("  A1 baseline done")
    plot_A2_posttest(bam, non_bam)
    print("  A2 posttest done")
    plot_A3_change(bam, non_bam)
    print("  A3 change done")
    plot_A4_table(bam, non_bam)
    print("  A4 table done")

    print("\n-- Part B: All metrics together ---------------")
    plot_B1_pre(bam, non_bam)
    print("  B1 pre done")
    plot_B2_post(bam, non_bam)
    print("  B2 post done")
    plot_B3_change(bam, non_bam)
    print("  B3 change done")
    plot_B4_stats_table(bam, non_bam)
    print("  B4 stats table done")

    print("\n-- Console summary ----------------------------")
    print(f"\n  {'Metric':<15} {'Pre BAM':>8} {'Pre NB':>8} {'Pre p':>7}"
          f" {'dBAM':>7} {'dNB':>6} {'MWU p':>7} {'Cohen h':>10}")
    print("  " + "-" * 75)
    for metric in M_LABELS:
        d_b   = bam[f"delta_{metric}"].dropna()
        d_n   = non_bam[f"delta_{metric}"].dropna()
        pre_b = bam[f"pre_{metric}"].dropna()
        pre_n = non_bam[f"pre_{metric}"].dropna()
        _, p_pre = stats.mannwhitneyu(pre_b, pre_n, alternative="two-sided")
        _, p_mwu = stats.mannwhitneyu(d_b,   d_n,   alternative="two-sided")
        h = _cohen_h_delta(d_b.mean(), d_n.mean())
        print(f"  {metric:<15} {pre_b.mean():>8.2f} {pre_n.mean():>8.2f} {p_pre:>7.3f}"
              f" {d_b.mean():>7.2f} {d_n.mean():>6.2f} {p_mwu:>7.3f} {h:>+10.2f} ({_h_label(h)})")

    print(f"\n  Figures saved to: {OUT}\n")


if __name__ == "__main__":
    run()

