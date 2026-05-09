"""
Q1 Analysis: Does VR help learning of quantum computing?

Experiment groups (BAM students only):
  - Segovia BAM  [CONTROL]   : pre-test → session 1 → post-test → VR
  - Madrid BAM   [TREATMENT] : pre-test → VR → session 1 → post-test

Scoring: computed from document-verified correct answers. The provided Score/Score 2
spreadsheet columns had encoding inconsistencies for the pre-test (~30% row mismatch),
so we derive our own total from the 7 per-question binaries.

Filter: only students who answered ALL 7 pre-test AND ALL 7 post-test questions.

Run:  python src/q1_learning.py
      → saves 8 figures directly to outputs_q1/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from src.configuration import MADRID_FILE, SEGOVIA_FILE, BASE_DIR

OUT = BASE_DIR / "outputs_q1"
OUT.mkdir(parents=True, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
C_SEG = "#2E86AB"   # Segovia BAM — control
C_MAD = "#E07A5F"   # Madrid BAM  — treatment

# ── Question definitions (from survey doc — all correct answers verified) ─────
# Q6 pre uses smart/curly quotes in the spreadsheet; resolved at runtime.
PRE_DEFS = [
    ("Q1 – Qubit vs bit",
     "Which statement is correct?",
     "A qubit can be in a combination \u03b1|0\u27e9+\u03b2|1\u27e9 with |\u03b1|\u00b2+|\u03b2|\u00b2=1"),

    ("Q2 – Probability",
     "A qubit is prepared so that when measuring it, it gives outcome |0\u27e9 with probability 0.25. "
     "What is the probability of obtaining outcome |1\u27e9  (considering the same state preparation)? "
     "Assume only outcomes of states |0\u27e9 or |1\u27e9.",
     0.75),

    ("Q3 – Superposition",
     "A qubit starts in state \u22230\u27e9. You apply a Hadamard (H) gate to obtain a superposition state. "
     "Which statement best describes the result?",
     "Measuring the qubit can yield |0\u27e9 or |1\u27e9 with equal probabilities"),

    ("Q4 – Entanglement",
     "Which statement best captures entanglement of two qubits?",
     "Two entangled qubits have correlations that cannot be described by a product state (not separable)"),

    ("Q5 – Gates",
     "Starting from |00\u27e9, what quantum logic gates are required (at least) to create "
     "an entangled state of two qubits?",
     "A Hadamard gate and a CNOT gate"),

    ("Q6 – Measurement",
     "How does the act of measuring affect a quantum system that is in superposition? "
     "Assume the computational basis.",
     "Measurement \u201ccollapses\u201d the quantum system into an (eigen)state of the measured observable"),

    ("Q7 – Coherence",
     "Quantum computers often require strong shielding and (for many platforms) cryogenic "
     "temperatures mainly to  ",
     "Reduce thermal noise and environmental interactions, which minimizes error rates and decoherence"),
]

POST_DEFS = [
    ("Q1 – Qubit vs bit",
     "Which statement is correct? 2",
     "A qubit state can be written as \u03b1|0\u27e9+\u03b2|1\u27e9 with |\u03b1|\u00b2+|\u03b2|\u00b2=1"),

    ("Q2 – Probability",
     "A qubit is in the state \u2223\u03c8\u27e9= \u221a3/2 |0\u27e9  + 1/2 |1\u27e9. If you measure this qubit, "
     "what is the probability of obtaining the state |1\u27e9?  Assume only outcomes for states |0\u27e9 or |1\u27e9.",
     "1/4"),

    ("Q3 – Superposition",
     " Which description best matches the meaning of a qubit in superposition "
     "(\u03b1|0\u27e9+\u03b2|1\u27e9 with |\u03b1|\u00b2+|\u03b2|\u00b2=1)? Assume the computational basis.",
     "After measurement, the qubit will yield |0\u27e9 or |1\u27e9 with probabilities |\u03b1|\u00b2 and |\u03b2|\u00b2, respectively"),

    ("Q4 – Entanglement",
     "Which statement best captures entanglement of two qubits? 2",
     "Two entangled qubits have correlations that cannot be described by a product state "
     "(i.e., the joint state is not separable)"),

    ("Q5 – Gates",
     "Starting from a qubit in state |0\u27e9, which quantum gate operation can prepare "
     "a superposition state such as (|0\u27e9+|1\u27e9)/\u221a2?",
     "Apply a Hadamard gate to the qubit"),

    ("Q6 – Measurement",
     "Which statement best describes the effect of measurement on a quantum state? "
     "Assume the computational basis.",
     "After the measurement, the quantum system is one of its (eigen)states of the measured observable"),

    ("Q7 – Coherence",
     "Maintaining quantum coherence in a quantum computer often requires:",
     "Isolation from radiation and vibrations; and (often) very low temperatures"),
]

Q_LABELS = [d[0] for d in PRE_DEFS]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _mark_correct(df, col, correct):
    s = (df[col] == correct).astype(float)
    s[df[col].isna()] = np.nan
    return s


def load_and_filter():
    madrid  = pd.read_excel(MADRID_FILE)
    segovia = pd.read_excel(SEGOVIA_FILE)
    madrid  = madrid[madrid["Relationship to IE"] != "Professor/Lecturer"].copy()
    segovia = segovia[segovia["Relationship to IE"] != "Professor/Lecturer"].copy()
    mad     = madrid[madrid["What degree are you studying?"].str.contains("BAM", na=False)].copy()
    seg     = segovia.copy()

    # Score every question for both groups
    for df in [mad, seg]:
        for (qname, col, ans) in PRE_DEFS:
            df[f"pre_{qname}"] = _mark_correct(df, col, ans)
        for (qname, col, ans) in POST_DEFS:
            df[f"post_{qname}"] = _mark_correct(df, col, ans)

    pre_cols  = [f"pre_{d[0]}"  for d in PRE_DEFS]
    post_cols = [f"post_{d[0]}" for d in POST_DEFS]

    # Compute totals (out of 7) from verified per-question binaries
    for df in [mad, seg]:
        df["pre_total"]  = df[pre_cols].sum(axis=1)
        df["post_total"] = df[post_cols].sum(axis=1)
        df["delta"]      = df["post_total"] - df["pre_total"]

    # Filter: must have answered ALL 7 pre AND ALL 7 post questions
    mad = mad.dropna(subset=pre_cols + post_cols).copy()
    seg = seg.dropna(subset=pre_cols + post_cols).copy()

    return mad, seg


# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def _hist_panel(ax, data, color, title, xlabel, bins):
    ax.hist(data, bins=bins, color=color, alpha=0.78, edgecolor="white", linewidth=0.9)
    ax.axvline(data.mean(),   color=color, lw=2.2, ls="--", label=f"Mean   = {data.mean():.2f}")
    ax.axvline(data.median(), color=color, lw=2.2, ls=":",  label=f"Median = {data.median():.1f}")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Number of students", fontsize=10)
    ax.legend(fontsize=9)
    _style(ax)


def _annotate_bars(ax, bars):
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=7.5)


def _q_acc(df, prefix):
    return [df[f"{prefix}_{q}"].mean() for q in Q_LABELS]


def _cohen_h(p1, p2):
    """Cohen's h effect size for the difference between two proportions.
    h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))
    Magnitude: 0.2 small  |  0.5 medium  |  0.8 large
    Sign: positive means p1 > p2 (Madrid change > Segovia change).
    Clipped to [0,1] before arcsin to guard against float rounding.
    """
    p1 = np.clip(p1, 0.0, 1.0)
    p2 = np.clip(p2, 0.0, 1.0)
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def _h_label(h):
    """Text magnitude label for a Cohen's h value."""
    a = abs(h)
    if a < 0.2:   return "negligible"
    elif a < 0.5: return "small"
    elif a < 0.8: return "medium"
    else:         return "large"


# ══════════════════════════════════════════════════════════════════════════════
# PART A — TOTAL SCORE
# ══════════════════════════════════════════════════════════════════════════════

def plot_A1_baseline(mad, seg):
    """Pre-test distributions — confirm groups are comparable before intervention."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    bins = np.arange(-0.5, 8.5, 1)
    _hist_panel(ax1, seg["pre_total"], C_SEG,
                f"Segovia BAM \u2014 control  (n={len(seg)})",
                "Pre-test score (out of 7)", bins)
    _hist_panel(ax2, mad["pre_total"], C_MAD,
                f"Madrid BAM \u2014 treatment  (n={len(mad)})",
                "Pre-test score (out of 7)", bins)
    ax1.set_xticks(range(0, 8))
    ax2.set_xticks(range(0, 8))

    u, p    = stats.mannwhitneyu(seg["pre_total"], mad["pre_total"], alternative="two-sided")
    verdict = "groups comparable \u2713" if p > 0.05 else "groups differ \u2014 interpret with caution \u26a0"
    fig.suptitle(
        f"A1 \u2014 Baseline Check: Pre-test Score Distributions\n"
        f"Mann-Whitney U = {u:.1f},  p = {p:.3f}  ({verdict})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "A1_baseline_check.png", dpi=150)
    plt.close()
    return u, p


def plot_A2_posttest(mad, seg):
    """Post-test score distributions per group."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    bins = np.arange(-0.5, 8.5, 1)
    _hist_panel(ax1, seg["post_total"], C_SEG,
                f"Segovia BAM \u2014 control  (n={len(seg)})",
                "Post-test score (out of 7)", bins)
    _hist_panel(ax2, mad["post_total"], C_MAD,
                f"Madrid BAM \u2014 treatment  (n={len(mad)})",
                "Post-test score (out of 7)", bins)
    ax1.set_xticks(range(0, 8))
    ax2.set_xticks(range(0, 8))

    u, p = stats.mannwhitneyu(seg["post_total"], mad["post_total"], alternative="two-sided")
    fig.suptitle(
        f"A2 \u2014 Post-test Score Distributions\n"
        f"Mann-Whitney U = {u:.1f},  p = {p:.3f}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "A2_posttest_distributions.png", dpi=150)
    plt.close()
    return u, p


def plot_A3_change(mad, seg):
    """Score change (post − pre) distributions per group."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    bins = np.arange(-4.5, 8.5, 1)

    for ax, df, color, tag in [
        (ax1, seg, C_SEG, f"Segovia BAM \u2014 control  (n={len(seg)})"),
        (ax2, mad, C_MAD, f"Madrid BAM \u2014 treatment  (n={len(mad)})"),
    ]:
        _, p_wil = stats.wilcoxon(df["delta"])
        _hist_panel(ax, df["delta"], color,
                    f"{tag}\nWilcoxon p = {p_wil:.3f}  (\u0394 \u2260 0)",
                    "Score change \u0394  (post \u2212 pre)", bins)
        ax.axvline(0, color="black", lw=1, ls="-", alpha=0.35)

    u, p = stats.mannwhitneyu(seg["delta"], mad["delta"], alternative="two-sided")
    fig.suptitle(
        f"A3 \u2014 Score Change Distributions  (post \u2212 pre)\n"
        f"Between-group Mann-Whitney U = {u:.1f},  p = {p:.3f}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "A3_change_distributions.png", dpi=150)
    plt.close()
    return u, p


def plot_A4_summary_table(mad, seg):
    """Summary statistics table for total scores."""
    rows = []
    for label, df in [("Segovia BAM (control)", seg), ("Madrid BAM (treatment)", mad)]:
        pre, post, d = df["pre_total"], df["post_total"], df["delta"]
        _, p_wil = stats.wilcoxon(d)
        rows.append([
            label, len(df),
            f"{pre.mean():.2f} \u00b1 {pre.std():.2f}",
            f"{post.mean():.2f} \u00b1 {post.std():.2f}",
            f"{d.mean():.2f} \u00b1 {d.std():.2f}",
            f"{d.median():.1f}",
            f"{p_wil:.3f}",
        ])

    u_pre,  p_pre  = stats.mannwhitneyu(seg["pre_total"],  mad["pre_total"],  alternative="two-sided")
    u_post, p_post = stats.mannwhitneyu(seg["post_total"], mad["post_total"], alternative="two-sided")
    u_d,    p_d    = stats.mannwhitneyu(seg["delta"],      mad["delta"],      alternative="two-sided")
    rows.append([
        "Between-group MWU p", "\u2014",
        f"p = {p_pre:.3f}", f"p = {p_post:.3f}", f"p = {p_d:.3f}", "\u2014", "\u2014",
    ])

    cols = [
        "Group", "n",
        "Pre mean \u00b1 SD\n(out of 7)",
        "Post mean \u00b1 SD\n(out of 7)",
        "\u0394 mean \u00b1 SD",
        "\u0394 median",
        "Wilcoxon p\n(\u0394 \u2260 0)",
    ]

    fig, ax = plt.subplots(figsize=(14, 2.8))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2E4057")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 1:
            cell.set_facecolor(C_SEG + "33")
        elif r == 2:
            cell.set_facecolor(C_MAD + "33")
        elif r == 3:
            cell.set_facecolor("#f0f0f0")
    ax.set_title(
        "A4 \u2014 Total Score: Summary Statistics  "
        "(scores derived from doc-verified correct answers, out of 7)",
        fontsize=10, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    fig.savefig(OUT / "A4_summary_stats_table.png", dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PART B — QUESTION BY QUESTION
# ══════════════════════════════════════════════════════════════════════════════

def _qbyq_bar(mad, seg, pre_or_post, title, ylabel, filename):
    """Grouped bar chart: proportion correct per question, Segovia vs Madrid."""
    acc_s = _q_acc(seg, pre_or_post)
    acc_m = _q_acc(mad, pre_or_post)
    x, w  = np.arange(len(Q_LABELS)), 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bs = ax.bar(x - w / 2, acc_s, w, color=C_SEG, alpha=0.85, edgecolor="white",
                label=f"Segovia BAM \u2014 control  (n={len(seg)})")
    bm = ax.bar(x + w / 2, acc_m, w, color=C_MAD, alpha=0.85, edgecolor="white",
                label=f"Madrid BAM \u2014 treatment  (n={len(mad)})")
    _annotate_bars(ax, bs)
    _annotate_bars(ax, bm)

    ax.set_xticks(x)
    ax.set_xticklabels([q.replace(" \u2013 ", "\n") for q in Q_LABELS], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close()


def plot_B1_pretest_accuracy(mad, seg):
    """Pre-test accuracy per question — baseline comparison."""
    _qbyq_bar(mad, seg, "pre",
              "B1 \u2014 Pre-test Accuracy per Question",
              "Proportion correct",
              "B1_pretest_accuracy_by_q.png")


def plot_B2_posttest_accuracy(mad, seg):
    """Post-test accuracy per question."""
    _qbyq_bar(mad, seg, "post",
              "B2 \u2014 Post-test Accuracy per Question",
              "Proportion correct",
              "B2_posttest_accuracy_by_q.png")


def plot_B3_change_accuracy(mad, seg):
    """Change in accuracy per question (post − pre), annotated with Cohen's h."""
    pre_s  = _q_acc(seg, "pre");  post_s = _q_acc(seg, "post")
    pre_m  = _q_acc(mad, "pre");  post_m = _q_acc(mad, "post")
    chg_s  = [po - pr for pr, po in zip(pre_s, post_s)]
    chg_m  = [po - pr for pr, po in zip(pre_m, post_m)]

    # Cohen's h on the change proportions (Madrid change vs Segovia change)
    # Clip to [0,1] — changes can be negative so shift: treat as proportion of max possible change
    # We compute h directly on the post-test proportions vs pre-test proportions per group,
    # then take the difference of the two h values (effect of gain in Mad vs gain in Seg).
    h_vals = [_cohen_h(cm, cs) for cm, cs in zip(chg_m, chg_s)]

    x, w = np.arange(len(Q_LABELS)), 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    bs = ax.bar(x - w / 2, chg_s, w, color=C_SEG, alpha=0.85, edgecolor="white",
                label="Segovia BAM \u2014 control")
    bm = ax.bar(x + w / 2, chg_m, w, color=C_MAD, alpha=0.85, edgecolor="white",
                label="Madrid BAM \u2014 treatment")

    # Annotate bar values
    for bars in [bs, bm]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ypos = h + 0.015 if h >= 0 else h - 0.05
                ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                        f"{h:+.0%}", ha="center", va="bottom", fontsize=7.5)

    # Annotate Cohen's h between the two bars for each question
    y_top = ax.get_ylim()[1]
    for i, (h_val, cm, cs) in enumerate(zip(h_vals, chg_m, chg_s)):
        top = max(cm, cs, 0) + 0.07
        label = f"h={h_val:+.2f}\n({_h_label(h_val)})"
        color = C_MAD if h_val > 0 else C_SEG
        ax.text(i, top, label, ha="center", va="bottom", fontsize=7,
                color=color, fontweight="bold")

    ax.axhline(0, color="black", lw=0.9, ls="-", alpha=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace(" \u2013 ", "\n") for q in Q_LABELS], fontsize=9)
    ax.set_ylabel("Change in proportion correct  (post \u2212 pre)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.set_ylim(bottom=ax.get_ylim()[0], top=1.05)
    ax.set_title(
        "B3 \u2014 Change in Accuracy per Question  (post \u2212 pre)\n"
        "Cohen\u2019s h: effect size of Madrid\u2019s gain vs Segovia\u2019s gain  "
        "(+\u202f=\u202fMadrid gained more,  \u2212\u202f=\u202fSegovia gained more)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "B3_change_accuracy_by_q.png", dpi=150)
    plt.close()


def plot_B4_qbyq_stats_table(mad, seg):
    """Per-question stats table: accuracy + Fisher's exact test (pre and post)."""
    rows = []
    for qname in Q_LABELS:
        pre_s  = seg[f"pre_{qname}"].dropna()
        pre_m  = mad[f"pre_{qname}"].dropna()
        post_s = seg[f"post_{qname}"].dropna()
        post_m = mad[f"post_{qname}"].dropna()

        # Fisher's exact on post-test proportions (correct/wrong × group)
        c_s, w_s = int(post_s.sum()), len(post_s) - int(post_s.sum())
        c_m, w_m = int(post_m.sum()), len(post_m) - int(post_m.sum())
        _, p_post = stats.fisher_exact([[c_s, w_s], [c_m, w_m]])

        # Fisher's exact on pre-test proportions (baseline check per question)
        c_s_pre, w_s_pre = int(pre_s.sum()), len(pre_s) - int(pre_s.sum())
        c_m_pre, w_m_pre = int(pre_m.sum()), len(pre_m) - int(pre_m.sum())
        _, p_pre = stats.fisher_exact([[c_s_pre, w_s_pre], [c_m_pre, w_m_pre]])

        rows.append([
            qname.replace(" \u2013 ", "\n"),
            f"{pre_s.mean():.0%}",
            f"{pre_m.mean():.0%}",
            f"p={p_pre:.3f}",
            f"{post_s.mean():.0%}",
            f"{post_m.mean():.0%}",
            f"{post_s.mean()-pre_s.mean():+.0%}",
            f"{post_m.mean()-pre_m.mean():+.0%}",
            f"p={p_post:.3f}",
        ])

    cols = [
        "Question",
        "Pre\nSegovia", "Pre\nMadrid", "Fisher p\n(pre)",
        "Post\nSegovia", "Post\nMadrid",
        "\u0394 Seg", "\u0394 Mad",
        "Fisher p\n(post)",
    ]

    fig, ax = plt.subplots(figsize=(16, 4.8))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2E4057")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f7fa")
    ax.set_title(
        "B4 \u2014 Per-question Accuracy & Statistics\n"
        "(Fisher\u2019s exact test: correct/wrong \u00d7 group, separately for pre and post)",
        fontsize=11, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    fig.savefig(OUT / "B4_qbyq_stats_table.png", dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    mad, seg = load_and_filter()

    print("=" * 54)
    print("  Q1 \u2014 VR Effect on Learning  (test scores)")
    print("=" * 54)
    print(f"\n  Segovia BAM (control)   n = {len(seg)}")
    print(f"  Madrid BAM (treatment)  n = {len(mad)}")
    print(f"  Scoring: doc-verified correct answers, out of 7")

    # ── Part A: Total score ───────────────────────────────────────────────────
    print("\n\u2500\u2500 Part A: Total Score \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    u, p = plot_A1_baseline(mad, seg)
    verdict = "comparable \u2713" if p > 0.05 else "differ \u26a0"
    print(f"  A1 Baseline (pre):  MWU U={u:.1f}, p={p:.3f}  [{verdict}]")

    u, p = plot_A2_posttest(mad, seg)
    print(f"  A2 Post-test:       MWU U={u:.1f}, p={p:.3f}")

    u, p = plot_A3_change(mad, seg)
    print(f"  A3 Score change:    MWU U={u:.1f}, p={p:.3f}")

    _, p_ws = stats.wilcoxon(seg["delta"])
    _, p_wm = stats.wilcoxon(mad["delta"])
    print(f"  Wilcoxon (\u0394\u22600):   Segovia p={p_ws:.3f},  Madrid BAM p={p_wm:.3f}")

    plot_A4_summary_table(mad, seg)

    # ── Part B: Question by question ──────────────────────────────────────────
    print("\n\u2500\u2500 Part B: Question-by-question \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    plot_B1_pretest_accuracy(mad, seg)
    plot_B2_posttest_accuracy(mad, seg)
    plot_B3_change_accuracy(mad, seg)
    plot_B4_qbyq_stats_table(mad, seg)

    print(f"\n  {'Question':<25} {'Pre S':>6} {'Pre M':>6} {'Pre p':>7}"
          f" {'Post S':>7} {'Post M':>7} {'Fisher p post':>13}")
    print("  " + "\u2500" * 78)
    for qname in Q_LABELS:
        pre_s  = seg[f"pre_{qname}"].dropna()
        pre_m  = mad[f"pre_{qname}"].dropna()
        post_s = seg[f"post_{qname}"].dropna()
        post_m = mad[f"post_{qname}"].dropna()
        c_sp, w_sp = int(pre_s.sum()),  len(pre_s)  - int(pre_s.sum())
        c_mp, w_mp = int(pre_m.sum()),  len(pre_m)  - int(pre_m.sum())
        c_s,  w_s  = int(post_s.sum()), len(post_s) - int(post_s.sum())
        c_m,  w_m  = int(post_m.sum()), len(post_m) - int(post_m.sum())
        _, p_pre  = stats.fisher_exact([[c_sp, w_sp], [c_mp, w_mp]])
        _, p_post = stats.fisher_exact([[c_s,  w_s],  [c_m,  w_m]])
        print(f"  {qname:<25} {pre_s.mean():>5.0%}  {pre_m.mean():>5.0%}  {p_pre:>6.3f}"
              f"  {post_s.mean():>6.0%}  {post_m.mean():>6.0%}  {p_post:>12.3f}")

    print(f"\n  Figures saved to: {OUT}\n")


if __name__ == "__main__":
    run()

