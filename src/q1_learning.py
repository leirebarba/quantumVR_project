"""
Q1 Analysis: Does VR help learning of quantum computing?
 
Groups (both BAM students, comparable baseline):
  - Segovia BAM  [CONTROL]   : test → session 1 → post-test → VR
  - Madrid BAM   [TREATMENT] : test → VR → session 1 → post-test
 
Only students who answered BOTH pre and post-test are included.
 
Outputs: outputs/figures/q1_learning/
  Part A – Total score
    A1_baseline_check.png         Pre-test score distributions (verify comparable baselines)
    A2_posttest_distributions.png Post-test score distributions
    A3_change_distributions.png   Score change distributions
    A4_summary_stats_table.png    Summary statistics table
 
  Part B – Question by question
    B1_pretest_accuracy_by_q.png  Pre-test accuracy per question per group
    B2_posttest_accuracy_by_q.png Post-test accuracy per question per group
    B3_change_accuracy_by_q.png   Change in accuracy per question per group
    B4_qbyq_stats_table.png       Per-question statistics table
"""
 
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from src.configuration import get_output_dir, MADRID_FILE, SEGOVIA_FILE
 
OUT = get_output_dir("q1_learning")
 
C_SEG = "#2E86AB"   # Segovia BAM (control)
C_MAD = "#E07A5F"   # Madrid BAM (treatment)
 
# ── Question definitions ───────────────────────────────────────────────────────
# Q5 is present in both pre and post but was NOT included in the pre-test total Score column
# (likely excluded from grading). It is shown in plots but flagged.
Q_LABELS = [
    "Q1 – Qubit vs bit",
    "Q2 – Probability",
    "Q3 – Hadamard/super.",
    "Q4 – Entanglement",
    "Q5 – Gates (info)",
    "Q6 – Measurement",
    "Q7 – Coherence",
]
 
PRE_COLS = {
    "Q1 – Qubit vs bit":    ("Which statement is correct?",
                              "A qubit can be in a combination \u03b1|0\u27e9+\u03b2|1\u27e9 with |\u03b1|\u00b2+|\u03b2|\u00b2=1"),
    "Q2 – Probability":     ("A qubit is prepared so that when measuring it, it gives outcome |0\u27e9 with probability 0.25. "
                              "What is the probability of obtaining outcome |1\u27e9  (considering the same state preparation)? "
                              "Assume only outcomes of states |0\u27e9 or |1\u27e9.",
                              0.75),
    "Q3 – Hadamard/super.": ("A qubit starts in state \u22230\u27e9. You apply a Hadamard (H) gate to obtain a superposition state. "
                              "Which statement best describes the result?",
                              "Measuring the qubit can yield |0\u27e9 or |1\u27e9 with equal probabilities"),
    "Q4 – Entanglement":    ("Which statement best captures entanglement of two qubits?",
                              "Two entangled qubits have correlations that cannot be described by a product state (not separable)"),
    "Q5 – Gates (info)":    ("Starting from |00\u27e9, what quantum logic gates are required (at least) to create "
                              "an entangled state of two qubits?",
                              "A Hadamard gate and a CNOT gate"),
    "Q6 – Measurement":     None,   # filled at runtime (curly-quote safe)
    "Q7 – Coherence":       ("Quantum computers often require strong shielding and (for many platforms) cryogenic "
                              "temperatures mainly to  ",
                              "Reduce thermal noise and environmental interactions, which minimizes error rates and decoherence"),
}
 
POST_COLS = {
    "Q1 – Qubit vs bit":    ("Which statement is correct? 2",
                              "A qubit state can be written as \u03b1|0\u27e9+\u03b2|1\u27e9 with |\u03b1|\u00b2+|\u03b2|\u00b2=1"),
    "Q2 – Probability":     ("A qubit is in the state \u2223\u03c8\u27e9= \u221a3/2 |0\u27e9  + 1/2 |1\u27e9. If you measure this qubit, "
                              "what is the probability of obtaining the state |1\u27e9?  Assume only outcomes for states |0\u27e9 or |1\u27e9.",
                              "1/4"),
    "Q3 – Hadamard/super.": (" Which description best matches the meaning of a qubit in superposition "
                              "(\u03b1|0\u27e9+\u03b2|1\u27e9 with |\u03b1|\u00b2+|\u03b2|\u00b2=1)? Assume the computational basis.",
                              "After measurement, the qubit will yield |0\u27e9 or |1\u27e9 with probabilities |\u03b1|\u00b2 and |\u03b2|\u00b2, respectively"),
    "Q4 – Entanglement":    ("Which statement best captures entanglement of two qubits? 2",
                              "Two entangled qubits have correlations that cannot be described by a product state "
                              "(i.e., the joint state is not separable)"),
    "Q5 – Gates (info)":    ("Starting from a qubit in state |0\u27e9, which quantum gate operation can prepare "
                              "a superposition state such as (|0\u27e9+|1\u27e9)/\u221a2?",
                              "Apply a Hadamard gate to the qubit"),
    "Q6 – Measurement":     ("Which statement best describes the effect of measurement on a quantum state? "
                              "Assume the computational basis.",
                              "After the measurement, the quantum system is one of its (eigen)states of the measured observable"),
    "Q7 – Coherence":       ("Maintaining quantum coherence in a quantum computer often requires:",
                              "Isolation from radiation and vibrations; and (often) very low temperatures"),
}
 
 
def _binary(series, correct):
    result = (series == correct).astype(float)
    result[series.isna()] = np.nan
    return result
 
 
def load_and_filter():
    madrid  = pd.read_excel(MADRID_FILE)
    segovia = pd.read_excel(SEGOVIA_FILE)
    madrid  = madrid[madrid["Relationship to IE"] != "Professor/Lecturer"].copy()
    segovia = segovia[segovia["Relationship to IE"] != "Professor/Lecturer"].copy()
 
    mad = madrid[madrid["What degree are you studying?"].str.contains("BAM", na=False)].copy()
    seg = segovia.copy()
 
    # Q6 pre: read correct answer from data to avoid encoding issues
    q6_pre_col = ("How does the act of measuring affect a quantum system that is in superposition? "
                  "Assume the computational basis.")
    q6_pre_correct = mad[q6_pre_col].dropna().iloc[0]   # first entry = "collapses" answer
    PRE_COLS["Q6 – Measurement"] = (q6_pre_col, q6_pre_correct)
 
    for df in [mad, seg]:
        for qname, qa in PRE_COLS.items():
            col, correct = qa
            df[f"pre_{qname}"] = _binary(df[col], correct)
        for qname, qa in POST_COLS.items():
            col, correct = qa
            df[f"post_{qname}"] = _binary(df[col], correct)
 
    # Filter: must have both Score AND Score 2
    mad = mad.dropna(subset=["Score", "Score 2"]).copy()
    seg = seg.dropna(subset=["Score", "Score 2"]).copy()
 
    return mad, seg
 
 
# ── Style helpers ─────────────────────────────────────────────────────────────
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
 
 
def _fmt_pct(ax):
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
 
 

# PART A – TOTAL SCORE

def plot_A1_baseline(mad, seg):
    """Check baseline comparability: pre-test score distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    results = {}
    for ax, df, color, tag in [(axes[0], seg, C_SEG, "control"), (axes[1], mad, C_MAD, "treatment")]:
        scores = df["Score"].dropna()
        grp    = "Segovia BAM (control)" if tag == "control" else "Madrid BAM (treatment)"
        ax.hist(scores, bins=np.arange(0.5, 8.5, 1), color=color, alpha=0.75, edgecolor="white", linewidth=0.8)
        ax.axvline(scores.mean(),   color=color, lw=2, ls="--", label=f"Mean = {scores.mean():.2f}")
        ax.axvline(scores.median(), color=color, lw=2, ls=":",  label=f"Median = {scores.median():.1f}")
        ax.set_title(f"{grp}\nn = {len(df)}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Pre-test score (out of 6)", fontsize=10)
        ax.set_ylabel("Number of students", fontsize=10)
        ax.set_xticks(range(0, 8))
        ax.legend(fontsize=9); _style(ax)
        results[tag] = scores
 
    u, p = stats.mannwhitneyu(results["control"], results["treatment"], alternative="two-sided")
    verdict = "groups comparable ✓" if p > 0.05 else "groups differ — interpret with caution ⚠"
    fig.suptitle(f"A1 — Baseline Check: Pre-test Score Distributions\n"
                 f"Mann-Whitney U = {u:.1f},  p = {p:.3f}  ({verdict})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "A1_baseline_check.png", dpi=150)
    plt.close()
    return u, p
 
 
def plot_A2_posttest(mad, seg):
    """Post-test score distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, df, color, grp in [
        (axes[0], seg, C_SEG, f"Segovia BAM (control)\nn = {len(seg)}"),
        (axes[1], mad, C_MAD, f"Madrid BAM (treatment)\nn = {len(mad)}"),
    ]:
        scores = df["Score 2"].dropna()
        ax.hist(scores, bins=np.arange(0.5, 9.5, 1), color=color, alpha=0.75, edgecolor="white", linewidth=0.8)
        ax.axvline(scores.mean(),   color=color, lw=2, ls="--", label=f"Mean = {scores.mean():.2f}")
        ax.axvline(scores.median(), color=color, lw=2, ls=":",  label=f"Median = {scores.median():.1f}")
        ax.set_title(grp, fontsize=11, fontweight="bold")
        ax.set_xlabel("Post-test score (out of 7)", fontsize=10)
        ax.set_ylabel("Number of students", fontsize=10)
        ax.set_xticks(range(0, 9))
        ax.legend(fontsize=9); _style(ax)
 
    u, p = stats.mannwhitneyu(seg["Score 2"], mad["Score 2"], alternative="two-sided")
    fig.suptitle(f"A2 — Post-test Score Distributions\nMann-Whitney U = {u:.1f},  p = {p:.3f}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "A2_posttest_distributions.png", dpi=150)
    plt.close()
    return u, p
 
 
def plot_A3_change(mad, seg):
    """Score change distributions (post \u2212 pre)."""
    mad = mad.copy(); mad["change"] = mad["Score 2"] - mad["Score"]
    seg = seg.copy(); seg["change"] = seg["Score 2"] - seg["Score"]
 
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, df, color, grp in [
        (axes[0], seg, C_SEG, f"Segovia BAM (control)\nn = {len(seg)}"),
        (axes[1], mad, C_MAD, f"Madrid BAM (treatment)\nn = {len(mad)}"),
    ]:
        changes = df["change"].dropna()
        _, p_wil = stats.wilcoxon(changes)
        ax.hist(changes, bins=np.arange(-3.5, 8.5, 1), color=color, alpha=0.75, edgecolor="white", linewidth=0.8)
        ax.axvline(0,              color="black", lw=1, ls="-", alpha=0.4)
        ax.axvline(changes.mean(),   color=color, lw=2, ls="--", label=f"Mean \u0394 = {changes.mean():.2f}")
        ax.axvline(changes.median(), color=color, lw=2, ls=":",  label=f"Median \u0394 = {changes.median():.1f}")
        ax.set_title(f"{grp}\nWilcoxon p = {p_wil:.3f} (\u0394\u2260 0)",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Score change (post \u2212 pre)", fontsize=10)
        ax.set_ylabel("Number of students", fontsize=10)
        ax.legend(fontsize=9); _style(ax)
 
    u, p = stats.mannwhitneyu(seg["change"], mad["change"], alternative="two-sided")
    fig.suptitle(f"A3 — Score Change Distributions (post \u2212 pre)\n"
                 f"Between-group Mann-Whitney U = {u:.1f},  p = {p:.3f}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "A3_change_distributions.png", dpi=150)
    plt.close()
    return u, p
 
 
def plot_A4_summary_table(mad, seg):
    """Summary statistics table for total scores."""
    mad = mad.copy(); mad["change"] = mad["Score 2"] - mad["Score"]
    seg = seg.copy(); seg["change"] = seg["Score 2"] - seg["Score"]
 
    rows = []
    for label, df in [("Segovia BAM (control)", seg), ("Madrid BAM (treatment)", mad)]:
        pre, post, d = df["Score"].dropna(), df["Score 2"].dropna(), df["change"].dropna()
        _, p_wil = stats.wilcoxon(d)
        rows.append([label, len(df),
                     f"{pre.mean():.2f} \u00b1 {pre.std():.2f}",
                     f"{post.mean():.2f} \u00b1 {post.std():.2f}",
                     f"{d.mean():.2f} \u00b1 {d.std():.2f}",
                     f"{d.median():.1f}",
                     f"{p_wil:.3f}"])
 
    u_pre,  p_pre  = stats.mannwhitneyu(seg["Score"],   mad["Score"],   alternative="two-sided")
    u_post, p_post = stats.mannwhitneyu(seg["Score 2"], mad["Score 2"], alternative="two-sided")
    u_d,    p_d    = stats.mannwhitneyu(seg["change"],  mad["change"],  alternative="two-sided")
    rows.append(["Between-group MWU p", "\u2014",
                 f"p = {p_pre:.3f}", f"p = {p_post:.3f}", f"p = {p_d:.3f}", "\u2014", "\u2014"])
 
    cols = ["Group", "n", "Pre mean \u00b1 SD", "Post mean \u00b1 SD",
            "\u0394 mean \u00b1 SD", "\u0394 median", "Wilcoxon p\n(\u0394 \u2260 0)"]
 
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2E4057"); cell.set_text_props(color="white", fontweight="bold")
        elif r == 1: cell.set_facecolor(C_SEG + "33")
        elif r == 2: cell.set_facecolor(C_MAD + "33")
        elif r == 3: cell.set_facecolor("#f0f0f0")
    ax.set_title("A4 — Total Score: Summary Statistics", fontsize=11, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(OUT / "A4_summary_stats_table.png", dpi=150, bbox_inches="tight")
    plt.close()
 
 

# PART B – QUESTION BY QUESTION
def _q_acc(df, prefix):
    """Helper to compute accuracy per question for a given prefix (pre/post)."""
    return {q: df[f"{prefix}_{q}"].dropna().mean() for q in Q_LABELS}
 
 
def _plot_qbyq_bar(mad, seg, prefix, title, ylabel, filename, note_q5=False):
    """Helper to plot question-by-question accuracy bar plots for pre/post tests."""
    acc_s = _q_acc(seg, prefix)
    acc_m = _q_acc(mad, prefix)
    x, w  = np.arange(len(Q_LABELS)), 0.35
 
    fig, ax = plt.subplots(figsize=(12, 5))
    bs = ax.bar(x - w/2, [acc_s[q] for q in Q_LABELS], w,
                color=C_SEG, alpha=0.85, edgecolor="white",
                label=f"Segovia BAM — control  (n={len(seg)})")
    bm = ax.bar(x + w/2, [acc_m[q] for q in Q_LABELS], w,
                color=C_MAD, alpha=0.85, edgecolor="white",
                label=f"Madrid BAM — treatment  (n={len(mad)})")
 
    for bars in [bs, bm]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=7.5)
 
    if note_q5:
        idx = Q_LABELS.index("Q5 – Gates (info)")
        ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.07, color="gray")
        ax.text(idx, 1.07, "not in\npre score", ha="center", fontsize=7, color="gray", style="italic")
 
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace(" – ", "\n") for q in Q_LABELS], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, 1.18)
    _fmt_pct(ax); _style(ax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close()
 
 
def plot_B1_pretest_accuracy(mad, seg):
    """Pre-test accuracy per question per group."""
    _plot_qbyq_bar(mad, seg, "pre",
                   "B1 — Pre-test Accuracy per Question",
                   "Proportion correct", "B1_pretest_accuracy_by_q.png", note_q5=True)
 
 
def plot_B2_posttest_accuracy(mad, seg):
    """Post-test accuracy per question per group."""
    _plot_qbyq_bar(mad, seg, "post",
                   "B2 — Post-test Accuracy per Question",
                   "Proportion correct", "B2_posttest_accuracy_by_q.png", note_q5=False)
 
 
def plot_B3_change_accuracy(mad, seg):
    """Change in accuracy per question per group."""
    pre_s  = _q_acc(seg, "pre");  post_s = _q_acc(seg, "post")
    pre_m  = _q_acc(mad, "pre");  post_m = _q_acc(mad, "post")
    chg_s  = {q: post_s[q] - pre_s[q] for q in Q_LABELS}
    chg_m  = {q: post_m[q] - pre_m[q] for q in Q_LABELS}
    x, w   = np.arange(len(Q_LABELS)), 0.35
 
    fig, ax = plt.subplots(figsize=(12, 5))
    bs = ax.bar(x - w/2, [chg_s[q] for q in Q_LABELS], w,
                color=C_SEG, alpha=0.85, edgecolor="white", label="Segovia BAM — control")
    bm = ax.bar(x + w/2, [chg_m[q] for q in Q_LABELS], w,
                color=C_MAD, alpha=0.85, edgecolor="white", label="Madrid BAM — treatment")
 
    for bars in [bs, bm]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ypos = h + 0.015 if h >= 0 else h - 0.045
                ax.text(bar.get_x() + bar.get_width()/2, ypos,
                        f"{h:+.0%}", ha="center", va="bottom", fontsize=7.5)
 
    ax.axhline(0, color="black", lw=0.8, ls="-", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace(" – ", "\n") for q in Q_LABELS], fontsize=9)
    ax.set_ylabel("Change in proportion correct (post \u2212 pre)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    _style(ax)
    ax.set_title("B3 — Change in Accuracy per Question (post \u2212 pre)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "B3_change_accuracy_by_q.png", dpi=150)
    plt.close()
 
 
def plot_B4_qbyq_stats_table(mad, seg):
    """Per-question statistics table (Fisher's exact test on post-test correct/wrong × group)."""
    rows = []
    for qname in Q_LABELS:
        pre_s  = seg[f"pre_{qname}"].dropna()
        pre_m  = mad[f"pre_{qname}"].dropna()
        post_s = seg[f"post_{qname}"].dropna()
        post_m = mad[f"post_{qname}"].dropna()
 
        # Fisher's exact on post-test proportions (correct vs wrong × group)
        c_s = int(post_s.sum()); w_s = len(post_s) - c_s
        c_m = int(post_m.sum()); w_m = len(post_m) - c_m
        _, p_fish = stats.fisher_exact([[c_s, w_s], [c_m, w_m]])
 
        note = "\u2190 not in pre score" if qname == "Q5 – Gates (info)" else ""
        rows.append([
            qname.replace(" – ", "\n"),
            f"{pre_s.mean():.0%}",
            f"{pre_m.mean():.0%}",
            f"{post_s.mean():.0%}",
            f"{post_m.mean():.0%}",
            f"{post_s.mean()-pre_s.mean():+.0%}",
            f"{post_m.mean()-pre_m.mean():+.0%}",
            f"{p_fish:.3f}",
            note,
        ])
 
    cols = ["Question", "Pre\nSegovia", "Pre\nMadrid",
            "Post\nSegovia", "Post\nMadrid",
            "\u0394 Seg", "\u0394 Mad",
            "Fisher p\n(post correct\n\u00d7 group)", "Note"]
 
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.75)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2E4057"); cell.set_text_props(color="white", fontweight="bold")
        elif r > 0 and rows[r-1][-1] != "":
            cell.set_facecolor("#f5f5f5"); cell.set_text_props(color="#999999")
        elif r % 2 == 0:
            cell.set_facecolor("#f8f8f8")
    ax.set_title("B4 — Per-question Accuracy & Statistics\n"
                 "(Fisher\u2019s exact test: post-test correct/wrong \u00d7 group)",
                 fontsize=11, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(OUT / "B4_qbyq_stats_table.png", dpi=150, bbox_inches="tight")
    plt.close()
 
 
def run():
    mad, seg = load_and_filter()
    n_mad_orig, n_seg_orig = 24, 20
 
    print("=" * 50)
    print("  Q1 — VR Effect on Learning")
    print("=" * 50)
    print(f"\n  Segovia BAM (control)   n = {len(seg)}  [{n_seg_orig - len(seg)} excluded: missing post-test]")
    print(f"  Madrid BAM (treatment)  n = {len(mad)}  [{n_mad_orig - len(mad)} excluded: missing post-test]")
 
    print("\n── Part A: Total Score ──────────────────────")
    u, p = plot_A1_baseline(mad, seg)
    print(f"  A1 Baseline:   MWU U={u:.1f}, p={p:.3f}  "
          f"{'comparable ✓' if p > 0.05 else 'differ ⚠'}")
    u, p = plot_A2_posttest(mad, seg)
    print(f"  A2 Post-test:  MWU U={u:.1f}, p={p:.3f}")
 
    mad_c = mad.copy(); mad_c["change"] = mad_c["Score 2"] - mad_c["Score"]
    seg_c = seg.copy(); seg_c["change"] = seg_c["Score 2"] - seg_c["Score"]
    u, p  = plot_A3_change(mad_c, seg_c)
    print(f"  A3 Change:     MWU U={u:.1f}, p={p:.3f}")
 
    _, p_ws = stats.wilcoxon(seg_c["change"].dropna())
    _, p_wm = stats.wilcoxon(mad_c["change"].dropna())
    print(f"  Wilcoxon (\u0394\u22600):  Segovia p={p_ws:.3f}, Madrid BAM p={p_wm:.3f}")
    plot_A4_summary_table(mad_c, seg_c)
 
    print("\n── Part B: Question-by-question ─────────────")
    plot_B1_pretest_accuracy(mad, seg)
    plot_B2_posttest_accuracy(mad, seg)
    plot_B3_change_accuracy(mad, seg)
    plot_B4_qbyq_stats_table(mad, seg)
 
    print(f"\n  {'Question':<30} {'Pre Seg':>8} {'Pre Mad':>8} {'Post Seg':>9} {'Post Mad':>9} {'Fisher p':>9}")
    print("  " + "\u2500" * 78)
    for qname in Q_LABELS:
        pre_s  = seg[f"pre_{qname}"].dropna()
        pre_m  = mad[f"pre_{qname}"].dropna()
        post_s = seg[f"post_{qname}"].dropna()
        post_m = mad[f"post_{qname}"].dropna()
        c_s = int(post_s.sum()); w_s = len(post_s) - c_s
        c_m = int(post_m.sum()); w_m = len(post_m) - c_m
        _, p_fish = stats.fisher_exact([[c_s, w_s], [c_m, w_m]])
        flag = "  \u2190 not in pre Score" if qname == "Q5 – Gates (info)" else ""
        print(f"  {qname:<30} {pre_s.mean():>7.0%}  {pre_m.mean():>7.0%}  "
              f"{post_s.mean():>8.0%}  {post_m.mean():>8.0%}  {p_fish:>8.3f}{flag}")
 
    print(f"\n  Figures saved to: {OUT}\n")
 
 
if __name__ == "__main__":
    run()
 