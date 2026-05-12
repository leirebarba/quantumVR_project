import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from src.configuration import MADRID_FILE, SEGOVIA_FILE, VR_METRICS, BASE_DIR

OUT = BASE_DIR / "outputs_vr"
OUT.mkdir(parents=True, exist_ok=True)

VR_METRICS_SHORT = {
    VR_METRICS[0]: "Helped understand concepts",
    VR_METRICS[1]: "Increased motivation",
    VR_METRICS[2]: "Appropriate difficulty",
    VR_METRICS[3]: "Would recommend",
}


def load_groups():
    madrid  = pd.read_excel(MADRID_FILE)
    segovia = pd.read_excel(SEGOVIA_FILE)

    # Remove professors
    madrid  = madrid[madrid["Relationship to IE"] != "Professor/Lecturer"].copy()
    segovia = segovia[segovia["Relationship to IE"] != "Professor/Lecturer"].copy()

    # Identify BAM
    madrid["is_BAM"] = madrid["What degree are you studying?"].str.contains("BAM", na=False)

    # Groups
    seg_bam = segovia.copy()              # keep ALL Segovia (important fix)
    mad_bam = madrid[madrid["is_BAM"]].copy()

    return seg_bam, mad_bam


def fmt(series):
    """Return 'mean ± SD  (n)' string for a numeric series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "—"
    return f"{s.mean():.2f} ± {s.std():.2f}  (n={len(s)})"


def fmt_combined(series_list):
    """Combine multiple series and return mean ± SD  (n)."""
    s = pd.concat(series_list, ignore_index=True)
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return "—"
    return f"{s.mean():.2f} ± {s.std():.2f}  (n={len(s)})"


def run():
    seg_bam, mad_bam = load_groups()

    lines = []

    # ── Section: VR experience ratings ─────────────────────────────────────
    lines.append("=" * 78)
    lines.append("  VR EXPERIENCE RATINGS  (1–5 Likert scale)")
    lines.append("  Only BAM students answered these (they are the ones who did VR)")
    lines.append("  ⚠ Segovia BAM did VR after the post test | Madrid BAM did VR before post test")
    lines.append("=" * 78)

    col_w = 28
    header = (f"{'Question':<28} "
              f"{'Segovia BAM':<{col_w}} "
              f"{'Madrid BAM':<{col_w}} "
              f"{'Combined':<{col_w}}")
    lines.append(header)
    lines.append("-" * len(header))

    for col, short in VR_METRICS_SHORT.items():
        seg_val  = fmt(seg_bam[col])
        mad_val  = fmt(mad_bam[col])
        comb_val = fmt_combined([seg_bam[col], mad_bam[col]])

        lines.append(
            f"{short:<28} "
            f"{seg_val:<{col_w}} "
            f"{mad_val:<{col_w}} "
            f"{comb_val:<{col_w}}"
        )

    lines.append("")
    lines.append("  Note: combined column is descriptive only (different timing of VR).")
    lines.append("=" * 78)

    # Print + save
    output = "\n".join(lines)
    print(output)

    out_file = OUT / "vr_descriptive_stats.txt"
    out_file.write_text(output, encoding="utf-8")
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    run()
