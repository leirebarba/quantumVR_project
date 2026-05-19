import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.configuration import get_output_dir

SECTION_OUTPUT_DIR = get_output_dir("section3_plots")


def plot_seminar_comparison(df):
    """Compare initial test scores between seminar attendees and non-attendees."""

    plot_df = df[["Score", "seminar_attendee"]].copy()
    plot_df["Score"] = pd.to_numeric(plot_df["Score"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Score"])

    attendees     = plot_df.loc[plot_df["seminar_attendee"],  "Score"]
    non_attendees = plot_df.loc[~plot_df["seminar_attendee"], "Score"]

    bins = np.arange(plot_df["Score"].min() - 0.5, plot_df["Score"].max() + 1.5, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    ax1.hist(attendees, bins=bins, color="#2E86AB", alpha=0.78, edgecolor="white")
    ax1.axvline(attendees.mean(), color="#2E86AB", lw=2, ls="--",
                label=f"Mean = {attendees.mean():.2f}")
    ax1.set_title(f"Attended seminar  (n={len(attendees)})", fontweight="bold")
    ax1.set_xlabel("Initial test score")
    ax1.set_ylabel("Number of students")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.hist(non_attendees, bins=bins, color="#E07A5F", alpha=0.78, edgecolor="white")
    ax2.axvline(non_attendees.mean(), color="#E07A5F", lw=2, ls="--",
                label=f"Mean = {non_attendees.mean():.2f}")
    ax2.set_title(f"Did not attend  (n={len(non_attendees)})", fontweight="bold")
    ax2.set_xlabel("Initial test score")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Initial test score: seminar vs non-seminar", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(SECTION_OUTPUT_DIR / "seminar_initial_score_comparison.png", dpi=300)
    plt.close()

