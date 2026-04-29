import matplotlib.pyplot as plt
import pandas as pd

from src.configuration import get_output_dir

SECTION_OUTPUT_DIR = get_output_dir("section3_plots")


def plot_seminar_comparison(df):
    """Compare initial test scores between seminar attendees and non-attendees."""

    # 🔥 use ONLY initial test score
    plot_df = df[["Score", "seminar_attendee"]].copy()
    plot_df["Score"] = pd.to_numeric(plot_df["Score"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Score"])

    # compute means
    attendee_mean = plot_df.loc[plot_df["seminar_attendee"], "Score"].mean()
    non_attendee_mean = plot_df.loc[~plot_df["seminar_attendee"], "Score"].mean()

    values = [attendee_mean, non_attendee_mean]
    labels = ["Attended seminar", "Did not attend"]

    # plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values)

    ax.set_ylabel("Initial test score")
    ax.set_title("Initial test score: seminar vs non-seminar")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(SECTION_OUTPUT_DIR / "seminar_initial_score_comparison.png", dpi=300)
    plt.close()

    