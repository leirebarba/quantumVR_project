import matplotlib.pyplot as plt
import seaborn as sns

from src.configuration import VR_METRICS, get_output_dir

SECTION_OUTPUT_DIR = get_output_dir("section2_plots")

def plot_vr_feedback(df):
    """Plot mean VR feedback scores by site for each metric."""
    vr_df = df[df[VR_METRICS[0]].notna()].copy()
    summary = vr_df.groupby("site")[VR_METRICS].mean(numeric_only=True).T
    short_labels = ["Understanding", "Motivation", "Difficulty", "Recommendation"]

    fig, ax = plt.subplots(figsize=(10, 5))
    summary.plot(kind="bar", ax=ax)
    ax.set_xticklabels(short_labels, rotation=0)
    ax.set_title("VR Feedback by Site", fontsize=13, weight="bold")
    ax.set_ylabel("Mean Likert Score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Site", frameon=False)

    plt.tight_layout()
    plt.savefig(SECTION_OUTPUT_DIR / "vr_feedback_by_site.png", dpi=300, bbox_inches="tight")
    plt.close()
