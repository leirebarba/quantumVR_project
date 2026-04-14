from src.load_data import load_all_data, build_long_prepost, build_change_scores
from src.section1_plots import plot_test_scores_by_site, plot_metric_grid, plot_site_time_panels
from src.section2_plots import plot_vr_feedback, plot_vr_feedback_overall
from src.section3_plots import plot_seminar_comparison
from src.section4_plots import plot_all_metric_distributions
from src.section5_plots import plot_all_metric_dots

def main():
    madrid, segovia, all_data = load_all_data()

    long_df = build_long_prepost(all_data)
    change_df = build_change_scores(all_data)

    plot_test_scores_by_site(all_data)
    plot_metric_grid(long_df)
    plot_site_time_panels(all_data)
    plot_vr_feedback(all_data)
    plot_vr_feedback_overall(all_data)
    plot_seminar_comparison(change_df)
    plot_all_metric_distributions(all_data)
    plot_all_metric_dots(all_data)

    print("All figures saved in outputs/figures/")


if __name__ == "__main__":
    main()
