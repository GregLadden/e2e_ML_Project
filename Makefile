.PHONY: create_test_set plot_histogram plot_histogram_by_proximity plot_ocean_proximity_histogram plot_scatter plot_violin show_correlation show_correlation_with_ocean_proximity

create_test_set:
    python LAM.py create_test_set

plot_histogram:
    python LAM.py plot_histogram

plot_histogram_by_proximity:
    python LAM.py plot_histogram_by_proximity

plot_ocean_proximity_histogram:
    python LAM.py plot_ocean_proximity_histogram

plot_scatter:
    python LAM.py plot_scatter

plot_violin:
    python LAM.py plot_violin

show_correlation:
    python LAM.py show_correlation

show_correlation_with_ocean_proximity:
    python LAM.py show_correlation_with_ocean_proximity
