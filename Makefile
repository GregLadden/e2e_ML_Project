.PHONY: plot_histogram plot_histogram_by_proximity plot_ocean_proximity_histogram plot_scatter plot_violin show_correlation show_correlation_with_ocean_proximity

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
