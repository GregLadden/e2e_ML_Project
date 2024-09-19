create_test_set:
	PYTHONPATH=src python src/create_test_set.py

prepare_data:
	PYTHONPATH=src python src/prepare_data.py

train_model_using_linear_regression:
	PYTHONPATH=src python src/models/linear.py

train_model_using_random_forest_regression:
	PYTHONPATH=src python src/models/random_forest.py

train_model_using_decision_tree_regression:
	PYTHONPATH=src python src/models/decision_tree.py

plot_histogram:
	PYTHONPATH=src python main.py plot_histogram

plot_histogram_by_proximity:
	PYTHONPATH=src python main.py plot_histogram_by_proximity

plot_ocean_proximity_histogram:
	PYTHONPATH=src python main.py plot_ocean_proximity_histogram

plot_scatter:
	PYTHONPATH=src python main.py plot_scatter

plot_violin:
	PYTHONPATH=src python main.py plot_violin

show_correlation_with_ocean_proximity:
	PYTHONPATH=src python main.py show_correlation_with_ocean_proximity
