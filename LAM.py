
from zlib import crc32
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt  # for creating plots
import numpy as np
import pandas as pd
import plotly.express as px  # for creating plots
# import seaborn as sns  # for creating plots
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

##########################################################
########## Analyze Best Models and Their Errors ##########
##########################################################

# feature_importances = grid_search.best_estimator_.feature_importances_
# print(feature_importances)
#
# extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
# cat_encoder = full_pipeline.named_transformers_["cat"]
# cat_one_hot_attribs = list(cat_encoder.categories_[0])
# attributes = num_attribs + extra_attribs + cat_one_hot_attribs
# print(sorted(zip(feature_importances, attributes), reverse=True))

####################################
########## Env Arg Parser ##########
####################################

# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Run data analysis on housing data.")
#     parser.add_argument(
#         "command",
#         choices=[
#             "plot_histogram",
#             "plot_histogram_by_proximity",
#             "plot_ocean_proximity_histogram",
#             "plot_scatter",
#             "plot_violin",
#             "show_correlation",
#             "show_correlation_with_ocean_proximity",
#         ],
#         help="Command to run",
#     )
#     args = parser.parse_args()
#
#     if args.command == "plot_histogram":
#         plot_histogram()
#     elif args.command == "plot_histogram_by_proximity":
#         plot_histogram_by_proximity()
#     elif args.command == "plot_ocean_proximity_histogram":
#         plot_ocean_proximity_histogram()
#     elif args.command == "plot_scatter":
#         plot_scatter()
#     elif args.command == "plot_violin":
#         plot_violin()
#     # elif args.command == "show_correlation":
#     #     show_correlation()
#     elif args.command == "show_correlation_with_ocean_proximity":
#         show_correlation_with_ocean_proximity()
