import tarfile
import types
import urllib.request
from pathlib import Path  # for local storage
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

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))


def plot_histogram():
    housing = load_housing_data()
    plot = px.histogram(
        housing,
        x="median_house_value",
        color_discrete_sequence=["purple"],
        title="Distribution of Median House Values",
    )
    plot.show()


def plot_histogram_by_proximity():
    housing = load_housing_data()
    plot = px.histogram(
        housing,
        x="median_house_value",
        color="ocean_proximity",
        color_discrete_sequence=["purple", "green", "red", "blue", "brown"],
        title="Distribution of Median House Values",
    )
    plot.show()


def plot_ocean_proximity_histogram():
    housing = load_housing_data()
    plot = px.histogram(
        housing, x="ocean_proximity", title="Histogram of Ocean Proximity"
    )
    plot.show()


def plot_scatter():
    housing = load_housing_data()
    scatter = px.scatter(
        housing,
        x="population",
        y="median_house_value",
        opacity=0.6,
        color="ocean_proximity",
        title="Population vs Median House Value",
    )
    scatter.show()


def plot_violin():
    housing = load_housing_data()
    violin = px.violin(housing, x="ocean_proximity", y="median_house_value")
    violin.show()


# def show_correlation():
#     housing = load_housing_data()
#     attributes = ["longitude", "latitude", "population", "total_bedrooms", "households", "median_house_value", "median_income", "total_rooms", "housing_median_age"]
#     print(housing[attributes].corr())


def show_correlation_with_ocean_proximity():
    housing = load_housing_data()
    ocean_prox_values = {
        "<1H OCEAN": 0,
        "INLAND": 1,
        "NEAR OCEAN": 2,
        "NEAR BAY": 3,
        "ISLAND": 4,
    }
    ocean_prox_numeric = housing.ocean_proximity.map(ocean_prox_values)
    print(housing.median_house_value.corr(ocean_prox_numeric))


######################################
########## Create a Test Set ##########
######################################
# Load the housing data
housing = load_housing_data()

# Create the income_cat column
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])

# Randomly split the data (optional, if you want to keep this split)
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Prepare for stratified sampling
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Print proportions of each category in the stratified test set
strat_test_proportions = strat_test_set["income_cat"].value_counts() / len(strat_test_set)
# print("Proportions of each category in the stratified test set:")
# print(strat_test_proportions)

# Print the stratified test set DataFrame
# print("Stratified test set:")
# print(strat_test_set)

# Removing the income_cat category to get the data back to its original state. 
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# confirm income_cat column is removed. 
# print("Stratified test set:")
# print(strat_test_set)


######################################
########## Clean the Data ##########
######################################
# revert to a clean training set (by copying strat_train_set once again). 
# Let’s also separate the predictors and the labels, since we don’t necessarily want to apply the same transformations to the predictors and the target values
# (note that drop() creates a copy of the data and does not affect strat_train_set):
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

# Since the median can only be computed on numerical attributes, you need to create a copy of the data without the text attribute ocean_proximity:
housing_num = housing.drop("ocean_proximity", axis=1)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
# print(housing_tr.head())

# Handling Text and Categorical Attributes.

housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

# ordinal_encoder = OrdinalEncoder()
# housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# housing_cat_encoded[:10]

# One Hot Encoder is a better approach than OrdinalEncoder for our purpose. 
# The OrdinalEncoder converts categorical values into numerical values based on their order, 
# but this method can misrepresent non-ordinal categories by implying a false sense of similarity between adjacent values. 
# For categorical data without inherent order, such as location types, 
# using OneHotEncoder is preferable as it creates binary attributes for each category, 
# avoiding this issue and better reflecting the categorical nature of the data.

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

###################################################
########## Createing and Training Models ##########
###################################################

# Creating and training three models (Linear Regression, Decision Tree Regression, and Random Forest Regression)
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

# print("Predictions:", lin_reg.predict(some_data_prepared))
# print("Labels:", list(some_labels))
# print(some_data_prepared)

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)

#### Training using DecisionTreeRegressor ###
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse)

# Using Cross Validation to get a more accurate score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

# Calulating scores using Linear regression 
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# Calculating scores using Random Forest Regression
# Initialize and train the model
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

# Make predictions and evaluate performance
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)

# Print the RMSE for the predictions on the training data
print(f"Training RMSE: {forest_rmse}")

# Perform cross-validation and display scores
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

######################################
########## Fine Tune Models ##########
######################################

# Grid Search 
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)

# Random Search
param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

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
