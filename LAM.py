import tarfile
import urllib.request
from pathlib import Path  # for local storage

import matplotlib.pyplot as plt  # for creating plots
import pandas as pd
import plotly.express as px  # for creating plots

# import seaborn as sns  # for creating plots
from pandas.plotting import scatter_matrix

##################################
########## GET THE DATA ##########
##################################


# This function will fetch housing.tgz file from
# github repository and download it into the project,
# then unzip it and load the dataset into the pandas dataframe
def load_housing_data():
    # Set path for local storage
    tarball_path = Path("datasets/housing.tgz")

    # Create 'datasets' directory and download housing.tgz from github repo
    if not tarball_path.is_file():
        Path("datasets").mkdir(
            parents=True, exist_ok=True
        )  # sets the path within the project folder
        url = "https://github.com/ageron/data/raw/main/housing.tgz"  # url for retreiving tarfile
        urllib.request.urlretrieve(url, tarball_path)  # fetching tarfile
        with tarfile.open(
            tarball_path
        ) as housing_tarball:  # extract tarfile into datasets folder
            housing_tarball.extractall(path="datasets")

    # return downloaded dataset
    return pd.read_csv(Path("datasets/housing/housing.csv"))


# Stored dataset
housing = load_housing_data()

# Viewing basic info for first glance at dataset:
# print(housing.head())  # returns first 5 in dataset
print(
    housing.info()
)  # returns column names, null counts (if any), and data types of each column
# print(housing.describe()) # for looking at min, max, Q1-Q3 and median of each column to determine outliers

###################################
########## View THE DATA ##########
###################################

# We want to look at different plotted data so we can swap out the feature here for cross-referencing
# if we add the attribute 'marginal' we can set it equal to 'box' and have an additional box plot displayed
# how they look on a plot:
plot = px.histogram(
    housing,
    x="median_house_value",  # value for x can be changed to other features like longitude, etc
    color_discrete_sequence=["purple"],  # color of represented data
    title="Distribution of Median House Values",
)  # can be changed to reflect data being shown
plot.show()

# This version uses the 'color' attribute to use the selected value of x as its basis with categorical differences
# the value in color can be exchanged for another feature like 'total_rooms'
# if using something with multiple categorical values (ocean_proximity has 5), make sure to use that amount of
# colors in the color_discrete_sequence:
plot = px.histogram(
    housing,
    x="median_house_value",  # value for x can be changed to other features like longitude, etc
    color="ocean_proximity",
    color_discrete_sequence=[
        "purple",
        "green",
        "red",
        "blue",
        "brown",
    ],  # color of represented data
    title="Distribution of Median House Values",
)  # can be changed to reflect data being shown
# plot.show()

# Taking a closer look at our only categorical feature (ocean_proximity) we can plot on a histogram
# and determine how many of each appear in the dataset:
plot = px.histogram(housing, x="ocean_proximity", title="Histogram of Ocean Proximity")
# plot.show()

# Now we will look at some scatter plots to determine relationships between features
# each feature here can be exchanged but since we are predicting median_house_value, we should
# only change the other feature value:
scatter = px.scatter(
    housing,
    x="population",
    y="median_house_value",
    opacity=0.6,  # for viewing cluster areas
    color="ocean_proximity",  # can be used to view each proximity according to x and y features
    title="longitude vs median house value",
)
# scatter.show()

# When attempting to view categorical values vs our target feature we can use something like
# a violin display, else we get a straight line:
violin = px.violin(
    housing,
    x="ocean_proximity",
    y="median_house_value",
)
# violin.show()

# We should check correlation between our target feature and other features:
# print(housing.median_house_value.corr(housing.population))  # prints correlation in numerical form

# We can also check correlation of numerical features by listing each feature (may take time to display):
# attributes = ["longitude", "latitude", "population", "total_bedrooms", "households", "median_house_value",
#               "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(24, 16))
# plt.show()

# Or we can just print the results instead of creating a chart of correlation diagrams:
attributes = [
    "longitude",
    "latitude",
    "population",
    "total_bedrooms",
    "households",
    "median_house_value",
    "median_income",
    "total_rooms",
    "housing_median_age",
]
# print(housing[attributes].corr())  # doesn't show all so add and remove features as needed

# To include categorical data such as ocean_proximity, we need to convert it to numerical form:
ocean_prox_values = {
    "<1H OCEAN": 0,
    "INLAND": 1,
    "NEAR OCEAN": 2,
    "NEAR BAY": 3,
    "ISLAND": 4,
}
ocean_prox_numeric = housing.ocean_proximity.map(
    ocean_prox_values
)  # mapping numeric values to categorical
# print(housing.median_house_value.corr(ocean_prox_numeric))  # prints correlation in numerical form

# One more way to view correlation
# sns.heatmap(housing[attributes].corr(), cmap="Reds", annot=True, fmt=".2f")
# plt.show()

###########################################################
########## LINEAR REGRESSION WITH SINGLE FEATURE ##########
###########################################################
