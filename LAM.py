import tarfile
import urllib.request
from pathlib import Path  # for local storage

import matplotlib.pyplot as plt  # for creating plots
import pandas as pd
import plotly.express as px  # for creating plots

# import seaborn as sns  # for creating plots
from pandas.plotting import scatter_matrix


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data analysis on housing data.")
    parser.add_argument(
        "command",
        choices=[
            "plot_histogram",
            "plot_histogram_by_proximity",
            "plot_ocean_proximity_histogram",
            "plot_scatter",
            "plot_violin",
            "show_correlation",
            "show_correlation_with_ocean_proximity",
        ],
        help="Command to run",
    )
    args = parser.parse_args()

    if args.command == "plot_histogram":
        plot_histogram()
    elif args.command == "plot_histogram_by_proximity":
        plot_histogram_by_proximity()
    elif args.command == "plot_ocean_proximity_histogram":
        plot_ocean_proximity_histogram()
    elif args.command == "plot_scatter":
        plot_scatter()
    elif args.command == "plot_violin":
        plot_violin()
    # elif args.command == "show_correlation":
    #     show_correlation()
    elif args.command == "show_correlation_with_ocean_proximity":
        show_correlation_with_ocean_proximity()
