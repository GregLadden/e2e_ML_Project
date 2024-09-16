from data_loading import load_housing_data
import plotly.express as px  # for creating plots

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

 
