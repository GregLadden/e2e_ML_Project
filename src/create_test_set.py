import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

def load_housing_data(csv_path=None):
    # Load the housing data from the CSV file or a provided path
    return pd.read_csv(csv_path or "datasets/housing/housing.csv")

def split_train_test(housing, test_size=0.2, random_state=42):
    # Split the data into training and test sets randomly
    return train_test_split(housing, test_size=test_size, random_state=random_state)

def stratified_split(housing, test_size=0.2, random_state=42):
    # Perform stratified sampling based on the 'median_income' column
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(housing, pd.cut(housing["median_income"], 
                                                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
                                                               labels=[1, 2, 3, 4, 5])):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    return strat_train_set, strat_test_set

def create_test_set():
    # Load the housing data
    housing = load_housing_data()

    # Perform stratified split
    strat_train_set, strat_test_set = stratified_split(housing)

    # Print proportions of categories in the test set
    proportions = strat_test_set["median_income"].value_counts() / len(strat_test_set)
    print("Test set proportions:\n", proportions)

    # Optionally, save the stratified train and test sets to CSV files
    strat_train_set.to_csv("datasets/strat_train_set.csv", index=False)
    strat_test_set.to_csv("datasets/strat_test_set.csv", index=False)

if __name__ == "__main__":
    create_test_set()
