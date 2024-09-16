from create_test_set import stratified_split
from data_loading import load_housing_data
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def prepare_train_set(strat_train_set):
    # Prepare the training set by separating predictors and labels.
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    return housing, housing_labels

def handle_missing_values(housing):
    """Handle missing values in numerical attributes."""
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
    return housing_tr

def handle_categorical_attributes(housing):
    """Handle categorical attributes using OneHotEncoder."""
    housing_cat = housing[["ocean_proximity"]]
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    return housing_cat_1hot

def create_pipeline(housing):
    """Create and apply a full pipeline for both numerical and categorical attributes."""
    housing_num = housing.drop("ocean_proximity", axis=1)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)
    return housing_prepared, full_pipeline

def prepare_and_transform_data():
    """Load data, prepare it, handle missing values, handle categorical attributes, and create a pipeline."""
    # Load the data from the CSV file
    housing = load_housing_data()

    # Get stratified train and test sets
    strat_train_set, strat_test_set = stratified_split(housing)

    # Prepare the training set by separating predictors and labels
    housing, housing_labels = prepare_train_set(strat_train_set)

    # Handle missing values in the numerical attributes
    housing_tr = handle_missing_values(housing)

    # Handle categorical attributes using OneHotEncoder
    housing_cat_1hot = handle_categorical_attributes(housing)

    # Create and apply a full pipeline for both numerical and categorical attributes
    housing_prepared, full_pipeline = create_pipeline(housing)

    # Print to confirm data 
    print(housing_tr.head())
    print(housing_cat_1hot.toarray())
    print(housing_prepared)

    return housing_prepared, housing_labels, full_pipeline

if __name__ == "__main__":
    prepare_and_transform_data()
