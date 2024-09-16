from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
import numpy as np

def grid_search_forest(housing_prepared, housing_labels):
    # Define the parameter grid for Grid Search
    param_grid = [
        # Try 12 (3x4) combinations of hyperparameters
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        # Then try 6 (2x3) combinations with bootstrap set as False
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    
    # Initialize the RandomForestRegressor
    forest_reg = RandomForestRegressor(random_state=42)
    
    # Set up Grid Search with 5-fold cross-validation
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    
    # Fit the model
    grid_search.fit(housing_prepared, housing_labels)
    
    # Print the best parameters found
    print("Best parameters found through Grid Search:", grid_search.best_params_)
    
def random_search_forest(housing_prepared, housing_labels):
    # Define the parameter distributions for Random Search
    param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }
    
    # Initialize the RandomForestRegressor
    forest_reg = RandomForestRegressor(random_state=42)
    
    # Set up Random Search with 5-fold cross-validation
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    
    # Fit the model
    rnd_search.fit(housing_prepared, housing_labels)
    
    # Print the results of the Random Search
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print("RMSE:", np.sqrt(-mean_score), "Params:", params)
