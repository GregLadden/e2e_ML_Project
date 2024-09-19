from prepare_data import prepare_and_transform_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from cross_validation import evaluate_model, display_scores

# Prepare the data
housing_prepared, housing_labels, full_pipeline = prepare_and_transform_data()

# Initialize and train the Random Forest Regressor model
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

# Predict using the trained model
housing_predictions = forest_reg.predict(housing_prepared)

# Calculate and print the root mean squared error
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print("Random Forest Regression RMSE:", forest_rmse)

# Perform cross-validation and display scores
forest_rmse_scores = evaluate_model(forest_reg, housing_prepared, housing_labels)
display_scores(forest_rmse_scores)
