from prepare_data import prepare_and_transform_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from cross_validation import evaluate_model, display_scores

# Prepare the data
housing_prepared, housing_labels, full_pipeline = prepare_and_transform_data()

# Initialize and train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Prepare some data for prediction
some_data = housing_prepared[:5]  # Use the prepared data for consistency
some_labels = housing_labels.iloc[:5]
some_data_prepared = some_data

# Predict using the trained model
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
print("Transformed Data:", some_data_prepared)

# Predict on the entire dataset
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Linear Regression RMSE:", lin_rmse)

# Perform cross-validation and display scores
lin_rmse_scores = evaluate_model(lin_reg, housing_prepared, housing_labels)
display_scores(lin_rmse_scores)
