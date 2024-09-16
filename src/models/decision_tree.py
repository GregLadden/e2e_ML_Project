from prepare_data import prepare_and_transform_data, create_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from cross_validation import evaluate_model, display_scores

# Prepare the data
housing_prepared, housing_labels, full_pipeline = prepare_and_transform_data()

# Initialize and train the Decision Tree Regressor model
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

# Predict using the trained model
housing_predictions = tree_reg.predict(housing_prepared)

# Calculate and print the root mean squared error
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Decision Tree Regression RMSE:", tree_rmse)

# Perform cross-validation and display scores
tree_rmse_scores = evaluate_model(tree_reg, housing_prepared, housing_labels)
display_scores(tree_rmse_scores)
