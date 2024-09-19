from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate_model(model, X, y, cv=10):
    # Evaluate a model using cross-validation and return RMSE scores
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores

def display_scores(scores):
    # Display cross-validation scores, mean, and standard deviation
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
