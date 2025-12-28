import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def precision_at_k(recommended, relevant, k=3):
    recommended_k = recommended[:k]
    relevant = set(relevant)
    return len(set(recommended_k) & relevant) / k
