from ds import utils
import logging
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
utils.set_logger()

#Parameters
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)


def validate_data(X: np.ndarray, y: np.ndarray) -> None:
    """
    Function to validate that X and y are both numpy arrays with equal number of elements.
    Input: X, y
    Output: None
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Either X or y is not a numpy array.")
    elif len(X) != len(y):
        raise ValueError("Mismatch: X and y have different number of elements.")
    else:
        logger.info("Inputs validated.")
    return


def train_model(
                X: np.ndarray,
                y: np.ndarray, 
                param_grid: dict[str, any] = param_grid
                ):
    validate_data(X, y)

    #Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Setup Support Vector Classification
    svc = svm.SVC()
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

    #Grid search
    grid = GridSearchCV(svc, param_grid=param_grid, cv=cv)
    grid.fit(X_scaled, y)
    logger.info("Model trained.")
    return grid

if __name__ == "__main__":
    # Init logging
    #For sample, just use iris dataset
    from sklearn.datasets import load_iris
    iris_data = load_iris()
    X = iris_data.data
    y = iris_data.target
    grid = train_model(X,y,param_grid)
