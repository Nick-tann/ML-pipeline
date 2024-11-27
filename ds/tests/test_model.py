import pytest
import numpy as np
from ds.model import validate_data

def test_validate_data():
    # Test input type
    with pytest.raises(TypeError):
        validate_data("test","test")
    
    #Test input size
    with pytest.raises(ValueError):
        X = np.ndarray(shape=(3,3))
        y = np.ndarray(shape=(2,2))
        validate_data(X,y)