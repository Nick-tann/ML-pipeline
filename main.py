from ds import model
import numpy as np
import logging
import uvicorn
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
from sklearn.datasets import load_iris

#Init logging
logger = logging.getLogger(__name__)

#Input format for validation
input_format = {
    "sepal_length": float,
    "sepal_width": float,
    "petal_length": float,
    "petal_width": float
}

#Pydantic basemodel to specify json format
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


#sample_input = {
#   "sepal_length": 6.3,
#   "sepal_width": 2.7,
#   "petal_length": 4.9,
#   "petal_width": 1.8
# }

# def validate_inputs(
#         data: InputData, 
#         input_format: dict[str, type] = input_format
#         ):



#Load dataset
iris_data = load_iris()
X = iris_data.data
y = iris_data.target

#Train model
grid = model.train_model(X,y)
clf = grid.best_estimator_

app = FastAPI(title="model_prediction")

@app.get('/')
def index():
    return 'Hello World'

@app.post('/predict/')
def prediction(data: InputData):
    # validate_inputs(data,input_format)

    #Prepare values for model prediction
    vals = [
            data.sepal_length, \
            data.sepal_width, \
            data.petal_length, \
            data.petal_width
            ]
    parsed_data = np.array(vals).reshape(1, -1)
    prediction = clf.predict(parsed_data)[0]
    return {
        "Predicted number of petals" : prediction.item()
        }
if __name__ == "__main__":
    uvicorn.run("main:app", host = "127.0.0.1", port = 5000)
