from sklearn.linear_model import LinearRegression
import numpy as np
from train import load_data

miles, prices = load_data("data/data.csv")
miles_2d = np.array(miles).reshape(-1, 1)
reg = LinearRegression()
reg.fit(miles_2d, prices) 

def sklearn_prediction(input_mileage):
    input_2d = np.array([[input_mileage]]) 
    prediction = reg.predict(input_2d)
    return round(float(prediction[0]),0) 

