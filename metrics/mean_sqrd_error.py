import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
X_train, X_test, y_train, y_test = train_test_split(X, y)

reg1 = DecisionTreeRegressor()
reg1.fit(X_train, y_train)
dt_mse = mse(y_test, reg1.predict(X_test))

print "Decision Tree mean absolute error: {:.2f}".format(dt_mse)

reg2 = LinearRegression()
reg2.fit(X_train, y_train)
lr_mse = mse(y_test, reg2.predict(X_test))
print "Linear regression mean absolute error: {:.2f}".format(lr_mse)

results = {
 "Linear Regression": lr_mse,
 "Decision Tree": dt_mse
}