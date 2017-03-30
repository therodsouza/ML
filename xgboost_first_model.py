# First XGBoost model for pima indians dataset

from numpy import loadtxt
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
dataset = loadtxt('pima-indians-diabetes.data', delimiter=",")

# split data into X and y
X = dataset[:, 0:8]
y = dataset[:, 8]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# fit model into training data
model = xgboost.XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print "Accuracy: %.2f%%" % (accuracy * 100.0)