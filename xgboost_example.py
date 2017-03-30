import numpy
import xgboost
from sklearn import model_selection
from sklearn.metrics import accuracy_score

# load data
dataset = numpy.loadtxt('pima-indians-diabetes.data', delimiter=",")

# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

model = xgboost.XGBClassifier()
model.fit(X_train, y_train)

print model

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

print predictions

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))