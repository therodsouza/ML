import pandas
import sklearn
from sklearn import model_selection
from sklearn import pipeline
from sklearn import feature_selection
from sklearn import ensemble

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

y = dataset['class']
X = dataset.drop('class', axis=1)

select = feature_selection.SelectKBest(k='all')
clf = ensemble.RandomForestClassifier()

steps = [('feature_selection', select),
        ('random_forest', clf)]

pipeline = sklearn.pipeline.Pipeline(steps)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)

pipeline.fit(X_train, y_train)

y_prediction = pipeline.predict(X_test)

report = sklearn.metrics.classification_report(y_test, y_prediction)

print(report)

