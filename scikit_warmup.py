from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import pandas
from pandas import DataFrame

housing_data = datasets.load_boston()

print (housing_data.data.shape)

linear_regression_model = LinearRegression()
linear_regression_model.fit(housing_data.data, housing_data.target)

predictions = linear_regression_model.predict(housing_data.data)

score = metrics.r2_score(housing_data.target, predictions)

print score

# creating sample data
sample_data = {'name': ['Ray', 'Adam', 'Jason', 'Varun', 'Xiao'], 'health': ['fit', 'slim', 'obese', 'fit', 'slim']}

# storing sample data in the form of a dataframe
data = DataFrame(sample_data, columns = ['name', 'health'])

label_encoder = preprocessing.LabelEncoder()
# label_encoder.fit(data['health'])
# label_encoder.transform(data['health'])
label_encoder.fit_transform(data['health'])

pandas.get_dummies(data['health'])

ohe = preprocessing.OneHotEncoder()
label_encoded_data = label_encoder.fit_transform(data['health'])
ohe.fit_transform(label_encoded_data.reshape(-1, 1))

print label_encoded_data
