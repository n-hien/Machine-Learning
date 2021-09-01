import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#read the datasets 
training_set = pd.read_csv('train.csv')

#column prices
y = training_set.SalePrice

#list of the predictive features
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#select the data corresponding to features in feature_names
X = training_set[feature_names]

#build a random forest model
forest_model = RandomForestRegressor()
forest_model.fit(X,y)

#read test data
test_data = pd.read_csv('test.csv')

#select the test data corresponding to features in feature_names
test_X = test_data[feature_names]

# make predictions which we will submit. 
test_preds = forest_model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

print(output.head())




