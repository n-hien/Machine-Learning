# The dataset, which is used here, is from the course "Intro to Machine Learning" of kaggle.com

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#save filepath for easier access
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

#read the data and store in DataFrame
home_data = pd.read_csv(iowa_file_path)

#print a summary of the data 
home_data.describe()

#see the list of all columns in the dataset
home_data.columns

#columns prices, is the prediction target called y
y = home_data.SalePrice

#list of the predictive features
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

#select the data corresponding to features in feature_names
X = home_data[feature_names]

#review data
X.describe()
X.head()

#split data into training and validation data
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=1)

#decision tree is the model being used
#create a DecisionTreeRegressor and save it iowa_model
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)

#make predictions
val_predictions=iowa_model.predict(val_X)

#calculate the mean absolute error
value_mae = mean_absolute_error(val_y, val_predictions)




