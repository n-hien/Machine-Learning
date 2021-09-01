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

#function to help compare MAE from different values for max_leaf_nodes
def get_mae(max_leaf_nodes,train_X, val_X, train_y, val_y):
	model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 1)
	model.fit(train_X, train_y)
	preds_val = model.predict(val_X)
	mae = mean_absolute_error(val_y, preds_val)
	return (mae)

# loop that tries the following values for max_leaf_nodes from a set of possible values
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = min(my_mae, key=my_mae.get)

# Fit Model Using All Data
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=1)
final_model.fit(X,y)

###build a random forest model
forest_model = RandomForestRegressor(random_state =1)
forest_model.fit(train_X,train_y)

#Calculate the mean absolute error of the random forest model
forest_val_preds = forest_model.predict(val_X)
forest_val_mae = mean_absolute_error(val_y,forest_val_preds)


