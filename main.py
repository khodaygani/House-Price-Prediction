# Libraries

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.linear_model import LassoCV
#from sklearn.model_selection import cross_val_score




### Getting the train and test data

train_data = pd.read_csv("input/train.csv")
test_data = pd.read_csv("input/test.csv")

# The ground truth for train set
y = train_data['SalePrice']

# Used only for creating the csv output file as ordered
Id = test_data['Id']




#### Data preparation 

# merging train and test data to help in filling NAs, NaN and ...
data = pd.concat([train_data,test_data])

# Removing indexes that were generated from old dataframes
data = data.reset_index()

# Removing old index and useless Id columns to have a simpler model
data =data.drop(columns = ['index','Id','SalePrice'])




### Filling missing values 

# selecting string type column and filling them with "None"
null_data = data.isnull().sum()
str_col_list = list(data.select_dtypes(include = ['object']).columns)

for i in range(len(null_data)):
     if null_data[i] > 0 and data.columns[i] in str_col_list:
            data[data.columns[i]] = data[data.columns[i]].fillna('None')
            
# As for remaining fields which are numerical we use the average of the other cases' values          
data = data.fillna(data.mean())

# Adding a feature called Total Area
data['TotalArea'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']


# Log transform on SalePrice
y = np.log1p(y)

# Selecting the numerical feature to check the skewness on them
numeric = data.dtypes[data.dtypes != "object"].index

# Log-transforming the columns where the skewness is more than 0.2 (The treshhold was decided based on trial and error)
for i in numeric:
    if skew(data[i]) > 0.2:
        data[i] = np.log1p(data[i])
        
        
# Taking care of the categorrical data and turning them into dummy variables, unless we can not use Lasso model from Scikit
# Then splitting the data again
data = pd.get_dummies(data)
train_x = data[0:1460]
test_x= data[1460:]


### Developing the regression model
    
# Model
model_lasso = LassoCV(alphas = [0.0005]).fit(train_x, y)


"""
# The rmse calculator function
def rmse_value(model):
    rmse= np.sqrt(-cross_val_score(model, train_x, y, scoring="neg_mean_squared_error", cv = 20))
    return(rmse)
    
rmse_value(model_lasso).mean()
"""

total = pd.DataFrame()
total['Id'] = Id

lasso_preds = np.expm1(model_lasso.predict(test_x))
lasso_preds = pd.DataFrame(lasso_preds)

total['SalePrice'] = lasso_preds
total.to_csv("pred.csv", index = False)
