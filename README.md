# House-Price-Prediction

I have tried to make the model as small as possible, at the same time have a score in the acceptable interval.
Data Tidying:

• Removing outliers did not have a positive effect, so I deleted that part to make the model as simple
as possible

• Merged both train and test data to be able to find more reliable values for missing values.

• After finding the not-numerical columns, analyzing the reasoning behind being missing, decided to
fill them with “None”

• For numerical columns, filled them with mean of the column.




Feature Engineering:

• Created a variable for some features that had correlation (had multi-collinearity)

• Applied log transform on the features having skewness over 0.2 and also on SalePrice (y).

• Using dummy variables to get rid of the categorical data.

• Split data again to test and train set accordingly.




Model Selection:

• Using LASSO model, with alpha 0.0005, based on trial and error, with default 3 fold cross
validation.

Kaggle Score = 0.11939

