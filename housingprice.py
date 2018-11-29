import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from mlxtend.regressor import StackingRegressor

#First Attempt 
path="C:/Users/annie/OneDrive/Desktop/Machine Learning"
training_path = path+'/train.csv'
val_path=path+'/test.csv'
data_train = pd.read_csv(training_path)
data_test=pd.read_csv(val_path)
print(data_train.describe())
print(data_train.columns)

train_y=data_train.SalePrice
print(train_y.head())
#Shud find meaningful parameters to to anaylze,find each parameter's relationship to SalePrice
print("Parameters used to predict Sale Price")
predictors=['YearBuilt','1stFlrSF','OverallQual','GrLivArea']
train_X=data_train[predictors]
print(train_X.head())
pred_X=data_test[predictors]
print("Missing values for each column:")

nan1=data_train.isnull().sum()

#Splits data to  training and testing subsets 


###End of 1st try.. now improve...
#Must anaylze relationships
sns.distplot(train_y);
print("Skewness: %f" % train_y.skew())
print("Kurtosis: %f" % train_y.kurt())
corrmat = data_train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, annot=True);
#shows that OverQual,YearBuilt,YearRemodAdd,GrLivArea,Garage has strong 
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
#lt.figure(figsize=(10,10))
g = sns.heatmap(data_train[top_corr_features].corr(),annot=True,cmap="RdYlBu")
plt.figure(figsize=(10,10))
sns.barplot(data_train.OverallQual,train_y)

train_y=np.log1p(train_y)
#sns.distplot(train_y)

from sklearn.model_selection import train_test_split
train_X1,test_X,train_y1,test_y=train_test_split(train_X,train_y)

# Initialize models
lr = LinearRegression(
    n_jobs = -1
)

rd = Ridge(
    alpha = 4.84
)

rf = RandomForestRegressor(
    n_estimators = 12,
    max_depth = 3,
    n_jobs = -1
)

gb = GradientBoostingRegressor(
    n_estimators = 40,
    max_depth = 2
)

nn = MLPRegressor(
    hidden_layer_sizes = (90, 90),
    alpha = 2.75
)

model = StackingRegressor(
    regressors=[rf, gb, nn, rd],
    meta_regressor=lr
)
# Fit the model on our data
model.fit(train_X, train_y)

y_pred = model.predict(train_X)
print(sqrt(mean_squared_error(train_y, y_pred)))
Y_pred = model.predict(test_X)



#process data and imputes values into missing data
#from sklearn.preprocessing import Imputer
#le_imputer=Imputer()
#train_X=le_imputer.fit_transform(train_X)
#val_X=le_imputer.transform(train_X)
#print(train_X)
#print(val_X)
#forest_model=RandomForestRegressor()

#forest_model.fit(train_X,train_y)
#preds1=forest_model.predict(pred_X)
#predsfinal=np.expm1(preds1)
#id1=data_test["Id"]
#id1=np.array(id1)
#id1=id1.astype(int)
#final=pd.DataFrame([id1,predsfinal]).T
#final.columns=["ID","SalePrice"]
#final.to_csv('prediction.csv',index=False)

sub = pd.DataFrame()
sub['Id'] = data_test['Id']
sub['SalePrice'] = np.expm1(Y_pred)
sub.to_csv('prediction.csv',index=False)




