# -*- coding: utf-8 -*-

import pandas as pd

#import matplotlib.pyplot as plt

dataset=pd.read_csv('cs-training.csv')
x=dataset.iloc[:,[2,3,4,5,6,7,8,9,10,11]].values
y=dataset.iloc[:,1].values

#Handling the missing values
from sklearn.preprocessing import Imputer
imputer1=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer1=imputer1.fit(x[:,4:5])
x[:,4:5]=imputer1.transform(x[:,4:5])
imputer2=Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imputer2=imputer2.fit(x[:,9:10])
x[:,9:10]=imputer2.transform(x[:,9:10])

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion='entropy')
classifier.fit(x,y)

test_data=pd.read_csv('cs-test.csv')
xt=test_data.iloc[:,[2,3,4,5,6,7,8,9,10,11]].values


from sklearn.preprocessing import Imputer
imputer1=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer1=imputer1.fit(xt[:,4:5])
xt[:,4:5]=imputer1.transform(xt[:,4:5])
imputer2=Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imputer2=imputer2.fit(xt[:,9:10])
xt[:,9:10]=imputer2.transform(xt[:,9:10])

y_pred=classifier.predict(xt)

#code for input


x_opt=x[:,[0,1,2,3,4,5,6,7,8,9]]
import statsmodels.formula.api as sm
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

x_opt=x[:,[1,2,3,4,5,6,7,8,9]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

x_opt=x[:,[1,2,4,5,6,7,8,9]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

x_opt=x[:,[1,2,5,6,7,8,9]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()



