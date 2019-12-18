# -*- coding: utf-8 -*-
"""


@author: Parveen Yadav
"""

# LinearRegression is a machine learning library for linear regression 
from sklearn.linear_model import LinearRegression 

# pandas and numpy are used for data manipulation 
import pandas as pd 
import numpy as np 

# matplotlib and seaborn are used for plotting graphs 
import matplotlib.pyplot as plt 
import seaborn 

# fix_yahoo_finance is used to fetch data 
import yfinance as yf

# Read data 
Df = yf.download('GLD','2008-01-01','2017-12-31')

# Only keep close columns 
Df=Df[['Close']] 

# Drop rows with missing values 
Df= Df.dropna() 

# Plot the closing price of GLD 
Df.Close.plot(figsize=(10,5)) 
plt.ylabel("Gold ETF Prices")
print ("Gold ETF Price Series")
plt.show()

#Define explanatory variables
Df['S_3'] = Df['Close'].shift(1).rolling(window=3).mean() 
Df['S_9']= Df['Close'].shift(1).rolling(window=9).mean() 
Df= Df.dropna() 
X = Df[['S_3','S_9']] 
X.head()

#Define dependent variable
y = Df['Close']
y.head()

#Split the data into train and test dataset
t=.8 
t = int(t*len(Df)) 

# Train dataset 
X_train = X[:t] 
y_train = y[:t]  

# Test dataset 
X_test = X[t:] 
y_test = y[t:]

#Create a linear regression model
linear = LinearRegression().fit(X_train,y_train)
print "Linear Regression equation"
print ("Gold ETF Price (y) =", \
round(linear.coef_[0],2), "* 3 Days Moving Average (x1)", \
round(linear.coef_[1],2), "* 9 Days Moving Average (x2) +", \
round(linear.intercept_,2), "(constant)")

#Predicting the Gold ETF prices
predicted_price = linear.predict(X_test)  
predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])  
predicted_price.plot(figsize=(10,5))  
y_test.plot()  
plt.legend(['predicted_price','actual_price'])  
plt.ylabel("Gold ETF Price")  
plt.show()

# R square
r2_score = linear.score(X[t:],y[t:])*100  
float("{0:.2f}".format(r2_score))