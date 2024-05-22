#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install yfinance')


# In[8]:


import pandas as pd
import numpy as np
import math 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
import statsmodels.api as sm
from sklearn.metrics import r2_score
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# In[11]:


#importing the historical stock data from yfinance
goog = yf.Ticker('GOOG')
#goog.info
history = goog.history(period = '12mo')
df = pd.DataFrame(history)
df.head(10)

#Alternative Way (Keep index for later visualization)

df = pd.DataFrame()
df = yf.download('GOOG', period = '12mo')
df['Date'] = pd.to_datetime(df.index) #Might not necessary
df.head()

# In[14]:


#Creating a plot of historical stock price
x = df.index
y = df['Close']
def df_plot(data, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

stock_name= "GOOG"
title = (stock_name,"History stock performance till date")
df_plot(df , x , y , title=title,xlabel='Date', ylabel='Price',dpi=100)

# In[45]:


df['Date'] = pd.to_datetime(df.Date)
df.describe()


# In[49]:


#Set the x and y for the regression
df['Close_lag']= df['Close'].shift(-1)
df = df.dropna()
x = df[['Open','High','Low','Volume']]
y = df['Close_lag']
#Split the data into training and test datasets with a test size of 20% of the total dataset.
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, shuffle=False,random_state = 0)

#Make sure there is the correct number of observations.
print(train_x.shape )
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)


# In[ ]:
#Fitting the model on train data
train_x = sm.add_constant(train_x)
test_x = sm.add_constant(test_x)
results = sm.OLS(endog=train_y, exog=train_x).fit()
results.summary()

# In[ ]:

predictions = results.predict(test_x)


# Evaulating the accuracy of our model 
r_squared=r2_score(test_y,predictions)
mse=mean_squared_error(test_y, predictions)
print(f'R-squared : {r_squared}')
print(f'Mean Squared Error: {mse}')


#Insight from the model's coefficients
coefficients = pd.DataFrame({'Feature': train_x.columns, 'Coefficient': results.params})
print(coefficients)



#Plot of regression model
plt.figure(figsize=(10, 6))
plt.plot(test_y.index, test_y, label='Actual Closing Price')
plt.plot(test_x.index, predictions, label='Predicted Closing Price', linestyle = '--')
plt.xlabel('Test Dates')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted Closing Price')
plt.legend()
plt.show()

# Plot residuals
residuals = test_y - predictions
plt.figure(figsize=(10, 6))
plt.plot(test_y.index, residuals, label='Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Test Dates')
plt.ylabel('Residual')
plt.title('Residuals of OLS Regression Model')
plt.legend()
plt.show() 


#------------TRYING TO IMPROVE THE MODEL W/ MORE PREDICTOR VARIABLES ---------------------------------------------
# Load data
stock_data = yf.download('GOOG', start='2023-01-01', end='2023-12-31')

# Create lagged features
lags = [1, 2, 3]
for lag in lags:
    stock_data[f'Open_Lag{lag}'] = stock_data['Open'].shift(lag)
    stock_data[f'High_Lag{lag}'] = stock_data['High'].shift(lag)
    stock_data[f'Low_Lag{lag}'] = stock_data['Low'].shift(lag)
    stock_data[f'Close_Lag{lag}'] = stock_data['Close'].shift(lag)
    stock_data[f'Volume_Lag{lag}'] = stock_data['Volume'].shift(lag)

window_size = 5
stock_data['SMA'] = stock_data['Close'].rolling(window=window_size).mean()
stock_data['Close_lag']= stock_data['Close'].shift(-1)
stock_data = stock_data.dropna()


# Define feature columns and target
feature_columns = [f'{col}_Lag{lag}' for col in ['Open', 'High', 'Low', 'Close', 'Volume'] for lag in lags]
X = stock_data[feature_columns]
y2 = stock_data['Close_lag']

X_train, X_test, y2_train, y2_test = train_test_split(X, y2, shuffle=False, random_state = 0)


model2 = LinearRegression()
model2.fit(X_train, y2_train)

y_pred = model2.predict(X_test)

# Plot 
dates2 = np.arange(62)
plt.figure(figsize=(10, 6))
plt.plot(dates2, y2_test, label='Actual')
plt.plot(dates2, y_pred, label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.show()






