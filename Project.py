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
df['Date'] = pd.to_datetime(df.index) 
data.head(2)

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


# In[41]:


# Data Processing and scaling
df.drop(columns=['index','Dividends','Stock Splits']).head(2) # We are dropping un necessary columns from the set


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















