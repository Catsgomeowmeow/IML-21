#!/usr/bin/env python
# coding: utf-8

# In[102]:


import numpy as np
import sklearn as skl
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt 


# Load the Data

# In[103]:


test = pd.read_csv('test.csv')
train =pd.read_csv('train.csv')
sample = pd.read_csv('sample.csv')


# In[104]:


test


# In[105]:


train_y = train['y']
train_x=train.drop(columns=['y','Id'])


# Perform linear regression on the data using sklearn's linear regression method

# In[106]:


reg = LinearRegression().fit(train_x,train_y)
prediction = reg.predict(train_x)


# Check the mean squared error of our training data

# In[107]:


mean_squared_error(prediction,train_y)**0.5


# As expected it's small 

# Use our "trained" regression model to predict the x value

# In[108]:


test_x = test.drop(columns=['Id'])
pred_test=reg.predict(test_x)


# Since we know the source of the data we can check against the true mean

# In[109]:


true_sol=np.mean(test_x,axis=1)


# In[110]:


mean_squared_error(true_sol,pred_test)**0.5


# Output save file

# In[111]:


data = {"Id":test["Id"],"y":pd.Series(pred_test)}
df = pd.DataFrame(data)
df.to_csv("results.csv")


# In[ ]:




