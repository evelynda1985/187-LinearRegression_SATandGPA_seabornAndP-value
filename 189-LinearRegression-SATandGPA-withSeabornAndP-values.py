#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set() #reset the current graph style, we need to re-run everythign again


# In[2]:


data = pd.read_csv('1.01. Simple linear regression.csv')


# In[3]:


data


# In[4]:


#provides statics data from each column
data.describe()


# In[5]:


#We are going to create a linear regression which predits GPA based on the SAT score (reading, writing and math)
y = data['GPA']
x1 = data['SAT']


# In[6]:


plt.scatter(x1,y)
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()


# In[7]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit() #fit will apply a specific estimation technique (OLS in this case) to obtain the fit model
results.summary()


# In[8]:


plt.scatter(x1,y)
yhat = 0.0017*x1 + 0.275
fig = plt.plot(x1, yhat, lw=4, c='orange', label='regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()


# In[9]:


plt.scatter(x1,y)
yhat = 0.0017*x1
fig = plt.plot(x1, yhat, lw=4, c='green', label='regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.xlim(0)
plt.ylim(0)
plt.show()


# In[10]:


plt.scatter(x1,y)
yhat = 0*x1 + 0.275
fig = plt.plot(x1, yhat, lw=4, c='red', label='regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()


# In[ ]:


#are thosse variables useful?
# Does it help us explain the variability we have in this case?
# p-value "P>\t\ = p-value < 0.05" means tha variable is significant
#SAT p-value is = 0.000
#we could observed that the p-value didn't match what we are looking and SAT is a significant variable

