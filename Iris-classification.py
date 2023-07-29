#!/usr/bin/env python
# coding: utf-8

# In[30]:


# OIBSIP: TASK 1: IRIS FLOWER CLASSIFICATION

# 1. Importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[10]:


# 2. Reading the Dataset


df= pd.read_csv(r"C:\Users\bhara\Downloads\archive\Iris.csv")
df.head()


# In[6]:


df.head(10)


# In[7]:


df.tail()


# In[9]:


df.shape


# In[11]:


df.isnull().sum()


# In[12]:


df.dtypes


# In[13]:


data=df.groupby('Species')


# In[14]:


data.head()


# In[15]:


df['Species'].unique()


# In[16]:


df.info()


# In[14]:


plt.boxplot(df['SepalLengthCm'])


# In[15]:


plt.boxplot(df['SepalWidthCm'])


# In[16]:


plt.boxplot(df['PetalLengthCm'])


# In[17]:


plt.boxplot(df['PetalWidthCm'])


# In[18]:


sns.heatmap(df.corr())


# In[ ]:


# 3. data preparation



df.drop('Id', axis=1, inplace=True)


# In[7]:


sp={'Iris-Setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3}


# In[11]:


colors_palete = {'setosa': "red", 'versicolor': "yellow", 'virginica': "blue"}


# In[12]:


df


# In[12]:


X=df.iloc[:,0:4]


# In[14]:


X


# In[14]:


y=df.iloc[:,4]


# In[15]:


y


# In[21]:


# 4. Training model

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state=10)


# In[22]:


model=LinearRegression()


# In[23]:


model.fit(X,y)


# In[24]:


model.score(X,y)


# In[25]:


model.coef_


# In[26]:


model.intercept_


# In[27]:


y_pred=model.predict(X_test)


# In[31]:


print("Mean squared error: %.2f" % np.mean((y_pred - y_test) ** 2))
