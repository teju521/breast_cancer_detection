#!/usr/bin/env python
# coding: utf-8

# In[33]:


from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import warnings

data = pd.read_csv('cancer.csv')
data.head(5)


# In[34]:


data.shape
data.isnull().sum()


# In[35]:


dignosis = pd.get_dummies(data['diagnosis'], drop_first = True)
data['diagnosis'] = dignosis


# In[36]:


data['diagnosis'].value_counts()

#data.groupby('diagnosis').count()


# In[37]:


data.drop(['radius_mean', 'radius_se', 'radius_worst','perimeter_mean', 'area_mean', 'area_se', 'perimeter_se'], axis=1, inplace=True)


# In[38]:


plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")


# In[39]:


X = data.iloc[:,2:31]
Y = data.iloc[:,1]


# In[40]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)


# In[41]:


model = LogisticRegression()
model.fit(X_train,Y_train)


# In[42]:


Y_pred = model.predict(X_test)
Y_prob = model.predict_proba(X_test)
Y_pred


# In[43]:


cm = confusion_matrix(Y_test, Y_pred)


# In[50]:


score = model.score(X_test,Y_test)
print("Testing score of logistic regression model=",score)


# In[ ]:




