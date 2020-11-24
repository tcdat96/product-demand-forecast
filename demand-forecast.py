#!/usr/bin/env python
# coding: utf-8

# # Product Demand Forecast

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib as plt


# ## Explore data

# ### Read input data

# In[17]:


demand_df = pd.read_csv('Historical Product Demand.csv')
print('Original: ' + str(demand_df.shape))
demand_df.dropna(inplace=True)
print('After   : ' + str(demand_df.shape))
demand_df.head(4)


# In[29]:


print('Date range: ({}, {})'.format(demand_df.Date.min(), demand_df.Date.max()))
demand_df.describe()


# In[ ]:




