#!/usr/bin/env python
# coding: utf-8

# # Product Demand Forecast

# In[2]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib as plt


# ## Explore data

# ### Read input data

# In[117]:


orig_demand_df = pd.read_csv('Historical Product Demand.csv')
print('Original: ' + str(orig_demand_df.shape))
orig_demand_df.head(4)


# ### Data preprocessing

# In[119]:


# convert demand to int
demand_df = orig_demand_df.copy()
demand_df.Order_Demand = orig_demand_df.Order_Demand.str.replace('\(|\)', '').astype(int)
# convert to datetime
demand_df.Date = pd.to_datetime(demand_df.Date)
# remove NA
demand_df.dropna(inplace=True)
print('After   : ' + str(demand_df.shape))


# A little summary for our data

# In[143]:


print('Date range: ({}, {})'.format(demand_df.Date.min(), demand_df.Date.max()))
demand_df.drop('Date', axis=1).describe(include='all').iloc[:4,:]


# ### Generic plots
