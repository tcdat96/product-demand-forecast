#!/usr/bin/env python
# coding: utf-8

# # Product Demand Forecast

# In[49]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib as plt

from statsmodels.tsa.stattools import adfuller

from tqdm.notebook import tqdm


# ## Explore data

# ### Read input data

# In[3]:


orig_demand_df = pd.read_csv('Historical Product Demand.csv')
print('Original: ' + str(orig_demand_df.shape))
orig_demand_df.head(4)


# ### Data preprocessing

# Use more compact and consistent column names

# In[21]:


demand_df = orig_demand_df.copy()
demand_df.columns = ['ID', 'Warehouse', 'Category', 'Date', 'Demand']


# Then, Let's check for NAs

# In[7]:


demand_df.isna().sum()


# As we can see, only date column has NAs. Since we literally have no way to figure out the missing date, we need to drop all of them.

# In[22]:


# remove NA
demand_df.dropna(inplace=True)
print('After: ' + str(demand_df.shape))


# In[23]:


# convert demand to int
demand_df.Demand = demand_df.Demand.str.replace('\(|\)', '').astype(int)
# convert to datetime
demand_df.Date = pd.to_datetime(demand_df.Date)


# In[24]:


# Remove redundant prefixes
demand_df.ID = [cat.split('_')[1] for cat in demand_df.ID]
demand_df.Category = [cat.split('_')[1] for cat in demand_df.Category]


# convert the data to daily

# In[38]:


daily_demand_df = demand_df.groupby(['ID', 'Date', 'Category']).sum()
daily_demand_df = daily_demand_df.reset_index().sort_values(['Date', 'ID'])
daily_demand_df


# A little summary for our data

# In[40]:


print('Date range: ({}, {})'.format(daily_demand_df.Date.min(), daily_demand_df.Date.max()))
daily_demand_df.drop('Date', axis=1).describe(include='all').iloc[:4,:]


# ### Generic plots

# In[8]:


sns.set(rc={'figure.figsize':(15, 8)})


# In[10]:


# # sns.lineplot(data=demand_df, x='Date', y='Order_Demand', hue='Product_Category')

# cumsum_demand_df = demand_df.sort_values('Date')
# cumsum_demand_df['cumsum'] = cumsum_demand_df.groupby('Product_Category').cumsum()

# sns.lineplot(data=cumsum_demand_df, x='Date', y='cumsum', hue='Product_Category')

# del cumsum_demand_df


# In[ ]:


demand_by_cat_df = demand_df.groupby('Product_Category').sum().reset_index()
sns.set_style("whitegrid")
g = sns.barplot(data=demand_by_cat_df, y='Order_Demand', x='Product_Category')
g.set_yscale("log")
del demand_by_cat_df, g


# In[ ]:


g = sns.boxplot(data=demand_df, y='Order_Demand', x='Product_Category', showfliers=True)
g.set_yscale("log")
del g


# In[ ]:


g = sns.boxplot(data=demand_df, y='Order_Demand', x='Warehouse', showfliers=True)
g.set_yscale("log")
del g


# ## ARIMA

# ### Check for stationarity

# In[42]:


# simple function to filter dataframe with given parameters
def filter_demand(df, ID=None, category=None, Demand=-1):
    df = df.copy()
    if ID is not None:
        df = df[df.ID == ID]
    if category is not None:
        df = df[df.category == category]
    if Demand > -1:
        df = df[df.Demand > Demand]
    return df


# In[97]:


def is_stationary(df, print_stats=False):
    result = adfuller(df)
    if print_stats:
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
    return result[1] < 0.001


# Check stationarity of 100 items with highest frequency

# In[99]:


n = 100
check_ids = daily_demand_df.ID.value_counts().index[:n]
not_stationary = []
for check_id in tqdm(check_ids):
    df = filter_demand(daily_demand_df, ID=str(check_id)).Demand
    if not is_stationary(df):
        not_stationary.append(check_id)
print('{}/{} are not stationary'.format(len(not_stationary), n))


# Only one of them is not stationary. Let's write a simple function to apply differencing.

# In[100]:


import warnings
def apply_differencing(df, periods=1, ffill=False, stationary_check=True):
    df = df.diff(periods=periods)
    df = df.ffill() if ffill else df.dropna()
        
    if stationary_check and not is_stationary(df):
        warnings.warn('DataFrame is still not stationary.')
    return df


# In[101]:


not_stationary = []
for check_id in tqdm(check_ids):
    df = filter_demand(daily_demand_df, ID=str(check_id)).Demand
    if not is_stationary(apply_differencing(df)):
        not_stationary.append(check_id)
print('{}/{} are not stationary'.format(len(not_stationary), n))


# Okay, now we have the function, we only need to call it when we build the model.
# 
# ## Build ARIMA model

# In[ ]:





# In[ ]:




