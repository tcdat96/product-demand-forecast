#!/usr/bin/env python
# coding: utf-8

# # Product Demand Forecast

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from itertools import product 

from time import time
from datetime import timedelta
from tqdm.notebook import tqdm


# In[2]:


def print_elapsed_time(start):
    elapsed = timedelta(seconds=time() - start)
    # remove millis
    elapsed = str(elapsed).split('.')[0]
    print('Elapsed time: {}'.format(elapsed))


# ## Explore data

# ### Read input data

# In[3]:


orig_demand_df = pd.read_csv('Historical Product Demand.csv')
print('Original: ' + str(orig_demand_df.shape))
orig_demand_df.head(4)


# ### Data preprocessing

# Use more compact and consistent column names

# In[4]:


demand_df = orig_demand_df.copy()
demand_df.columns = ['ID', 'Warehouse', 'Category', 'Date', 'Demand']


# Then, Let's check for NAs

# In[5]:


demand_df.isna().sum()


# As we can see, only date column has NAs. Since we literally have no way to figure out the missing date, we need to drop all of them.

# In[6]:


# remove NA
demand_df.dropna(inplace=True)
print('After: ' + str(demand_df.shape))


# In[7]:


# convert demand to int
demand_df.Demand = demand_df.Demand.str.replace('\(|\)', '').astype(int)
# convert to datetime
demand_df.Date = pd.to_datetime(demand_df.Date)


# In[8]:


# Remove redundant prefixes
demand_df.ID = [cat.split('_')[1] for cat in demand_df.ID]
demand_df.Category = [cat.split('_')[1] for cat in demand_df.Category]


# convert the data to daily

# In[9]:


daily_demand_df = demand_df.groupby(['ID', 'Date', 'Category']).sum()
daily_demand_df = daily_demand_df.reset_index().sort_values(['Date', 'ID'])
daily_demand_df


# A little summary for our data

# In[10]:


def get_date_range(df, debugging=False):
    dates = df.Date if 'Date' in df.columns else df.index
    date_range = (dates.min(), dates.max())
    if debugging:
        print('({} - {})'.format(*date_range))
    return date_range


# In[11]:


print('Date range: ')
date_range = get_date_range(daily_demand_df, debugging=True)
daily_demand_df.drop('Date', axis=1).describe(include='all').iloc[:4,:]


# ### Generic plots

# In[12]:


sns.set(rc={'figure.figsize':(15, 8)})


# In[13]:


# # sns.lineplot(data=demand_df, x='Date', y='Order_Demand', hue='Product_Category')

# cumsum_demand_df = demand_df.sort_values('Date')
# cumsum_demand_df['cumsum'] = cumsum_demand_df.groupby('Product_Category').cumsum()

# sns.lineplot(data=cumsum_demand_df, x='Date', y='cumsum', hue='Product_Category')

# del cumsum_demand_df


# In[14]:


# demand_by_cat_df = demand_df.groupby('Product_Category').sum().reset_index()
# sns.set_style("whitegrid")
# g = sns.barplot(data=demand_by_cat_df, y='Order_Demand', x='Product_Category')
# g.set_yscale("log")
# del demand_by_cat_df, g


# In[15]:


# g = sns.boxplot(data=demand_df, y='Order_Demand', x='Product_Category', showfliers=True)
# g.set_yscale("log")
# del g


# In[16]:


# g = sns.boxplot(data=demand_df, y='Order_Demand', x='Warehouse', showfliers=True)
# g.set_yscale("log")
# del g


# ### Split dataset

# In[17]:


train_df, test_df = train_test_split(daily_demand_df, test_size=0.1)
print('Train: {}'.format(train_df.shape))
print('Test : {}'.format(test_df.shape))


# ## Exploratory time-series analysis

# ### Stationarity check

# In[18]:


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


# In[19]:


def is_stationary(df, print_stats=False):
    result = adfuller(df)
    if print_stats:
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
    return result[1] < 0.001


# Check stationarity of 100 items with highest frequency

# In[20]:


n = 100
check_ids = daily_demand_df.ID.value_counts().index[:n]
not_stationary = []
for check_id in tqdm(check_ids):
    df = filter_demand(daily_demand_df, ID=str(check_id)).Demand
    if not is_stationary(df):
        not_stationary.append(check_id)
print('{}/{} are not stationary'.format(len(not_stationary), n))


# Only one of them is not stationary. Let's write a simple function to apply differencing.

# In[21]:


import warnings
def apply_differencing(df, periods=1, ffill=False, stationary_check=True):
    if isinstance(df, pd.DataFrame) and 'Demand' in df.columns:
        df.Demand = df.Demand.diff(periods=periods)
    else:
        df = df.diff(periods=periods)
    df = df.ffill() if ffill else df.dropna()
        
    if stationary_check and not is_stationary(df.Demand if isinstance(df, pd.DataFrame) else df):
        warnings.warn('DataFrame is still not stationary.')
    return df


# In[22]:


not_stationary = []
for check_id in tqdm(check_ids):
    df = filter_demand(daily_demand_df, ID=str(check_id)).Demand
    if not is_stationary(apply_differencing(df)):
        not_stationary.append(check_id)
print('{}/{} are not stationary'.format(len(not_stationary), n))


# Okay, now we have the function, we only need to call it when we build the model.

# ## Finding correlation  

# First, since we're dealing with one product at a time, we need a simple function to filter dataset based on product code (ID). Since a lot of days will have no demand for that particulat product, we need to fill those missing dates with 0.

# In[23]:


# a simple function to get appropriate training data
def get_train_data(df, date_range=None, differencing=0):
    # we don't need these columns now
    df = df.drop(columns=['ID', 'Category'])
    
    df = df.set_index('Date')
    
    # fill date gap with 0
    if date_range is not None:
        idx = pd.date_range(*date_range)
        df = df.reindex(idx, fill_value=0)
        
    # apply diffencing of 0
    if differencing > 0:
        df = apply_differencing(df, periods=differencing)
    
    return df


# Next, we need a specific product to find out the correlation
# 
# **Clarification**: Each product's demand might be completely different from each other, so the correlation of one product might not hold true to others, eventually produces inaccurate results. In this project, I will only consider products with high demand, e.g. staples, so there will be high chance that they have similar correlation.

# In[24]:


# getting the item with highest demand
highest_freq_item_df = filter_demand(train_df, ID=str(check_ids[0]))
train_date_range = get_date_range(train_df, debugging=True)
item_df = get_train_data(highest_freq_item_df, train_date_range)
item_diff_df = get_train_data(highest_freq_item_df, train_date_range, differencing=True)
del highest_freq_item_df


# In[25]:


fig, ax = plt.subplots(1, 4, figsize=(18,4))
lags = 30
sm.graphics.tsa.plot_acf(item_df, lags=lags, ax=ax[0], title='Autocorrelation')
sm.graphics.tsa.plot_pacf(item_df, lags=lags, ax=ax[1], title='Partial autocorrelation')
sm.graphics.tsa.plot_acf(item_diff_df, lags=lags, ax=ax[2], title='Autocorrelation')
sm.graphics.tsa.plot_pacf(item_diff_df, lags=lags, ax=ax[3], title='Partial autocorrelation')
pass


# As expected, the data have both daily and weekly seasonality.
# 
# In this project, I will use **_TBATS_**, a method specifically designed to handle datasets with multiple seasonalities. The traditional **_SARIMA_** model will also be conducted to compare the results.

# ## SARIMA
# 
# ### Finding right parameters

# In[26]:


from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

def_range = range(0,4)

@ignore_warnings(category=ConvergenceWarning)
def optimize_SARIMA(df, p_range=def_range, d=1, q_range=def_range,                     P_range=def_range, D=1, Q_range=def_range, s_range=def_range):   
    results = []
    best_aic = float('inf')
    
    params = list(product(p_range, q_range, P_range, Q_range, s_range))
    warnings.simplefilter("ignore")
    for param in tqdm(params):
        try: model = SARIMAX(df, order=(param[0], d, param[1]),
                             seasonal_order=(param[2], D, param[3], param[4])).fit(disp=-1)
        except: continue
        if model.aic < best_aic:
            best_model = model
            best_aic = model.aic
            best_param = param
        results.append([param, model.aic])
        
    results = pd.DataFrame(results, columns=['param', 'aic']).sort_values('aic')
    return results.reset_index(drop=True)


# In[ ]:


start_time = time()
d = 1
D = 0
tried_models = optimize_SARIMA(item_df, d=d, D=D, s_range=(1,2,5,7))
print_elapsed_time(start_time)


# In[ ]:


tried_models[:5]


# So, we have the params for our SARIMA model now

# In[ ]:


best_model = tried_models.param[0]
order = (best_model[0], d, best_model[1])
seasonal_order = (best_model[2], D, best_model[3], best_model[4])


# ### Build SARIMA model

# In[ ]:


# train_ratio = 0.1
# train_df, test_df = train_test_split(daily_demand_df, test_size=0.1)
# print('Train: {}'.format(train_df.shape))
# print('Test : {}'.format(test_df.shape))


# In[ ]:





# In[ ]:




