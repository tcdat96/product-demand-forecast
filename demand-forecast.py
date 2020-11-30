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

from tbats import TBATS, BATS

from itertools import product 

from time import time
from datetime import timedelta
from datetime import datetime

from tqdm.notebook import tqdm


# In[2]:


def print_elapsed_time(start):
    elapsed = timedelta(seconds=time() - start)
    # remove millis
    elapsed = str(elapsed).split('.')[0]
    print('Elapsed time: {}'.format(elapsed))


# In[3]:


def calc_datetime_delta(d1, d2, date_format='%Y-%m-%d'):
    d1 = datetime.strptime(d1, date_format)
    d2 = datetime.strptime(d2, date_format)
    delta = d2 - d1
    return delta.days


# ## Explore data

# ### Read input data

# In[4]:


orig_train_df = pd.read_csv('dataset/demand-forecasting-kernels-only/train.csv')
print('Original: ' + str(orig_train_df.shape))
orig_train_df


# ### Data preprocessing

# Use more compact and consistent column names

# In[5]:


train_df = orig_train_df.copy()
train_df.columns = ['date', 'store', 'item', 'sales']


# Then, Let's check for NAs

# In[6]:


train_df.isna().sum()


# As we can see, only date column has NAs. Since we literally have no way to figure out the missing date, we need to drop all of them.

# In[9]:


# remove NA
train_df.dropna(inplace=True)
print('After: ' + str(train_df.shape))


# In[10]:


# # convert demand to int
# train_df.sales = demand_df.Demand.str.replace('\(|\)', '').astype(int)
# # convert to datetime
# demand_df.Date = pd.to_datetime(demand_df.Date)


# In[11]:


# # Remove redundant prefixes
# demand_df.ID = [cat.split('_')[1] for cat in demand_df.ID]
# demand_df.Category = [cat.split('_')[1] for cat in demand_df.Category]


# convert the data to daily

# In[12]:


def group_demand_by(df, columns):
    df = df.groupby(columns).sum()
    df = df.reset_index().sort_values(columns)
    return df


# In[13]:


train_df = group_demand_by(train_df, ['date', 'item', 'store'])
train_df


# A little summary for our data

# In[14]:


def get_date_range(df, debugging=False):
    dates = df.date if 'date' in df.columns else df.index
    date_range = (dates.min(), dates.max())
    if debugging:
        print('({} - {})'.format(*date_range))
    return date_range


# In[15]:


print('Date range: ')
date_range = get_date_range(train_df, debugging=True)
train_df.drop('date', axis=1).describe(include='all').iloc[:4,:]


# ### Generic plots

# In[16]:


sns.set(rc={'figure.figsize':(15, 8)})


# In[ ]:


# # sns.lineplot(data=demand_df, x='Date', y='Order_Demand', hue='Product_Category')

# cumsum_demand_df = demand_df.sort_values('Date')
# cumsum_demand_df['cumsum'] = cumsum_demand_df.groupby('Product_Category').cumsum()

# sns.lineplot(data=cumsum_demand_df, x='Date', y='cumsum', hue='Product_Category')

# del cumsum_demand_df


# In[ ]:


# demand_by_cat_df = demand_df.groupby('Product_Category').sum().reset_index()
# sns.set_style("whitegrid")
# g = sns.barplot(data=demand_by_cat_df, y='Order_Demand', x='Product_Category')
# g.set_yscale("log")
# del demand_by_cat_df, g


# In[ ]:


# g = sns.boxplot(data=demand_df, y='Order_Demand', x='Product_Category', showfliers=True)
# g.set_yscale("log")
# del g


# In[ ]:


# g = sns.boxplot(data=demand_df, y='Order_Demand', x='Warehouse', showfliers=True)
# g.set_yscale("log")
# del g


# ### Split dataset

# In[17]:


test_df = pd.read_csv('dataset/demand-forecasting-kernels-only/test.csv', index_col='id')
# test_df = group_demand_by(test_df, ['date', 'item'])
test_df


# In[18]:


# train_df, test_df = train_test_split(daily_demand_df, test_size=0.1, shuffle=False)
# print('Train: {}'.format(train_df.shape))
# print('Test : {}'.format(test_df.shape))


# ## Exploratory time-series analysis

# ### Stationarity check

# In[19]:


# simple function to filter dataframe with given parameters
def filter_demand(df, item=None, store=None, category=None, sales=-1):
    df = df.copy()
    if item is not None:
        df = df[df.item == item]
    if store is not None:
        df = df[df.store == store]
    if category is not None:
        df = df[df.category == category]
    if sales > -1:
        df = df[df.sales > sales]
    return df


# In[20]:


def is_stationary(df, print_stats=False):
    result = adfuller(df)
    if print_stats:
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
    return result[1] < 0.001


# Check stationarity of 100 items with highest frequency

# In[22]:


check_items = train_df.item.value_counts().index
n = min(100, len(check_items))
check_items = check_items[:n]

not_stationary = []
for check_item in tqdm(check_items):
    df = filter_demand(train_df, item=check_item).sales
    if not is_stationary(df):
        not_stationary.append(check_items)
print('{}/{} are not stationary'.format(len(not_stationary), n))


# Only one of them is not stationary. Let's write a simple function to apply differencing.

# In[23]:


import warnings
def apply_differencing(df, periods=1, ffill=False, stationary_check=True):
    if isinstance(df, pd.DataFrame) and 'sales' in df.columns:
        df.sales = df.sales.diff(periods=periods)
    else:
        df = df.diff(periods=periods)
    df = df.ffill() if ffill else df.dropna()
        
    if stationary_check and not is_stationary(df.sales if isinstance(df, pd.DataFrame) else df):
        warnings.warn('DataFrame is still not stationary.')
    return df


# In[24]:


not_stationary = []
for check_item in tqdm(check_items):
    df = filter_demand(train_df, item=check_item).sales
    if not is_stationary(apply_differencing(df)):
        not_stationary.append(check_item)
print('{}/{} are not stationary'.format(len(not_stationary), n))


# Okay, now we have the function, we only need to call it when we build the model.

# ## Finding correlation  

# First, since we're dealing with one product at a time, we need a simple function to filter dataset based on product code (ID). Since a lot of days will have no demand for that particulat product, we need to fill those missing dates with 0.

# In[25]:


# a simple function to get appropriate training data
def get_train_data(df, date_range=None, differencing=0):
    # we don't need these columns now
    df = df.drop(columns=['store', 'item'])
    
    df = df.set_index('date')
    
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

# In[28]:


# getting the item with highest demand
highest_freq_item_df = filter_demand(train_df, item=check_items[0], store=1)
train_date_range = get_date_range(train_df, debugging=True)
item_df = get_train_data(highest_freq_item_df, train_date_range)
item_diff_df = get_train_data(highest_freq_item_df, train_date_range, differencing=True)
del highest_freq_item_df


# In[29]:


fig, ax = plt.subplots(1, 4, figsize=(18,4))
lags = 30
sm.graphics.tsa.plot_acf(item_df, lags=lags, ax=ax[0], title='Autocorrelation')
sm.graphics.tsa.plot_pacf(item_df, lags=lags, ax=ax[1], title='Partial autocorrelation')
sm.graphics.tsa.plot_acf(item_diff_df, lags=lags, ax=ax[2], title='Autocorrelation')
sm.graphics.tsa.plot_pacf(item_diff_df, lags=lags, ax=ax[3], title='Partial autocorrelation')
pass


# It is quite obvious that the item also has weekly seasonality.
# 
# In this project, I will use **_TBATS_**, a method specifically designed to handle datasets with multiple seasonalities. The traditional **_SARIMA_** model will also be conducted to compare the results.

# ## SARIMA
# 
# ### Finding right parameters

# In[30]:


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


# In[ ]:


# tried_models.to_csv('SARIMA_GridSearch.csv', index=False)
# tried_models = pd.read_csv('SARIMA_GridSearch.csv')


# So, we have the params for our SARIMA model now

# In[ ]:


best_model = tried_models.param[0]
order = (best_model[0], d, best_model[1])
seasonal_order = (best_model[2], D, best_model[3], best_model[4])


# ### Build SARIMA model

# In[63]:


def pred_demand(train_df, test_df, model_func, forecast_steps=196, train_inst_min=None):
    warnings.simplefilter("ignore")

    unique_items = test_df.item.unique()
    unique_stores = test_df.store.unique()
    combinations = list(product(test_df.store.unique(), test_df.item.unique()))
    
    if train_inst_min is None:
        train_inst_min = 0
    
    preds = {}
    for param in tqdm(combinations):
        # get train data
        item_train_df = filter_demand(train_df, item=param[1], store=param[0])
        if isinstance(train_inst_min, int) and len(item_train_df) < train_inst_min:
            continue
        item_train_df = get_train_data(item_train_df, train_date_range)
        # train and predict
        pred = model_func(item_train_df, forecast_steps)
        if pred is not None:
            if str(param[0]) not in preds:
                preds[str(param[0])] = {}
            preds[str(param[0])][str(param[1])] = pred
            pd.DataFrame(preds).to_csv('{}.csv'.format(model_func.__name__))
#         if len(preds) > 0: break
        
    return preds


# In[41]:


def model_sarima(train_df, steps):
    # train
    try: 
        model = SARIMAX(df, order=order, seasonal_order=seasonal_order).fit(disp=-1)
    except: return None
    # predict
    return model.forecast(steps).reset_index(drop=True)        


# In[ ]:


test_size = calc_datetime_delta(*get_date_range(test_df))
pred = pred_demand(train_df, test_df, model_sarima, 365, 1100)


# ## TBATS
# 
# Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and Seasonal components.

# In[42]:


def model_tbats(train_df, steps):
    estimator = TBATS(seasonal_periods=(7, 365.25))
    model = estimator.fit(train_df)
    return model.forecast(steps=steps)


# In[64]:


pred_tbats = pred_demand(train_df, test_df, model_tbats, 365)


# In[76]:


pred_tbats = pd.read_csv('model_tbats.csv')
pred = np.fromstring(pred_tbats['1'][0][1:-1].replace('\n', ' '), dtype=float, sep=' ')
sns.lineplot(data=pred)


# In[ ]:


item = next(iter(pred_tbats.keys()))
sns.lineplot(data=train_df[train_df.item == int(item)].sales[-400:])


# In[ ]:


sns.lineplot(data=pred_tbats[item])

