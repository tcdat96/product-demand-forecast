#!/usr/bin/env python
# coding: utf-8

# # Product Demand Forecast

import pandas as pd
import numpy as np

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tbats import TBATS, BATS

from itertools import product 
from ast import literal_eval as make_tuple

from time import time
from datetime import timedelta
from datetime import datetime

from multiprocessing import Pool
import functools

from pathlib import Path

from tqdm import tqdm
import warnings


def print_elapsed_time(start):
    elapsed = timedelta(seconds=time() - start)
    # remove millis
    elapsed = str(elapsed).split('.')[0]
    print('Elapsed time: {}'.format(elapsed))


def calc_datetime_delta(d1, d2, date_format='%Y-%m-%d'):
    d1 = datetime.strptime(d1, date_format)
    d2 = datetime.strptime(d2, date_format)
    delta = d2 - d1
    return delta.days


# convert the data to daily
def group_demand_by(df, columns):
    df = df.groupby(columns).sum()
    df = df.reset_index().sort_values(columns)
    return df



def get_date_range(df, debugging=False):
    dates = df.date if 'date' in df.columns else df.index
    date_range = (dates.min(), dates.max())
    if debugging:
        print('({} - {})'.format(*date_range))
    return date_range


# ### Stationarity check

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


def is_stationary(df, print_stats=False):
    result = adfuller(df)
    if print_stats:
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
    return result[1] < 0.001


def apply_differencing(df, periods=1, ffill=False, stationary_check=True):
    if isinstance(df, pd.DataFrame) and 'sales' in df.columns:
        df.sales = df.sales.diff(periods=periods)
    else:
        df = df.diff(periods=periods)
    df = df.ffill() if ffill else df.dropna()
        
    if stationary_check and not is_stationary(df.sales if isinstance(df, pd.DataFrame) else df):
        warnings.warn('DataFrame is still not stationary.')
    return df


def check_stationary(df, items, differencing=False):
    items_df = []
    for item in items:
        item_df = filter_demand(df, item=item).sales
        if differencing:
            item_df = apply_differencing(item_df)
        items_df.append(item_df)

    result = []
    with Pool(4) as p:
        result = list(tqdm(p.imap(is_stationary, items_df), total=len(items_df)))
    return result


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


def test_model_sarima(df, param):
    warnings.simplefilter("ignore")
    try: 
        model = SARIMAX(df, order=param[0], seasonal_order=param[1])
        model = model.fit(disp=-1)
    except: return
    return [param, model.aic]

def_range = range(0,4)
@ignore_warnings(category=ConvergenceWarning)
def optimize_SARIMA(df, p_range=def_range, d=1, q_range=def_range, P_range=def_range, D=1, Q_range=def_range, s_range=def_range):   
    results = []
    best_aic = float('inf')
    
    def _get_sarima_param(param):
        order = (param[0], d, param[1])
        seasonal_order = (param[2], D, param[3], param[4])
        return [order, seasonal_order]

    params = list(product(p_range, q_range, P_range, Q_range, s_range))
    params = [_get_sarima_param(param) for param in params]
    
    results = []
    with Pool(4) as pool:
        mp_param = functools.partial(test_model_sarima, df)
        results = list(tqdm(pool.imap(mp_param, params), total=len(params)))
        results = [result for result in results if result is not None]
    results = pd.DataFrame(results, columns=['param', 'aic']).sort_values('aic')
    
    return results.reset_index(drop=True)


def pred_demand_by(model_func, forecast_steps, train_inst_min, kwargs, param):
    warnings.simplefilter("ignore")
    # get train data
    item_train_df = filter_demand(train_df, item=param[1], store=param[0])
    if isinstance(train_inst_min, int) and len(item_train_df) < train_inst_min:
        return
    item_train_df = get_train_data(item_train_df, train_date_range)
    # train and predict
    pred = model_func(item_train_df, forecast_steps, kwargs)
    # print('{} {} {}'.format(*param, len(pred) if pred is not None else None))
    return [*param, pred]
        

def pred_demands(train_df, test_df, model_func, forecast_steps=196, train_inst_min=None, multiprocessing=True, kwargs={}):
    unique_items = test_df.item.unique()
    unique_stores = test_df.store.unique()
    combinations = list(product(test_df.store.unique(), test_df.item.unique()))

    if train_inst_min is None:
        train_inst_min = 0

    preds = []
    if multiprocessing:
        with Pool(4) as p:
            mp_param = functools.partial(pred_demand_by, model_func, forecast_steps, train_inst_min, kwargs)
            preds = list(tqdm(p.imap(mp_param, combinations), total=len(combinations)))
    else:
        for combination in tqdm(combinations):
            pred = pred_demand_by(model_func, forecast_steps, train_inst_min, combination, kwargs)
            preds.append(pred)

    # convert to DataFrame
    for pred in preds:
        pred[2] = ' '.join(map(str, pred[2]))
    return pd.DataFrame(preds, columns=['store', 'item', 'pred'])


def model_sarima(df, steps, kwargs):
    # train
    try: 
        model = SARIMAX(df, order=kwargs['order'], seasonal_order=kwargs['seasonal_order'])
        model = model.fit(disp=-1)
    except: return None
    # predict
    return model.forecast(steps).reset_index(drop=True)        



def model_tbats(train_df, steps, kwargs):
    estimator = TBATS(seasonal_periods=(7, 365.25), n_jobs=1)
    model = estimator.fit(train_df)
    return model.forecast(steps=steps)



if __name__ == "__main__":
    
    visualize = False
    stationary_check = False

    # ## Explore data
    # ### Read input data
    orig_train_df = pd.read_csv('dataset/demand-forecasting-kernels-only/train.csv')
    print('Original: ' + str(orig_train_df.shape))
    orig_train_df


    # ### Data preprocessing

    # Use more compact and consistent column names
    train_df = orig_train_df.copy()
    train_df.columns = ['date', 'store', 'item', 'sales']


    train_df = group_demand_by(train_df, ['date', 'item', 'store'])

    # remove NA
    train_df.dropna(inplace=True)
    print('After: ' + str(train_df.shape))



    print('Date range: ')
    date_range = get_date_range(train_df, debugging=True)
    train_df.drop('date', axis=1).describe(include='all').iloc[:4,:]


    # ### Split dataset
    test_df = pd.read_csv('dataset/demand-forecasting-kernels-only/test.csv', index_col='id')
    test_df


    # ## Exploratory time-series analysis
    # Check stationarity of 100 items with highest frequency
    check_items = train_df.item.value_counts().index
    n = min(100, len(check_items))
    check_items = check_items[:n]

    if stationary_check:
        stationary_check = check_stationary(train_df, check_items)
        print('Original: {}/{} are not stationary'.format(len(stationary_check) - np.sum(stationary_check), n))
        # Only one of them is not stationary. Let's write a simple function to apply differencing.
        stationary_check = check_stationary(train_df, check_items, differencing=True)
        print('After differencing: {}/{} are not stationary'.format(len(stationary_check) - np.sum(stationary_check), n))


    # Okay, now we have the function, we only need to call it when we build the model.

    # ## Finding correlation  

    # First, since we're dealing with one product at a time, we need a simple function to filter dataset based on product code (ID). Since a lot of days will have no demand for that particulat product, we need to fill those missing dates with 0.

    # Next, we need a specific product to find out the correlation
    # **Clarification**: Each product's demand might be completely different from each other, so the correlation of one product might not hold true to others, eventually produces inaccurate results. In this project, I will only consider products with high demand, e.g. staples, so there will be high chance that they have similar correlation.

    # getting the item with highest demand
    highest_freq_item_df = filter_demand(train_df, item=check_items[0], store=1)
    train_date_range = get_date_range(train_df, debugging=True)
    item_df = get_train_data(highest_freq_item_df, train_date_range)
    item_diff_df = get_train_data(highest_freq_item_df, train_date_range, differencing=True)
    del highest_freq_item_df

    if visualize:
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

    d = 1
    D = 0

    tried_params_filepath = "sarima_params_gridsearch.csv"
    if Path(tried_params_filepath).is_file():
        print('Reading from {}...'.format(tried_params_filepath))
        tried_models = pd.read_csv(tried_params_filepath)
    else:
        print('{} does not exist. Running SARIMA grid search...'.format(tried_params_filepath))
        start_time = time()
        tried_models = optimize_SARIMA(item_df, d=d, D=D, s_range=(1, 2, 5, 7))
        tried_models.to_csv(tried_params_filepath, index=False)
        print_elapsed_time(start_time)


    print('Parameters of best SARIMA models:')
    # print(tried_models[:5])


    # So, we have the params for our SARIMA model now
    best_model = make_tuple(tried_models.param[0])
    best_model_param = {'order': (best_model[0], d, best_model[1]), \
                        'seasonal_order': (best_model[2], D, best_model[3], best_model[4])}
    print(best_model_param)

    # ### Build SARIMA model
    test_size = calc_datetime_delta(*get_date_range(test_df))

    sarima_pred_filepath = 'sarima_prediction.csv'
    if Path(sarima_pred_filepath).is_file():
        print('Reading from {}...'.format(sarima_pred_filepath))
        pred_tbats = pd.read_csv(sarima_pred_filepath)
    else:
        print('{} does not exist. Running SARIMA...'.format(sarima_pred_filepath))
        start_time = time()
        pred_sarima = pred_demands(train_df, test_df, model_sarima, 365, kwargs=best_model_param)
        pred_sarima.to_csv(sarima_pred_filepath, index=False)
        print_elapsed_time(start_time)


    # ## TBATS
    # Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and Seasonal components.
    tbats_pred_filepath = 'tbats_prediction.csv'
    if Path(tbats_pred_filepath).is_file():
        print('Reading from {}...'.format(tbats_pred_filepath))
        pred_tbats = pd.read_csv(tbats_pred_filepath)
    else:
        print('{} does not exist. Running TBATS...'.format(tbats_pred_filepath))
        start_time = time()
        # apparently, tbats already uses multiprocessing underneath
        pred_tbats = pred_demands(train_df, test_df, model_tbats, 365, multiprocessing=True) 
        pred_tbats.to_csv(tbats_pred_filepath, index=False)
        print_elapsed_time(start_time)