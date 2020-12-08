#!/usr/bin/env python
# coding: utf-8

# # Product Demand Forecast

import pandas as pd
import numpy as np

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import OneHotEncoder
import joblib

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tbats import TBATS, BATS

from xgboost import XGBRegressor

from itertools import product 
from ast import literal_eval

from time import time
from datetime import timedelta
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar

from multiprocessing import Pool
import functools

from pathlib import Path

from tqdm import tqdm
import warnings


PROCESS_COUNT = 8


def print_elapsed_time(start):
    elapsed = timedelta(seconds=time() - start)
    # remove millis
    elapsed = str(elapsed).split('.')[0]
    print('Elapsed time: {}'.format(elapsed))


def get_timestamp_string(ts, ts_format='%Y-%m-%d %H:%M:%S'):
    return datetime.fromtimestamp(start_time).strftime(ts_format)


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
    with Pool(PROCESS_COUNT) as p:
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

def _get_fourier_terms(df, periods=0):
    date_range = get_date_range(df)
    date_range = pd.date_range(start=date_range[0], end=date_range[1] + timedelta(days=periods))

    exog = pd.DataFrame({'date': date_range})
    exog = exog.set_index(pd.DatetimeIndex(exog['date'], freq='D'))
    exog['sin365'] = np.sin(2 * np.pi * exog.index.dayofyear / 365.25)
    exog['cos365'] = np.cos(2 * np.pi * exog.index.dayofyear / 365.25)
    exog['sin365_2'] = np.sin(4 * np.pi * exog.index.dayofyear / 365.25)
    exog['cos365_2'] = np.cos(4 * np.pi * exog.index.dayofyear / 365.25)
    exog = exog.drop(columns=['date'])

    exog_to_train = exog.iloc[:-periods] if periods != 0 else exog
    exog_to_test = exog.iloc[-periods:] if periods != 0 else None
    return exog_to_train, exog_to_test


def test_model_sarima(df, param, exog=None):
    warnings.simplefilter("ignore")
    try: 
        model = SARIMAX(df, order=param[0], seasonal_order=param[1], exog=exog)
        model = model.fit(disp=-1)
    except Exception as e: 
        # print(e)
        return
    return [param, model.aic]


def_range = range(0, 4)
def_s_range = (1, 2, 5, 7)
@ignore_warnings(category=ConvergenceWarning)
def optimize_SARIMA(df, p_range=def_range, d=1, q_range=def_range, P_range=def_range, D=1, 
                    Q_range=def_range, s_range=def_s_range, fourier=False, multiprocessing=False):   
    results = []
    best_aic = float('inf')

    def _get_sarima_param(param):
        order = (param[0], d, param[1])
        seasonal_order = (param[2], D, param[3], param[4])
        return [order, seasonal_order]

    params = list(product(p_range, q_range, P_range, Q_range, s_range))
    params = [_get_sarima_param(param) for param in params]
    
    exog_to_train, _ = _get_fourier_terms(df) if fourier else None

    results = []
    if multiprocessing:
        with Pool(PROCESS_COUNT) as pool:
            mp_param = functools.partial(test_model_sarima, df, exog=exog_to_train)
            results = list(tqdm(pool.imap(mp_param, params, chunksize=10), total=len(params)))
            results = [result for result in results if result is not None]
    else:
        for param in tqdm(params):
            result = test_model_sarima(df, param, exog_to_train)
            if result is not None:
                results.append(result)

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
        

def pred_demands(train_df, test_df, model_func, forecast_steps=196, train_inst_min=None, multiprocessing=False, kwargs={}):
    unique_items = test_df.item.unique()
    unique_stores = test_df.store.unique()
    combinations = list(product(test_df.store.unique(), test_df.item.unique()))

    if train_inst_min is None:
        train_inst_min = 0

    preds = []
    if multiprocessing:
        with Pool(PROCESS_COUNT) as p:
            mp_param = functools.partial(pred_demand_by, model_func, forecast_steps, train_inst_min, kwargs)
            preds = list(tqdm(p.imap(mp_param, combinations), total=len(combinations)))
    else:
        for combination in tqdm(combinations):
            pred = pred_demand_by(model_func, forecast_steps, train_inst_min, kwargs, combination)
            preds.append(pred)

    # convert to DataFrame
    for pred in preds:
        pred[2] = ' '.join(map(str, pred[2]))
    return pd.DataFrame(preds, columns=['store', 'item', 'pred'])


def model_sarima(df, steps, kwargs):
    exog_to_train, exog_to_test = None, None
    if 'fourier' in kwargs and kwargs['fourier']:
        exog_to_train, exog_to_test = _get_fourier_terms(df, steps)

    # train
    try: 
        model = SARIMAX(df, order=kwargs['order'], seasonal_order=kwargs['seasonal_order'], exog=exog_to_train)
        model = model.fit(disp=-1)
    except: return None
    # predict
    return model.forecast(steps, exog=exog_to_test).reset_index(drop=True)        



def model_tbats(train_df, steps, kwargs):
    estimator = TBATS(seasonal_periods=(7, 365.25), n_jobs=1)
    model = estimator.fit(train_df)
    return model.forecast(steps=steps)


def _create_sales_lag_feats(df, gpby_cols, target_col, lags):
    gpby = df.groupby(gpby_cols)
    for i in lags:
        df['_'.join([target_col, 'lag', str(i)])] = \
                gpby[target_col].shift(i).values + np.random.normal(scale=1.6, size=(len(df),))
    return df


def _create_sales_ewm_feats(df, gpby_cols, target_col, alpha=[0.9], shift=[1]):
    gpby = df.groupby(gpby_cols)
    for a in alpha:
        for s in shift:
            df['_'.join([target_col, 'lag', str(s), 'ewm', str(a)])] = \
                gpby[target_col].shift(s).ewm(alpha=a).mean().values
    return df


def add_new_features(df):
    if df.index.name != 'date':
        df = df.set_index('date')

    dates = df.index
    df['weekday'] = dates.dayofweek
    df['is_weekend'] = (df['weekday'] >= 4).astype(int)
    df['day'] = dates.day
    df['day_of_year'] = dates.dayofyear
    df['is_month_start'] = (dates.is_month_start).astype(int)
    df['is_month_end'] = (dates.is_month_end).astype(int)
    df['week_of_year'] = dates.isocalendar().week.astype(int)
    df['month'] = dates.month
    df['quarter'] = (df.month - 1) // 3
    df['year'] = dates.year

    # 12 month lag
    prior_year_sales = df.reset_index()[['date','sales','store','item']]
    prior_year_sales['date'] += pd.Timedelta('365 days')
    prior_year_sales.columns = ['date','12m_lag','store','item']
    df = df.merge(prior_year_sales, on=['date','store','item'])

    # is holiday or not
    holidays = USFederalHolidayCalendar().holidays(start=df.date.min(), end=df.date.max())
    df['holiday'] = df.date.isin(holidays).astype(int)
    
    df = _create_sales_lag_feats(df, gpby_cols=['store','item'], target_col='sales', 
                               lags=[91,98,105,112,119,126,182,364])

    df = _create_sales_ewm_feats(df, gpby_cols=['store','item'], 
                               target_col='sales', 
                               alpha=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5], 
                               shift=[91,98,105,112,119,126,182,364,546,728])

    df = df.set_index('date')
    
    # convert categorical columns to numerics
    #num_cols = ['sales', '12m_lag']
    #cat_cols = [col for col in df.keys() if col not in num_cols]
    cat_cols=['store', 'item', 'weekday', 'month', 'quarter']
    df = pd.get_dummies(df, columns=cat_cols)

    # print(list(df.keys()))

    return df


def split_features_labels(df):
    X_train = df.drop(columns=['sales'])
    y_train = df[['sales']].values.ravel()
    return X_train, y_train


def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def tune_xgb(train_df):
    cutoff_date = '2017-10-01'
    validation_df = train_df[train_df.index >= cutoff_date]
    train_df = train_df[train_df.index < cutoff_date]

    X_train, y_train = split_features_labels(train_df)
    X_test, y_test = split_features_labels(validation_df)

    n_estimators = range(50, 400, 100) 
    eta = (0.01, 0.05, 0.10)
    max_depth = (6, 10)
    # min_child_weight = (1, 3, 5)
    gamma = (.0, .1, .2)
    params = list(product(n_estimators, eta, max_depth, gamma))

    result = []
    for param in tqdm(params):
        # build XGBoost model
        xgb_model = XGBRegressor(n_estimators = param[0], learning_rate=param[1], max_depth=param[2], gamma=param[3])
        xgb_model.fit(X_train, y_train)
        # predict
        pred_xgb = xgb_model.predict(X_test)
        # evaluate
        eval_smape = smape(pred_xgb, y_test)
        result.append([param, eval_smape])

    result = pd.DataFrame(result, columns=['param', 'smape']).sort_values('smape')
    return result.reset_index(drop=True)


def plot_corr(df, method='spearman', size=10):
    corr = df.corr(method=method)
    sns.set(rc={'figure.figsize':(size+1,size)})
    ax = sns.heatmap(corr, cmap='Greens', annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)



if __name__ == "__main__":
    
    visualize = False
    stationary_check = False
    pred_count = 365

    # ## Explore data
    # ### Read input data
    orig_train_df = pd.read_csv('dataset/train.csv')
    print('Original: ' + str(orig_train_df.shape))
    orig_train_df


    # ### Data preprocessing

    # Use more compact and consistent column names
    train_df = orig_train_df.copy()
    train_df.columns = ['date', 'store', 'item', 'sales']

    train_df.date = pd.to_datetime(train_df.date)
    train_df = group_demand_by(train_df, ['date', 'item', 'store'])

    # remove NA
    train_df.dropna(inplace=True)
    print('After: ' + str(train_df.shape))

    print('Date range: ')
    date_range = get_date_range(train_df, debugging=True)
    train_df.drop('date', axis=1).describe(include='all').iloc[:4,:]


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

    tried_params_filepath = "backup/sarima_params.csv"
    if Path(tried_params_filepath).is_file():
        print('Reading from {}...'.format(tried_params_filepath))
        tried_models = pd.read_csv(tried_params_filepath)
        tried_models.param = tried_models.param.apply(lambda param: literal_eval(param))
    else:
        print('{} does not exist. Finding best SARIMA model...'.format(tried_params_filepath))
        tried_models = optimize_SARIMA(item_df, d=d, D=D)
        tried_models.to_csv(tried_params_filepath, index=False)


    # So, we have the params for our SARIMA model now
    print('Parameters of best SARIMA model:')
    order, seasonal_order = tried_models.param[0]
    sarima_param = {'order': order, 'seasonal_order': seasonal_order}
    print(sarima_param)


    # ### Build SARIMA model
    sarima_pred_filepath = 'backup/sarima_prediction.csv'
    if Path(sarima_pred_filepath).is_file():
        print('Reading from {}...'.format(sarima_pred_filepath))
        pred_sarima = pd.read_csv(sarima_pred_filepath)
        pred_sarima.pred = pred_sarima.pred.apply(lambda pred: np.fromstring(pred, dtype=float, sep=' '))
    else:
        print('{} does not exist. Running SARIMA...'.format(sarima_pred_filepath))
        pred_sarima = pred_demands(train_df, test_df, model_sarima, pred_count, kwargs=sarima_param)
        pred_sarima.to_csv(sarima_pred_filepath, index=False)



    # SARIMA model with FOURIER
    sarima_fourier_params_filepath = "backup/sarima_fourier_params.csv"
    if Path(sarima_fourier_params_filepath).is_file():
        print('Reading from {}...'.format(sarima_fourier_params_filepath))
        tried_models = pd.read_csv(sarima_fourier_params_filepath)
        tried_models.param = tried_models.param.apply(lambda param: literal_eval(param))
    else:
        print('{} does not exist. Finding best SARIMA with FOURIER model...'.format(sarima_fourier_params_filepath))
        tried_models = optimize_SARIMA(item_df, d=d, D=D, fourier=True, multiprocessing=True)
        tried_models.to_csv(sarima_fourier_params_filepath, index=False)


    print('Parameters of best SARIMA with FOURIER:')
    order, seasonal_order = tried_models.param[0]
    sarima_fourier_param = {'order': order, 'seasonal_order': seasonal_order}
    print(sarima_fourier_param)


    sarima_fourier_pred_filepath = 'backup/sarima_fourier_prediction.csv'
    if Path(sarima_fourier_pred_filepath).is_file():
        print('Reading from {}...'.format(sarima_fourier_pred_filepath))
        pred_sarima_fourier = pd.read_csv(sarima_fourier_pred_filepath)
        pred_sarima_fourier.pred = pred_sarima_fourier.pred.apply(lambda pred: np.fromstring(pred, dtype=float, sep=' '))
    else:
        print('{} does not exist. Running SARIMA with FOURIER...'.format(sarima_fourier_pred_filepath))
        # indicate Fourier transformation
        sarima_fourier_param = sarima_param.copy()
        sarima_fourier_param['fourier'] = True
        # train model
        pred_sarima_fourier = pred_demands(train_df, test_df, model_sarima, pred_count, kwargs=sarima_fourier_param)
        pred_sarima_fourier.to_csv(sarima_fourier_pred_filepath, index=False)



    # ## TBATS
    # Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and Seasonal components.
    tbats_pred_filepath = 'backup/tbats_prediction.csv'
    if Path(tbats_pred_filepath).is_file():
        print('Reading from {}...'.format(tbats_pred_filepath))
        pred_tbats = pd.read_csv(tbats_pred_filepath)
        pred_tbats.pred = pred_tbats.pred.apply(lambda pred: np.fromstring(pred, dtype=float, sep=' '))
    else:
        print('{} does not exist. Running TBATS...'.format(tbats_pred_filepath))
        # apparently, tbats already uses multiprocessing underneath
        pred_tbats = pred_demands(train_df, test_df, model_tbats, pred_count, multiprocessing=True) 
        pred_tbats.to_csv(tbats_pred_filepath, index=False)



    # read the test file
    test_df = pd.read_csv('dataset/test.csv', index_col='id')
    test_size = calc_datetime_delta(*get_date_range(test_df))
    test_df.date = pd.to_datetime(test_df.date)
    first_test_date = test_df.date.min()


    # ## XGBoost
    # add features to train and test file
    xgb_pred_filepath = 'xgb_prediction.csv'
    if Path(xgb_pred_filepath).is_file():
        print('Reading from {}...'.format(xgb_pred_filepath))
        pred_xgb = np.loadtxt(xgb_pred_filepath)
    else:
        print('{} does not exist. Running XGBoost...'.format(xgb_pred_filepath))
        
        start_time = time()
        print('Start time: ' + get_timestamp_string(start_time))

        # adding new features
        full_df = pd.concat([train_df, test_df])
        full_df = add_new_features(full_df)
        train_added_df = full_df[full_df.index < first_test_date]
        test_added_df = full_df[full_df.index >= first_test_date]


        # tune parameters
        xgb_params_filepath = "xgb_params.csv"
        if Path(xgb_params_filepath).is_file():
            print('Reading from {}...'.format(xgb_params_filepath))
            xgb_params = pd.read_csv(xgb_params_filepath)
            xgb_params.param = xgb_params.param.apply(lambda param: literal_eval(param))
        else:
            print('{} does not exist. Finding best SARIMA model...'.format(xgb_params_filepath))
            xgb_params = tune_xgb(train_added_df)
            xgb_params.to_csv(xgb_params_filepath, index=False)


        # split into features and labels
        X_train, y_train = split_features_labels(train_added_df)
        X_test, _ = split_features_labels(test_added_df)

        # build XGBoost model
        # xgb_model = XGBRegressor(tree_method='gpu_hist', n_jobs=PROCESS_COUNT, verbosity=3)
        xgb_model = XGBRegressor(n_jobs=PROCESS_COUNT, verbosity=3)
        xgb_model.fit(X_train, y_train)
        # predict
        pred_xgb = xgb_model.predict(X_test)
        np.savetxt(xgb_pred_filepath, pred_xgb, delimiter=",")
        # save model
        joblib.dump(xgb_model, 'xgb_model.pkl', compress=9)

        print_elapsed_time(start_time)


    # testing
    submissions_filepath = 'submissions.csv'
    if not Path(submissions_filepath).is_file():
        #TODO: consider refactoring this
        # iterate each test case, and find its prediction
        pred_dfs = {'sarima': pred_sarima, 'sarima_fourier': pred_sarima_fourier, 'tbats': pred_tbats, 'xgb': pred_xgb}
        def combine_prediction(input):
            index, row = input
            offset = (row.date - first_test_date).days
            result = []
            for name, pred_df in pred_dfs.items():
                if isinstance(pred_df, pd.DataFrame):
                    df = filter_demand(pred_df, store=row.store, item=row['item'])
                    result.append(df.pred.values[0][offset])
                else:
                    result.append(pred_df[index])
            return result

        # distribute the workload
        with Pool(PROCESS_COUNT) as p:
            rows_iter = ((index, row) for index, row in test_df.iterrows())
            submissions = list(tqdm(p.imap(combine_prediction, rows_iter), total=test_df.shape[0]))
            submissions = pd.DataFrame(submissions, columns=pred_dfs.keys())
            submissions.index.name = 'id'
            submissions.to_csv(submissions_filepath)