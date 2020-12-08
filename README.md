-------------------------- Prerequisites ------------------------------
1. numpy
2. pandas
3. sklearn
4. seaborn
5. statsmodels
6. tbats
7. xgboost
8. tqdm
These can be easily installed with pip, something like
    pip install -U numpy pandas sklearn seaborn statsmodels tbats xgboost tqdm
or if you already have the common ones
    pip install -U tbats xgboost tqdm



-------------------------- File Structure -----------------------------
1. All the code is written in demand-forecast.ipynb
2. A python script demand-forecast.py is also included if you want to run it with command
3. The folder 'backup' saves all prediction, so we don't need to rerun it every time.
   If you want to rerun any part, simply delete the corresponding saved file.