import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Tools.demo.sortvisu import steps

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import load_forecaster
from skforecast.utils import save_forecaster
import os
from xgboost import XGBRFRegressor


def creazione_modelli(df: pd.DataFrame):
    for provincia in df['Territorio'].unique():
        if not os.path.exists(f'./{provincia}'):
            os.makedirs(f'./{provincia}')
        for tipo in df['TIPO_ALLOGGIO2'].unique():
            if not os.path.exists(f'./{provincia}/{tipo}'):
                os.makedirs(f'./{provincia}/{tipo}')
            for paesi in df['Paese di residenza dei clienti'].unique():
                df_train = df[df['Territorio'] == provincia][df['TIPO_ALLOGGIO2'] == tipo][
                    df['Paese di residenza dei clienti'] == paesi][df['Indicatori'] == 'presenze'].fillna(0)

                df_train['TIME'] = pd.to_datetime(df_train['TIME'])
                df_train.index = pd.DatetimeIndex(df_train['TIME'], freq='MS')

                test_pipe = make_pipeline(
                    XGBRFRegressor(booster='gbtree', nthread=2, eta=0.3, reg_lambda=0.5, reg_alpha=0.5))

                # forecasting
                test_forecaster = ForecasterAutoreg(
                    regressor=test_pipe,
                    lags=10
                )

                param_grid = {'xgbrfregressor__eta': [0, 0.3, 0.6, 1],
                              'xgbrfregressor__reg_lambda': [0, 0.5, 1],
                              'xgbrfregressor__reg_alpha': [0, 0.5, 1]}

                lags_grid = [5, 10, [1, 2, 3, 10]]

                test_grid = grid_search_forecaster(
                    forecaster=test_forecaster,
                    y=df_train['Value'].astype(float),
                    param_grid=param_grid,
                    lags_grid=lags_grid,
                    steps=12,
                    metric='mean_absolute_error',
                    refit=False,
                    initial_train_size=int(len(df_train['Value']) * 0.8),
                    return_best=True,
                    verbose=False
                )
                print(test_grid)

                final_pipe = make_pipeline(
                    XGBRFRegressor(booster='gbtree', nthread=2, eta=test_grid['xgbrfregressor__eta'][0],
                                   reg_lambda=test_grid['xgbrfregressor__reg_lambda'][0],
                                   reg_alpha=test_grid['xgbrfregressor__reg_alpha'][0]))

                final_forecaster = ForecasterAutoreg(
                    regressor=final_pipe,
                    lags=test_grid['lags'][0]
                )

                final_forecaster.fit(df_train['Value'].astype(float))

                save_forecaster(final_forecaster, file_name=f'./{provincia}/{tipo}/{paesi}.py', verbose=False)


if __name__ == '__main__':
    df = pd.read_json('./dataset/dataset_con_sostituzione.json', orient='record')
    df = df.drop(['Flag Codes', 'Flags'], axis=1)

    creazione_modelli(df)



