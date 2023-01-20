from skforecast.utils import load_forecaster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


def testing(df: pd.DataFrame):
    for provincia in df['Territorio'].unique():

        for tipo in df['TIPO_ALLOGGIO2'].unique():

            for paesi in df['Paese di residenza dei clienti'].unique():
                df_train = df[df['Territorio'] == provincia][df['TIPO_ALLOGGIO2'] == tipo][
                    df['Paese di residenza dei clienti'] == paesi][df['Indicatori'] == 'presenze'].fillna(0)

                df_train = df_train.set_index(pd.DatetimeIndex(df_train['TIME'], freq='MS'))

                modello = load_forecaster(f'./{provincia}/{tipo}/{paesi}.py')
                pred = modello.predict(steps=24)

                plt.figure(figsize=(12, 8))
                plt.plot(df_train.index, df_train['Value'])
                plt.plot(pred.index, pred)
                plt.title(f'{provincia},{tipo},{paesi}')
                plt.show()
                # print(round(pred, 3))


if __name__ == '__main__':
    df = pd.read_json('./dataset/dataset_con_sostituzione.json', orient='record')

    df = df.drop(['Flag Codes', 'Flags'], axis=1)

    testing(df)
