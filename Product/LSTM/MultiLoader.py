#! /usr/bin/env python3

# coding: utf-8

import math
import numpy as np
import pandas as pd


class MultiLoader:

    dtypes = {
        'Open': np.float64,
        'High': np.float64,
        'Low': np.float64,
        'Close': np.float64,
        'Adj close': np.float64,
        'Volume': np.int64,
    }

    def __init__(self, stock_files):
        self.data = self.__load_multiple(stock_files)
        self.stock_count = len(stock_files)

    def __load_multiple(self, stock_files):
        df_all = pd.DataFrame()

        for f in stock_files:
            df = pd.read_csv(f, dtype=self.dtypes, index_col=0)
            df.columns = pd.MultiIndex.from_tuples(
                [(c, f) for c in df.columns])
            df_all = pd.concat([df_all, df], axis=1, sort=True)

            # Supplements
            df_all.loc[:, ('Ommyo', f)] = df_all['Close'][f] - \
                df_all['Open'][f]
            df_all.loc[:, ('Ommyo rate', f)] = (
                df_all['Ommyo'][f]) / df_all['Close'][f]
            df_all.loc[:, ('Adj open', f)] = (df_all['Open'][f]) * \
                df_all['Adj close'][f] / df_all['Close'][f]
            df_all.loc[:, ('Adj high', f)] = (df_all['High'][f]) * \
                df_all['Adj close'][f] / df_all['Close'][f]
            df_all.loc[:, ('Adj low', f)] = (df_all['Low'][f]) * \
                df_all['Adj close'][f] / df_all['Close'][f]

        return df_all.dropna()

    def get_raw_data(self):
        return self.data

    def extract(self, column):
        return self.data[column]


if __name__ == '__main__':

    stock_data_files = [',Nikkei225.csv', ',TOPIX.csv', ',6501.csv']
    ml = MultiLoader(stock_data_files)
    data = ml.extract('Adj close')
    print(data)
