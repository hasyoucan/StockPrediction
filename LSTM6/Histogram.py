# coding: utf-8

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import pylab

from MultiLoader import MultiLoader


def load_data(stock_data_files):
    """
    Scraper が吐き出したファイルを読むのです。
    日付と調整後終値を返すのです。
    """
    multi_loader = MultiLoader(stock_data_files)
    # values = multi_loader.extract('adj_start')
    values = multi_loader.extract('Adj close')
    # values = multi_loader.extract('ommyo_log')
    # values = multi_loader.extract('ommyo_rate')
    return values


def pct_change(values):
    ret_val = pd.Series(values).pct_change()
    return ret_val[1:]


def log_diff(values):
    series = pd.Series(values)
    # 全要素の対数を出す
    log_values = series.apply(math.log10)
    # 対数の差を出す
    ret_val = log_values.diff()
    return ret_val[1:]


if __name__ == '__main__':

    stock_data_files = [
        ',Nikkei225.csv', ',TOPIX.csv', ',6501.csv',
    ]

    values = load_data(stock_data_files)

    for (i, stock) in enumerate(stock_data_files):
        # 変化率を出す
        print(stock, i)
        rod = pct_change(values[stock])
        # rod = values[i]

        pylab.clf()
        pylab.hist(rod, bins=50, rwidth=0.8)
        pylab.savefig(',LSTM4Histogram%s.png' % (stock))

        stdev = np.std(rod)
        average = np.average(rod)
        print('average:', average, 'stdev:', stdev)

        threshold = 0.01
        categorized = pd.cut(rod, [-1, -threshold, 0, threshold, 1])
        # categorized = pd.qcut(rod, 4)
        # print(categorized)
        count = categorized.value_counts()
        print(count)
