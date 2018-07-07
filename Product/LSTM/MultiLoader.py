#! /usr/bin/env python3

# coding: utf-8

import math
from collections import OrderedDict
import numpy as np
import pandas as pd


class MultiLoader:

    def __init__(self, date_file, stock_files):
        self.data = self.__load_multiple(date_file, stock_files)
        self.stock_count = len(stock_files)

    def __load_multiple(self, date_file, stock_files):
        # 日付を読み込む
        dates = self.__load_date_file(date_file)
        for stock_file in stock_files:
            lines = [line[:-1]
                     for line in open(stock_file, 'r', encoding='utf-8')]
            for line in lines:
                splited_line = line.split(',')
                start = float(splited_line[1])
                high = float(splited_line[2])
                low = float(splited_line[3])
                end = float(splited_line[4])
                adj_end = float(splited_line[5])
                adj_start = start * adj_end / end
                adj_high = high * adj_end / end
                adj_low = low * adj_end / end
                stock_data = {
                    'date':       splited_line[0],
                    'start':      start,
                    'high':       high,
                    'low':        low,
                    'end':        end,
                    'adj_start':  adj_start,
                    'adj_high':   adj_high,
                    'adj_low':    adj_low,
                    'adj_end':    adj_end,
                    'ommyo':      end - start,
                    'ommyo_rate': (end - start) / end,
                    'ommyo_log':  (math.log10(end / start)),
                }
                stocks = dates[splited_line[0]]
                stocks.append(stock_data)

        # 規定数に満たない日付のデータを削除
        dates = self.__trim_incomplete_data(dates, len(stock_files))
        return dates

    def __load_date_file(self, date_file):
        """
        日付ファイルをロードするのです。
        """
        lines = [line[:-1] for line in open(date_file, 'r', encoding='utf-8')]
        dates = OrderedDict()
        for line in lines:
            dates[line] = []
        return dates

    def __trim_incomplete_data(self, dates, regular_length):
        """
        不完全なデータを除去するのです。
        """
        delete_keys = []
        for data_key in dates.keys():
            if len(dates[data_key]) != regular_length:
                delete_keys.append(data_key)
        for key in delete_keys:
            del dates[key]
        return dates

    def get_raw_data(self):
        return self.data

    def extract(self, column):
        ret_val = np.empty((self.stock_count, len(self.data.keys())))
        for date_index, val in enumerate(self.data.values()):
            for index, stock in enumerate(val):
                ret_val[index][date_index] = stock[column]
        return ret_val


if __name__ == '__main__':
    stock_data_files = [
        ',6501.txt', ',6502.txt', ',6503.txt', ',6702.txt', ',6752.txt',
        ',6758.txt', ',6770.txt', ',6803.txt', ',6857.txt', ',7752.txt',
    ]
    date_file = ',date.txt'

    ml = MultiLoader(date_file, stock_data_files)
    data = ml.extract('adj_end')
    print(data)
