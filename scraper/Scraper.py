#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import os
from datetime import date, datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup


class Scraper:

    beautiful_soup_parser = 'html.parser'
    url_pattern = 'http://info.finance.yahoo.co.jp/history/' \
        '?code=%s&sy=1983&sm=1&sd=1&ey=%d&em=%d&ed=%d&tm=d&p=%d'
    csv_headers = ['Date', 'Open', 'High',
                   'Low', 'Close', 'Adj close', 'Volume']

    def __init__(self, ticker, file_name):
        self.ticker = ticker
        self.file_name = file_name

    def load_stock_data(self):
        if os.path.exists(self.file_name):
            return pd.read_csv(self.file_name,
                               header=None,
                               index_col=0,
                               names=self.csv_headers)
        else:
            return pd.DataFrame({}, columns=self.csv_headers[1:])

    def save_stock_data(self, stock_data):
        stock_data.sort_index().to_csv(self.file_name, header=False)

    def merge_stock_data(self, stock_data):
        today = date.today()
        end_year = today.year
        end_month = today.month
        end_day = today.day

        def _get_stock(page, stocks):
            soup = self.__get_page(end_year, end_month, end_day, page)
            if page == 1:
                print(soup.title.text)
            trs = soup.select('table.boardFin tr')
            if len(trs) == 0:
                return stocks
            else:
                len_stocks = len(stocks)
                dates = []
                opens = []
                highs = []
                lows = []
                closes = []
                adj_closes = []
                volumes = []
                for tr in trs:
                    bs_tr = BeautifulSoup(str(tr), self.beautiful_soup_parser)
                    tds = bs_tr.find_all('td')
                    if len(tds) == 7:
                        # 普通の銘柄
                        dates.append(self.__convert_date(tds[0].text))
                        opens.append(tds[1].text.replace(',', ''))
                        highs.append(tds[2].text.replace(',', ''))
                        lows.append(tds[3].text.replace(',', ''))
                        closes.append(tds[4].text.replace(',', ''))
                        adj_closes.append(tds[6].text.replace(',', ''))
                        volumes.append(tds[5].text.replace(',', ''))
                    elif len(tds) == 5:
                        # 日経平均とか
                        dates.append(self.__convert_date(tds[0].text))
                        opens.append(tds[1].text.replace(',', ''))
                        highs.append(tds[2].text.replace(',', ''))
                        lows.append(tds[3].text.replace(',', ''))
                        closes.append(tds[4].text.replace(',', ''))
                        adj_closes.append(tds[4].text.replace(',', ''))
                        volumes.append(None)

                # 1 ペイジ分のデイタ
                df_page = pd.DataFrame({
                    self.csv_headers[1]: opens,
                    self.csv_headers[2]: highs,
                    self.csv_headers[3]: lows,
                    self.csv_headers[4]: closes,
                    self.csv_headers[5]: adj_closes,
                    self.csv_headers[6]: volumes,
                }, index=dates)

                # 既存のデイタにくっつける
                new_stock_data = pd.concat([stocks, df_page], sort=False)

                # グループ化して1件だけ抽出
                grouped = new_stock_data.groupby(level=0)
                new_new_stock_data = grouped.last()

                if len(new_new_stock_data) == len_stocks:
                    # 増えてない→おしまい
                    return new_new_stock_data
                else:
                    # 増えた→つづく
                    time.sleep(1)   # 武士の情け
                    return _get_stock(page + 1, new_new_stock_data)

        return _get_stock(1, stock_data)

    def __get_page(self, end_year, end_month, end_day, page):
        url = self.url_pattern % (
            self.ticker, end_year, end_month, end_day, page)
        res = requests.get(url)
        print(url)
        return BeautifulSoup(res.text, self.beautiful_soup_parser)

    def __convert_date(self, date_str):
        dt = datetime.strptime(date_str, '%Y年%m月%d日')
        return dt.strftime('%Y-%m-%d')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: Scraper.py [TICKER] [FILE]")
        exit()

    ticker_ = sys.argv[1]
    file_name_ = sys.argv[2]

    scraper = Scraper(ticker_, file_name_)

    # 株価データをロード
    stock_data = scraper.load_stock_data()

    # 既存のデータと最新データをマージ
    new_stock_data = scraper.merge_stock_data(stock_data)

    # ファイルに保存
    scraper.save_stock_data(new_stock_data)
