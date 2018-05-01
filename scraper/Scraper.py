#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#
# Yahoo! ファイナンスから株価データをダウンロードするのです。
#


import sys
import time
from datetime import date

import requests
from bs4 import BeautifulSoup

from scraper import Stock

beautiful_soup_parser = 'html.parser'

url_pattern = 'http://info.finance.yahoo.co.jp/history/' \
              '?code=%s&sy=1983&sm=1&sd=1&ey=%d&em=%d&ed=%d&tm=d&p=%d'


def save_stock_data(ticker, file_name):
    stock_data = get_stock(ticker)
    f = open(file_name, 'w', encoding='UTF-8')
    [f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
        s.date, s.start, s.high, s.low, s.end, s.volume, s.adj_end)) for s in stock_data[::-1]]
    f.close()


def get_stock(ticker):
    today = date.today()
    end_year = today.year
    end_month = today.month
    end_day = today.day

    def _get_stock(page, stocks):
        soup = get_page(ticker, end_year, end_month, end_day, page)
        time.sleep(0.5)
        if page == 1:
            print(soup.title.text)
        trs = soup.select('table.boardFin tr')
        if len(trs) == 0:
            return stocks
        else:
            len_stocks = len(stocks)
            for tr in trs:
                bs_tr = BeautifulSoup(str(tr), beautiful_soup_parser)
                tds = bs_tr.find_all('td')
                if len(tds) == 7:
                    # 普通の銘柄
                    stock = Stock.Stock(
                            date=tds[0].text,
                            start=tds[1].text.replace(',', ''),
                            high=tds[2].text.replace(',', ''),
                            low=tds[3].text.replace(',', ''),
                            end=tds[4].text.replace(',', ''),
                            volume=tds[5].text.replace(',', ''),
                            adj_end=tds[6].text.replace(',', '')
                    )
                    stocks.append(stock)
                elif len(tds) == 5:
                    # 日経平均とか
                    stock = Stock.Stock(
                            date=tds[0].text,
                            start=tds[1].text.replace(',', ''),
                            high=tds[2].text.replace(',', ''),
                            low=tds[3].text.replace(',', ''),
                            end=tds[4].text.replace(',', ''),
                            volume='',
                            adj_end=tds[4].text.replace(',', ''))
                    stocks.append(stock)

            if len(stocks) == len_stocks:
                return stocks
            else:
                return _get_stock(page + 1, stocks)

    return _get_stock(1, [])


def get_page(ticker, end_year, end_month, end_day, page):
    url = url_pattern % (ticker, end_year, end_month, end_day, page)
    res = requests.get(url)
    print(url)
    return BeautifulSoup(res.text, beautiful_soup_parser)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: Scraper.py [TICKER] [FILE]")
        exit()

    # 株価データをファイルに保存するのです。
    ticker_ = sys.argv[1]
    file_name_ = sys.argv[2]
    save_stock_data(ticker_, file_name_)
