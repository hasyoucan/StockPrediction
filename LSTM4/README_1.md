# 4 分類

[このスライド](https://www.slideshare.net/tetsuoishigaki/stock-prediction-82133990/) で高い予測を出していたので、これを再現してみます。

## ヒストグラム

以下の指標と、出来高の高い銘柄を適当に選んで、調整後終値の変化率のヒストグラムを出してみます。

* 日経平均株価
* TOPIX
* 日立
* 曙ブレーキ
* 三菱UFJ
* みずほFG

ヒストグラムを出すプログラムは以下の通りです。

```python
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import pylab

from MultiLoader import MultiLoader


def load_data(date_file, stock_data_files):
    multi_loader = MultiLoader(date_file, stock_data_files)
    adj_ends = multi_loader.extract('adj_end')
    return adj_ends


def rate_of_decline(values):
    ret_val = pd.Series(values).pct_change()
    return ret_val[1:]


if __name__ == '__main__':

    stock_data_files = [
        ',Nikkei225.txt', ',TOPIX.txt', ',6501.txt', ',7238.txt', ',8306.txt', ',8411.txt'
    ]
    date_file = ',date.txt'

    adj_ends = load_data(date_file, stock_data_files)

    for (i, stock) in enumerate(stock_data_files):
        # 変化率を出す
        print(stock, i)
        rod = rate_of_decline(adj_ends[i])

        pylab.clf()
        pylab.hist(rod, bins=50, rwidth=0.8)
        pylab.savefig('LSTM4Histogram%s.png' % (stock))
```

グラフは以下のようになりました。偏りがみられますが正規分布に近いグラフになります。



## 4 つにカテゴライズ

分布が均等になるように 4つの領域に分割します。`pandas.qcut()` で分類してみます。

```python
count = pd.qcut(rod, 4).value_counts()
print(count)
```

```
,Nikkei225.txt 0
(0.00814, 0.0771]     579
(-0.00641, 0.0007]    579
(-0.107, -0.00641]    579
(0.0007, 0.00814]     578
dtype: int64
,TOPIX.txt 1
(0.00728, 0.0802]                   579
(-0.00647, 0.000708]                579
(-0.09570000000000001, -0.00647]    579
(0.000708, 0.00728]                 578
dtype: int64
,6501.txt 2
(-0.0109, 0.0]       610
(0.0115, 0.166]      579
(-0.171, -0.0109]    579
(0.0, 0.0115]        547
dtype: int64
,7238.txt 3
(-0.0138, 0.0]       652
(0.0132, 0.349]      579
(-0.152, -0.0138]    579
(0.0, 0.0132]        505
dtype: int64
,8306.txt 4
(-0.0112, 0.0]       635
(0.0109, 0.158]      579
(-0.094, -0.0112]    579
(0.0, 0.0109]        522
dtype: int64
,8411.txt 5
(-0.00976, 0.0]       735
(-0.106, -0.00976]    580
(0.00894, 0.152]      579
(0.0, 0.00894]        421
dtype: int64
```

日経平均と TOPIX はきれいに別れましたが、個別銘柄は偏りがあります。
`pandas.qcut()` で自動分類すると、学習データとテストデータで分割点が変わるので、`pandas.qcut()` が出した分割点を固定します。
とりあえず、`0.01` で固定します。


```
threshold = 0.01
count = pd.cut(rod, [-1, -threshold, 0, threshold, 1]).value_counts()
print(count)
```

```
,Nikkei225.txt 0
(0.0, 0.01]      750
(-0.01, 0.0]     679
(0.01, 1.0]      476
(-1.0, -0.01]    410
dtype: int64
,TOPIX.txt 1
(0.0, 0.01]      810
(-0.01, 0.0]     704
(0.01, 1.0]      422
(-1.0, -0.01]    379
dtype: int64
,6501.txt 2
(0.01, 1.0]      648
(-1.0, -0.01]    631
(-0.01, 0.0]     558
(0.0, 0.01]      478
dtype: int64
,7238.txt 3
(-1.0, -0.01]    708
(0.01, 1.0]      699
(-0.01, 0.0]     523
(0.0, 0.01]      385
dtype: int64
,8306.txt 4
(-1.0, -0.01]    646
(0.01, 1.0]      611
(-0.01, 0.0]     568
(0.0, 0.01]      490
dtype: int64
,8411.txt 5
(-0.01, 0.0]     752
(-1.0, -0.01]    563
(0.01, 1.0]      522
(0.0, 0.01]      478
dtype: int64
```

