# LSTM で株価予測してみる (その2)


[前回](http://qiita.com/deadbeef/items/3966e702a3b361258dfe) はあまり精度が出なくて轟沈したわけですが、やり方を変えてチャレンジしてみました。変更ポイントは以下のとおりです。

* 株価(調整後終値)の変化率だけではなく、テクニカル指標も学習させる。
* ラベルデータは調整後終値が上昇したかどうかではなく、陽線を引いた場合を 1 とする。
* 評価は、Precision/Recall を使う。


## テクニカル指標

株価や為替といった相場の動向をわかりやすくするために[テクニカル指標](https://ja.wikipedia.org/wiki/%E3%83%86%E3%82%AF%E3%83%8B%E3%82%AB%E3%83%AB%E6%8C%87%E6%A8%99%E4%B8%80%E8%A6%A7)という考え方が用いられています。その中で下記に示す代表的な指標を算出して学習させたいと思います。

* 移動平均
* 移動平均乖離率
* MACD
* MACD シグナル
* RSI
* ROC
* Fast ストキャスティクス
* Slow ストキャスティクス

実装は以下のようになります。

```python
import numpy as np
import pandas as pd
from pandas import DataFrame

def moving_average(values, period):
    """
    移動平均を計算するのです。
    """
    return DataFrame(values).rolling(period).mean()


def moving_average_deviation_rate(values, period):
    """
    移動平均乖離率を計算するのです。
    """
    _values = DataFrame(values)
    ma = moving_average(_values, period)
    return (_values - ma) / ma


def macd(values, short_period, long_period, signal_period):
    """
    MACD とその signal を計算するのです。
    """
    _values = DataFrame(values)
    shorts = _values.ewm(span=short_period).mean()
    longs = _values.ewm(span=long_period).mean()
    _macd = shorts - longs
    return _macd, _macd.ewm(span=signal_period).mean()


def roc(values, period):
    """
    ROC を計算するのです。
    """
    _values = DataFrame(values)
    pasts = _values.shift(period)
    return (_values - pasts) / _values


def rsi(values, period):
    """
    Wilder の RSI を計算するのです。
    """
    _values = DataFrame(values)
    _diff = _values.diff(1)
    _posi = _diff.clip_lower(0).ewm(alpha=1/period).mean()
    _nega = _diff.clip_upper(0).ewm(alpha=1/period).mean()
    return _posi / (_posi - _nega)


def stochastic_K(values_end, values_high, values_low, period):
    """
    ストキャスティクス の %K を計算するのです。
    """
    end = DataFrame(values_end)
    high = DataFrame(values_high)
    low = DataFrame(values_low)
    
    hline = high.rolling(period).max()
    lline = low.rolling(period).min()
    return (end - lline) / (hline - lline)


def stochastic_D(values_end, values_high, values_low, period_K, period):
    """
    ストキャスティクス の %D を計算するのです。
    """
    end = DataFrame(values_end)
    high = DataFrame(values_high)
    low = DataFrame(values_low)
    
    hline = high.rolling(period_K).max()
    lline = low.rolling(period_K).min()
    
    sumlow = (end - lline).rolling(period).sum()
    sumhigh = (hline - lline).rolling(period).sum()
    
    return sumlow / sumhigh


def stochastic_slowD(values_end, values_high, values_low, period_K, period_D, period):
    """
    ストキャスティクス の SlowD を計算するのです。
    """
    d = stochastic_D(values_end, values_high, values_low, period_K, period_D)
    return d.rolling(period).mean()
```


## 実装

### データの読み込み

データファイルは前回と同じタブ区切り形式です。高値、安値、終値と(終値-始値)も返すように変更します。

```python
def load_data(file_name):
    lines = [line[:-1] for line in open(file_name, 'r', encoding='utf-8')]
    split = [line.split('\t') for line in lines if
             not (line.startswith('#') or len(line) == 0)]
    # 日付, 高値, 安値, 調整後終値、(終値-始値))を返すのです。
    return ([line[0] for line in split],
            [float(line[2]) for line in split],
            [float(line[3]) for line in split],
            [float(line[4]) for line in split],
            [float(line[6]) for line in split],
            [float(line[4]) - float(line[1]) for line in split])
```


### 学習データの生成

学習データにテクニカル指標を盛り込むようにします。前述の指標をすべて盛り込みましたので、学習データは 1日あたり 13 次元になります。
移動平均は調整後終値の変化率から算出するようにします。移動平均は他の指標に比べて大きな数値になるためか、学習がうまくいかなくなるようです。(予測させるとほぼ同じ値しか返さなくなる)
テクニカル指標の計算に用いる期間は、とりあえず [Yahoo! ファイナンスで使われている日数](https://info.finance.yahoo.co.jp/dictionary/)にしました。
ラベルデータは[終値-始値]を使用し、この値が 0 より大きい場合は 1 となるようにします。これは、寄り付きに成り行きで買って引けで売る、という取引で利益を出すのを想定しています。ざら場に張り付かなくてもよくするためです。

```python
# samples = length_of_sequences
def create_train_data(high, low, end, adj_end, up_down_rate, ommyou, samples):
    tech_period = {
        'ma_short': 25,
        'ma_long' : 75,

        'madr_short' : 5,
        'madr_long' : 25,

        'macd_short' : 12,
        'macd_long' : 26,
        'macd_signal' : 9,

        'rsi' : 14,
        'roc' : 12,

        'fast_stoc_k' : 5,
        'fast_stoc_d' : 3,

        'slow_stoc_k' : 15,
        'slow_stoc_d' : 3,
        'slow_stoc_sd' : 3,
    }

    ma_short = Technical.moving_average(up_down_rate.values, tech_period['ma_short'])
    ma_long = Technical.moving_average(up_down_rate.values, tech_period['ma_long'])
    madr_short = Technical.moving_average_deviation_rate(adj_end, tech_period['madr_short'])
    madr_long = Technical.moving_average_deviation_rate(adj_end, tech_period['madr_long'])
    macd, macd_signal = Technical.macd(adj_end,
                                       tech_period['macd_short'],
                                       tech_period['macd_long'],
                                       tech_period['macd_signal'])
    rsi = Technical.rsi(adj_end, tech_period['rsi'])
    roc = Technical.roc(adj_end, tech_period['roc'])
    fast_stoc_k = Technical.stochastic_K(end, high, low, tech_period['fast_stoc_k'])
    fast_stoc_d = Technical.stochastic_D(end, high, low,
                                         tech_period['fast_stoc_k'],
                                         tech_period['fast_stoc_d'])
    slow_stoc_d = Technical.stochastic_D(end, high, low,
                                         tech_period['slow_stoc_k'],
                                         tech_period['slow_stoc_d'])
    slow_stoc_sd = Technical.stochastic_slowD(end, high, low,
                                              tech_period['slow_stoc_k'],
                                              tech_period['slow_stoc_d'],
                                              tech_period['slow_stoc_sd'])

    # 先頭のこの日数分のデータは捨てる
    chop = max(tech_period.values())

    _x = []
    _y = []
    # サンプルのデータを学習、 1 サンプルずつ後ろにずらしていく
    for i in np.arange(chop, len(adj_end) - samples - 1):
        s = i + samples  # samplesサンプル間の変化を素性にする
        feature = up_down_rate.iloc[i:s]
        _ma_short = ma_short.iloc[i:s]
        _ma_long = ma_long.iloc[i:s]
        _madr_short = madr_short.iloc[i:s]
        _madr_long = madr_long.iloc[i:s]
        _macd = macd.iloc[i:s]
        _macd_signal = macd_signal.iloc[i:s]
        _rsi = rsi.iloc[i:s]
        _roc = roc.iloc[i:s]
        _fast_stoc_k = fast_stoc_k.iloc[i:s]
        _fast_stoc_d = fast_stoc_d.iloc[i:s]
        _slow_stoc_d = slow_stoc_d.iloc[i:s]
        _slow_stoc_sd = slow_stoc_sd.iloc[i:s]

        if ommyou[s] > 0:
            _y.append([1]) # 上がった
        else:
            _y.append([0]) # 上がらなかった
        _x.append([[
            feature.values[x],
            _ma_short.values[x][0],
            _ma_long.values[x][0],
            _madr_short.values[x][0],
            _madr_long.values[x][0],
            _macd.values[x][0],
            _macd_signal.values[x][0],
            _rsi.values[x][0],
            _roc.values[x][0],
            _fast_stoc_k.values[x][0],
            _fast_stoc_d.values[x][0],
            _slow_stoc_d.values[x][0],
            _slow_stoc_sd.values[x][0]
        ] for x in range(len(feature.values))])

    # 上げ下げの結果と教師データのセットを返す
    return np.array(_x), np.array(_y)
```

### モデルの構築

モデルは前回と変わりありません。入力データにあわせて `batch_input_shape` だけ変えています。

```python
# hidden_neurons = 128
# length_of_sequences = 25
# in_out_neurons = 1

def create_model():
    model = Sequential()
    model.add(LSTM(hidden_neurons,
                   batch_input_shape=(None, length_of_sequences, 13)))
    model.add(Dropout(0.5))
    model.add(Dense(in_out_neurons))
    model.add(Activation("sigmoid"))
    
    return model
```


### Precision/Recall

ラベルデータに陽線を引いたかどうかを使うことにしましたので、評価に使う指標も変更します。重要なのは取引がうまくいくかどうかですので、Precision を出すことにします。つまり、「陽線を引くと予測して実際に陽線を引く確率」で評価します。Recall、F 値もついでに出しておきます。


## 結果

今回は `epochs` を 20 にしました。使用したのは日経平均のデータで、1991/1/4～2017/4/7 の範囲です。


### すべての指標を使う

下記の 13 次元データで学習した結果です。

1. 調整後終値変化率
1. 移動平均(25日)
1. 移動平均(75日)
1. 移動平均乖離率(5日)
1. 移動平均乖離率(25日)
1. MACD(12日, 26日)
1. MACD シグナル(9日)
1. RSI(14日)
1. ROC(12日)
1. Fast ストキャスティクス(%K)(5日)
1. Fast ストキャスティクス(%D)(5日, 3日)
1. Slow ストキャスティクス(%D)(15日, 3日)
1. Slow ストキャスティクス(Slow D)(3日)

```
Precision: 0.5402298850574713 , Recall: 0.4351851851851852 , F: 0.482051282051282
Precision: 0.5673758865248227 , Recall: 0.24691358024691357 , F: 0.34408602150537637
Precision: 0.5138248847926268 , Recall: 0.3441358024691358 , F: 0.4121996303142329
Precision: 0.530952380952381 , Recall: 0.3441358024691358 , F: 0.4176029962546816
Precision: 0.5088105726872246 , Recall: 0.7129629629629629 , F: 0.5938303341902312
```

いまいちです。




### 組み合わせを絞る

指標をごった煮するのではなくて、特定の種類の指標を使って学習させます。

#### 移動平均

1. 調整後終値変化率
1. 移動平均(25日)
1. 移動平均(75日)

```
true_positive: 0
true_negative: 624
false_positive: 0
false_negative: 648
Traceback (most recent call last):
  File "StockPrediction2.py", line 234, in <module>
    print_predict_result(preds, test_y)
  File "StockPrediction2.py", line 196, in print_predict_result
    precision = true_positive / (true_positive + false_positive)
ZeroDivisionError: division by zero
```

予測値が 0.5 に達しなかったため、positive な予測が出ませんでした。



#### 移動平均乖離率

1. 調整後終値変化率
1. 移動平均乖離率(5日)
1. 移動平均乖離率(25日)

```
Precision: 0.5232558139534884 , Recall: 0.1388888888888889 , F: 0.21951219512195125
Precision: 0.5263157894736842 , Recall: 0.1388888888888889 , F: 0.21978021978021975
Precision: 0.5268817204301075 , Recall: 0.15123456790123457 , F: 0.23501199040767384
Precision: 0.4931506849315068 , Recall: 0.05555555555555555 , F: 0.09986130374479889
Precision: 0.5133333333333333 , Recall: 0.11882716049382716 , F: 0.1929824561403509
```

だめですね。


#### MACD

1. 調整後終値変化率
1. MACD(12日, 26日)
1. MACD シグナル(9日)

```
Precision: 0.515527950310559 , Recall: 0.25617283950617287 , F: 0.3422680412371134
Precision: 0.5209003215434084 , Recall: 0.5 , F: 0.5102362204724409
Precision: 0.5335120643431636 , Recall: 0.30709876543209874 , F: 0.3898139079333986
Precision: 0.5728155339805825 , Recall: 0.09104938271604938 , F: 0.15712383488681758
Precision: 0.5451713395638629 , Recall: 0.2700617283950617 , F: 0.3611971104231166
```

まあまあ。


#### RSI

1. 調整後終値変化率
1. RSI(14日)

```
true_positive: 0
true_negative: 624
false_positive: 0
false_negative: 648
Traceback (most recent call last):
  File "StockPrediction2.py", line 234, in <module>
    print_predict_result(preds, test_y)
  File "StockPrediction2.py", line 196, in print_predict_result
    precision = true_positive / (true_positive + false_positive)
ZeroDivisionError: division by zero
```

はい。


#### ROC

1. 調整後終値変化率
1. ROC(12日)

```
Precision: 0.5450236966824644 , Recall: 0.17746913580246915 , F: 0.26775320139697323
Precision: 0.5287958115183246 , Recall: 0.1558641975308642 , F: 0.2407628128724672
Precision: 0.5 , Recall: 0.07098765432098765 , F: 0.12432432432432432
Precision: 0.5089686098654709 , Recall: 0.7006172839506173 , F: 0.5896103896103896
Precision: 0.5 , Recall: 0.09259259259259259 , F: 0.15625
```

いまいちですね。


#### ストキャスティクス

1. 調整後終値変化率
1. Fast ストキャスティクス(%K)(5日)
1. Fast ストキャスティクス(%D)(5日, 3日)
1. Slow ストキャスティクス(%D)(15日, 3日)
1. Slow ストキャスティクス(Slow D)(3日)

```
Precision: 0.494949494949495 , Recall: 0.22685185185185186 , F: 0.31111111111111106
Precision: 0.4909090909090909 , Recall: 0.16666666666666666 , F: 0.2488479262672811
Precision: 0.4879518072289157 , Recall: 0.125 , F: 0.199017199017199
Precision: 0.5241935483870968 , Recall: 0.10030864197530864 , F: 0.1683937823834197
```

だめですね。



## まとめ

* 調整後終値変化率だけを用いた場合と、テクニカル指標を足した場合とでは、結果に大差ないことがわかりました。
* 学習に使う指標を絞っても結果に大差ないことがわかりました。
* 不労所得への道は険しい…



## 参考

* [Pythonでテクニカル指標を作成する](http://nbviewer.jupyter.org/url/forex.toyolab.com/ipynb/TA_ohlc.ipynb)