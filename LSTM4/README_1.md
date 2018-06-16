[このスライド](https://www.slideshare.net/tetsuoishigaki/stock-prediction-82133990/)で高い予測を出していたので、これを再現してみます。

## 準備

### ヒストグラム

指標と、出来高の高い銘柄を適当に選んで、調整後終値の変化率のヒストグラムを出してみます。

* 日経平均株価
* TOPIX
* 日立(6501)
* 曙ブレーキ(7238)
* 三菱UFJ(8306)
* みずほFG(8411)


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

#### 日経平均


#### TOPIX



#### 日立


#### 曙ブレーキ

この銘柄は最近大きなニュースを出しましたので値上がりの方に偏っています。




### 4 つに分類


分布が均等になるように 4つに分割します。`pandas.qcut()` で分類してみます。

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
`pandas.qcut()` での自動分割は、0% で分割してくれるとは限りません。`pandas.qcut()` による分割点を参考に、`pandas.cut()` で `0` と `±0.01` で分割するようにします。


```python
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

これにより値動きを

* 1% 以下の値下がり
* 1% 未満の値下がり
* 1% 以下の値上り
* 1% より大きい値上り

の 4つに分類できました。


## 実装

### 学習データの作成

学習データに使う銘柄は、以下のとおりとします。今までは初期値を 1とした騰落率の累積積を学習させてましたが、今回は騰落率の対数を学習させます。

* 日経平均株価
* TOPIX
* ドル円
* 日立 (6501)

ラベルデータは、日立の騰落率を 4 つに分類し、4 次元のベクトルにしたものにします。


```python
import math
import numpy as np
import pandas as pd

from MultiLoader import MultiLoader

training_days = 75
threshold = 0.01
category_threshold = [-1, -threshold, 0, threshold, 1]

def load_data(date_file, stock_data_files, target_stock_name):
    multi_loader = MultiLoader(date_file, stock_data_files)
    adj_ends = multi_loader.extract('adj_end')
    adj_starts = multi_loader.extract('adj_start')
    return (adj_starts, adj_ends)

def pct_change(values):
    returns = pd.Series(values).pct_change()
    returns[0] = 0
    return returns

def log_diff(values):
    series = pd.Series(values)
    log_values = series.apply(math.log10)
    ret_val = log_values.diff().mul(100)
    ret_val[0] = 0.0
    return ret_val

def convert_data(values):
    return log_diff(values)

def create_train_data(adj_start, adj_ends, y_data, samples):

    udr_end = np.asarray([convert_data(v) for v in adj_ends])
    transposed = udr_end.transpose()

    _x = []
    _y = []
    length = len(udr_end[0])
    for i in np.arange(0, length - samples):
        s = i + samples
        _x.append(transposed[i:s])

        __y = [0, 0, 0, 0]
        __y[y_data[s]] = 1
        _y.append(__y)

    # 上げ下げの結果と教師データのセットを返す
    return np.array(_x), np.array(_y)


target_stock = ',6501.txt'
stock_data_files = [
    ',Nikkei225.txt', ',TOPIX.txt', ',usdjpy.txt', target_stock,
]
date_file = ',date.txt'

adj_starts, adj_ends = load_data(date_file, stock_data_files, target_stock)
y_data = pct_change(adj_starts[stock_data_files.index(target_stock)])
y_data = pd.cut(y_data, category_threshold, labels=False)

X, Y = create_train_data(adj_starts, adj_ends, y_data, training_days)

# データを学習用と検証用に分割
split_pos = int(len(X) * 0.8)
train_x = X[:split_pos]
train_y = Y[:split_pos]
test_x = X[split_pos:]
test_y = Y[split_pos:]
```


### LSTM モデル

[Keras](https://keras.io/ja/) を使って LSTM を実装します。バックエンドは [Tensorflow](https://www.tensorflow.org/) を使いました。活性化関数に `softmax`、損失関数に `categorical_crossentropy` を使います。


```python
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

hidden_neurons = 400
epochs = 50

def create_model(dimension):
    model = Sequential()
    model.add(LSTM(hidden_neurons,
                   activation='tanh',
                   recurrent_activation='hard_sigmoid',
                   use_bias=True,
                   kernel_initializer='random_uniform',
                   bias_initializer='zeros',
                   dropout=0.5,
                   recurrent_dropout=0.5,
                   return_sequences=False,
                   batch_input_shape=(None, training_days, dimension)))
    model.add(Dropout(0.5))
    model.add(Dense(4,
                    kernel_initializer='random_uniform',
                    bias_initializer='zeros'))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="RMSprop", metrics=['categorical_accuracy'])
    return model


dimension = len(X[0][0])
model = create_model(dimension)
es = EarlyStopping(patience=10, verbose=1)
history = model.fit(train_x, train_y, batch_size=10,
                    epochs=epochs, verbose=1, validation_split=0.2, callbacks=[es])
```


### 学習履歴

損失と精度を出します。

```python
def print_train_history(history):
    print("Epoch, Loss, Val loss, Acc, Val Acc")
    for i in range(len(history.history['loss'])):
        loss = history.history['loss'][i]
        val_loss = history.history['val_loss'][i]
        acc = history.history['categorical_accuracy'][i]
        val_acc = history.history['val_categorical_accuracy'][i]
        print("%d,%f,%f,%f,%f" % (i, loss, val_loss, acc, val_acc))

print_train_history(history)
```

### Precision, Recall, F値

値上がりに相当するカテゴリを True/Positive として、Precision、Recall、F値 を出します。予測値は確率で与えられますので、最も高い値になったカテゴリで判断します。


```python
def print_predict_result(preds, test_y):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(preds)):
        predict = np.argmax(preds[i])
        test    = np.argmax(test_y[i])
        positive = True if predict == 2 or predict == 3 else False
        true     = True if test == 2 or test == 3 else False
        if true and positive:
            tp += 1
        if not true and positive:
            fp += 1
        if true and not positive:
            tn += 1
        if not true and not positive:
            fn += 1

    print("TP = %d, FP = %d, TN = %d, FN = %d" % (tp, fp, tn, fn))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_value = 2 * recall * precision / (recall + precision)
    print("Precision = %f, Recall = %f, F = %f" % (precision, recall, f_value))


preds = model.predict(test_x)
print_predict_result(preds, test_y)
```

## 結果

### 損失、精度

損失のグラフです。



精度のグラフです。



### Precision, Recall, F値

Precision が値上がりすると予測して実際に値上がりする確率ですが、安定して 7割以上の値を出しています。

```
Precision = 0.758427, Recall = 0.441176, F = 0.557851
Precision = 0.770950, Recall = 0.443730, F = 0.563265
Precision = 0.716157, Recall = 0.523962, F = 0.605166
Precision = 0.762376, Recall = 0.481250, F = 0.590038
Precision = 0.824427, Recall = 0.361204, F = 0.502326
```

## 結論

4分類の予測で 7割以上の高い Precision を出せることがわかりました。ただし、上げ幅が小さいと手数料で負けてしまいますので、それを加味した分類にしてみるなどの工夫も必要です。


## ソースコード

ソースコードは[ここ](https://github.com/SakaiTakao/StockPrediction)で公開しています。過去分のものもあります。今回のコードは `LSTM4` です。


## 追記

### 他の銘柄

他の銘柄でも試してみました。

#### 曙ブレーキ

```
Precision = 0.756881, Recall = 0.457064, F = 0.569948
Precision = 0.823529, Recall = 0.416216, F = 0.552962
Precision = 0.792929, Recall = 0.430137, F = 0.557726
```

#### 三菱UFJ

```
Precision = 0.823834, Recall = 0.447887, F = 0.580292
Precision = 0.796117, Recall = 0.465909, F = 0.587814
Precision = 0.818750, Recall = 0.394578, F = 0.532520
```


#### みずほFG

```
Precision = 0.942857, Recall = 0.228374, F = 0.367688
Precision = 0.931818, Recall = 0.154717, F = 0.265372
Precision = 0.927083, Recall = 0.288026, F = 0.439506
```
