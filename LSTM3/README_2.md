# LSTM で株価予測してみる (その6、3日分の予測)

[前回の方法](https://qiita.com/deadbeef/items/be3252538de2f5684f86)で、値動きなら予測可能であることがわかったので、3日分の値動きを予測させてみます。これは、

1. 0日目: 当日の取引終了後に値動きを予測。翌日の始値と翌々日の始値を出す。翌々日の始値が翌日の始値より高ければ買い。
1. 1日目: 寄付で成行で買う。
1. 2日目: 寄付で成行で売る。

のような取引を想定しています。


## データ


以下のようなタブ区切り形式ファイルを想定しています。

```
# 日付	始値	高値	安値	終値	出来高	調整後終値
1983-01-04	838	838	820	821	10620999	781.9
1983-01-05	820	832	807	832	16797999	792.38
1983-01-06	836	838	821	823	16977999	783.81
1983-01-07	840	840	823	823	16026999	783.81
```

始値も調整後の値が必要ですがこれは、調整後高値から計算できます。


銘柄は以下のものを使います

* 日経平均株価
* TOPIX 指数
* 日立製作所(6501)



## 実装

### データの読み込みと加工


```python
import numpy as np
import pandas as pd
from MultiLoader import MultiLoader

def load_data(date_file, stock_data_files, target_stock_name):
    """
    Scraper が吐き出したファイルを読むのです。
    日付と調整後終値を返すのです。
    """
    multi_loader = MultiLoader(date_file, stock_data_files)

    high = multi_loader.extract('high')
    low = multi_loader.extract('low')
    end = multi_loader.extract('end')
    adj_ends = multi_loader.extract('adj_end')
    adj_starts = multi_loader.extract('adj_start')

    return (high, low, end, adj_starts, adj_ends)

    
def convert_data(values):
    # 騰落率を出すのです。
    returns = pd.Series(values).pct_change()
    # 累積積を出すのです。
    ret_index = (1 + returns).cumprod()
    # 最初は 1 なのです。
    ret_index[0] = 1.0
    return ret_index


def create_train_data(high, low, end, adj_ends, up_down_rate, y_data, samples):

    chop = 0

    # 銘柄×日付→日付×銘柄に変換
    transposed = up_down_rate.transpose()

    _x = []
    _y = []
    # サンプルのデータを学習、 1 サンプルずつ後ろにずらしていく
    length = len(up_down_rate[0])
    for i in np.arange(chop, length - samples - 2):
        s = i + samples  # samplesサンプル間の変化を素性にする
        features = transposed[i:s]

        _y.append(y_data[s:s+in_out_neurons])
        _x.append(features)

    # 上げ下げの結果と教師データのセットを返す
    return np.array(_x), np.array(_y)

in_out_neurons = 3
length_of_sequences = 25


stock_data_files = [
    ',Nikkei225.txt', ',Topix.txt', ',6501.txt',
]
date_file = ',date.txt'

high, low, end, adj_starts, adj_ends = load_data(date_file, stock_data_files, ',6501.txt')
up_down_rate = np.asarray([convert_data(adj_end) for adj_end in adj_ends])
y_data = convert_data(adj_starts[stock_data_files.index(',6501.txt')])

# 学習データを生成
X, Y = create_train_data(high, low, end, adj_ends,
                         up_down_rate, y_data, length_of_sequences)
```



### 訓練データとテストデータの分割

時系列で検証するため、後半の2割をテストデータに利用します。

```python
split_pos = int(len(X) * 0.8)
train_x = X[:split_pos]
train_y = Y[:split_pos]
test_x = X[split_pos:]
test_y = Y[split_pos:]
```



### LSTM

[Keras](https://keras.io/ja/) を使って LSTM を実装します。バックエンドは [Tensorflow](https://www.tensorflow.org/) を使いました。

```python
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


hidden_neurons = 128
in_out_neurons = 3
epochs = 50


def create_model(dimension):
    model = Sequential()
    model.add(LSTM(hidden_neurons,
                   kernel_initializer='random_uniform',
                   return_sequences=False,
                   batch_input_shape=(None, length_of_sequences, dimension)))
    # model.add(Dropout(0.5))
    model.add(Dense(in_out_neurons, kernel_initializer='random_uniform'))
    model.add(Activation("linear"))
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss="mean_squared_error",
                  optimizer=optimizer)

    return model


dimension = len(X[0][0])
model = create_model(dimension)
es = EarlyStopping(monitor='loss', patience=10, verbose=1)
history = model.fit(train_x, train_y, batch_size=10,
                    epochs=epochs, verbose=1, validation_split=0.2, callbacks=[es])
```



### 検証

`model.evaluate` でスコアを出します。また、`model.predict` を実行して予測値と正解を `print` します。
Precision, Recall を計算するため、以下のようにカテゴライズします。
* 予測値が正の場合 Positive、0または負の場合 Negative
* 真の値が正の場合 True、0または負の場合 False

```python
from sklearn import metrics


def print_train_history(history):
    print("Epoch,Loss,Val loss")
    for i in range(len(history.history['loss'])):
        loss = history.history['loss'][i]
        val_loss = history.history['val_loss'][i]
        print("%d,%f,%f" % (i, loss, val_loss))


def print_predict_result(preds, test_y, initial_value):
    print("i,predict,test")
    for i in range(0, len(preds), 3):
        for j in range(3):
            predict = preds[i][j] * initial_value
            test    = test_y[i][j] * initial_value
            print("%d,%f,%f" % (i+j, predict, test))

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(preds)):
        predict = preds[i]
        test    = test_y[i]
        updown_p = predict[1] - predict[0]
        updown_t = test[1] - test[0]
        if updown_p > 0 and updown_t > 0:
            tp += 1
        elif updown_p > 0 and updown_t <= 0:
            fp += 1
        elif updown_p <= 0 and updown_t <= 0:
            fn += 1
        elif updown_p <= 0 and updown_t > 0:
            tn += 1

    print("TP = %d, FP = %d, TN = %d, FN = %d" % (tp, fp, tn, fn))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_value = 2 * recall * precision / (recall + precision)
    print("Precision = %f, Recall = %f, F = %f" % (precision, recall, f_value))


# 学習の履歴
print_train_history(history)

# 検証
score = model.evaluate(test_x, test_y, batch_size=10, verbose=1)
print("score:", score)

# 検証(2)
initial_value = adj_starts[stock_data_files.index(',6501.txt')][0]
preds = model.predict(test_x)
print_predict_result(preds, test_y, initial_value)
```



## 結果

LSTM の出力数(`hidden_neurons`)と、学習に使う日数 (`length_of_sequences`) を変えて検証してみます。
結果の検証は損失関数の出力 (二乗平均平方根) で行います。
`loss` が学習データの損失、`Val loss` が検証データの損失です。両者の差がわかりやすいよう、片対数グラフとしました。

### hidden_neurons = 4, 25日

### hidden_neurons = 4, 5日

### hidden_neurons = 128, 25日

### hidden_neurons = 128, 5日

### hidden_neurons = 300, 25日

### hidden_neurons = 300, 5日



## hidden_neurons = 128, 25日で、Precision, Recall を計算

`hidden_neurons` が 128、`length_of_sequences` が 25日の場合が比較的良い結果が得られたので、この組み合わせで Precision, Recall を計算します。

```
Precision = 0.502232, Recall = 0.661765, F = 0.571066
Precision = 0.495899, Recall = 1.000000, F = 0.663011 (Negative なし)
Precision = 0.509669, Recall = 0.534783, F = 0.521924
Precision = 0.495899, Recall = 1.000000, F = 0.663011 (Negative なし)
Precision = 0.666667, Recall = 0.002954, F = 0.005882
Precision = 0.495899, Recall = 1.000000, F = 0.663011 (Negative なし)
Precision = 0.503861, Recall = 0.763158, F = 0.606977
Precision = 0.529915, Recall = 0.179710, F = 0.268398
ZeroDivisionError: division by zero  (Positive なし)
```


## 結論

* 不労所得への道は険しい。
