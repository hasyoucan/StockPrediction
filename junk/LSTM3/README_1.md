# LSTM で株価予測してみる (その5)

今回は原点に立ち返って、値動きそのものを予測してみます。


## データ


以下のようなタブ区切り形式ファイルを想定しています。

```
# 日付	始値	高値	安値	終値	出来高	調整後終値
1983-01-04	838	838	820	821	10620999	781.9
1983-01-05	820	832	807	832	16797999	792.38
1983-01-06	836	838	821	823	16977999	783.81
1983-01-07	840	840	823	823	16026999	783.81
```

銘柄は以下のものを使います

* 日経平均株価
* TOPIX 指数
* ドル円レート
* 日立製作所(6501)



## 実装

### データの読み込みと加工

まず、調整後終値から騰落率を算出します。そして騰落率の累積積を計算して、初期値を1とした変動率を算出します。
ラベルデータには、日立の騰落率の累積積を使い、この値を予測させます。

以下がデータ読み込みを行うコードです。`MultiLoader` は [ここ](https://qiita.com/deadbeef/items/196a8af7d5767f6cd01e)のものを使います。
色々と使ってない変数が多いですが気にしないでください。

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
    ommyo_rate = multi_loader.extract('ommyo_rate')

    # (終値-始値)/終値→ Y
    target_index = stock_data_files.index(target_stock_name)
    y_data = ommyo_rate[target_index]
    
    return (high, low, end, adj_ends, ommyo_rate, y_data)

    
def convert_data(values):
    # 騰落率を出すのです。
    returns = pd.Series(values).pct_change()
    # 累積積を出すのです。
    ret_index = (1 + returns).cumprod()
    # 最初は 1 なのです。
    ret_index[0] = 1.0
    return ret_index

def create_train_data(high, low, end, adj_ends, up_down_rate, ommyo_rate, y_data, samples):
    chop = 0

    # 銘柄×日付→日付×銘柄に変換
    transposed = up_down_rate.transpose()

    _x = []
    _y = []
    # サンプルのデータを学習、 1 サンプルずつ後ろにずらしていく
    length = len(up_down_rate[0])
    for i in np.arange(chop, length - samples - 1):
        s = i + samples  # samplesサンプル間の変化を素性にする
        features = transposed[i:s]

        _y.append([y_data[s]])
        _x.append(features)

    # 上げ下げの結果と教師データのセットを返す
    return np.array(_x), np.array(_y)


length_of_sequences = 25

if __name__ == '__main__':

    stock_data_files = [
        ',Nikkei225.txt', ',Topix.txt', ',usdjpy.txt' ',6501.txt',
    ]
    date_file = ',date.txt'
    # 調整後終値, y
    high, low, end, adj_ends, ommyo_rate, y_data = load_data(date_file, stock_data_files, ',6501.txt')
    up_down_rate = np.asarray([convert_data(adj_end) for adj_end in adj_ends])
    y_data = convert_data(adj_ends[stock_data_files.index(',6501.txt')])

    # 学習データを生成
    X, Y = create_train_data(high, low, end, adj_ends,
                             up_down_rate, ommyo_rate, y_data, length_of_sequences)
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

[Keras](https://keras.io/ja/) を使って LSTM を実装します。バックエンドは [Tensorflow](https://www.tensorflow.org/) を使いました。活性化関数に `linear`、損失関数に `mean_squared_error` を使います。

```python
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


hidden_neurons = 128
in_out_neurons = 1
epochs = 20


def create_model(dimension):
    model = Sequential()
    model.add(LSTM(hidden_neurons,
                   batch_input_shape=(None, length_of_sequences, dimension)))
    model.add(Dropout(0.5))
    model.add(Dense(in_out_neurons))
    model.add(Activation("linear"))
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss="mean_squared_error",
                  optimizer=optimizer, metrics=['accuracy'])

    return model

model = create_model(len(stock_data_files))
history = model.fit(train_x, train_y, batch_size=10,
                    epochs=epochs, verbose=1, validation_split=0.2)
```



### 検証

`model.evaluate` でスコアを出します。また、`model.predict` を実行して予測値と正解を `print` します。

```python
from sklearn import metrics


def print_predict_result(preds, test_y):
    print("i,predict,test,error")
    for i in range(len(preds)):
        predict = preds[i][0]
        test    = test_y[i][0]
        print("%d,%f,%f,%f" % (i, predict, test, (predict - test) / test))
    print('mean_squared_error:', metrics.mean_squared_error(test_y, preds))

score = model.evaluate(test_x, test_y, batch_size=10, verbose=1)
print("score:", score)

preds = model.predict(test_x)
print_predict_result(preds, test_y)
```



### 結果


#### 日経平均 & 日立

学習データに日経平均と日立の騰落率の累積積を与えて日立の騰落率の累積積を予測させます。



#### TOPIX & 日立

学習データに TOPIX と日立の騰落率の累積積を与えて日立の騰落率の累積積を予測させます。



#### ドル円 & 日立

学習データに TOPIX と日立の騰落率の累積積を与えて日立の騰落率の累積積を予測させます。



#### 日経平均 & TOPIX & 日立

学習データに日経平均、TOPIX と日立の騰落率の累積積を与えて日立の騰落率の累積積を予測させます。



### 1日の上昇率を予測

1日の上昇率 ((終値 - 始値) / 終値) の値は `ommyo_rate` に入っています。この値で予測してみます。
学習データに日経平均、TOPIX と日立の 1日の上昇率を与えて日立の 1日の上昇率を予測させます。


このようなノイジーなデータを予測するのは難しそうです。