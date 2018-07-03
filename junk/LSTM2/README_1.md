# LSTM で株価予測してみる (複数銘柄を組み合わせる)


[今まで](https://qiita.com/deadbeef/items/3966e702a3b361258dfe)いくつかの方法を試して[轟沈](https://qiita.com/deadbeef/items/49e226b0b9c7236c4875)してきたわけですが、今回はちょっと方法を変えて試してみます。
今回は複数の銘柄の株価データを使って学習させてみます。銘柄の組み合わせはいくつか考えられますが、まずは同一業種の銘柄を組み合わせてみます。


## データ


以下のようなタブ区切り形式ファイルを想定しています。

```
# 日付	始値	高値	安値	終値	出来高	調整後終値
1983-01-04	838	838	820	821	10620999	781.9
1983-01-05	820	832	807	832	16797999	792.38
1983-01-06	836	838	821	823	16977999	783.81
1983-01-07	840	840	823	823	16026999	783.81
```



### 業種

株式の銘柄は必ずどこかの業種に分類されています。とりあえず「電気機器」から、2018年5月1日の出来高ランキング上位の銘柄から、10銘柄を組み合わせてみます。なるべく多くの時系列データを利用できるよう、最近上場されたものは除外します。

|株式コード|銘柄|
|----:|----|
|6803|ティアック|
|6501|日立|
|6702|富士通|
|6758|ソニー|
|6503|三菱電|
|~6740~|~ＪＤＩ~|
|6502|東芝|
|~6723~|~ルネサス~|
|6857|アドバンテス|
|6752|パナソニック|
|7752|リコー|
|6770|アルプス電|



### 日付を揃える

複数の株価データから日付の部分を抽出し、`sort` と `uniq` で株価データに含まれるすべての日付を抽出します。これを「日付ファイル」として保存し、株価データ読み込み時に日付ファイルの内容と照合して日付ごとの株価データのセットを作成します。
ただし、株価データは日付によっては欠損していることがありますので、日付のデータセットに含まれる銘柄数が想定より少ない場合は除外するようにします。

```sh
IN_FILES=,*.txt
OUT_FILE=,date.txt
cat $IN_FILES | awk '{print $1}' | sort | uniq > $OUT_FILE
```



## 実装

### データの読み込みと加工

まず、調整後終値から騰落率を算出します。そして騰落率の累積積を計算して、初期値を1とした変動率を算出します。
ラベルデータとしては、終値が始値より騰がった場合は 1、騰がらなかった場合は 0 とします。

`length_of_sequences` は変動を予測するもとの日数を示します。たとえば、`length_of_sequences=5` とした場合、5日間の変動を元に予測します。長期の変動を元に予測させたい場合はこの数字を大きくします。


`MultiLoader` クラスは複数の株価データの読み込みを行うクラスです。

```python
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
                splited_line = line.split('\t')
                stock_data = {
                    'date':    splited_line[0],
                    'high':    float(splited_line[2]),
                    'low':     float(splited_line[3]),
                    'end':     float(splited_line[4]),
                    'adj_end': float(splited_line[6]),
                    'ommyo':   float(splited_line[4]) - float(splited_line[1])
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

    def extract(self, column):
        ret_val = np.empty((self.stock_count, len(self.data.keys())))
        for date_index, val in enumerate(self.data.values()):
            for index, stock in enumerate(val):
                ret_val[index][date_index] = stock[column]
        return ret_val
```


これを使って株価データをロードします。ラベルデータに使う銘柄は日立 (6501) にします。

```python
import numpy as np
import pandas as pd
from MultiLoader import MultiLoader

def load_data(date_file, stock_data_files, target_stock_name):
    multi_loader = MultiLoader(date_file, stock_data_files)

    # 調整後終値
    adj_ends = multi_loader.extract('adj_end')

    # 終値-始値→ Y
    ommyo = multi_loader.extract('ommyo')
    target_index = stock_data_files.index(target_stock_name)
    y_data = ommyo[target_index]
    return (adj_ends, y_data)

def convert_data(values):
    # 騰落率を出すのです。
    returns = pd.Series(values).pct_change()
    # 累積席を出すのです。
    ret_index = (1 + returns).cumprod()
    # 最初は 1 なのです。
    ret_index[0] = 1.0
    return ret_index

def create_train_data(data, ommyou, samples):

    # 銘柄×日付→日付×銘柄に変換
    _data_transposed = data.transpose()
    _x = []
    _y = []
    # サンプルのデータを学習、 1 サンプルずつ後ろにずらしていく
    length = len(data[0])
    for i in np.arange(0, length - samples - 1):
        s = i + samples  # samplesサンプル間の変化を素性にする
        features = _data_transposed[i:s]
        if ommyou[s] > 0:
            _y.append([1])  # 上がった
        else:
            _y.append([0])  # 上がらなかった
        _x.append(features)

    # 上げ下げの結果と教師データのセットを返す
    return np.array(_x), np.array(_y)


length_of_sequences = 25

stock_data_files = [
    ',6501.txt', ',6502.txt', ',6503.txt', ',6702.txt', ',6752.txt',
    ',6758.txt', ',6770.txt', ',6803.txt', ',6857.txt', ',7752.txt',
]
date_file = ',date.txt'
# 調整後終値, y
adj_ends, y_data = load_data(date_file, stock_data_files, ',6501.txt')
up_down_rate = np.asarray([convert_data(adj_end) for adj_end in adj_ends])

# 学習データを生成
X, Y = create_train_data(up_down_rate, y_data, length_of_sequences)
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

[Keras](https://keras.io/ja/) を使って LSTM を実装します。バックエンドは [Tensorflow](https://www.tensorflow.org/) を使いました。0|1 の分類ですので活性化関数に `sigmoid`、損失関数に `binary_crossentropy` を使います。

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
    model.add(Activation("sigmoid"))

    return model

model = create_model(len(stock_data_files))
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=['accuracy'])
history = model.fit(train_x, train_y, batch_size=10,
                    epochs=epochs, verbose=1, validation_split=0.2)
```


### 検証

`model.evaluate` でスコアを出します。また、`model.predict` を実行して Precision と Recall を算出します。

```python
def print_predict_result(preds, test_y):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(preds)):
        p = 0 if preds[i][0] < 0.5 else 1
        t = test_y[i][0]
        if p == 1 and t == 1:
            true_positive += 1
        if p == 0 and t == 0:
            true_negative += 1
        if p == 1 and t == 0:
            false_positive += 1
        if p == 0 and t == 1:
            false_negative += 1

    print("true_positive:", true_positive)
    print("true_negative:", true_negative)
    print("false_positive:", false_positive)
    print("false_negative:", false_negative)

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f_value = 2.0 * precision * recall / (precision + recall)
    print("Precision:", precision, ", Recall:", recall, ", F:", f_value)



score = model.evaluate(test_x, test_y, batch_size=10, verbose=1)
print("score:", score)

preds = model.predict(test_x)
print_predict_result(preds, test_y)
```



### 結果

1983年1月4日から2018年5月1日まで、10銘柄を組み合わせて検証しました。

```
Epoch 1/20
2018-05-03 15:09:14.589849: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
5632/5632 [==============================] - 24s 4ms/step - loss: 0.6909 - acc: 0.5627 - val_loss: 0.6995 - val_acc: 0.5653
Epoch 2/20
5030/5632 [=========================>....] - ETA: 2s - loss: 0.6787 - acc: 0.5751 


.........


Epoch 20/20
5632/5632 [==============================] - 21s 4ms/step - loss: 0.6728 - acc: 0.5850 - val_loss: 0.6879 - val_acc: 0.5653
Epoch,Acc,Loss,Val acc, Val loss
0,0.562678,0.690944,0.565341,0.699541
1,0.577237,0.678620,0.544034,0.690786
2,0.583097,0.675354,0.565341,0.688084
3,0.587180,0.674890,0.565341,0.689957
4,0.583984,0.675691,0.565341,0.688123
5,0.583274,0.673056,0.565341,0.687446
6,0.588778,0.674388,0.565341,0.687056
7,0.585050,0.675308,0.565341,0.687081
8,0.586293,0.673709,0.558949,0.688856
9,0.583984,0.673126,0.509943,0.692804
10,0.590021,0.672878,0.551847,0.689544
11,0.588068,0.672859,0.545455,0.691463
12,0.585760,0.674387,0.565341,0.688578
13,0.585050,0.672735,0.561790,0.689673
14,0.586293,0.672865,0.565341,0.690853
15,0.587003,0.673080,0.506392,0.693095
16,0.587891,0.672880,0.565341,0.687788
17,0.585050,0.672130,0.551136,0.689740
18,0.583629,0.672761,0.550426,0.690164
19,0.585050,0.672832,0.565341,0.687882
1761/1761 [==============================] - 2s 924us/step
score: [0.6970385907388154, 0.5394662181333274]
true_positive: 55
true_negative: 895
false_positive: 65
false_negative: 746
Precision: 0.4583333333333333 , Recall: 0.0686641697877653 , F: 0.11943539630836046
```

5回動かします。

```
Precision: 0.4583333333333333 , Recall: 0.0686641697877653 , F: 0.11943539630836046
Precision: 0.45524296675191817 , Recall: 0.2222222222222222 , F: 0.2986577181208054
ZeroDivisionError: division by zero (positive なし)
Precision: 0.4481012658227848 , Recall: 0.2209737827715356 , F: 0.29598662207357856
ZeroDivisionError: division by zero
```

はい、いまいちですね。


不労所得の道は険しい。