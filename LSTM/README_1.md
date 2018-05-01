# LSTM で株価予測してみる


好きな言葉は「不労所得」です。


## データ


データは、Yahoo! ファイナンスなど、適当なところから取得します。ここでは以下のようなタブ区切り形式ファイルを想定しています。

```text
# 日付	始値	高値	安値	終値	出来高	調整後終値
2017年3月31日	237.5	250	233	241.4	321569000	241.4
2017年4月3日	225.4	234.6	218.7	228.2	276744000	228.2
2017年4月4日	223.8	224.7	203.8	206.8	264374000	206.8
2017年4月5日	210	216	207.1	214.9	146285000	214.9
```


## 実装

### データの読み込みと加工

まず、調整後終値から騰落率を算出します。そして騰落率の累積積を計算して、初期値を1とした変動率を算出します。
ラベルデータとしては、株価が上がった場合は 1、上がらなかった場合は 0 とします。

`length_of_sequences` は変動を予測するもとの日数を示します。たとえば、`length_of_sequences=5` とした場合、5日間の変動を元に予測します。長期の変動を元に予測させたい場合はこの数字を大きくします。

```python
import sys
import numpy as np
import pandas as pd


length_of_sequences = 5

def load_data(file_name):
    lines = [line[:-1] for line in open(file_name, 'r', encoding='utf-8')]
    split = [line.split('\t') for line in lines if
            not (line.startswith('#') or len(line) == 0)]
    # 日付と調整後終値を返すのです。
    return [line[0] for line in split], [float(line[6]) for line in split]


def convert_data(values):
    # 騰落率を出すのです。
    returns = pd.Series(values).pct_change()
    # 累積積を出すのです。
    ret_index = (1 + returns).cumprod()
    # 最初は 1 なのです。
    ret_index[0] = 1.0
    return ret_index


def create_train_data(values, samples):
    train_x = []
    train_y = []
    # サンプルのデータを学習、 1 サンプルずつ後ろにずらしていく
    for i in np.arange(0, len(values) - samples - 1):
        s = i + samples  # samplesサンプル間の変化を素性にする
        feature = values.iloc[i:s]
        if feature[s - 1] < values[s]:
            train_y.append([1])            # 上がった
        else:
            train_y.append([0])            # 上がらなかった
        train_x.append([[x] for x in feature.values])

    # 上げ下げの結果と教師データのセットを返す
    return np.array(train_x), np.array(train_y)
    

file_name = 'Nikkei225.txt'
_, adj_end = load_data(file_name)
adj_end = convert_data(adj_end)
X, Y = create_train_data(adj_end, length_of_sequences)
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
epochs = 25

def create_model():
    model = Sequential()
    model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons)))
    model.add(Dropout(0.5))
    model.add(Dense(in_out_neurons))
    model.add(Activation("sigmoid"))
    
    return model

model = create_model()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
history = model.fit(train_x, train_y, batch_size=10, epochs=epochs, verbose=1)
```


### 検証

`model.evaluate` で検証します。また、テストデータで予測させて正答した割合も出します。予測値が 0.5 以上だった場合は上がったと予測、0.5 未満の場合は下がったと予測したとします。

```python
score = model.evaluate(test_x, test_y, batch_size=10, verbose=1)
print("score:", score)


preds = model.predict(test_x)
correct = 0
for i in range(len(preds)):
    p = 0 if preds[i][0] < 0.5 else 1
    t = test_y[i][0]
    if p == t:
        correct += 1

print("正答率:", correct / len(preds))
```


### 結果

1991年1月4日から2017年4月5日までの日経平均株価 (6,456日分) で検証しました。結果は以下のようになりました。

```
Epoch 1/25
5160/5160 [==============================] - 2s - loss: 0.6945 - acc: 0.4926
Epoch 2/25
5160/5160 [==============================] - 1s - loss: 0.6939 - acc: 0.4944
.........
Epoch 25/25
5160/5160 [==============================] - 1s - loss: 0.6932 - acc: 0.5103
 660/1290 [==============>...............] - ETA: 0sscore: [0.69332546873610146, 0.4906976814417876]
正答率: 0.4906976744186046
```

5回動かします。

```
正答率: 0.506201550387597
正答率: 0.541860465116279
正答率: 0.5054263565891473
正答率: 0.5348837209302325
```

いまいちですね。

短期がだめなら長期ではどうでしょうか。`length_of_sequences = 25` とします。

```
正答率: 0.5419906687402799
正答率: 0.48522550544323484
正答率: 0.5116640746500778
正答率: 0.4618973561430793
正答率: 0.5108864696734059
```

だめですね。

`batch_size` や LSTM のユニット数 `hidden_neurons` を変えても傾向は変わりませんでした。

モデルをに手を加えれば改善するかもですが、私にはまだそこまでの知識がありません…

また、取引にどう結びつけるかという課題もあります。たとえば翌日は上がるという結果が出ても陰線を引く場合があります。ざら場に張り付かなくてもいいようにラベルデータについても工夫する必要があります。

不労所得の道は険しい。