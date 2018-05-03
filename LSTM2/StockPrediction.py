# coding: utf-8

import numpy as np
import pandas as pd

import Technical
from MultiLoader import MultiLoader

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

# from sklearn import cross_validation as cv


hidden_neurons = 128
length_of_sequences = 25
in_out_neurons = 1

epochs = 20


def load_data(date_file, stock_data_files, target_stock_name):
    """
    Scraper が吐き出したファイルを読むのです。
    日付と調整後終値を返すのです。
    """
    multi_loader = MultiLoader(date_file, stock_data_files)

    # 調整後終値
    adj_ends = multi_loader.extract('adj_end')

    # 終値-始値→ Y
    ommyo = multi_loader.extract('ommyo')
    target_index = stock_data_files.index(target_stock_name)
    y_data = ommyo[target_index]
    return (adj_ends, y_data)

    # lines = [line[:-1] for line in open(file_name, 'r', encoding='utf-8')]
    # split = [line.split('\t') for line in lines if
    #          not (line.startswith('#') or len(line) == 0)]
    # # 日付, 高値, 安値, 調整後終値、(終値-始値))を返すのです。
    # return ([line[0] for line in split],
    #         [float(line[2]) for line in split],
    #         [float(line[3]) for line in split],
    #         [float(line[4]) for line in split],
    #         [float(line[6]) for line in split],
    #         [float(line[4]) - float(line[1]) for line in split])


def convert_data(values):
    # 騰落率を出すのです。
    returns = pd.Series(values).pct_change()
    # 累積席を出すのです。
    ret_index = (1 + returns).cumprod()
    # 最初は 1 なのです。
    ret_index[0] = 1.0
    return ret_index


def create_train_data(data, ommyou, samples):
    tech_period = {
        'ma_short': 25,
        'ma_long': 75,

        'madr_short': 5,
        'madr_long': 25,

        'macd_short': 12,
        'macd_long': 26,
        'macd_signal': 9,

        'rsi': 14,
        'roc': 12,

        'fast_stoc_k': 5,
        'fast_stoc_d': 3,

        'slow_stoc_k': 15,
        'slow_stoc_d': 3,
        'slow_stoc_sd': 3,
    }

    # ma_short = Technical.moving_average(up_down_rate.values, tech_period['ma_short'])
    # ma_long = Technical.moving_average(up_down_rate.values, tech_period['ma_long'])
    # madr_short = Technical.moving_average_deviation_rate(adj_end, tech_period['madr_short'])
    # madr_long = Technical.moving_average_deviation_rate(adj_end, tech_period['madr_long'])
    # macd, macd_signal = Technical.macd(adj_end,
    #                                    tech_period['macd_short'],
    #                                    tech_period['macd_long'],
    #                                    tech_period['macd_signal'])
    # rsi = Technical.rsi(adj_end, tech_period['rsi'])
    # roc = Technical.roc(adj_end, tech_period['roc'])
    # fast_stoc_k = Technical.stochastic_K(end, high, low, tech_period['fast_stoc_k'])
    # fast_stoc_d = Technical.stochastic_D(end, high, low,
    #                                      tech_period['fast_stoc_k'],
    #                                      tech_period['fast_stoc_d'])
    # slow_stoc_d = Technical.stochastic_D(end, high, low,
    #                                      tech_period['slow_stoc_k'],
    #                                      tech_period['slow_stoc_d'])
    # slow_stoc_sd = Technical.stochastic_slowD(end, high, low,
    #                                           tech_period['slow_stoc_k'],
    #                                           tech_period['slow_stoc_d'],
    #                                           tech_period['slow_stoc_sd'])

    # 先頭のこの日数分のデータは捨てる
    # chop = max(tech_period.values())
    chop = 0

    # 銘柄×日付→日付×銘柄に変換
    _data_transposed = data.transpose()

    _x = []
    _y = []
    # サンプルのデータを学習、 1 サンプルずつ後ろにずらしていく
    length = len(data[0])
    for i in np.arange(chop, length - samples - 1):
        s = i + samples  # samplesサンプル間の変化を素性にする
        features = _data_transposed[i:s]
        # feature = up_down_rate.iloc[i:s]
        # _ma_short = ma_short.iloc[i:s]
        # _ma_long = ma_long.iloc[i:s]
        # _madr_short = madr_short.iloc[i:s]
        # _madr_long = madr_long.iloc[i:s]
        # _macd = macd.iloc[i:s]
        # _macd_signal = macd_signal.iloc[i:s]
        # _rsi = rsi.iloc[i:s]
        # _roc = roc.iloc[i:s]
        # _fast_stoc_k = fast_stoc_k.iloc[i:s]
        # _fast_stoc_d = fast_stoc_d.iloc[i:s]
        # _slow_stoc_d = slow_stoc_d.iloc[i:s]
        # _slow_stoc_sd = slow_stoc_sd.iloc[i:s]

        if ommyou[s] > 0:
            _y.append([1])  # 上がった
        else:
            _y.append([0])  # 上がらなかった
        _x.append(features)
        # _x.append([[
        #     feature.values[x],
        #     # _ma_short.values[x][0],
        #     # _ma_long.values[x][0],
        #     # _madr_short.values[x][0],
        #     # _madr_long.values[x][0],
        #     _macd.values[x][0],
        #     _macd_signal.values[x][0]
        #     # _rsi.values[x][0],
        #     # _roc.values[x][0],
        #     # _fast_stoc_k.values[x][0],
        #     # _fast_stoc_d.values[x][0],
        #     # _slow_stoc_d.values[x][0],
        #     # _slow_stoc_sd.values[x][0]
        # ] for x in range(len(feature.values))])

    # 上げ下げの結果と教師データのセットを返す
    return np.array(_x), np.array(_y)


def create_model():
    model = Sequential()
    model.add(LSTM(hidden_neurons,
                   batch_input_shape=(None, length_of_sequences, 10)))
    model.add(Dropout(0.5))
    model.add(Dense(in_out_neurons))
    model.add(Activation("sigmoid"))

    return model


def print_train_history(history):
    print("Epoch,Acc,Loss,Val acc, Val loss")
    for i in range(len(history.history['acc'])):
        acc = history.history['acc'][i]
        val_acc = history.history['val_acc'][i]
        loss = history.history['loss'][i]
        val_loss = history.history['val_loss'][i]
        print("%d,%f,%f,%f,%f" % (i, acc, loss, val_acc, val_loss))


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


if __name__ == '__main__':

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

    # データを学習用と検証用に分割
    split_pos = int(len(X) * 0.8)
    train_x = X[:split_pos]
    train_y = Y[:split_pos]
    test_x = X[split_pos:]
    test_y = Y[split_pos:]

    # LSTM モデルを作成
    model = create_model()
    model.compile(loss="binary_crossentropy",
                  optimizer="adam", metrics=['accuracy'])
    history = model.fit(train_x, train_y, batch_size=10,
                        epochs=epochs, verbose=1, validation_split=0.2)

    # 学習の履歴
    print_train_history(history)

    # 検証
    score = model.evaluate(test_x, test_y, batch_size=10, verbose=1)
    print("score:", score)

    # 検証(2)
    preds = model.predict(test_x)
    print_predict_result(preds, test_y)
