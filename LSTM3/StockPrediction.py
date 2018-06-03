# coding: utf-8

import numpy as np
import pandas as pd

import Technical
from MultiLoader import MultiLoader

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn import metrics
# from sklearn import cross_validation as cv


hidden_neurons = 128
length_of_sequences = 25
in_out_neurons = 3

epochs = 50


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
    # 累積席を出すのです。
    ret_index = (1 + returns).cumprod()
    # 最初は 1 なのです。
    ret_index[0] = 1.0
    return ret_index


def create_train_data(high, low, end, adj_ends, up_down_rate, y_data, samples):
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

    ma_short = np.asarray([Technical.moving_average(
        v, tech_period['ma_short'])[0].values for v in up_down_rate])
    ma_long = np.asarray([Technical.moving_average(
        v, tech_period['ma_long'])[0].values for v in up_down_rate])
    madr_short = np.asarray([Technical.moving_average_deviation_rate(
        v, tech_period['madr_short'])[0].values for v in adj_ends])
    madr_long = np.asarray([Technical.moving_average_deviation_rate(
        v, tech_period['madr_long'])[0].values for v in adj_ends])
    macd = np.asarray([Technical.macd(v,
                                      tech_period['macd_short'],
                                      tech_period['macd_long'],
                                      tech_period['macd_signal'])[0][0].values for v in adj_ends])
    macd_signal = np.asarray([Technical.macd(v,
                                             tech_period['macd_short'],
                                             tech_period['macd_long'],
                                             tech_period['macd_signal'])[1][0].values for v in adj_ends])
    rsi = np.asarray([Technical.rsi(v, tech_period['rsi'])[
                     0].values for v in adj_ends])
    roc = np.asarray([Technical.roc(v, tech_period['roc'])[
                     0].values for v in adj_ends])

    fast_stoc_k = []
    fast_stoc_d = []
    slow_stoc_d = []
    slow_stoc_sd = []
    for i, e in enumerate(end):
        h = high[i]
        l = low[i]
        fast_stoc_k.append(Technical.stochastic_K(
            e, h, l, tech_period['fast_stoc_k'])[0].values)
        fast_stoc_d.append(Technical.stochastic_D(e, h, l,
                                                  tech_period['fast_stoc_k'],
                                                  tech_period['fast_stoc_d'])[0].values)
        slow_stoc_d.append(Technical.stochastic_D(e, h, l,
                                                  tech_period['slow_stoc_k'],
                                                  tech_period['slow_stoc_d'])[0].values)
        slow_stoc_sd.append(Technical.stochastic_slowD(e, h, l,
                                                       tech_period['slow_stoc_k'],
                                                       tech_period['slow_stoc_d'],
                                                       tech_period['slow_stoc_sd'])[0].values)
    fast_stoc_k = np.asarray(fast_stoc_k)
    fast_stoc_d = np.asarray(fast_stoc_d)
    slow_stoc_d = np.asarray(slow_stoc_d)
    slow_stoc_sd = np.asarray(slow_stoc_sd)

    # 先頭のこの日数分のデータは捨てる
    # chop = max(tech_period.values())
    chop = 0

    # 銘柄×日付→日付×銘柄に変換
    transposed = up_down_rate.transpose()
    # transposed = rsi.transpose()
    # transposed = roc.transpose()
    # transposed = np.concatenate((macd, macd_signal)).transpose()
    # transposed = np.concatenate((ma_short, ma_long)).transpose()
    # transposed = np.concatenate((madr_short, madr_long)).transpose()
    # transposed = fast_stoc_k.transpose()
    # transposed = np.concatenate((fast_stoc_k, fast_stoc_d)).transpose()
    # transposed = np.concatenate((slow_stoc_d, slow_stoc_sd)).transpose()

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


if __name__ == '__main__':

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

    # データを学習用と検証用に分割
    split_pos = int(len(X) * 0.8)
    train_x = X[:split_pos]
    train_y = Y[:split_pos]
    test_x = X[split_pos:]
    test_y = Y[split_pos:]

    # LSTM モデルを作成
    dimension = len(X[0][0])
    model = create_model(dimension)
    es = EarlyStopping(monitor='loss', patience=10, verbose=1)
    history = model.fit(train_x, train_y, batch_size=10,
                        epochs=epochs, verbose=1, validation_split=0.2, callbacks=[es])

    # 学習の履歴
    print_train_history(history)

    # 検証
    score = model.evaluate(test_x, test_y, batch_size=10, verbose=1)
    print("score:", score)

    # 検証(2)
    initial_value = adj_starts[stock_data_files.index(',6501.txt')][0]
    preds = model.predict(test_x)
    print_predict_result(preds, test_y, initial_value)
