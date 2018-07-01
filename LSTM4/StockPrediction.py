#! /usr/bin/env python3

# coding: utf-8

import math
import argparse
import numpy as np
import pandas as pd

from MultiLoader import MultiLoader

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import matplotlib
matplotlib.use('Agg')
import pylab


hidden_neurons = 400
training_days = 75

epochs = 200
batch_size = 256

threshold = 0.01
category_threshold = [-1, -threshold, 0, threshold, 1]


def load_data(date_file, stock_data_files):
    """
    Scraper が吐き出したファイルを読むのです。
    日付と調整後終値を返すのです。
    """
    multi_loader = MultiLoader(date_file, stock_data_files)

    adj_starts = multi_loader.extract('adj_start')
    high = multi_loader.extract('high')
    low = multi_loader.extract('low')
    adj_ends = multi_loader.extract('adj_end')
    ommyo_rate = multi_loader.extract('ommyo_rate')

    return (adj_starts, high, low, adj_ends, ommyo_rate)


# def rate_of_decline(values):
#     # 騰落率を出すのです。
#     returns = pd.Series(values).pct_change()
#     # 累積席を出すのです。
#     ret_index = (1 + returns).cumprod()
#     # 最初は 1 なのです。
#     ret_index[0] = 1.0
#     return ret_index


def pct_change(values):
    returns = pd.Series(values).pct_change()
    returns[0] = 0
    return returns


# def log_diff(values):
#     series = pd.Series(values)
#     # 全要素の対数を出す
#     log_values = series.apply(math.log10)
#     # 対数の差を出す
#     ret_val = log_values.diff()
#     # 初期値は 0
#     ret_val[0] = 0.0
#     return ret_val


def create_train_data(adj_starts, high, low, adj_ends, ommyo_rate, y_data, samples):

    udr_start = np.asarray([pct_change(v) for v in adj_starts])
    udr_high = np.asarray([pct_change(v) for v in high])
    udr_low = np.asarray([pct_change(v) for v in low])
    udr_end = np.asarray([pct_change(v) for v in adj_ends])

    # 銘柄×日付→日付×銘柄に変換
    # transposed = udr_start.transpose()
    # transposed = udr_end.transpose()
    transposed = np.concatenate(
        (udr_start, udr_high, udr_low, udr_end, ommyo_rate)).transpose()

    _x = []
    _y = []
    # サンプルのデータを学習、1 サンプルずつ後ろにずらしていく
    length = len(udr_end[0])
    for i in np.arange(0, length - samples):
        s = i + samples  # samplesサンプル間の変化を素性にする
        _x.append(transposed[i:s])
        __y = [0, 0, 0, 0]
        __y[y_data[s]] = 1
        _y.append(__y)

    # 上げ下げの結果と教師データのセットを返す
    return np.array(_x), np.array(_y)


def create_model(dimension):
    model = Sequential()
    model.add(LSTM(hidden_neurons,
                   use_bias=True,
                   dropout=0.5,
                   recurrent_dropout=0.5,
                   return_sequences=False,
                   batch_input_shape=(None, training_days, dimension)))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="RMSprop", metrics=['categorical_accuracy'])

    return model


def print_train_history(history):
    print("Epoch, Loss, Val loss, Acc, Val Acc")
    for i in range(len(history.history['loss'])):
        loss = history.history['loss'][i]
        val_loss = history.history['val_loss'][i]
        acc = history.history['categorical_accuracy'][i]
        val_acc = history.history['val_categorical_accuracy'][i]
        print("%d,%f,%f,%f,%f" % (i, loss, val_loss, acc, val_acc))


def draw_train_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    xdata = np.arange(0, len(loss))

    pylab.clf()
    pylab.plot(xdata, loss, label='loss', color='blue')
    pylab.plot(xdata, val_loss, label='val_loss', color='red')
    pylab.title('Loss')
    pylab.xlabel('Epochs')
    pylab.ylabel('value')
    pylab.legend(loc='upper right')
    pylab.savefig(',train_history_loss.png')

    pylab.clf()
    pylab.plot(xdata, acc, label='acc', color='blue')
    pylab.plot(xdata, val_acc, label='val_acc', color='red')
    pylab.title('Accuracy')
    pylab.xlabel('Epochs')
    pylab.ylabel('value')
    pylab.legend(loc='lower right')
    pylab.savefig(',train_history_acc.png')


def print_predict_result(preds, test_y):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(preds)):
        predict = np.argmax(preds[i])
        test = np.argmax(test_y[i])
        positive = True if predict == 2 or predict == 3 else False
        true = True if test == 2 or test == 3 else False
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


def test_predict(stock_data_files, target_stock, date_file):
    adj_starts, high, low, adj_ends, ommyo_rate = load_data(
        date_file, stock_data_files)

    _y_data = pct_change(adj_starts[stock_data_files.index(target_stock)])
    # _y_data = pct_change(adj_ends[stock_data_files.index(target_stock)])
    # _y_data = ommyo_rate[stock_data_files.index(target_stock)]
    y_data = pd.cut(_y_data, category_threshold, labels=False)

    # 学習データを生成
    X, Y = create_train_data(
        adj_starts, high, low, adj_ends, ommyo_rate, y_data, training_days)

    # データを学習用と検証用に分割
    split_pos = int(len(X) * 0.8)
    train_x = X[:split_pos]
    train_y = Y[:split_pos]
    test_x = X[split_pos:]
    test_y = Y[split_pos:]

    # LSTM モデルを作成
    dimension = len(X[0][0])
    model = create_model(dimension)
    es = EarlyStopping(patience=10, verbose=1)
    history = model.fit(train_x, train_y, batch_size=batch_size,
                        epochs=epochs, verbose=1, validation_split=0.2, callbacks=[es])

    # 学習の履歴
    print_train_history(history)
    draw_train_history(history)

    # 検証
    preds = model.predict(test_x)
    print_predict_result(preds, test_y)



###############################################################################
if __name__ == '__main__':

    target_stock = ',TOPIX.txt'
    stock_data_files = [
        ',Nikkei225.txt', ',TOPIX.txt', ',6501.txt'
    ]
    date_file = ',date.txt'

    test_predict(stock_data_files, target_stock, date_file)
