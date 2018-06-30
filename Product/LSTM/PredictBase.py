# coding: utf-8

import math
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


class PredictBase:

    def __init__(self):
        # 学習に使う日数
        self.training_days = 75

        # LSTM の隠れ層
        self.hidden_neurons = 400

        # EPOCH 回数
        self.epochs = 50

        # 株価変動の閾値
        threshold = 0.01
        self.category_threshold = [-1, -threshold, 0, threshold, 1]

    def load_data(self, date_file, stock_data_files):
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

    def pct_change(self, values):
        returns = pd.Series(values).pct_change()
        returns[0] = 0
        return returns

    def create_train_data(self, adj_starts, high, low, adj_ends, ommyo_rate, y_data, samples):

        udr_start = np.asarray([self.pct_change(v) for v in adj_starts])
        udr_high = np.asarray([self.pct_change(v) for v in high])
        udr_low = np.asarray([self.pct_change(v) for v in low])
        udr_end = np.asarray([self.pct_change(v) for v in adj_ends])

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

    def create_model(self, dimension):
        model = Sequential()
        model.add(LSTM(self.hidden_neurons,
                       use_bias=True,
                       dropout=0.5,
                       recurrent_dropout=0.5,
                       return_sequences=False,
                       batch_input_shape=(None, self.training_days, dimension)))
        model.add(Dropout(0.5))
        model.add(Dense(4))
        model.add(Activation("softmax"))
        model.compile(loss="categorical_crossentropy",
                      optimizer="RMSprop", metrics=['categorical_accuracy'])

        return model

    def print_train_history(self, history):
        print("Epoch, Loss, Val loss, Acc, Val Acc")
        for i in range(len(history.history['loss'])):
            loss = history.history['loss'][i]
            val_loss = history.history['val_loss'][i]
            acc = history.history['categorical_accuracy'][i]
            val_acc = history.history['val_categorical_accuracy'][i]
            print("%d,%f,%f,%f,%f" % (i, loss, val_loss, acc, val_acc))

    def draw_train_history(self, history):
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
        pylab.legend(loc=2)
        pylab.savefig(',train_history_loss.png')

        pylab.clf()
        pylab.plot(xdata, acc, label='acc', color='blue')
        pylab.plot(xdata, val_acc, label='val_acc', color='red')
        pylab.title('Accuracy')
        pylab.xlabel('Epochs')
        pylab.ylabel('value')
        pylab.legend(loc=2)
        pylab.savefig(',train_history_acc.png')
