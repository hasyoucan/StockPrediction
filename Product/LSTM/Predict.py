# coding: utf-8

import numpy as np
import pandas as pd

from PredictBase import PredictBase

from keras.callbacks import EarlyStopping


class Predict(PredictBase):

    def __init__(self):
        super().__init__()
        self.draw_graph = False

    def set_draw_graph(self, v):
        self.draw_graph = v

    def predict(self, stock_data_files, target_stock):
        adj_starts, high, low, adj_ends, ommyo_rate = self.load_data(
            stock_data_files)
        _y_data = self.pct_change(adj_starts[target_stock])
        y_data = pd.cut(_y_data, self.category_threshold, labels=False).values

        # 学習データを生成
        X, Y = self.create_train_data(
            adj_starts, high, low, adj_ends, ommyo_rate, y_data, self.training_days)

        # データを学習用と検証用に分割
        train_x = X
        train_y = Y
        test_x = self.__create_prediction_data(
            adj_starts, high, low, adj_ends, ommyo_rate, self.training_days)

        # LSTM モデルを作成
        dimension = len(X[0][0])
        model = self.create_model(dimension)
        es = EarlyStopping(patience=10, verbose=1)
        history = model.fit(train_x, train_y, batch_size=self.batch_size,
                            epochs=self.epochs, verbose=1, validation_split=0.1, callbacks=[es])

        self.print_train_history(history)
        if self.draw_graph:
            self.draw_train_history(history)

        # 検証
        preds = model.predict(test_x)
        self.__print_predict_result(preds)

    def __create_prediction_data(self, adj_starts, high, low, adj_ends, ommyo_rate, samples):

        udr_start = self.pct_change(adj_starts)
        udr_high = self.pct_change(high)
        udr_low = self.pct_change(low)
        udr_end = self.pct_change(adj_ends)

        transposed = np.concatenate(
            (udr_start, udr_high, udr_low, udr_end, ommyo_rate), axis=1)

        _x = []
        # サンプルのデータを学習、1 サンプルずつ後ろにずらしていく
        length = len(udr_end)
        for i in np.arange(length - samples, length - samples + 1):
            s = i + samples  # samplesサンプル間の変化を素性にする
            _x.append(transposed[i:s])

        # 上げ下げの結果と教師データのセットを返す
        return np.array(_x)

    def __print_predict_result(self, preds):
        print("i,predict,,,")
        print("i,0,1,2,3")
        for i in range(0, len(preds)):
            predict = preds[i]
            print("%d, %f,%f,%f,%f" %
                  (i, predict[0], predict[1], predict[2], predict[3]))
