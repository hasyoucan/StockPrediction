# coding: utf-8

import numpy as np
import pandas as pd

from PredictBase import PredictBase

from keras.callbacks import EarlyStopping


class TestValidate(PredictBase):

    def __init__(self):
        super().__init__()
        self.draw_graph = False

    def set_draw_graph(self, v):
        self.draw_graph = v

    def test_predict(self, stock_data_files, target_stock):
        adj_starts, high, low, adj_ends, ommyo_rate = self.load_data(
            stock_data_files)

        _y_data = self.pct_change(adj_starts[target_stock])
        # y_data = pct_change(adj_ends[stock_data_files.index(target_stock)])
        # y_data = ommyo_rate[stock_data_files.index(target_stock)]
        y_data = pd.cut(_y_data, self.category_threshold, labels=False).values

        # 学習データを生成
        X, Y = self.create_train_data(
            adj_starts, high, low, adj_ends, ommyo_rate, y_data, self.training_days)

        # データを学習用と検証用に分割
        split_pos = int(len(X) * 0.9)
        train_x = X[:split_pos]
        train_y = Y[:split_pos]
        test_x = X[split_pos:]
        test_y = Y[split_pos:]

        # LSTM モデルを作成
        dimension = len(X[0][0])
        model = self.create_model(dimension)
        es = EarlyStopping(patience=10, verbose=1)
        history = model.fit(train_x, train_y, batch_size=self.batch_size,
                            epochs=self.epochs, verbose=1, validation_split=0.1, callbacks=[es])

        # 学習の履歴
        self.print_train_history(history)
        if self.draw_graph:
            self.draw_train_history(history)

        # 検証
        preds = model.predict(test_x)
        self.__print_predict_result(preds, test_y)

    def __print_predict_result(self, preds, test_y):
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
        print("Precision = %f, Recall = %f, F = %f" %
              (precision, recall, f_value))
