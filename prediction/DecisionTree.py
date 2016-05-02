#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# http://qiita.com/ynakayama/items/6a472e5ebbe9365186bd

import sys
import numpy as np
import pandas as pd
from sklearn import tree


def main():
    if len(sys.argv) != 3:
        print("USAGE: DecisionTree.py [FILE] [SPAN]")
        exit()

    file_name = sys.argv[1]
    samples = int(sys.argv[2])

    result = prediction(file_name, samples)
    print(file_name, "上がるぞぉぉぉ" if result == 1 else "下がるぞぉぉぉ")


def prediction(file_name, samples):
    # ファイルを読むのです
    date_, adj_end = load_data(file_name)

    # pandas.Series に変換するのです
    adj_end = convert_data(adj_end)

    # 学習データを作るのです。
    train_x, train_y = create_train_data(adj_end, samples)

    # 決定木のインスタンスを作って学習するのです。
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_x, train_y)

    # 軽く検証するのです
    verify(adj_end, clf, train_y, samples)

    # 予測するのです!!
    last_data = adj_end.ix[len(adj_end) - samples - 1:len(adj_end)].values
    last_dates = date_[len(adj_end) - samples - 1:len(adj_end)]
    # print(list(zip(last_dates, last_data)))
    return clf.predict([last_data])[0]


# ファイルを読み込むのです
def load_data(file_name):
    lines = [line[:-1] for line in open(file_name, 'r', encoding='utf-8')]
    split = [line.split('\t') for line in lines if
             not (line.startswith('#') or len(line) == 0)]
    return [line[0] for line in split], [float(line[6]) for line in split]


# pandas.Series に変換するのです
def convert_data(values):
    returns = pd.Series(values).pct_change()  # 騰落率を出すのです
    ret_index = (1 + returns).cumprod()  # 累積積を出すのです。
    ret_index[0] = 1.0  # 最初の値は 1 なのです。
    return ret_index


# 学習データを作るのです。
def create_train_data(arr, samples):
    train_x = []
    train_y = []
    # サンプルのデータを学習、 1 サンプルずつ後ろにずらしていく
    for i in np.arange(0, len(arr) - samples - 1):
        s = i + samples  # samplesサンプル間の変化を素性にする
        feature = arr.ix[i:s]
        if feature[s - 1] < arr[s]:
            # 上がった
            train_y.append(1)
        else:
            # 上がらなかった
            train_y.append(0)
        train_x.append(feature.values)

    # 上げ下げの結果と教師データのセットを返す
    return np.array(train_x), np.array(train_y)


def verify(arr, clf, train_y, samples):
    test_y = []
    for i in np.arange(0, len(arr) - samples - 1):
        s = i + samples
        test_x = [arr.ix[i:s].values]
        result = clf.predict(test_x)
        test_y.append(result[0])

    for x, y in zip(train_y, np.array(test_y)):
        if x != y:
            print("はわわ…!")


if __name__ == '__main__':
    main()
