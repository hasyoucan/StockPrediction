# StockPrediction


## 必要なもの

* Python 3
* pyenv


## インストール手順

```
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
pyenv install 3.6.6
source mkvenv.src.sh
```

必要なパッケージをインストール
```
pip install requests
pip install beautifulsoup4
pip install --no-wheel numpy
pip install sklearn
pip install scipy
pip install pandas
pip install tensorflow
pip install keras
pip install matplotlib
```


## `mkvenv.src.sh`

virtualenv に `StockPrediction` という名前で環境を作成します。すでに環境がある場合は `activate` します。本リポジトリにある Python プログラムを動かす前に、 `source mkvenv.src.sh` します。


## Scraper

株価データをダウンロードします。ダウンロードする銘柄は `targets.txt` で設定します。
`targets.txt` はタブ区切りテキストファイルです。1列目に銘柄コード、2列目に株価データを保存するファイル名を指定します。
`RunScraper.sh` を実行すると Scraper が動きます。



## LSTM6

LSTM を使って株価の変動を予測します。
Scraper が生成した株価データを用いて、株価の変動を予測します。
`LSTM6/StockPrediction.py` を実行すると学習とテストが実行されます。



## Product/LSTM

LSTM を使って株価の変動を予測します。
Scraper が生成した株価データを用いて、株価の変動を予測します。
`RunLstmPredict.sh` を実行すると翌日の値動きを予測します。


## junk

現状までの格闘の記録です。結果がイマイチなものが全てここに詰まっています。


## 成績

* https://qiita.com/deadbeef/items/3966e702a3b361258dfe
* https://qiita.com/deadbeef/items/49e226b0b9c7236c4875
* https://qiita.com/deadbeef/items/196a8af7d5767f6cd01e
* https://qiita.com/deadbeef/items/be3252538de2f5684f86
* https://qiita.com/deadbeef/items/30d031ff88b0c4e879cc
* https://qiita.com/deadbeef/items/762e1f01ae30bcbb529c
* https://qiita.com/deadbeef/items/8831a34990da7f84304b


## 参考

* http://qiita.com/ynakayama/items/6a472e5ebbe9365186bd
