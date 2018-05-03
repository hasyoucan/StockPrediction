# StockPrediction


## 必要なもの

* Python 3
* venv

## インストール手順

```
sudo pip3 install venv
source mkvenv.src.sh
```

Python のバージョンを確認。
```
python -V
Python 3.6.5
```

必要なパッケージをインストール
```
pip install requests
pip install beautifulsoup4
pip install numpy
pip install sklearn
pip install scipy
pip install pandas
pip install tensorflow
pip install keras
```



## Scraper

株価データをダウンロードします。ダウンロードする銘柄は `targets.txt` で設定します。
`targets.txt` はタブ区切りテキストファイルです。1列目に銘柄コード、2列目に株価データを保存するファイル名を指定します。
`RunScraper.py` を実行すると Scraper が動きます。



## CreateDateFile

複数の株価データをまとめるのに便利なよう、複数の株価データから日付部分を抽出するスクリプトを用意しています。`CreateDateFile.sh` を実行すると、`,date.txt` ファイルが作成されます。



## Prediction

決定木を使って株価の変動を予測します。
Scraper が生成した株価データを用いて、株価の変動を予測します。利用する株価データは `targets.txt` で設定します。
1銘柄につき、短期(5日)、長期(25日) の変動予測を10回行います。10回予測した平均値を 0 以上 1 以下の値で算出します。
平均値が 0 の場合は株価は上昇しない(下がるか横ばい)という予測結果になります。平均値が 1 の場合は株価は上昇するという予測結果になります。
`RunPredict.py` を実行すると Prediction が動きます。


## LSTM

LSTM を使って株価の変動を予測します。
Scraper が生成した株価データを用いて、株価の変動を予測します。利用する株価データは `targets.txt` で設定します。
`LSTM/StockPrediction2.py` を実行すると学習とテストが実行されます。


## LSTM2

LSTM を使って株価の変動を予測します。
Scraper が生成した**複数の**株価データを用いて、株価の変動を予測します。利用する株価データは `targets.txt` で設定します。
`LSTM2/StockPrediction.py` を実行すると学習とテストが実行されます。


## 成績

* LSTM
    * https://qiita.com/deadbeef/items/3966e702a3b361258dfe
    * https://qiita.com/deadbeef/items/49e226b0b9c7236c4875
* LSTM2
    * https://qiita.com/deadbeef/items/196a8af7d5767f6cd01e


## 参考

* http://qiita.com/ynakayama/items/6a472e5ebbe9365186bd
