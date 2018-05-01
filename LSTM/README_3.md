# LSTM で株価予測(その3)

[LSTM で株価予測してみる (その2)](http://qiita.com/deadbeef/items/49e226b0b9c7236c4875) で予測がいまいちだった原因について調査しました。

## 学習曲線

学習に使うデータサイズを変えながら、精度がどのように変化するかをプロットしてみます。`epoch` は 20、`hidden_neurons` は、128、64、8、2 と変化させます。
株価データは、今回は KDDI の時系列データを使います。

### hidden_neurons=128

縦軸が精度、横軸がサンプルの数です。`Acc` が訓練データの精度、`Val Acc` が検証データの精度です。

![hidden_neurons=128](https://qiita-image-store.s3.amazonaws.com/0/66283/9d52746c-511f-cc35-0d78-01865f334c1b.png)


### hidden_neurons=64

![hidden_neurons=64](https://qiita-image-store.s3.amazonaws.com/0/66283/2864f934-2a96-6b09-316d-91e2e54f1d25.png)



### hidden_neurons=8

![hidden_neurons=8](https://qiita-image-store.s3.amazonaws.com/0/66283/e4589aab-370b-e62e-120b-73e2af4d51b9.png)


### hidden_neurons=2

![hidden_neurons=2](https://qiita-image-store.s3.amazonaws.com/0/66283/89881c13-c889-47ab-bf0a-e255e8f2610f.png)




## Precision

LSTM の `hidden_neurons` を下げていったところ、`hidden_neurons=8` ぐらいから訓練データの精度と検証データの精度の乖離が小さくなりました。`hidden_neurons=8` で全時系列データで予測し、Precision を出します。

```
Precision: 0.5 , Recall: 0.009578544061302681 , F: 0.018796992481203006
Precision: 0.125 , Recall: 0.0019157088122605363 , F: 0.0037735849056603774
Precision: 0.8 , Recall: 0.007662835249042145 , F: 0.015180265654648957
Precision: 0.5 , Recall: 0.05938697318007663 , F: 0.10616438356164383
Precision: 0.5 , Recall: 0.032567049808429116 , F: 0.061151079136690635
```

Precision がまったく安定しません。これは、true-positive、false-positive が少ないのが原因です。何回か実行すると、true-positive、false-positive が両方とも 0 になることもあります。

```
true_positive: 17
true_negative: 604
false_positive: 17
false_negative: 505

...

true_positive: 0
true_negative: 621
false_positive: 0
false_negative: 522

...

true_positive: 4
true_negative: 614
false_positive: 7
false_negative: 518
```


## まとめ

LSTM の `hideen_neurons` を下げていったところ、訓練データの精度と検証データの精度の乖離が小さくなりましたが、Precision には大きな変化がありませんでした。結局現状のデータではこれが限界なのかもしれません。


## 参考

* [そのモデル、過学習してるの？未学習なの？と困ったら](http://chezou.hatenablog.com/entry/2016/05/29/215739)