import numpy as np

from prediction import DecisionTree


def main():
    files = [
        '4517_biofel.txt',
        '4686_js.txt',
        '4839_wowwow.txt',
        '6178_JP.txt',
        '6920_laser.txt',
        '7181_kampo.txt',
        '7182_jpbank.txt',
        '9433_kddi.txt',
        'Nikkei225.txt',
    ]
    spans = [5, 25]
    count = 10

    results = []
    for s in spans:
        for f in files:
            _res = []
            for c in np.arange(count):
                preds = DecisionTree.prediction(f, s)
                print("%d 日, %s" % (s, f), preds)
                _res.append(preds)

            ave = np.average(_res)
            print("%d 日, %s" % (s, f), ave)
            results.append(ave)

    print("結果")
    print('\t'.join([str(r) for r in results]))


if __name__ == '__main__':
    main()
