import numpy as np

from prediction import DecisionTree


def main():
    target_file = 'targets.txt'

    lines = [line[:-1] for line in open(target_file, 'r', encoding='utf-8')]
    split = [line.split('\t') for line in lines if
             not (line.startswith('#') or len(line) == 0)]
    files = [line[1] for line in split]
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
