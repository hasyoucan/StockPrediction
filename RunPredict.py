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
            predicts = []
            scores = []
            for c in np.arange(count):
                predict, score = DecisionTree.prediction(f, s)
                print("%d 日, %s" % (s, f), predict, "Score:", score)
                predicts.append(predict)
                scores.append(score)

            ave_predict = np.average(predicts)
            print("Average: %d 日, %s" % (s, f), ave_predict, "Score:", np.average(scores))
            results.append(ave_predict)

    print("結果")
    print('\t'.join([str(r) for r in results]))


if __name__ == '__main__':
    main()
