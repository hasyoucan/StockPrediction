import numpy as np

from prediction import DecisionTree
from prediction import SVM


def main():
    target_file = 'targets.txt'

    lines = [line[:-1] for line in open(target_file, 'r', encoding='utf-8')]
    split = [line.split('\t') for line in lines if
             not (line.startswith('#') or len(line) == 0)]
    files = [line[1] for line in split]
    spans = [5, 25]
    count = 10

    results = []
    score_results = []

    def do_prediction(label, func):
        for s in spans:
            for f in files:
                predicts = []
                scores = []
                for c in np.arange(count):
                    predict, score = func(f, s)
                    print("%s: %d 日, %s, Prediction: %f, Score: %f" % (label, s, f, predict, score))
                    predicts.append(predict)
                    scores.append(score)

                ave_predict = np.average(predicts)
                ave_score = np.average(scores)
                print("%s: Average: %d 日, %s, Prediction: %f, Score: %f" % (
                    label, s, f, ave_predict, ave_score))
                results.append(ave_predict)
                score_results.append(ave_score)

    do_prediction("DT", DecisionTree.prediction)
    do_prediction("SVM", SVM.prediction)

    print("結果")
    print('\t'.join([str(r) for r in results]))
    print('\t'.join([str(r) for r in score_results]))


if __name__ == '__main__':
    main()
