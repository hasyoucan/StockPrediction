import numpy as np
from sklearn import linear_model as lm
from sklearn import svm
from sklearn import tree

from prediction import Predictor


def main():
    target_file = 'targets.txt'

    lines = [line[:-1] for line in open(target_file, 'r', encoding='utf-8')]
    split = [line.split('\t') for line in lines if
             not (line.startswith('#') or len(line) == 0)]
    files = [line[1] for line in split]
    spans = [5, 25]

    def do_prediction(predictor, count):
        results = []
        score_results = []
        for s in spans:
            for f in files:
                predicts = []
                scores = []
                for _ in np.arange(count):
                    predict, score = predictor.prediction(f, s)
                    print("%d 日, %s, Prediction: %f, Score: %f" % (s, f, predict, score))
                    predicts.append(predict)
                    scores.append(score)

                ave_predict = np.average(predicts)
                ave_score = np.average(scores)
                print("Average: %d 日, %s, Prediction: %f, Score: %f" %
                      (s, f, ave_predict, ave_score))
                results.append(ave_predict)
                score_results.append(ave_score)
        return results, score_results

    print("Decision Tree")
    dt_preds, dt_scores = do_prediction(Predictor.Predictor(tree.DecisionTreeClassifier()), 10)
    print("SVM")
    svm_preds, svm_scores = do_prediction(Predictor.Predictor(svm.SVC(kernel='rbf')), 5)
    print("SGD")
    sgd_preds, sgd_scores = do_prediction(Predictor.Predictor(lm.SGDClassifier()), 10)

    print("Decision Tree")
    print('\t'.join([str(r) for r in dt_preds]))
    print('\t'.join([str(r) for r in dt_scores]))

    print("SVM")
    print('\t'.join([str(r) for r in svm_preds]))
    print('\t'.join([str(r) for r in svm_scores]))

    print("SGD")
    print('\t'.join([str(r) for r in sgd_preds]))
    print('\t'.join([str(r) for r in sgd_scores]))


if __name__ == '__main__':
    main()
