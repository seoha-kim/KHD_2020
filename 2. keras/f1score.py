def GetF1score(y, y_pred, target):
        tp = 0
        fp = 0
        fn = 0
        for i, y_hat in enumerate(y_pred):
            if (y[i] == target) and (y_hat == target):
                tp += 1
            if (y[i] == target) and (y_hat != target):
                fn += 1
            if (y[i] != target) and (y_hat == target):
                fp += 1
        f1s = tp / (tp + (fp + fn) / 2)
        return f1s

def CategoricalF1Score(y, y_pred, num_classes):
    F1scores = []
    for t in range(num_classes):
        F1scores.append(GetF1score(y, y_pred, str(t)))
    return F1scores

def WeightedF1Score(y, y_pred, num_classes):
    F1scores = CategoricalF1Score(y, y_pred, num_classes)
    SCORE = 0
    for i, s in enumerate(F1scores):
        SCORE += (i + 1) * s / 10
    SCORE = round(SCORE, 10)
    return SCORE