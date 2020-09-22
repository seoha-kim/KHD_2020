import os
import argparse
import numpy as np


class F1Score():
    def __init__(self, y, y_pred, num_classes=4):
        super(F1Score, self).__init__()
        self.y = y
        self.y_pred = y_pred
        self.num_classes = num_classes

    def GetF1score(self):
        tp = 0
        fp = 0
        fn = 0
        for i, y_hat in enumerate(self.y_pred):
            if (self.y[i] == self.num_classes) and (y_hat == self.num_classes):
                tp += 1
            if (self.y[i] == self.num_classes) and (y_hat != self.num_classes):
                fn += 1
            if (self.y[i] != self.num_classes) and (y_hat == self.num_classes):
                fp += 1
        f1s = tp / ( tp + (fp + fn)/2 )
        return f1s

    def CategoricalF1Score(self):
        F1scores = []
        for t in range(self.num_classes):
            F1scores.append(self.GetF1score())
        return F1scores

    def WeightedF1Score(self):
        F1scores = self.CategoricalF1Score()
        SCORE = 0
        for i, s in enumerate(F1scores):
            SCORE += (i + 1) * s / 10
        SCORE = round(SCORE, 10)
        return SCORE