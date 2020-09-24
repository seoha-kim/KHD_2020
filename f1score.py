import torch
import torch.nn as nn


class Customf1(nn.Module):
    def __init__(self):
        super(Customf1, self).__init__()

    def forward(self, y, y_pred, num_classes):
        F1scores = []
        for t in range(num_classes):
            tp = (y * y_pred).sum(dim=0).to(torch.float32)
            fp = ((1 - y) * y_pred).sum(dim=0).to(torch.float32)
            fn = (y * (1 - y_pred)).sum(dim=0).to(torch.float32)

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            f1 = 2 * (precision * recall) / (precision + recall)
            F1scores.append(f1)
        loss = sum([(i + 1) * score for i, score in enumerate(F1scores)]) / 10
        return loss