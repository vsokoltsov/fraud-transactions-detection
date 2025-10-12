import numpy as np
from typing import List
from sklearn.metrics import confusion_matrix

def pr_at_thresholds(thr_list, thresholds, precision, recall):
    pts = []
    for t in thr_list:
        i = np.searchsorted(thresholds, t)
        i = np.clip(i, 0, len(thresholds)-1)
        p, r = precision[i+1], recall[i+1]
        pts.append((t, p, r, i))
    return pts

def cm_for_threshold(thr, y_pred_proba, y_test):
    y_pred_opt = (y_pred_proba >= thr).astype(int)
    return confusion_matrix(y_test, y_pred_opt)

def explain_confusion_matrix(matrix, thr):
    metrics = []
    mtx_precision = matrix[1][1] / (matrix[1][1] + matrix[0][1])
    mtx_recall = matrix[1][1] / (matrix[1][1] + matrix[1][0])
    mtx_accuracy = (matrix[1][1] + matrix[0][0]) / np.sum(matrix)
    mtx_false_positive_rate = matrix[0][1] / (matrix[0][1] + matrix[0][0])
    mtx_false_negative_rate = matrix[1][0] / (matrix[1][0] + matrix[1][1])
    metrics.extend([
        f"For the threshold {thr}:",
        f"Precision: {mtx_precision:.3f} (Among all of the transactions, ~{(mtx_precision * 100):.3f}% are fraud)",
        f"Recall: {mtx_recall:.3f} (Model found ~{(mtx_recall * 100):.3f}% fraud transactions)",
        f"Accuracy: {mtx_accuracy:.3f} (Overall accuracy of the model)",
        f"False blocks of transactions: ~{(mtx_false_positive_rate * 100):.3f}%",
        f"Missed ~{(mtx_false_negative_rate * 100):.3f}% of fraud transactions"
    ])
    return metrics