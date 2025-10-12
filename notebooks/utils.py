import numpy as np
import pandas as pd
from typing import List
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

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

def thresholds_grid(y_pred_proba, y_test):
    """
    Build thresholds dataframe with scores and confusion matrix results
    :param y_pred_proba: Vector of model's predictions
    :param y_test: Vector of expected results
    """

    grid = np.sort(np.unique(np.round(y_pred_proba, y_test, 3)))
    rows = []

    for t in sorted(set(grid)):
        y_hat = (y_pred_proba >= t).astype(int)
        
        pr = precision_score(y_test, y_hat, zero_division=0)
        rc = recall_score(y_test, y_hat, zero_division=0)
        f1 = f1_score(y_test, y_hat, zero_division=0)
        f05 = fbeta_score(y_test, y_hat, beta=0.5, zero_division=0)
        f2 = fbeta_score(y_test, y_hat, beta=2, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
        
        rows.append((t, pr, rc, f1, f05, f2, tp, fp, fn, tn))

    return pd.DataFrame(
        rows,
        columns=['thr', 'precision', 'recall', 'f1', 'f0.5', 'f2', 'TP', 'FP', 'FN', 'TN']
    ).sort_values('thr')