# Logistic Regression

## Precision-recall curve (init)

![curve_init](../data/plots/logreg_precision_recall_curve_init.png)

## Precision-recall curve with thresholds (init)

![curve_thresholds_init](../data//plots/logreg_precision_recall_curve_with_thrs_init.png)

## Precision-recall curve (best params)

![curve_best](../data/plots/logreg_precision_recall_curve_best.png)

## Precision-recall curve with thresholds (best)

![curve_thresholds_best](../data/plots/logreg_precision_recall_curve_with_thrs_best.png)

## Confusion matrix

For the threshold 0.79:
* Precision: 0.027 (Among all of the transactions, `~2.721%` are fraud)
* Recall: 0.913 (Model found `~91.295%` fraud transactions)
* Accuracy: 0.894 (Overall accuracy of the model)
* False blocks of transactions: `~10.567%`
* Missed `~8.705%` of fraud transactions

![conf_matrix](../data/plots/logreg_confusion_matrix_best.png)