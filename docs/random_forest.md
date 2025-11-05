# Random forest

## Precision-recall curve (init)

![](../reports/figures/rf_precision_recall_curve_init.png)

## Precision-recall curve with thresholds (init)

![](../reports/figures/rf_precision_recall_curve_with_thrs_init.png)

## Confusion matrix (init)

For the threshold 0.001:
* Precision: 0.028 (Among all of the transactions, `~2.829%` are fraud)
* Recall: 0.901 (Model found `~90.111%` fraud transactions)
* Accuracy: 0.904 (Overall accuracy of the model)
* False blocks of transactions: `~9.617%`
* Missed `~9.889%` of fraud transactions

![](../reports/figures/rf_confusion_matrix_init.png)

## Precision-recall curve (best params)

![](../reports/figures/rf_precision_recall_curve_best.png)

## Precision-recall curve with thresholds (best)

![](../reports/figures/rf_precision_recall_curve_with_thrs_best.png)

## Confusion matrix (best)

For the threshold 0.1:
* Precision: 0.047 (Among all of the transactions, `~4.709%` are fraud)
* Recall: 0.967 (Model found `~96.652%` fraud transactions)
* Accuracy: 0.937 (Overall accuracy of the model)
* False blocks of transactions: `~6.331%`
* Missed `~3.348%` of fraud transactions

![](../reports/figures/rf_confusion_matrix_best.png)

## Sklearn tree

![](../reports/figures/rf_sklearn_tree.png)

## Snap feature importance

![](../reports/figures/rf_snap_plot.png)

## dtreeviz tree visualization

![](../reports/figures/random_forest_visual.svg)

## dtreeviz tree visualization (vertical)

![](../reports/figures/random_forest_visual_lr.svg)