from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np


class EvaluationOnType:


    def get_optimal_threshold_from_train_set(self, train_data, type_label):
        # type_label is "tip" or "opinion" or "opinion_with_reason"

        y_true = [int(int(d[f'is_{type_label}']) == 1) for d in train_data]
        y_scores = [d['sentence_types']['flan_t5_xxl'][type_label] for d in train_data]

        # Youden's J statistic: (gives a similar result to simple optimal accuracy)

        # Compute False Positive and True Positive Rates:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)  # , drop_intermediate=False)
        roc_auc = auc(fpr, tpr)

        # Compute Youden's J Statistic
        J_stat = tpr - fpr
        opt_threshold = thresholds[np.argmax(J_stat)]

        return opt_threshold

    def compute_accuracy_on_test_set(self, test_data, type_label, type_theshold):
        # convert to lists of 0 or 1:
        y_true = [int(int(d[f'is_{type_label}']) == 1) for d in test_data]
        y_pred = [int(d['sentence_types']['flan_t5_xxl'][type_label] >= type_theshold) for d in test_data]
        y_scores = [d['sentence_types']['flan_t5_xxl'][type_label] for d in test_data]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # compute the precision_recall scores (without the given threshold), as a way to evaluate the results
        # like in https://dl.acm.org/doi/pdf/10.1145/3437963.3441755
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        # print(list(zip(precisions, recalls, thresholds)))
        # find the closest values for precisions .75, .8, .85, .9:
        prec_rec_vals = {0.5: np.nan, 0.55: np.nan, 0.6: np.nan, 0.65: np.nan, 0.7: np.nan,
                         0.75: np.nan, 0.8: np.nan, 0.85: np.nan, 0.9: np.nan, 0.95: np.nan, 0.99: np.nan}
        for p in prec_rec_vals:
            closest_to_p = min(precisions, key=lambda x: abs(x - p))  # the closest to the needed p
            if abs(closest_to_p - p) < 0.015:  # if the distance is more than .25, then don't use this values
                idx = int(np.where(precisions == closest_to_p)[0])
                prec_rec_vals[p] = recalls[idx]

        # get the precision for Positive predicted values only:
        # keep only the cases where 1 is predicted and then compute precision - this tells us
        # how accurate the model is when saying true
        y_pos_true, y_pos_pred = zip(*[(yt, yp) for yt, yp in zip(y_true, y_pred) if yp == 1])
        prec_pos = precision_score(y_pos_true, y_pos_pred)

        return acc, f1, prec_rec_vals, prec_pos