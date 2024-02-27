from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import pandas as pd

# eer reference: https://stackoverflow.com/questions/28339746/equal-error-rate-in-python

# TODO need to check which eer is the correct one


# def equal_error_rate(y_true, y_pred):
#     fpr, tpr, thresholds = roc_curve(y_true, y_pred)
#     eer = 1 - tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
#     return eer


# faster but might be wrong?
def equal_error_rate(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return eer


def false_reject_rate(y_true, y_pred, zero_division=0.0):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if fn + tp == 0:
        return zero_division
    return fn / (fn + tp)


def false_accept_rate(y_true, y_pred, zero_division=0.0):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if fn + tp == 0:
        return zero_division
    return fp / (fp + tn)


def evaluate_model(y_true, y_pred):
    dict_scores = {
        "eer": equal_error_rate(y_true, y_pred),
        "frr": false_reject_rate(y_true, y_pred, zero_division=0.0),
        "far": false_accept_rate(y_true, y_pred, zero_division=0.0),
        "acc": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0.0),
        "precision": precision_score(y_true, y_pred, zero_division=0.0),
        "f1": f1_score(y_true, y_pred, zero_division=0.0),
    }
    return dict_scores


def evaluation_to_df(dict_evaluation):
    first_element = next(iter(dict_evaluation["train"].values()))
    avg_scores = {
        "train": {
            metric: sum(
                dict_evaluation["train"][key][metric]
                for key in dict_evaluation["train"]
            )
            / len(dict_evaluation["train"])
            for metric in first_element
        },
        "test": {
            metric: sum(
                dict_evaluation["test"][key][metric] for key in dict_evaluation["test"]
            )
            / len(dict_evaluation["test"])
            for metric in first_element
        },
    }

    df = pd.DataFrame(avg_scores).T
    return df
