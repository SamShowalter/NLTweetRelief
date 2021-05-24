#################################################################################
#
#             Project Title:  Utilities for Project              Class: CS272
#             Author:         Sam Showalter
#             Date:           2021-05-18
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import numpy as np
import torch

#################################################################################
#   Function-Class Declaration
#################################################################################

def confusion_matrix(y_true, y_pred):
    """
    Fast confusion matrix calculation I found on the web
    Should make computing everything else pretty easy
    """

    N = max(max(y_true), max(y_pred)) + 1
    y_true = torch.tensor(y_true, dtype=torch.long)
    y_pred = torch.tensor(y_pred, dtype=torch.long)
    conf_mat = N * y_true + y_pred
    conf_mat = torch.bincount(conf_mat)
    if len(conf_mat) < N * N:
        conf_mat = torch.cat(conf_mat,
                             torch.zeros(N * N - len(conf_mat),
                                         dtype=torch.long))
    conf_mat = conf_mat.reshape(N, N)
    return conf_mat.detach().cpu().numpy()


def precision_from_cm(conf_mat, per_label = False):
    """TODO: Docstring for precision_from_cm.

    :conf_mat: TODO
    :per_label: TODO
    :returns: TODO

    """
    pred_sums = conf_mat.sum(axis = 1)
    precision = np.zeros(conf_mat.shape[0])
    for i in range(precision.shape[0]):
        precision[i] = conf_mat[i,i]/pred_sums[i]

    if not per_label:
        precision = precision.mean()

    return precision


def recall_from_cm(conf_mat, per_label = False):
    """TODO: Docstring for recall_from_cm.

    :conf_mat: TODO
    :per_label: TODO
    :returns: TODO

    """
    recall = np.zeros(conf_mat.shape[0])
    label_sums = conf_mat.sum(axis =0)
    for i in range(recall.shape[0]):
        recall[i] = conf_mat[i,i]/label_sums[i]

    if not per_label:
        recall = recall.mean()

    return recall

#################################################################################
#   Main Method
#################################################################################

#################################################################################
#   Main Method
#################################################################################

if __name__ == "__main__":
    preds = [1,2,2,2,3,3,3,4,4,5,0,0,0]
    true =[1,2,5,2,3,2,3,4,3,5,0,1,0]
    cm = confusion_matrix(true,preds)
    print(cm)
    print(recall_from_cm(cm, per_label = False))
    print(precision_from_cm(cm, per_label = False))


