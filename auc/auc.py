import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    fpr = np.array(fpr)
    tpr = np.array(tpr)

    M = len(fpr)

    result = 0.0

    for i in range(M - 1):
        result += ((tpr[i] + tpr[i+1]) / 2) * (fpr[i+1] - fpr[i])

    return result