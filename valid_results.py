from sklearn.metrics import roc_auc_score


def auc(y_test, y_proba):
    res = roc_auc_score(y_test, y_proba)
    print('auc=', res)
    return res