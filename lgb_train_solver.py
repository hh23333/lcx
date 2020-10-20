import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')


def lgb_train(X_train1, y_train1, X_test1, sub_user_index):
    # 提交结果
    sub = sub_user_index[['客户编号', '开户日期']].copy()
    sub['label'] = 0

    # 训练测试集
    X_train = X_train1.values
    y_train = y_train1.values
    X_test = X_test1.values

    del X_train1, y_train1, X_test1

    print('================================')
    print(X_train.shape)
    print(X_test.shape)
    print('================================')

    xx_logloss = []
    oof_preds = np.zeros(X_train.shape[0])
    N = 5
    skf = StratifiedKFold(n_splits=N, random_state=1024, shuffle=True)

    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': -1,
        'verbose': -1,
    }
    for k, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        print('train _K_ flod', k)

        lgb_train = lgb.Dataset(X_train[train_index], y_train[train_index])
        lgb_evals = lgb.Dataset(X_train[test_index], y_train[test_index], reference=lgb_train)

        lgbm = lgb.train(params, lgb_train, num_boost_round=50000, valid_sets=[lgb_train, lgb_evals],
                         valid_names=['train', 'valid'], early_stopping_rounds=100, verbose_eval=200)

        sub['label'] += lgbm.predict(X_test, num_iteration=lgbm.best_iteration) / N
        oof_preds[test_index] = lgbm.predict(X_train[test_index], num_iteration=lgbm.best_iteration)
        xx_logloss.append(lgbm.best_score['valid']['binary_logloss'])
        print(xx_logloss)
    a = np.mean(xx_logloss)
    a = round(a, 5)
    print(a)

    sub = sub.sort_values(by='label', ascending=False)
    sub = sub.head(50000)
    sub = sub[['客户编号', '开户日期', 'label']]
    sub.to_csv('./res/sub_F12_7.csv', index=False, index_label=False)