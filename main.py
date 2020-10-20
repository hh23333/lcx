import pandas as pd
import pickle
from get_data import *
from lgb_train_solver import *

# 数据准备 提速
train_profile_path = './data/train_profile.csv'
train_log_path = './data/train_log.csv'
train_label_path = './data/train_flag.csv'
test_profile_path = './data/A_profile.csv'
test_log_path = './data/A_log.csv'

train_profile_data = pd.read_csv(train_profile_path, sep=',')
train_log_data = pd.read_csv(train_log_path, sep=',')
train_label = pd.read_csv(train_label_path, sep=',')
test_profile_data = pd.read_csv(test_profile_path, sep=',')
test_log_data = pd.read_csv(test_log_path, sep=',')

# 清理数据
train_profile_data, test_profile_data = \
    clean_data(train_profile_data, test_profile_data)
train_log_data, test_log_data = clean_log_data(train_log_data, test_log_data)

# 储存提速
pickle.dump(train_profile_data, open('./cache/train_profile_data.pkl', 'wb'))
pickle.dump(train_log_data, open('./cache/train_log_data.pkl', 'wb'))
pickle.dump(train_label, open('./cache/train_label.pkl', 'wb'))
pickle.dump(test_profile_data, open('./cache/test_profile_data.pkl', 'wb'))
pickle.dump(test_log_data, open('./cache/test_log_data.pkl', 'wb'))


ignore_feat = ['标签', '客户编号', '开户日期']

"""
train set 1
"""

label_date = '2016-03'
train_start_date = '2015-01'
train_end_date = '2016-03'
test_date = '2016-04'

training_data = get_train_data(label_date, train_start_date, train_end_date)
test_data = get_test_data(test_date)

# train
feats = [f for f in training_data.columns if f not in ignore_feat]
print(feats)
print(len(feats))
label = training_data['label'].copy()
train = training_data[feats].copy()

# test
feats = [f for f in test_data.columns if f not in ignore_feat]
print(feats)
print(len(feats))
sub_user_index = test_data[['客户标签']].copy()
test = test_data[feats].copy()
print('test shape: ', test.shape)

lgb_train(train, label, test, sub_user_index)
